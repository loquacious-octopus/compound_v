"""Production ensemble helpers for prompt analysis, code generation, and repair."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from miner_reference.ensemble_tracks.candidate_ensemble import candidate_ensemble_track
from miner_reference.ensemble_tracks.direct_vision import (
    build_track_spec as build_direct_vision_track,
    validator_repair_prompt as direct_vision_validator_repair_prompt,
)
from miner_reference.ensemble_tracks.object_identity import build_track as build_object_identity_track
from miner_reference.ensemble_tracks.structured_repair import make_track_spec as make_structured_repair_track
from miner_reference.local_harness import (
    ChutesClient,
    ChutesCritic,
    ChutesSettings,
    HarnessPaths,
    LocalRenderService,
    PromptSpec,
    _code_system_prompt,
    _extract_js_module,
    _image_data_url,
    _parse_jsonish_or_raw,
    _summarize_critique,
    download_prompt,
    prompt_from_url,
    run_validator,
)


ROUND0_BASE = ""
BASELINE_REPO = ""
DEFAULT_VISION_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
DEFAULT_CODE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"


@dataclass(frozen=True)
class TrackSpec:
    """One modeling direction to test against the baseline."""

    name: str
    description: str
    candidates: int
    analysis_prompt: str
    code_prompt: str
    repair_prompt: str | None = None
    use_retrieved_js: bool = False
    use_vision_for_code: bool = False


@dataclass
class CandidateResult:
    prompt_stem: str
    track: str
    candidate_id: str
    js_path: str
    render_path: str | None
    validator_passed: bool
    render_passed: bool
    critique_score: float
    pairwise_vs_baseline: str | None
    pairwise_vs_public: str | None
    summary: dict[str, Any]
    error: str | None = None


def default_tracks() -> list[TrackSpec]:
    return [
        TrackSpec(
            name="baseline_qwen25",
            description="Two-stage planner/coder baseline using Qwen2.5-compatible models.",
            candidates=1,
            analysis_prompt=(
                "Create a compact object structural description for procedural Three.js reconstruction. "
                "Identify object identity, camera/view, silhouette, symmetry, primary volumes, secondary "
                "parts, attachments, material zones, colors, and must-have details. Return strict JSON."
            ),
            code_prompt=(
                "Generate one validator-safe raw Three.js module from the structural description. "
                "Favor the correct object identity, major silhouette, proportions, colors, and 3-6 defining "
                "details. Keep code simple and robust."
            ),
        ),
        TrackSpec(
            name="winner_plus_direct",
            description="Direct winner-style generation with denser scene brief and stronger anti-generic instructions.",
            candidates=2,
            analysis_prompt=(
                "Return strict JSON for a low-poly procedural reconstruction plan. Include: object_identity, "
                "negative_identity_checks, camera, silhouette_axes, symmetry, part_graph with parent/child "
                "attachments, material_zones, color_palette, distinctive_details, and validator_safe_geometry_plan. "
                "Be concrete: name every visible appendage, rim, hole, support, handle, wheel, label, seam, or stripe."
            ),
            code_prompt=(
                "Generate a validator-compliant raw Three.js module. Do not collapse the object into one or two "
                "generic primitives. Build named groups for body, appendages, repeated details, and material zones. "
                "Use Box/Cylinder/Sphere/Cone/Torus/Tube/Lathe/Extrude/DataTexture only when useful and safe. "
                "The final render should be recognizable from multiple views."
            ),
            repair_prompt=(
                "Patch the JS with real structural changes. If the critique mentions missing or wrong parts, add, "
                "remove, resize, or reposition geometry. Do not only rename variables, comments, or colors."
            ),
        ),
        make_structured_repair_track(TrackSpec),
        _with_candidate_count(candidate_ensemble_track(TrackSpec), 3),
        build_object_identity_track(TrackSpec),
        build_direct_vision_track(TrackSpec),
    ]


def run_adaptation(
    *,
    project_root: Path,
    runs_dir: Path,
    limit: int,
    seed: int,
    tracks: list[str] | None,
    render_port: int,
    render_static_port: int,
    validator_timeout_seconds: float,
    download_timeout_seconds: float,
) -> Path:
    project_root = project_root.resolve()
    repo_root = project_root.parent
    run_id = datetime.now(UTC).strftime("ensemble-%Y%m%dT%H%M%SZ")
    root = runs_dir / run_id
    if root.exists():
        shutil.rmtree(root)
    paths = HarnessPaths.create(root)
    for extra in ("public", "reports"):
        (root / extra).mkdir(parents=True, exist_ok=True)

    selected = [t for t in default_tracks() if tracks is None or t.name in tracks]
    if not selected:
        raise RuntimeError("No tracks selected")

    _ensure_chutes_defaults()
    settings = ChutesSettings.from_env_require(vision=True, code=True)
    client = ChutesClient(settings)
    critic = ChutesCritic(settings=settings)
    rules = "\n\n".join(
        path.read_text(encoding="utf-8")
        for path in (project_root / "output_specifications.md", project_root / "runtime_specifications.md")
        if path.exists()
    )
    example = ""
    prompts = _round0_baseline_prompts(limit=limit)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "baseline_repo": BASELINE_REPO,
        "vision_model": settings.vision_model,
        "code_model": settings.code_model,
        "seed": seed,
        "tracks": [asdict(t) for t in selected],
        "prompt_count": len(prompts),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results: list[CandidateResult] = []
    start = time.monotonic()
    renderer = LocalRenderService(repo_root=repo_root, port=render_port, static_port=render_static_port)
    renderer.start()
    try:
        for index, prompt in enumerate(prompts, start=1):
            print(f"[ensemble] prompt {index}/{len(prompts)} {prompt.stem}", flush=True)
            prompt_path = paths.prompts / f"{prompt.stem}{prompt.extension}"
            ok, error = download_prompt(prompt, prompt_path, timeout_seconds=download_timeout_seconds)
            if not ok:
                print(f"[ensemble] download failed {prompt.stem}: {error}", flush=True)
                continue
            public = _download_public_artifacts(root=root, prompt=prompt)
            if public.get("grid") is None and public.get("js") is not None:
                public["grid"] = _render_public_js(renderer=renderer, public=public, prompt=prompt)

            baseline_result: CandidateResult | None = None
            for track in selected:
                track_results = _run_track_for_prompt(
                    project_root=project_root,
                    paths=paths,
                    prompt=prompt,
                    prompt_path=prompt_path,
                    public=public,
                    track=track,
                    seed=seed,
                    client=client,
                    settings=settings,
                    critic=critic,
                    renderer=renderer,
                    rules=rules,
                    example=example,
                    baseline=baseline_result,
                    validator_timeout_seconds=validator_timeout_seconds,
                )
                results.extend(track_results)
                if track.name == "baseline_qwen25":
                    baseline_result = _best_result(track_results)
            _write_partial_summary(root=root, manifest=manifest, results=results, elapsed=time.monotonic() - start)
    finally:
        renderer.stop()

    summary = _write_partial_summary(root=root, manifest=manifest, results=results, elapsed=time.monotonic() - start)
    _write_markdown_report(root / "reports" / "summary.md", summary)
    return root


def _run_track_for_prompt(
    *,
    project_root: Path,
    paths: HarnessPaths,
    prompt: PromptSpec,
    prompt_path: Path,
    public: dict[str, Path | None],
    track: TrackSpec,
    seed: int,
    client: ChutesClient,
    settings: ChutesSettings,
    critic: ChutesCritic,
    renderer: LocalRenderService,
    rules: str,
    example: str,
    baseline: CandidateResult | None,
    validator_timeout_seconds: float,
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    analysis = _analyze_prompt(client=client, settings=settings, prompt=prompt, prompt_path=prompt_path, track=track, seed=seed)
    analysis_dir = paths.analysis / track.name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / f"{prompt.stem}.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    retrieved_js = _read_text(public.get("js")) if track.use_retrieved_js else None
    for index in range(1, track.candidates + 1):
        candidate_id = f"{track.name}_{index:02d}"
        candidate_dir = paths.candidates / prompt.stem / track.name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        js_path = candidate_dir / f"{candidate_id}.js"
        render_path = paths.renders / f"{prompt.stem}.{candidate_id}.png"
        try:
            source = _generate_source(
                client=client,
                settings=settings,
                prompt=prompt,
                track=track,
                analysis=analysis,
                prompt_path=prompt_path,
                seed=seed,
                candidate_index=index,
                rules=rules,
                example=example,
                retrieved_js=retrieved_js,
            )
            js_path.write_text(source, encoding="utf-8")
            validator_result = run_validator(project_root, js_path, timeout_seconds=validator_timeout_seconds)
            passed = bool(validator_result.get("passed"))
            if not passed:
                source, js_path, validator_result = _attempt_validator_repair(
                    project_root=project_root,
                    paths=paths,
                    prompt=prompt,
                    track=track,
                    source=source,
                    validator_result=validator_result,
                    seed=seed,
                    client=client,
                    settings=settings,
                    rules=rules,
                    example=example,
                    validator_timeout_seconds=validator_timeout_seconds,
                    candidate_id=candidate_id,
                )
                passed = bool(validator_result.get("passed"))
            render_passed = False
            critique: dict[str, Any] = {}
            score = 0.0
            if passed:
                renderer.render_grid(source, render_path)
                render_passed = render_path.exists()
                if render_passed:
                    critique = critic.critique(prompt=prompt, prompt_path=prompt_path, render_path=render_path)
                    score = _score(critique)
                    if track.repair_prompt and score < 6.0:
                        source, js_path, render_path, validator_result, render_passed, critique, score = _attempt_repair(
                            project_root=project_root,
                            paths=paths,
                            prompt=prompt,
                            prompt_path=prompt_path,
                            track=track,
                            source=source,
                            critique=critique,
                            seed=seed,
                            client=client,
                            settings=settings,
                            renderer=renderer,
                            rules=rules,
                            example=example,
                            validator_timeout_seconds=validator_timeout_seconds,
                            candidate_id=candidate_id,
                        )
                        passed = bool(validator_result.get("passed"))
            pairwise_baseline = None
            if baseline and baseline.render_path and render_passed:
                pairwise_baseline = _pairwise_judge(
                    client=client,
                    settings=settings,
                    prompt=prompt,
                    prompt_path=prompt_path,
                    a_path=Path(baseline.render_path),
                    b_path=render_path,
                    a_label="baseline_qwen25",
                    b_label=track.name,
                )
            pairwise_public = None
            public_render = public.get("grid")
            if public_render and render_passed:
                pairwise_public = _pairwise_judge(
                    client=client,
                    settings=settings,
                    prompt=prompt,
                    prompt_path=prompt_path,
                    a_path=public_render,
                    b_path=render_path,
                    a_label="public_baseline",
                    b_label=track.name,
                )
            results.append(
                CandidateResult(
                    prompt_stem=prompt.stem,
                    track=track.name,
                    candidate_id=candidate_id,
                    js_path=str(js_path),
                    render_path=str(render_path) if render_passed else None,
                    validator_passed=passed,
                    render_passed=render_passed,
                    critique_score=score,
                    pairwise_vs_baseline=pairwise_baseline,
                    pairwise_vs_public=pairwise_public,
                    summary=_summarize_critique(critique) if critique else {"validator": validator_result},
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                CandidateResult(
                    prompt_stem=prompt.stem,
                    track=track.name,
                    candidate_id=candidate_id,
                    js_path=str(js_path),
                    render_path=None,
                    validator_passed=False,
                    render_passed=False,
                    critique_score=0.0,
                    pairwise_vs_baseline=None,
                    pairwise_vs_public=None,
                    summary={},
                    error=str(exc),
                )
            )
    return results


def _analyze_prompt(
    *,
    client: ChutesClient,
    settings: ChutesSettings,
    prompt: PromptSpec,
    prompt_path: Path,
    track: TrackSpec,
    seed: int,
) -> dict[str, Any]:
    result = client.chat_complete(
        model=settings.vision_model,
        messages=[
            {"role": "system", "content": "You are a visual planner for procedural low-poly Three.js reconstruction. Return strict JSON only."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{track.analysis_prompt}\nSeed: {seed}. Prompt stem: {prompt.stem}."},
                    {"type": "image_url", "image_url": {"url": _image_data_url(prompt_path)}},
                ],
            },
        ],
        temperature=0.05,
        max_tokens=int(os.environ.get("ENSEMBLE_ANALYSIS_MAX_TOKENS", "1024")),
        label=f"{track.name}.analysis",
    )
    parsed = _parse_jsonish_or_raw(result.content)
    return parsed if isinstance(parsed, dict) else {"raw": result.content}


def _generate_source(
    *,
    client: ChutesClient,
    settings: ChutesSettings,
    prompt: PromptSpec,
    track: TrackSpec,
    analysis: dict[str, Any],
    prompt_path: Path,
    seed: int,
    candidate_index: int,
    rules: str,
    example: str,
    retrieved_js: str | None,
) -> str:
    retrieval_block = ""
    if retrieved_js:
        retrieval_block = f"\nPrior public JS reference for the same prompt. Use only as structural inspiration:\n```js\n{retrieved_js[:12000]}\n```\n"
    variant_block = _variant_instruction(track=track, candidate_index=candidate_index)
    user = f"""Generate candidate {candidate_index} for track `{track.name}`.

Track goal:
{track.description}

Track coding instruction:
{track.code_prompt}

Candidate-specific style:
{variant_block}

Seed: {seed}
Prompt stem: {prompt.stem}
Scene analysis:
{json.dumps(analysis, indent=2)}
{retrieval_block}
Return only JavaScript source with exactly `export default function generate(THREE)`.
Use no imports, no external assets, no randomness/time/network access.
Fit the object into [-0.5, 0.5] with Y-up and +Z forward.
Never reference THREE outside the generate function. Put helper functions inside generate, or pass THREE into
helpers and use it only through that parameter. Do not define top-level constants that call new THREE.*.
Prefer visible object-defining geometry and procedural material cues over generic placeholders.
"""
    if track.use_vision_for_code:
        model = settings.vision_model
        messages = [
            {"role": "system", "content": _code_system_prompt(rules, example)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": _image_data_url(prompt_path)}},
                ],
            },
        ]
    else:
        model = settings.code_model
        messages = [
            {"role": "system", "content": _code_system_prompt(rules, example)},
            {"role": "user", "content": user},
        ]
    result = client.chat_complete(
        model=model,
        messages=messages,
        temperature=0.12 + 0.04 * max(0, candidate_index - 1),
        max_tokens=int(os.environ.get("ENSEMBLE_CODE_MAX_TOKENS", "4096")),
        label=f"{track.name}.generate",
    )
    return _extract_js_module(result.content)


def _variant_instruction(*, track: TrackSpec, candidate_index: int) -> str:
    if track.name == "candidate_ensemble":
        return (
            "Diverse ensemble style. Internally choose a different plausible implementation hypothesis for this "
            "candidate, but do not over-specialize the prompt wording. The selector will compare rendered outputs."
        )
    if track.name == "structured_repair":
        return (
            "Code for editability. Name every major group and keep dimensions/materials as simple local constants "
            "inside generate so repair operations can visibly resize, reposition, or swap parts."
        )
    if track.name == "winner_plus_direct":
        return (
            "Winner-plus direct style. Use a strong, simple raw-JS scene with more distinctive parts than the baseline, "
            "but avoid custom BufferGeometry unless necessary."
        )
    if track.name == "motif_hybrid":
        return (
            "Motif hybrid style. Prefer compact reusable helpers for visible generic motifs, but rebuild composition "
            "when any retrieved source does not match the target category."
        )
    return "Baseline style. Produce the simplest validator-safe recognizable object."


def _attempt_validator_repair(
    *,
    project_root: Path,
    paths: HarnessPaths,
    prompt: PromptSpec,
    track: TrackSpec,
    source: str,
    validator_result: dict[str, Any],
    seed: int,
    client: ChutesClient,
    settings: ChutesSettings,
    rules: str,
    example: str,
    validator_timeout_seconds: float,
    candidate_id: str,
) -> tuple[str, Path, dict[str, Any]]:
    prompt_text = f"""Repair this Three.js module so it passes the validator.

Return only the complete JavaScript module. Preserve the same object intent.

Seed: {seed}
Prompt stem: {prompt.stem}
Track: {track.name}
Validator failure JSON:
{json.dumps(validator_result, indent=2)}

Repair policy:
- If THREE is referenced outside generate, move the entire helper function or constant inside generate.
- Do not leave top-level helper functions that accept a THREE parameter; this validator still rejects many top-level helper uses.
- Do not use top-level new THREE.*, THREE.Vector3, THREE.DataTexture, or THREE constants.
- All functions that call new THREE.*, THREE.Vector*, THREE.Shape, THREE.Curve, THREE.Geometry, or THREE.Material must be nested inside generate.
- Keep exactly `export default function generate(THREE)`.
- No imports, randomness, time, network, DOM, renderer, scene, camera, eval, Function, or external assets.
- If an API is disallowed, replace it with simpler allowed geometry/materials.
- If the fix is complex, simplify geometry rather than failing validation.

Track-specific validator repair guidance:
{_track_validator_repair_prompt(track)}

Current source:
```js
{source[:20000]}
```
"""
    result = client.chat_complete(
        model=settings.code_model,
        messages=[
            {"role": "system", "content": _code_system_prompt(rules, example)},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.03,
        max_tokens=int(os.environ.get("ENSEMBLE_REPAIR_MAX_TOKENS", "4096")),
        label=f"{track.name}.validator_repair",
    )
    repaired = _extract_js_module(result.content)
    candidate_dir = paths.candidates / prompt.stem / track.name
    js_path = candidate_dir / f"{candidate_id}.validator_repair.js"
    js_path.write_text(repaired, encoding="utf-8")
    repaired_validator = run_validator(project_root, js_path, timeout_seconds=validator_timeout_seconds)
    if repaired_validator.get("passed"):
        return repaired, js_path, repaired_validator
    return source, candidate_dir / f"{candidate_id}.js", validator_result


def _track_validator_repair_prompt(track: TrackSpec) -> str:
    if track.name == "direct_vision_code":
        return direct_vision_validator_repair_prompt()
    return "No additional track-specific validator repair guidance."


def _with_candidate_count(track: TrackSpec, candidates: int) -> TrackSpec:
    return TrackSpec(
        name=track.name,
        description=track.description,
        candidates=candidates,
        analysis_prompt=track.analysis_prompt,
        code_prompt=track.code_prompt,
        repair_prompt=track.repair_prompt,
        use_retrieved_js=track.use_retrieved_js,
        use_vision_for_code=track.use_vision_for_code,
    )


def _attempt_repair(
    *,
    project_root: Path,
    paths: HarnessPaths,
    prompt: PromptSpec,
    prompt_path: Path,
    track: TrackSpec,
    source: str,
    critique: dict[str, Any],
    seed: int,
    client: ChutesClient,
    settings: ChutesSettings,
    renderer: LocalRenderService,
    rules: str,
    example: str,
    validator_timeout_seconds: float,
    candidate_id: str,
) -> tuple[str, Path, Path, dict[str, Any], bool, dict[str, Any], float]:
    prompt_text = f"""Repair this Three.js module for visual fidelity.

Track repair instruction:
{track.repair_prompt}

Seed: {seed}
Prompt stem: {prompt.stem}
Critique JSON:
{json.dumps(critique, indent=2)}

Current source:
```js
{source[:18000]}
```

Return only the complete JavaScript module. Preserve validator compliance.
"""
    result = client.chat_complete(
        model=settings.code_model,
        messages=[
            {"role": "system", "content": _code_system_prompt(rules, example)},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.08,
        max_tokens=int(os.environ.get("ENSEMBLE_REPAIR_MAX_TOKENS", "4096")),
        label=f"{track.name}.repair",
    )
    repaired = _extract_js_module(result.content)
    candidate_dir = paths.candidates / prompt.stem / track.name
    js_path = candidate_dir / f"{candidate_id}.repair.js"
    render_path = paths.renders / f"{prompt.stem}.{candidate_id}.repair.png"
    js_path.write_text(repaired, encoding="utf-8")
    validator_result = run_validator(project_root, js_path, timeout_seconds=validator_timeout_seconds)
    render_passed = False
    repaired_critique: dict[str, Any] = {}
    score = 0.0
    if validator_result.get("passed"):
        renderer.render_grid(repaired, render_path)
        render_passed = render_path.exists()
        if render_passed:
            repaired_critique = ChutesCritic(settings=settings).critique(
                prompt=prompt,
                prompt_path=prompt_path,
                render_path=render_path,
            )
            score = _score(repaired_critique)
    if score <= _score(critique):
        return source, candidate_dir / f"{candidate_id}.js", paths.renders / f"{prompt.stem}.{candidate_id}.png", {"passed": True}, True, critique, _score(critique)
    return repaired, js_path, render_path, validator_result, render_passed, repaired_critique, score


def _pairwise_judge(
    *,
    client: ChutesClient,
    settings: ChutesSettings,
    prompt: PromptSpec,
    prompt_path: Path,
    a_path: Path,
    b_path: Path,
    a_label: str,
    b_label: str,
) -> str:
    result = client.chat_complete(
        model=settings.vision_model,
        messages=[
            {"role": "system", "content": "You judge which render better matches the prompt image. Return strict JSON only."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Prompt stem: {prompt.stem}. Compare render A ({a_label}) and render B ({b_label}) "
                            "against the prompt. Return JSON with keys winner equal to A, B, or Draw; confidence; reason."
                        ),
                    },
                    {"type": "text", "text": "Prompt image:"},
                    {"type": "image_url", "image_url": {"url": _image_data_url(prompt_path)}},
                    {"type": "text", "text": f"Render A ({a_label}):"},
                    {"type": "image_url", "image_url": {"url": _image_data_url(a_path)}},
                    {"type": "text", "text": f"Render B ({b_label}):"},
                    {"type": "image_url", "image_url": {"url": _image_data_url(b_path)}},
                ],
            },
        ],
        temperature=0.02,
        max_tokens=1024,
        label="pairwise",
    )
    parsed = _parse_jsonish_or_raw(result.content)
    if isinstance(parsed, dict):
        winner = str(parsed.get("winner") or "").strip()
        if winner in {"A", "B", "Draw"}:
            return winner
    return "Unknown"


def _round0_baseline_prompts(*, limit: int) -> list[PromptSpec]:
    prompts_text = urllib.request.urlopen(f"{ROUND0_BASE}/prompts.txt", timeout=20).read().decode("utf-8")  # noqa: S310
    submitted = _baseline_submitted()
    rows: list[PromptSpec] = []
    for line in prompts_text.splitlines():
        line = line.strip()
        if not line:
            continue
        prompt = prompt_from_url(line)
        if prompt.stem in submitted:
            rows.append(prompt)
        if len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError("Could not map round-0 prompts to submitted artifacts")
    return rows


def _baseline_submitted() -> dict[str, dict[str, Any]]:
    submissions = json.loads(urllib.request.urlopen(f"{ROUND0_BASE}/submissions.json", timeout=20).read())  # noqa: S310
    hotkey = next(hk for hk, item in submissions.items() if item.get("repo") == BASELINE_REPO)
    return json.loads(urllib.request.urlopen(f"{ROUND0_BASE}/{hotkey}/submitted.json", timeout=20).read())  # noqa: S310


def _download_public_artifacts(*, root: Path, prompt: PromptSpec) -> dict[str, Path | None]:
    submitted = _baseline_submitted()
    item = submitted.get(prompt.stem) or {}
    out = root / "public" / prompt.stem
    out.mkdir(parents=True, exist_ok=True)
    js_path = _download_url(item.get("js"), out / "public.js")
    grid_url = f"{item.get('views')}/grid.png" if item.get("views") else None
    grid_path = _download_url(grid_url, out / "grid.png")
    return {"js": js_path, "grid": grid_path}


def _download_url(url: str | None, destination: Path) -> Path | None:
    if not url:
        return None
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    try:
        with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310
            destination.write_bytes(response.read())
        return destination
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _render_public_js(*, renderer: LocalRenderService, public: dict[str, Path | None], prompt: PromptSpec) -> Path | None:
    js_path = public.get("js")
    if js_path is None or not js_path.exists():
        return None
    destination = js_path.parent / "public_rerender.png"
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    try:
        source = js_path.read_text(encoding="utf-8", errors="replace")
        renderer.render_grid(source, destination)
        return destination if destination.exists() else None
    except Exception as exc:  # noqa: BLE001
        (js_path.parent / "public_rerender_error.txt").write_text(f"{prompt.stem}: {exc}\n", encoding="utf-8")
        return None


def _best_result(results: list[CandidateResult]) -> CandidateResult | None:
    passing = [r for r in results if r.validator_passed and r.render_passed]
    if not passing:
        return None
    return sorted(passing, key=lambda r: r.critique_score, reverse=True)[0]


def _score(critique: dict[str, Any]) -> float:
    value = critique.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_text(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _ensure_chutes_defaults() -> None:
    import os

    os.environ.setdefault("CHUTES_VISION_MODEL", DEFAULT_VISION_MODEL)
    os.environ.setdefault("CHUTES_CODE_MODEL", DEFAULT_CODE_MODEL)
    os.environ.setdefault("CHUTES_TIMEOUT", "300")
    os.environ.setdefault("CHUTES_RETRIES", "4")


def _write_partial_summary(*, root: Path, manifest: dict[str, Any], results: list[CandidateResult], elapsed: float) -> dict[str, Any]:
    rows = [asdict(result) for result in results]
    by_track: dict[str, dict[str, Any]] = {}
    for track in {r.track for r in results}:
        track_results = [r for r in results if r.track == track]
        rendered = [r for r in track_results if r.render_passed]
        selected = _selected_results(track_results)
        vs_base = [r.pairwise_vs_baseline for r in track_results if r.pairwise_vs_baseline]
        vs_public = [r.pairwise_vs_public for r in track_results if r.pairwise_vs_public]
        selected_vs_base = [r.pairwise_vs_baseline for r in selected if r.pairwise_vs_baseline]
        by_track[track] = {
            "candidates": len(track_results),
            "validator_pass_rate": _avg(1.0 if r.validator_passed else 0.0 for r in track_results),
            "render_pass_rate": _avg(1.0 if r.render_passed else 0.0 for r in track_results),
            "avg_critique": _avg(r.critique_score for r in rendered),
            "wins_vs_baseline": sum(1 for v in vs_base if v == "B"),
            "losses_vs_baseline": sum(1 for v in vs_base if v == "A"),
            "draws_vs_baseline": sum(1 for v in vs_base if v == "Draw"),
            "wins_vs_public": sum(1 for v in vs_public if v == "B"),
            "losses_vs_public": sum(1 for v in vs_public if v == "A"),
            "draws_vs_public": sum(1 for v in vs_public if v == "Draw"),
            "errors": sum(1 for r in track_results if r.error),
            "best": max((r.critique_score for r in rendered), default=0.0),
            "selected_prompts": len(selected),
            "selected_avg_critique": _avg(r.critique_score for r in selected),
            "selected_wins_vs_baseline": sum(1 for v in selected_vs_base if v == "B"),
            "selected_losses_vs_baseline": sum(1 for v in selected_vs_base if v == "A"),
            "selected_draws_vs_baseline": sum(1 for v in selected_vs_base if v == "Draw"),
        }
    summary = {
        **manifest,
        "elapsed_seconds": round(elapsed, 3),
        "results_count": len(results),
        "by_track": by_track,
        "results": rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Ensemble Report",
        "",
        f"Created: {summary['created_at']}",
        f"Vision model: `{summary['vision_model']}`",
        f"Code model: `{summary['code_model']}`",
        f"Elapsed seconds: `{summary['elapsed_seconds']}`",
        "",
        "## Track Summary",
        "",
        "| Track | Pass | Render | Avg Critique | Cand W/L/D | Selected Avg | Selected W/L/D | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for track, item in sorted(summary["by_track"].items()):
        lines.append(
            f"| `{track}` | {item['validator_pass_rate']:.2f} | {item['render_pass_rate']:.2f} | "
            f"{item['avg_critique']:.2f} | "
            f"{item['wins_vs_baseline']}/{item['losses_vs_baseline']}/{item['draws_vs_baseline']} | "
            f"{item['selected_avg_critique']:.2f} | "
            f"{item['selected_wins_vs_baseline']}/{item['selected_losses_vs_baseline']}/{item['selected_draws_vs_baseline']} | "
            f"{item['errors']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selected_results(results: list[CandidateResult]) -> list[CandidateResult]:
    selected: list[CandidateResult] = []
    for stem in sorted({r.prompt_stem for r in results}):
        passing = [r for r in results if r.prompt_stem == stem and r.validator_passed and r.render_passed]
        if not passing:
            continue
        selected.append(
            max(
                passing,
                key=lambda r: (
                    1 if r.pairwise_vs_baseline == "B" else 0,
                    r.critique_score,
                    1 if r.pairwise_vs_baseline == "Draw" else 0,
                ),
            )
        )
    return selected


def _avg(values: Any) -> float:
    data = list(values)
    if not data:
        return 0.0
    return sum(float(v) for v in data) / len(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3341410624)
    parser.add_argument("--track", action="append", dest="tracks")
    parser.add_argument("--render-port", type=int, default=8765)
    parser.add_argument("--render-static-port", type=int, default=8766)
    parser.add_argument("--validator-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--download-timeout-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    root = run_adaptation(
        project_root=project_root,
        runs_dir=args.runs_dir,
        limit=args.limit,
        seed=args.seed,
        tracks=args.tracks,
        render_port=args.render_port,
        render_static_port=args.render_static_port,
        validator_timeout_seconds=args.validator_timeout_seconds,
        download_timeout_seconds=args.download_timeout_seconds,
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
