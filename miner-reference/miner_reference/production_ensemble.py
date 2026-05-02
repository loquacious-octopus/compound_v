"""Production-oriented ensemble generator.

This module adapts the optimization harness into a prompt-by-prompt generator
that can be called by the batch API service.
"""

from __future__ import annotations

import json
import os
import hashlib
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from loguru import logger

from miner_reference.ensemble_core import (
    DEFAULT_CODE_MODEL,
    DEFAULT_VISION_MODEL,
    TrackSpec,
    _analyze_prompt,
    _attempt_validator_repair,
    _generate_source,
    _score,
    _with_candidate_count,
    build_direct_vision_track,
    build_object_identity_track,
    candidate_ensemble_track,
)
from miner_reference.local_harness import (
    ChutesClient,
    ChutesCritic,
    ChutesSettings,
    HarnessPaths,
    LocalRenderService,
    PromptSpec,
    _summarize_critique,
    download_prompt,
    run_validator,
)

DEFAULT_PRODUCTION_VISION_MODEL = "Qwen/Qwen3-VL-30B-A3B-Thinking"
DEFAULT_PRODUCTION_CODE_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
CANONICAL_PLACEHOLDER_HASH = "4f6f7c0e531a2a95bdf135467c465f64df18af7205f6738e6180d8335e979068"
T = TypeVar("T")


@dataclass(frozen=True)
class GeneratedPrompt:
    stem: str
    source: bytes | None
    error: str | None = None
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class EnsembleCandidate:
    track: str
    candidate_id: str
    js_path: Path
    render_path: Path | None
    validator_passed: bool
    render_passed: bool
    critique_score: float
    source: str
    summary: dict[str, Any]
    error: str | None = None


class EnsembleGenerator:
    """Run the promoted ensemble tracks and select the best valid candidate."""

    def __init__(self) -> None:
        project_root = Path(os.environ.get("MINER_PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
        self._project_root = project_root
        self._repo_root = project_root.parent
        self._runs_dir = Path(os.environ.get("ENSEMBLE_RUNS_DIR", project_root / "runs" / "production")).resolve()
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._validator_timeout = float(os.environ.get("ENSEMBLE_VALIDATOR_TIMEOUT_SECONDS", "20"))
        self._download_timeout = float(os.environ.get("ENSEMBLE_DOWNLOAD_TIMEOUT_SECONDS", "60"))
        self._render_enabled = _env_flag("ENSEMBLE_RENDER", default=True)
        self._critique_enabled = _env_flag("ENSEMBLE_CRITIQUE", default=False)
        self._validator_repair_enabled = _env_flag("ENSEMBLE_VALIDATOR_REPAIR", default=True)
        self._allow_placeholder_fallback = _env_flag("ENSEMBLE_ALLOW_PLACEHOLDER_FALLBACK", default=False)
        self._placeholder_hash = CANONICAL_PLACEHOLDER_HASH
        self._render_port = int(os.environ.get("ENSEMBLE_RENDER_PORT", "8765"))
        self._render_static_port = int(os.environ.get("ENSEMBLE_RENDER_STATIC_PORT", "8766"))
        self._renderer: LocalRenderService | None = None

        _ensure_production_model_defaults()
        self._settings = ChutesSettings.from_env_require(vision=True, code=True)
        self._client = ChutesClient(self._settings)
        self._critic = ChutesCritic(settings=self._settings)
        self._rules = _production_rules(project_root)
        self._example = _style_example(project_root)
        self._tracks = _production_tracks()

        if self._render_enabled:
            try:
                self._renderer = LocalRenderService(
                    repo_root=self._repo_root,
                    port=self._render_port,
                    static_port=self._render_static_port,
                )
                self._renderer.start()
                logger.info("Started local render service for ensemble selection")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Render service unavailable; falling back to validator-only selection: {exc}")
                self._renderer = None

    def generate_prompt(self, *, stem: str, image_url: str, seed: int) -> GeneratedPrompt:
        started = time.monotonic()
        prompt = PromptSpec(url=image_url, stem=stem.lower(), extension=Path(image_url.split("?")[0]).suffix.lower() or ".png")
        run_root = self._runs_dir / f"{int(time.time())}-{prompt.stem}"
        paths = HarnessPaths.create(run_root)
        prompt_path = paths.prompts / f"{prompt.stem}{prompt.extension}"

        ok, error = download_prompt(prompt, prompt_path, timeout_seconds=self._download_timeout)
        if not ok:
            if not self._allow_placeholder_fallback:
                return GeneratedPrompt(
                    stem=stem,
                    source=None,
                    error=f"prompt download failed and placeholder fallback disabled: {error}",
                    provenance={"fallback_disabled": True},
                )
            return GeneratedPrompt(stem=stem, source=None, error=f"prompt download failed: {error}", provenance={"fallback_disabled": True})

        candidates: list[EnsembleCandidate] = []
        baseline: EnsembleCandidate | None = None
        for track in self._tracks:
            track_candidates = self._run_track(paths=paths, prompt=prompt, prompt_path=prompt_path, track=track, seed=seed)
            candidates.extend(track_candidates)
            if track.name == "baseline_qwen25":
                baseline = _best_candidate(track_candidates)
                if baseline is None:
                    logger.warning(f"{stem}: baseline track produced no valid candidate")

        selected = _select_candidate(candidates)
        provenance = {
            "mode": "ensemble",
            "seed": seed,
            "vision_model": self._settings.vision_model,
            "code_model": self._settings.code_model,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "selected": _candidate_summary(selected) if selected else None,
            "baseline": _candidate_summary(baseline) if baseline else None,
            "candidates": [_candidate_summary(c) for c in candidates],
        }
        (run_root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

        if selected:
            return GeneratedPrompt(stem=stem, source=selected.source.encode("utf-8"), provenance=provenance)

        if not self._allow_placeholder_fallback:
            return GeneratedPrompt(
                stem=stem,
                source=None,
                error=_failure_message(provenance),
                provenance=provenance | {"fallback_disabled": True},
            )

        return GeneratedPrompt(
            stem=stem,
            source=None,
            error=_failure_message(provenance),
            provenance=provenance | {"fallback_disabled": True},
        )

    def generate_batch(
        self,
        *,
        prompts: list[Any],
        seed: int,
        progress_callback: Callable[[str, GeneratedPrompt], None] | None = None,
    ) -> list[GeneratedPrompt]:
        """Generate a batch while minimizing local model swaps.

        The managed local router keeps only one 4-GPU vLLM process loaded at a
        time. This method runs all vision-analysis calls first, then all
        code-model generation/repair calls, and only then direct-vision
        generation calls. That keeps model transitions bounded by phase rather
        than by prompt.
        """
        started = time.monotonic()
        all_items = self._prepare_batch(prompts)
        fallback_results = [item["fallback"] for item in all_items if "fallback" in item]
        work_items = [item for item in all_items if "fallback" not in item]
        candidates_by_stem: dict[str, list[EnsembleCandidate]] = {item["prompt"].stem: [] for item in work_items}
        analyses: dict[tuple[str, str], dict[str, Any]] = {}

        for track in self._tracks:
            for item in work_items:
                prompt = item["prompt"]
                prompt_path = item["prompt_path"]
                try:
                    analysis = _with_transient_retries(
                        label=f"{prompt.stem}/{track.name}/analysis",
                        fn=lambda: _analyze_prompt(
                            client=self._client,
                            settings=self._settings,
                            prompt=prompt,
                            prompt_path=prompt_path,
                            track=track,
                            seed=seed,
                        ),
                    )
                    analyses[(prompt.stem, track.name)] = analysis
                    analysis_dir = item["paths"].analysis / track.name
                    analysis_dir.mkdir(parents=True, exist_ok=True)
                    (analysis_dir / f"{prompt.stem}.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"{prompt.stem}/{track.name}: analysis failed: {exc}")
                    candidates_by_stem[prompt.stem].append(
                        _failed_candidate(
                            paths=item["paths"],
                            prompt=prompt,
                            track=track,
                            candidate_id=f"{track.name}_analysis_failed",
                            error=f"analysis failed: {exc}",
                        )
                    )

        ordered_tracks = [track for track in self._tracks if not track.use_vision_for_code]
        ordered_tracks.extend(track for track in self._tracks if track.use_vision_for_code)
        for track in ordered_tracks:
            for item in work_items:
                prompt = item["prompt"]
                analysis = analyses.get((prompt.stem, track.name))
                if analysis is None:
                    continue
                candidates_by_stem[prompt.stem].extend(
                    self._generate_track_candidates(
                        paths=item["paths"],
                        prompt=prompt,
                        prompt_path=item["prompt_path"],
                        track=track,
                        analysis=analysis,
                        seed=seed,
                    )
                )

        results: list[GeneratedPrompt] = []
        for generated in fallback_results:
            results.append(generated)
            if progress_callback:
                progress_callback(generated.stem, generated)
        for item in work_items:
            prompt = item["prompt"]
            run_root = item["run_root"]
            candidates = candidates_by_stem[prompt.stem]
            selected = _select_candidate(candidates)
            provenance = {
                "mode": "ensemble_batch",
                "seed": seed,
                "vision_model": self._settings.vision_model,
                "code_model": self._settings.code_model,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "selected": _candidate_summary(selected) if selected else None,
                "candidates": [_candidate_summary(c) for c in candidates],
            }
            (run_root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            if selected:
                generated = GeneratedPrompt(stem=prompt.stem, source=selected.source.encode("utf-8"), provenance=provenance)
            elif not self._allow_placeholder_fallback:
                generated = GeneratedPrompt(
                    stem=prompt.stem,
                    source=None,
                    error=_failure_message(provenance),
                    provenance=provenance | {"fallback_disabled": True},
                )
            else:
                generated = GeneratedPrompt(
                    stem=prompt.stem,
                    source=None,
                    error=_failure_message(provenance),
                    provenance=provenance | {"fallback_disabled": True},
                )
            results.append(generated)
            if progress_callback:
                progress_callback(prompt.stem, generated)
        return results

    def _prepare_batch(self, prompts: list[Any]) -> list[dict[str, Any]]:
        work_items: list[dict[str, Any]] = []
        for raw in prompts:
            prompt = PromptSpec(
                url=raw.image_url,
                stem=raw.stem.lower(),
                extension=Path(raw.image_url.split("?")[0]).suffix.lower() or ".png",
            )
            run_root = self._runs_dir / f"{int(time.time())}-{prompt.stem}"
            paths = HarnessPaths.create(run_root)
            prompt_path = paths.prompts / f"{prompt.stem}{prompt.extension}"
            ok, error = download_prompt(prompt, prompt_path, timeout_seconds=self._download_timeout)
            if not ok:
                logger.warning(f"{prompt.stem}: prompt download failed: {error}; using fallback")
                if not self._allow_placeholder_fallback:
                    fallback = GeneratedPrompt(
                        stem=prompt.stem,
                        source=None,
                        error=f"prompt download failed and placeholder fallback disabled: {error}",
                        provenance={"fallback_disabled": True},
                    )
                    (run_root / "provenance.json").write_text(json.dumps(fallback.provenance, indent=2), encoding="utf-8")
                    work_items.append({"prompt": prompt, "run_root": run_root, "paths": paths, "prompt_path": prompt_path, "fallback": fallback})
                    continue
                fallback = GeneratedPrompt(
                    stem=prompt.stem,
                    source=None,
                    error=f"prompt download failed: {error}",
                    provenance={"fallback_disabled": True},
                )
                (run_root / "provenance.json").write_text(json.dumps(fallback.provenance, indent=2), encoding="utf-8")
                work_items.append({"prompt": prompt, "run_root": run_root, "paths": paths, "prompt_path": prompt_path, "fallback": fallback})
                continue
            work_items.append({"prompt": prompt, "run_root": run_root, "paths": paths, "prompt_path": prompt_path})
        return work_items

    def _generate_track_candidates(
        self,
        *,
        paths: HarnessPaths,
        prompt: PromptSpec,
        prompt_path: Path,
        track: TrackSpec,
        analysis: dict[str, Any],
        seed: int,
    ) -> list[EnsembleCandidate]:
        results: list[EnsembleCandidate] = []
        retry_count = int(os.environ.get("ENSEMBLE_FAILURE_RETRY_COUNT", "1"))
        max_candidates = track.candidates + max(0, retry_count)
        for index in range(1, max_candidates + 1):
            if index > track.candidates and _best_candidate(results) is not None:
                break
            candidate_id = f"{track.name}_{index:02d}"
            candidate_dir = paths.candidates / prompt.stem / track.name
            candidate_dir.mkdir(parents=True, exist_ok=True)
            js_path = candidate_dir / f"{candidate_id}.js"
            render_path = paths.renders / f"{prompt.stem}.{candidate_id}.png"
            try:
                source = _generate_source(
                    client=self._client,
                    settings=self._settings,
                    prompt=prompt,
                    track=track,
                    analysis=analysis,
                    prompt_path=prompt_path,
                    seed=seed,
                    candidate_index=index,
                    rules=self._rules,
                    example=self._example,
                    retrieved_js=None,
                )
                if self._is_placeholder_source(source):
                    raise RuntimeError("model returned the canonical placeholder car fixture")
                js_path.write_text(source, encoding="utf-8")
                validator_result = run_validator(self._project_root, js_path, timeout_seconds=self._validator_timeout)
                if _has_validator_failure(validator_result, "BOUNDING_BOX_OUT_OF_RANGE"):
                    source, js_path, validator_result = _attempt_bounds_normalization(
                        project_root=self._project_root,
                        paths=paths,
                        prompt=prompt,
                        track=track,
                        source=source,
                        validator_timeout_seconds=self._validator_timeout,
                        candidate_id=candidate_id,
                    )
                if _needs_static_sanitize(validator_result):
                    source, js_path, validator_result = _attempt_static_sanitize(
                        project_root=self._project_root,
                        paths=paths,
                        prompt=prompt,
                        track=track,
                        source=source,
                        validator_timeout_seconds=self._validator_timeout,
                        candidate_id=candidate_id,
                    )
                    if _has_validator_failure(validator_result, "BOUNDING_BOX_OUT_OF_RANGE"):
                        source, js_path, validator_result = _attempt_bounds_normalization(
                            project_root=self._project_root,
                            paths=paths,
                            prompt=prompt,
                            track=track,
                            source=source,
                            validator_timeout_seconds=self._validator_timeout,
                            candidate_id=candidate_id,
                        )
                if not validator_result.get("passed") and self._validator_repair_enabled:
                    source, js_path, validator_result = _attempt_validator_repair(
                        project_root=self._project_root,
                        paths=paths,
                        prompt=prompt,
                        track=track,
                        source=source,
                        validator_result=validator_result,
                        seed=seed,
                        client=self._client,
                        settings=self._settings,
                        rules=self._rules,
                        example=self._example,
                        validator_timeout_seconds=self._validator_timeout,
                        candidate_id=candidate_id,
                    )
                    if self._is_placeholder_source(source):
                        raise RuntimeError("validator repair returned the canonical placeholder car fixture")
                    if _needs_static_sanitize(validator_result):
                        source, js_path, validator_result = _attempt_static_sanitize(
                            project_root=self._project_root,
                            paths=paths,
                            prompt=prompt,
                            track=track,
                            source=source,
                            validator_timeout_seconds=self._validator_timeout,
                            candidate_id=candidate_id,
                        )
                    if _has_validator_failure(validator_result, "BOUNDING_BOX_OUT_OF_RANGE"):
                        source, js_path, validator_result = _attempt_bounds_normalization(
                            project_root=self._project_root,
                            paths=paths,
                            prompt=prompt,
                            track=track,
                            source=source,
                            validator_timeout_seconds=self._validator_timeout,
                            candidate_id=candidate_id,
                        )
                validator_passed = bool(validator_result.get("passed"))
                render_passed = False
                critique: dict[str, Any] = {}
                score = 0.0
                if validator_passed and self._renderer is not None:
                    self._renderer.render_grid(source, render_path)
                    render_passed = render_path.exists()
                    if render_passed:
                        if self._critique_enabled:
                            critique = self._critic.critique(prompt=prompt, prompt_path=prompt_path, render_path=render_path)
                            score = _score(critique)
                        else:
                            score = 1.0
                elif validator_passed:
                    render_passed = True
                    score = 0.1
                results.append(
                    EnsembleCandidate(
                        track=track.name,
                        candidate_id=candidate_id,
                        js_path=js_path,
                        render_path=render_path if render_path.exists() else None,
                        validator_passed=validator_passed,
                        render_passed=render_passed,
                        critique_score=score,
                        source=source,
                        summary=_summarize_critique(critique) if critique else {"validator": validator_result},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{prompt.stem}/{candidate_id}: candidate failed: {exc}")
                results.append(
                    _failed_candidate(paths=paths, prompt=prompt, track=track, candidate_id=candidate_id, error=str(exc))
                )
        return results

    def _run_track(
        self,
        *,
        paths: HarnessPaths,
        prompt: PromptSpec,
        prompt_path: Path,
        track: TrackSpec,
        seed: int,
    ) -> list[EnsembleCandidate]:
        results: list[EnsembleCandidate] = []
        try:
            analysis = _with_transient_retries(
                label=f"{prompt.stem}/{track.name}/analysis",
                fn=lambda: _analyze_prompt(
                    client=self._client,
                    settings=self._settings,
                    prompt=prompt,
                    prompt_path=prompt_path,
                    track=track,
                    seed=seed,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{prompt.stem}/{track.name}: analysis failed: {exc}")
            return [
                _failed_candidate(
                    paths=paths,
                    prompt=prompt,
                    track=track,
                    candidate_id=f"{track.name}_analysis_failed",
                    error=f"analysis failed: {exc}",
                )
            ]

        analysis_dir = paths.analysis / track.name
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / f"{prompt.stem}.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

        retry_count = int(os.environ.get("ENSEMBLE_FAILURE_RETRY_COUNT", "1"))
        max_candidates = track.candidates + max(0, retry_count)
        for index in range(1, max_candidates + 1):
            if index > track.candidates and _best_candidate(results) is not None:
                break
            candidate_id = f"{track.name}_{index:02d}"
            candidate_dir = paths.candidates / prompt.stem / track.name
            candidate_dir.mkdir(parents=True, exist_ok=True)
            js_path = candidate_dir / f"{candidate_id}.js"
            render_path = paths.renders / f"{prompt.stem}.{candidate_id}.png"
            try:
                source = _generate_source(
                    client=self._client,
                    settings=self._settings,
                    prompt=prompt,
                    track=track,
                    analysis=analysis,
                    prompt_path=prompt_path,
                    seed=seed,
                    candidate_index=index,
                    rules=self._rules,
                    example=self._example,
                    retrieved_js=None,
                )
                if self._is_placeholder_source(source):
                    raise RuntimeError("model returned the canonical placeholder car fixture")
                js_path.write_text(source, encoding="utf-8")
                validator_result = run_validator(self._project_root, js_path, timeout_seconds=self._validator_timeout)
                if _has_validator_failure(validator_result, "BOUNDING_BOX_OUT_OF_RANGE"):
                    source, js_path, validator_result = _attempt_bounds_normalization(
                        project_root=self._project_root,
                        paths=paths,
                        prompt=prompt,
                        track=track,
                        source=source,
                        validator_timeout_seconds=self._validator_timeout,
                        candidate_id=candidate_id,
                    )
                if _needs_static_sanitize(validator_result):
                    source, js_path, validator_result = _attempt_static_sanitize(
                        project_root=self._project_root,
                        paths=paths,
                        prompt=prompt,
                        track=track,
                        source=source,
                        validator_timeout_seconds=self._validator_timeout,
                        candidate_id=candidate_id,
                    )
                    if _has_validator_failure(validator_result, "BOUNDING_BOX_OUT_OF_RANGE"):
                        source, js_path, validator_result = _attempt_bounds_normalization(
                            project_root=self._project_root,
                            paths=paths,
                            prompt=prompt,
                            track=track,
                            source=source,
                            validator_timeout_seconds=self._validator_timeout,
                            candidate_id=candidate_id,
                        )
                if not validator_result.get("passed") and self._validator_repair_enabled:
                    source, js_path, validator_result = _attempt_validator_repair(
                        project_root=self._project_root,
                        paths=paths,
                        prompt=prompt,
                        track=track,
                        source=source,
                        validator_result=validator_result,
                        seed=seed,
                        client=self._client,
                        settings=self._settings,
                        rules=self._rules,
                        example=self._example,
                        validator_timeout_seconds=self._validator_timeout,
                        candidate_id=candidate_id,
                    )
                    if self._is_placeholder_source(source):
                        raise RuntimeError("validator repair returned the canonical placeholder car fixture")
                    if _needs_static_sanitize(validator_result):
                        source, js_path, validator_result = _attempt_static_sanitize(
                            project_root=self._project_root,
                            paths=paths,
                            prompt=prompt,
                            track=track,
                            source=source,
                            validator_timeout_seconds=self._validator_timeout,
                            candidate_id=candidate_id,
                        )
                    if _has_validator_failure(validator_result, "BOUNDING_BOX_OUT_OF_RANGE"):
                        source, js_path, validator_result = _attempt_bounds_normalization(
                            project_root=self._project_root,
                            paths=paths,
                            prompt=prompt,
                            track=track,
                            source=source,
                            validator_timeout_seconds=self._validator_timeout,
                            candidate_id=candidate_id,
                        )
                validator_passed = bool(validator_result.get("passed"))
                render_passed = False
                critique: dict[str, Any] = {}
                score = 0.0
                if validator_passed and self._renderer is not None:
                    self._renderer.render_grid(source, render_path)
                    render_passed = render_path.exists()
                    if render_passed:
                        critique = self._critic.critique(prompt=prompt, prompt_path=prompt_path, render_path=render_path)
                        score = _score(critique)
                elif validator_passed:
                    render_passed = True
                    score = 0.1

                results.append(
                    EnsembleCandidate(
                        track=track.name,
                        candidate_id=candidate_id,
                        js_path=js_path,
                        render_path=render_path if render_path.exists() else None,
                        validator_passed=validator_passed,
                        render_passed=render_passed,
                        critique_score=score,
                        source=source,
                        summary=_summarize_critique(critique) if critique else {"validator": validator_result},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{prompt.stem}/{candidate_id}: candidate failed: {exc}")
                results.append(
                    _failed_candidate(paths=paths, prompt=prompt, track=track, candidate_id=candidate_id, error=str(exc))
                )
        return results

    def _is_placeholder_source(self, source: str) -> bool:
        return _source_hash(source) == self._placeholder_hash


def create_generator() -> EnsembleGenerator:
    mode = os.environ.get("MINER_GENERATOR_MODE", "ensemble").strip().lower()
    if mode == "ensemble":
        return EnsembleGenerator()
    raise RuntimeError(f"Unsupported MINER_GENERATOR_MODE={mode!r}; expected 'ensemble'")


def _production_tracks() -> list[TrackSpec]:
    candidate_count = int(os.environ.get("ENSEMBLE_CANDIDATE_COUNT", "3"))
    include_object_identity = _env_flag("ENSEMBLE_INCLUDE_OBJECT_IDENTITY", default=True)
    tracks: list[TrackSpec] = [
        TrackSpec(
            name="baseline_qwen25",
            description="Baseline-safe planner/coder fallback.",
            candidates=1,
            analysis_prompt=(
                "Create a compact object structural description for procedural Three.js reconstruction. "
                "Identify object identity, camera/view, silhouette, symmetry, primary volumes, secondary parts, "
                "attachments, material zones, colors, and must-have details. Return strict JSON."
            ),
            code_prompt=(
                "Generate one validator-safe raw Three.js module from the structural description. "
                "Favor correct object identity, major silhouette, proportions, colors, and a few defining details. "
                "Use descriptive variable names and do not redeclare any const or let identifier in the module. "
                "Do not use TextGeometry, Font, EllipseGeometry, canvas text, sprites, labels, renderer, camera, scene, "
                "lights, window, document, DOM APIs, or texture loaders; represent lettering and elliptical details with simple primitives."
            ),
        ),
        _with_candidate_count(candidate_ensemble_track(TrackSpec), candidate_count),
        _with_candidate_count(build_direct_vision_track(TrackSpec), 1),
    ]
    if include_object_identity:
        tracks.append(_with_candidate_count(build_object_identity_track(TrackSpec), 1))

    wanted = {name.strip() for name in os.environ.get("ENSEMBLE_TRACKS", "").split(",") if name.strip()}
    if wanted:
        tracks = [track for track in tracks if track.name in wanted]
    return tracks


def _ensure_production_model_defaults() -> None:
    os.environ.setdefault("CHUTES_VISION_MODEL", os.environ.get("PRODUCTION_VISION_MODEL", DEFAULT_PRODUCTION_VISION_MODEL))
    os.environ.setdefault("CHUTES_CODE_MODEL", os.environ.get("PRODUCTION_CODE_MODEL", DEFAULT_PRODUCTION_CODE_MODEL))
    os.environ.setdefault("CHUTES_TIMEOUT", "300")
    os.environ.setdefault("CHUTES_RETRIES", "8")
    os.environ.setdefault("ENSEMBLE_TRANSIENT_RETRIES", "4")
    if "CHUTES_API_KEY" not in os.environ and os.environ.get("CHUTES_BASE_URL"):
        os.environ["CHUTES_API_KEY"] = "local-vllm"
    if os.environ.get("CHUTES_VISION_MODEL") in {DEFAULT_VISION_MODEL, ""}:
        os.environ["CHUTES_VISION_MODEL"] = DEFAULT_PRODUCTION_VISION_MODEL
    if os.environ.get("CHUTES_CODE_MODEL") in {DEFAULT_CODE_MODEL, ""}:
        os.environ["CHUTES_CODE_MODEL"] = DEFAULT_PRODUCTION_CODE_MODEL


def _production_rules(project_root: Path) -> str:
    rules_paths = [project_root / "output_specifications.md", project_root / "runtime_specifications.md"]
    sections = []
    for path in rules_paths:
        if path.exists():
            sections.append(path.read_text(encoding="utf-8"))
    if sections:
        return "\n\n".join(sections)
    raise RuntimeError("missing production rule files: output_specifications.md and runtime_specifications.md")


def _style_example(project_root: Path) -> str:
    if os.environ.get("ENSEMBLE_INCLUDE_STYLE_EXAMPLE", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return ""
    path = project_root / "examples" / "car.js"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _select_candidate(candidates: list[EnsembleCandidate]) -> EnsembleCandidate | None:
    valid = [candidate for candidate in candidates if candidate.validator_passed and candidate.render_passed]
    if not valid:
        return None
    return max(
        valid,
        key=lambda c: (
            c.critique_score,
            1 if c.track == "direct_vision_code" else 0,
            1 if c.track == "candidate_ensemble" else 0,
            -len(c.source),
        ),
    )


def _best_candidate(candidates: list[EnsembleCandidate]) -> EnsembleCandidate | None:
    valid = [candidate for candidate in candidates if candidate.validator_passed and candidate.render_passed]
    if not valid:
        return None
    return max(valid, key=lambda c: c.critique_score)


def _failed_candidate(
    *,
    paths: HarnessPaths,
    prompt: PromptSpec,
    track: TrackSpec,
    candidate_id: str,
    error: str,
) -> EnsembleCandidate:
    js_path = paths.candidates / prompt.stem / track.name / f"{candidate_id}.js"
    return EnsembleCandidate(
        track=track.name,
        candidate_id=candidate_id,
        js_path=js_path,
        render_path=None,
        validator_passed=False,
        render_passed=False,
        critique_score=0.0,
        source="",
        summary={},
        error=error,
    )


def _with_transient_retries(*, label: str, fn: Callable[[], T]) -> T:
    attempts = max(1, int(os.environ.get("ENSEMBLE_TRANSIENT_RETRIES", "3")))
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= attempts or not _is_transient_error(exc):
                raise
            delay = min(10.0, float(2 ** (attempt - 1)))
            logger.warning(f"{label}: transient error {attempt}/{attempts}; retrying in {delay:.0f}s: {exc}")
            time.sleep(delay)
    raise RuntimeError(f"{label}: transient retries exhausted")


def _is_transient_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "connection refused",
            "temporarily unavailable",
            "local model server unavailable",
            "chutes http 502",
            "chutes http 503",
            "chutes http 504",
            "timed out",
            "timeout",
        )
    )


def _has_validator_failure(validator_result: dict[str, Any], rule: str) -> bool:
    return any(failure.get("rule") == rule for failure in validator_result.get("failures", []))


def _has_forbidden_identifier(validator_result: dict[str, Any], text: str) -> bool:
    return any(
        failure.get("rule") in {"FORBIDDEN_IDENTIFIER", "IDENTIFIER_NOT_ALLOWED"}
        and text in str(failure.get("detail", ""))
        for failure in validator_result.get("failures", [])
    )


def _needs_static_sanitize(validator_result: dict[str, Any]) -> bool:
    if _has_forbidden_identifier(validator_result, "Math.random"):
        return True
    if _has_forbidden_identifier(validator_result, "window"):
        return True
    if any(
        failure.get("rule") == "FORBIDDEN_THREE_API"
        and any(api in str(failure.get("detail", "")) for api in ("THREE.AmbientLight", "THREE.DirectionalLight", "THREE.PointLight", "THREE.SpotLight"))
        for failure in validator_result.get("failures", [])
    ):
        return True
    return any(
        failure.get("rule") == "UNKNOWN_THREE_API"
        and any(api in str(failure.get("detail", "")) for api in ("THREE.TextGeometry", "THREE.Font", "THREE.EllipseGeometry"))
        for failure in validator_result.get("failures", [])
    )


def _attempt_static_sanitize(
    *,
    project_root: Path,
    paths: HarnessPaths,
    prompt: PromptSpec,
    track: TrackSpec,
    source: str,
    validator_timeout_seconds: float,
    candidate_id: str,
) -> tuple[str, Path, dict[str, Any]]:
    sanitized = _sanitize_static_source(source)
    if sanitized == source:
        candidate_dir = paths.candidates / prompt.stem / track.name
        return source, candidate_dir / f"{candidate_id}.js", {"passed": False, "failures": [{"rule": "STATIC_SANITIZE_NOOP"}]}

    candidate_dir = paths.candidates / prompt.stem / track.name
    js_path = candidate_dir / f"{candidate_id}.sanitized.js"
    js_path.write_text(sanitized, encoding="utf-8")
    validator_result = run_validator(project_root, js_path, timeout_seconds=validator_timeout_seconds)
    if validator_result.get("passed"):
        return sanitized, js_path, validator_result
    return sanitized, js_path, validator_result


def _sanitize_static_source(source: str) -> str:
    sanitized = re.sub(r"\bMath\.random\s*\(\s*\)", "0.5", source)
    sanitized = re.sub(
        r"new\s+THREE\.TextGeometry\s*\((?:[^()]|\([^()]*\))*\)",
        "new THREE.BoxGeometry(0.22, 0.05, 0.02)",
        sanitized,
    )
    sanitized = re.sub(r"new\s+THREE\.Font\s*\((?:[^()]|\([^()]*\))*\)", "{}", sanitized)
    sanitized = re.sub(
        r"new\s+THREE\.EllipseGeometry\s*\((?:[^()]|\([^()]*\))*\)",
        "new THREE.CircleGeometry(0.18, 32)",
        sanitized,
    )
    sanitized = re.sub(
        r"new\s+THREE\.(?:AmbientLight|DirectionalLight|PointLight|SpotLight)\s*\((?:[^()]|\([^()]*\))*\)",
        "new THREE.Group()",
        sanitized,
    )
    sanitized = re.sub(r"\bwindow\.innerWidth\b", "512", sanitized)
    sanitized = re.sub(r"\bwindow\.innerHeight\b", "512", sanitized)
    sanitized = re.sub(r"\bwindow\.devicePixelRatio\b", "1", sanitized)

    lines: list[str] = []
    for line in sanitized.splitlines():
        if "new THREE.TextGeometry" in line and "=" in line:
            lhs = line.split("=", 1)[0].rstrip()
            lines.append(f"{lhs} = new THREE.BoxGeometry(0.22, 0.05, 0.02);")
            continue
        if "new THREE.Font" in line and "=" in line:
            lhs = line.split("=", 1)[0].rstrip()
            lines.append(f"{lhs} = {{}};")
            continue
        if "new THREE.EllipseGeometry" in line and "=" in line:
            lhs = line.split("=", 1)[0].rstrip()
            lines.append(f"{lhs} = new THREE.CircleGeometry(0.18, 32);")
            continue
        if any(light in line for light in ("new THREE.AmbientLight", "new THREE.DirectionalLight", "new THREE.PointLight", "new THREE.SpotLight")) and "=" in line:
            lhs = line.split("=", 1)[0].rstrip()
            lines.append(f"{lhs} = new THREE.Group();")
            continue
        lines.append(line)
    return "\n".join(lines) + ("\n" if sanitized.endswith("\n") else "")


def _attempt_bounds_normalization(
    *,
    project_root: Path,
    paths: HarnessPaths,
    prompt: PromptSpec,
    track: TrackSpec,
    source: str,
    validator_timeout_seconds: float,
    candidate_id: str,
) -> tuple[str, Path, dict[str, Any]]:
    normalized = _inject_bounds_normalization(source)
    if normalized == source:
        candidate_dir = paths.candidates / prompt.stem / track.name
        return source, candidate_dir / f"{candidate_id}.js", {"passed": False, "failures": [{"rule": "NORMALIZATION_INJECTION_FAILED"}]}

    candidate_dir = paths.candidates / prompt.stem / track.name
    js_path = candidate_dir / f"{candidate_id}.normalized.js"
    js_path.write_text(normalized, encoding="utf-8")
    validator_result = run_validator(project_root, js_path, timeout_seconds=validator_timeout_seconds)
    return normalized, js_path, validator_result


def _inject_bounds_normalization(source: str) -> str:
    pattern = re.compile(r"(?P<indent>^[ \t]*)return\s+(?P<root>[A-Za-z_$][A-Za-z0-9_$]*)\s*;", re.MULTILINE)
    matches = list(pattern.finditer(source))
    if not matches:
        return source
    match = matches[-1]
    indent = match.group("indent")
    root = match.group("root")
    block = f"""{indent}const __box = new THREE.Box3().setFromObject({root});
{indent}const __size = new THREE.Vector3();
{indent}const __center = new THREE.Vector3();
{indent}__box.getSize(__size);
{indent}__box.getCenter(__center);
{indent}const __maxDim = Math.max(__size.x, __size.y, __size.z);
{indent}if (__maxDim > 0) {{
{indent}  const __scale = 0.45 / __maxDim;
{indent}  {root}.scale.setScalar(__scale);
{indent}  {root}.position.set(-__center.x * __scale, -__center.y * __scale, -__center.z * __scale);
{indent}}}
{match.group(0)}"""
    return source[: match.start()] + block + source[match.end() :]


def _candidate_summary(candidate: EnsembleCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    summary = {
        "track": candidate.track,
        "candidate_id": candidate.candidate_id,
        "validator_passed": candidate.validator_passed,
        "render_passed": candidate.render_passed,
        "critique_score": candidate.critique_score,
        "js_path": str(candidate.js_path),
        "render_path": str(candidate.render_path) if candidate.render_path else None,
        "error": candidate.error,
        "summary": candidate.summary,
    }
    if (
        candidate.source
        and not candidate.validator_passed
        and _env_flag("ENSEMBLE_DEBUG_FAILURE_SOURCES", default=False)
    ):
        summary["source_excerpt"] = candidate.source[:2000]
    return summary


def _failure_message(provenance: dict[str, Any]) -> str:
    snippets = []
    for candidate in provenance.get("candidates", [])[:6]:
        label = f"{candidate.get('track')}/{candidate.get('candidate_id')}"
        if candidate.get("error"):
            snippets.append(f"{label}: {candidate['error']}")
            continue
        summary = candidate.get("summary") or {}
        validator = summary.get("validator") if isinstance(summary, dict) else None
        if isinstance(validator, dict):
            failures = validator.get("failures") or []
            if failures:
                source_excerpt = candidate.get("source_excerpt")
                excerpt = f"; source_excerpt={source_excerpt!r}" if source_excerpt else ""
                snippets.append(f"{label}: validator {failures[:3]}{excerpt}")
                continue
            snippets.append(
                f"{label}: validator_passed={candidate.get('validator_passed')} render_passed={candidate.get('render_passed')}"
            )
    suffix = "; ".join(snippets) if snippets else "no candidate diagnostics"
    return f"all ensemble candidates failed; placeholder fallback disabled; {suffix}"


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.replace("\r\n", "\n").strip().encode("utf-8")).hexdigest()
