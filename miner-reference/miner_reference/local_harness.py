"""Local iteration harness for procedural Three.js miner outputs.

This module intentionally starts with a stub generator. It exercises the
competition-shaped file flow first: prompt URLs -> downloaded prompt images ->
generated ``{stem}.js`` files -> validator JSON -> run summaries.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import socket
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from miner_reference.dsl_compiler import DslError, compile_dsl_document


DEFAULT_PROMPT_LIST = Path("tests/Procedural_Competition_Input_Prompt_Samples.txt")
DEFAULT_RUNS_DIR = Path("runs")
REFERENCE_JS = Path("")
DEFAULT_RENDER_PORT = 8765
DEFAULT_RENDER_STATIC_PORT = 8766
DEFAULT_MODEL_PRICES_USD_PER_M_TOKENS = {
    "Qwen/Qwen2.5-VL-32B-Instruct": {"input": 0.0543, "output": 0.2174},
    "Qwen/Qwen3-Coder-Next-TEE": {"input": 0.12, "output": 0.75},
    "zai-org/GLM-4.6V": {"input": 0.3, "output": 0.9},
    "Qwen/Qwen2.5-Coder-32B-Instruct": {"input": 0.0272, "output": 0.1087},
    "Qwen/Qwen3-32B-TEE": {"input": 0.08, "output": 0.24},
    "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE": {"input": 0.1, "output": 0.6},
}


@dataclass(frozen=True)
class PromptSpec:
    """A prompt image URL and its competition stem."""

    url: str
    stem: str
    extension: str


@dataclass(frozen=True)
class HarnessPaths:
    """Directory layout for one harness run."""

    root: Path
    prompts: Path
    analysis: Path
    dsl: Path
    candidates: Path
    renders: Path
    critiques: Path
    results: Path

    @classmethod
    def create(cls, root: Path) -> "HarnessPaths":
        paths = cls(
            root=root,
            prompts=root / "prompts",
            analysis=root / "analysis",
            dsl=root / "dsl",
            candidates=root / "candidates",
            renders=root / "renders",
            critiques=root / "critiques",
            results=root / "results",
        )
        for directory in (
            paths.prompts,
            paths.analysis,
            paths.dsl,
            paths.candidates,
            paths.renders,
            paths.critiques,
            paths.results,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


class StubGenerator:
    """Generator interface placeholder.

    The real Chutes-backed implementation should keep this contract:
    ``generate(prompt, prompt_path, seed) -> bytes`` containing a complete
    Three.js ES module.
    """

    def __init__(self, reference_js: Path) -> None:
        self._reference_js = reference_js

    def generate(self, prompt: PromptSpec, prompt_path: Path, seed: int) -> bytes:  # noqa: ARG002
        return self._reference_js.read_bytes()

    def repair(
        self,
        prompt: PromptSpec,
        prompt_path: Path,
        seed: int,
        previous_source: str,
        validator_result: dict,
    ) -> bytes:  # noqa: ARG002
        return self.generate(prompt, prompt_path, seed)


@dataclass(frozen=True)
class ChutesSettings:
    """Environment-driven Chutes configuration."""

    api_key: str
    base_url: str
    vision_model: str
    code_model: str
    request_timeout_seconds: float
    max_tokens: int
    model_prices_usd_per_m_tokens: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "ChutesSettings":
        return cls.from_env_require(vision=True, code=True)

    @classmethod
    def from_env_require(cls, *, vision: bool, code: bool) -> "ChutesSettings":
        api_key = os.environ.get("CHUTES_API_KEY")
        if not api_key:
            raise RuntimeError("CHUTES_API_KEY is not set")

        shared_model = os.environ.get("CHUTES_MODEL")
        vision_model = os.environ.get("CHUTES_VISION_MODEL") or shared_model
        code_model = os.environ.get("CHUTES_CODE_MODEL") or shared_model
        missing = [
            name
            for name, value in (("CHUTES_VISION_MODEL", vision_model), ("CHUTES_CODE_MODEL", code_model))
            if (name == "CHUTES_VISION_MODEL" and vision and not value)
            or (name == "CHUTES_CODE_MODEL" and code and not value)
        ]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"{joined} not set. Set required model env vars, or set CHUTES_MODEL as a fallback.")

        return cls(
            api_key=api_key,
            base_url=os.environ.get("CHUTES_BASE_URL", "https://llm.chutes.ai/v1").rstrip("/"),
            vision_model=str(vision_model or ""),
            code_model=str(code_model or ""),
            request_timeout_seconds=float(os.environ.get("CHUTES_TIMEOUT", "300")),
            max_tokens=int(os.environ.get("CHUTES_MAX_TOKENS", "8192")),
            model_prices_usd_per_m_tokens=_model_prices_from_env(),
        )


@dataclass(frozen=True)
class ChutesChatResult:
    """Chutes chat completion content plus accounting metadata."""

    content: str
    usage_event: dict


class ChutesClient:
    """Minimal OpenAI-compatible chat-completions client using stdlib HTTP."""

    def __init__(self, settings: ChutesSettings) -> None:
        self._settings = settings

    def chat(self, *, model: str, messages: list[dict], temperature: float = 0.1, max_tokens: int | None = None) -> str:
        return self.chat_complete(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens).content

    def chat_complete(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int | None = None,
        label: str = "chat",
    ) -> ChutesChatResult:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self._settings.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._settings.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self._settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        attempts = max(1, int(os.environ.get("CHUTES_RETRIES", "4")))
        raw = b""
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._settings.request_timeout_seconds) as response:  # noqa: S310
                    raw = response.read()
                last_error = None
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = RuntimeError(f"Chutes HTTP {exc.code}: {detail}")
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= attempts:
                    raise last_error from exc
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                last_error = RuntimeError(f"Chutes request failed: {exc}")
                if attempt >= attempts:
                    raise last_error from exc
            time.sleep(min(2 ** (attempt - 1), 8))
        if last_error is not None:
            raise last_error

        body = json.loads(raw)
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Chutes response shape: {body}") from exc
        if not isinstance(content, str):
            content = _message_content_to_text(content)
        if not isinstance(content, str) or not content:
            raise RuntimeError("Chutes response message content was not text")
        usage = body.get("usage") if isinstance(body, dict) else None
        usage_event = _build_usage_event(
            label=label,
            model=model,
            usage=usage if isinstance(usage, dict) else {},
            prices=self._settings.model_prices_usd_per_m_tokens,
        )
        return ChutesChatResult(content=content, usage_event=usage_event)


def _message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text
    return ""


class ChutesGenerator:
    """Chutes-backed image -> scene plan -> Three.js source generator."""

    def __init__(self, *, project_root: Path, paths: HarnessPaths, settings: ChutesSettings) -> None:
        self._project_root = project_root
        self._paths = paths
        self._settings = settings
        self._client = ChutesClient(settings)
        self._rules = "\n\n".join(
            path.read_text(encoding="utf-8")
            for path in (project_root / "output_specifications.md", project_root / "runtime_specifications.md")
            if path.exists()
        )
        self._example = ""

    def generate(self, prompt: PromptSpec, prompt_path: Path, seed: int) -> bytes:
        analysis, analysis_usage = self._analyze(prompt=prompt, prompt_path=prompt_path, seed=seed)
        analysis_path = self._paths.analysis / f"{prompt.stem}.json"
        analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        self._write_usage_event(prompt=prompt, attempt=1, suffix="analysis", usage_event=analysis_usage)

        if _env_flag("CHUTES_TEMPLATE_FIRST"):
            template_source = _template_source_for_analysis(analysis)
            if template_source:
                return template_source.encode("utf-8")

        source, source_usage = self._generate_source(prompt=prompt, analysis=analysis, seed=seed)
        self._write_usage_event(prompt=prompt, attempt=1, suffix="generate", usage_event=source_usage)
        return _extract_js_module(source).encode("utf-8")

    def repair(
        self,
        prompt: PromptSpec,
        prompt_path: Path,
        seed: int,
        previous_source: str,
        validator_result: dict,
    ) -> bytes:
        analysis_path = self._paths.analysis / f"{prompt.stem}.json"
        analysis = analysis_path.read_text(encoding="utf-8") if analysis_path.exists() else "{}"
        prompt_text = f"""Fix this Three.js module so it passes the validator.

Keep the same visual intent. Return only the complete JavaScript module, no markdown.

Seed: {seed}
Generation strategy: {_generation_strategy_text()}
Prompt stem: {prompt.stem}
Scene analysis:
{analysis}

Validator result:
{json.dumps(validator_result, indent=2)}

Previous source:
```js
{previous_source}
```
"""
        result = self._client.chat_complete(
            model=self._settings.code_model,
            messages=[
                {"role": "system", "content": _code_system_prompt(self._rules, self._example)},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.05,
            label="repair",
        )
        self._write_usage_event(prompt=prompt, attempt=_next_repair_attempt(self._paths.candidates / prompt.stem), suffix="repair", usage_event=result.usage_event)
        return _extract_js_module(result.content).encode("utf-8")

    def _analyze(self, *, prompt: PromptSpec, prompt_path: Path, seed: int) -> tuple[dict, dict]:
        image_data_url = _image_data_url(prompt_path)
        messages = [
            {
                "role": "system",
                "content": (
                    "You analyze prompt images for procedural Three.js reconstruction. "
                    "Return compact JSON only. Do not include markdown."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this object for procedural Three.js generation. "
                            "Return strict JSON only with keys: category, confidence, camera_view, "
                            "overall_silhouette, symmetry, coordinate_plan, parts, materials, colors, "
                            "procedural_templates, must_have_details, avoid_confusions, and validator_risks. "
                            "For each part include name, primitive_hint, relative_position, relative_scale, "
                            "color, material, and visual_priority from 1 to 10. "
                            f"Seed: {seed}. Prompt stem: {prompt.stem}."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ]
        result = self._client.chat_complete(
            model=self._settings.vision_model,
            messages=messages,
            temperature=0.1,
            label="analysis",
        )
        parsed = _parse_jsonish_or_raw(result.content)
        return parsed if isinstance(parsed, dict) else {"raw": result.content}, result.usage_event

    def _generate_source(self, *, prompt: PromptSpec, analysis: dict, seed: int) -> tuple[str, dict]:
        if _env_flag("CHUTES_DIRECT_VISION_CODE"):
            return self._generate_source_direct_vision(prompt=prompt, analysis=analysis, seed=seed)
        prompt_text = f"""Generate a validator-compliant Three.js module for this image analysis.

Return only JavaScript source. No markdown.

Seed: {seed}
Generation strategy: {_generation_strategy_text()}
Prompt stem: {prompt.stem}
Scene analysis JSON:
{json.dumps(analysis, indent=2)}

Requirements:
- Use exactly `export default function generate(THREE)`.
- Use procedural geometry only.
- Fit inside [-0.5, 0.5], Y-up, +Z forward.
- Prefer robust simple geometry over risky complex code.
- Include a `fitToUnitCube(THREE, root)` helper unless the asset is already normalized.
- Follow the generation strategy above even when it costs more vertices/draw calls.
"""
        result = self._client.chat_complete(
            model=self._settings.code_model,
            messages=[
                {"role": "system", "content": _code_system_prompt(self._rules, self._example)},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.15,
            label="generate",
        )
        return result.content, result.usage_event

    def _generate_source_direct_vision(self, *, prompt: PromptSpec, analysis: dict, seed: int) -> tuple[str, dict]:
        image_path = self._paths.prompts / f"{prompt.stem}{prompt.extension}"
        result = self._client.chat_complete(
            model=self._settings.vision_model,
            messages=[
                {"role": "system", "content": _code_system_prompt(self._rules, self._example)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Generate a validator-compliant procedural Three.js module for this prompt image. "
                                "Return only JavaScript source. No markdown. The JS must use exactly "
                                "`export default function generate(THREE)`. Fit inside [-0.5, 0.5]. "
                                "Use procedural geometry only. Prefer category-specific part decomposition, "
                                "distinctive silhouettes, visible handles/spouts/holes/wheels/feet/labels, "
                                "and material/color details over generic blocks. "
                                "For TubeGeometry use THREE.CatmullRomCurve3 or another valid 3D curve, not THREE.Path. "
                                "Always instantiate Three.js classes as `new THREE.ClassName(...)`; never `new.ClassName(...)`. "
                                f"Seed: {seed}. Generation strategy: {_generation_strategy_text()}. "
                                f"Prompt stem: {prompt.stem}. Structured analysis: {json.dumps(analysis)}"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                    ],
                },
            ],
            temperature=0.12,
            label="direct_generate",
        )
        return result.content, result.usage_event

    def visual_repair(
        self,
        *,
        prompt: PromptSpec,
        prompt_path: Path,
        render_path: Path,
        seed: int,
        current_source: str,
        critique: dict,
        repair_index: int,
    ) -> bytes:
        if _env_flag("CHUTES_DIRECT_VISION_REPAIR"):
            return self._direct_visual_repair(
                prompt=prompt,
                prompt_path=prompt_path,
                render_path=render_path,
                seed=seed,
                current_source=current_source,
                critique=critique,
                repair_index=repair_index,
            )
        repair_plan, plan_usage = self._visual_repair_plan(
            prompt=prompt,
            prompt_path=prompt_path,
            render_path=render_path,
            critique=critique,
            repair_index=repair_index,
        )
        self._write_usage_event(
            prompt=prompt,
            attempt=repair_index,
            suffix="visual_plan",
            usage_event=plan_usage,
        )
        prompt_text = f"""Patch this validator-compliant Three.js module to improve visual match.

Return only the complete JavaScript module. Do not include markdown.

Seed: {seed}
Generation strategy: {_generation_strategy_text()}
Prompt stem: {prompt.stem}
Structured visual repair plan:
{json.dumps(repair_plan, indent=2)}

Current critique:
{json.dumps(critique, indent=2)}

Current source:
```js
{current_source}
```

Patch priorities:
- Preserve validator compliance and bounding box normalization.
- Change geometry, proportions, materials, orientation, and distinctive parts to match the repair plan.
- Add visible details using procedural primitives or DataTexture when useful.
- Always instantiate Three.js classes as `new THREE.ClassName(...)`; never write `new.ClassName(...)`.
- Do not merely rename variables or add comments.
"""
        result = self._client.chat_complete(
            model=self._settings.code_model,
            messages=[
                {"role": "system", "content": _code_system_prompt(self._rules, self._example)},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.12,
            label="visual_repair",
        )
        self._write_usage_event(
            prompt=prompt,
            attempt=repair_index,
            suffix="visual_repair",
            usage_event=result.usage_event,
        )
        return _extract_js_module(result.content).encode("utf-8")

    def _visual_repair_plan(
        self,
        *,
        prompt: PromptSpec,
        prompt_path: Path,
        render_path: Path,
        critique: dict,
        repair_index: int,
    ) -> tuple[dict, dict]:
        result = self._client.chat_complete(
            model=self._settings.vision_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce surgical visual repair plans for procedural Three.js code. "
                        "Return strict compact JSON only. Do not include markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Compare the prompt image to the render. Produce JSON with keys: "
                                "object_identity_correction, silhouette_changes, part_additions, "
                                "part_removals, proportion_changes, orientation_changes, material_changes, "
                                "texture_detail_changes, ranked_code_actions, and success_criteria. "
                                "Be concrete enough for a code model to patch primitives. "
                                f"Repair index: {repair_index}. Prompt stem: {prompt.stem}. "
                                f"Current critique: {json.dumps(critique)}"
                            ),
                        },
                        {"type": "text", "text": "Prompt image:"},
                        {"type": "image_url", "image_url": {"url": _image_data_url(prompt_path)}},
                        {"type": "text", "text": "Current render:"},
                        {"type": "image_url", "image_url": {"url": _image_data_url(render_path)}},
                    ],
                },
            ],
            temperature=0.05,
            label="visual_plan",
        )
        parsed = _parse_jsonish_or_raw(result.content)
        return parsed if isinstance(parsed, dict) else {"raw": result.content}, result.usage_event

    def _direct_visual_repair(
        self,
        *,
        prompt: PromptSpec,
        prompt_path: Path,
        render_path: Path,
        seed: int,
        current_source: str,
        critique: dict,
        repair_index: int,
    ) -> bytes:
        result = self._client.chat_complete(
            model=self._settings.vision_model,
            messages=[
                {"role": "system", "content": _code_system_prompt(self._rules, self._example)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Patch this procedural Three.js module while directly comparing the prompt image "
                                "and current render. Return only the complete JavaScript module. "
                                "Do not include markdown. Preserve validator compliance. "
                                "For TubeGeometry use THREE.CatmullRomCurve3 or another valid 3D curve, not THREE.Path. "
                                "Use `new THREE.ClassName(...)`, never `new.ClassName(...)`. "
                                f"Seed: {seed}. Repair index: {repair_index}. Prompt stem: {prompt.stem}. "
                                f"Current critique: {json.dumps(critique)}\n\nCurrent source:\n```js\n{current_source}\n```"
                            ),
                        },
                        {"type": "text", "text": "Prompt image:"},
                        {"type": "image_url", "image_url": {"url": _image_data_url(prompt_path)}},
                        {"type": "text", "text": "Current render:"},
                        {"type": "image_url", "image_url": {"url": _image_data_url(render_path)}},
                    ],
                },
            ],
            temperature=0.08,
            label="direct_visual_repair",
        )
        self._write_usage_event(
            prompt=prompt,
            attempt=repair_index,
            suffix="direct_visual_repair",
            usage_event=result.usage_event,
        )
        return _extract_js_module(result.content).encode("utf-8")

    def _write_usage_event(self, *, prompt: PromptSpec, attempt: int, suffix: str, usage_event: dict) -> None:
        candidate_dir = self._paths.candidates / prompt.stem
        candidate_dir.mkdir(parents=True, exist_ok=True)
        path = candidate_dir / f"attempt_{attempt:03d}.{suffix}.usage.json"
        path.write_text(json.dumps(usage_event, indent=2), encoding="utf-8")


class ChutesDslGenerator(ChutesGenerator):
    """Chutes-backed image -> DSL -> compiled Three.js generator."""

    def __init__(self, *, project_root: Path, paths: HarnessPaths, settings: ChutesSettings) -> None:
        super().__init__(project_root=project_root, paths=paths, settings=settings)
        self._dsl_examples = _load_dsl_examples(project_root)

    def generate(self, prompt: PromptSpec, prompt_path: Path, seed: int) -> bytes:
        analysis, analysis_usage = self._analyze(prompt=prompt, prompt_path=prompt_path, seed=seed)
        analysis_path = self._paths.analysis / f"{prompt.stem}.json"
        analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        self._write_usage_event(prompt=prompt, attempt=1, suffix="analysis", usage_event=analysis_usage)

        dsl, dsl_usage = self._generate_dsl(prompt=prompt, analysis=analysis, seed=seed)
        self._write_usage_event(prompt=prompt, attempt=1, suffix="dsl_generate", usage_event=dsl_usage)
        try:
            return self._compile_and_persist_dsl(prompt=prompt, dsl=dsl, attempt=1, suffix="attempt")
        except RuntimeError as exc:
            fixed = self._repair_dsl_compile_error(prompt=prompt, dsl=dsl, error=str(exc), attempt=1)
            return self._compile_and_persist_dsl(prompt=prompt, dsl=fixed, attempt=1, suffix="compile_fix")

    def repair(
        self,
        prompt: PromptSpec,
        prompt_path: Path,
        seed: int,
        previous_source: str,
        validator_result: dict,
    ) -> bytes:
        analysis_path = self._paths.analysis / f"{prompt.stem}.json"
        analysis = _read_json_if_exists(analysis_path, default={})
        previous_dsl = self._latest_dsl(prompt)
        prompt_text = f"""Repair this procedural object DSL so the compiled Three.js passes validation.

Return strict JSON only. Do not include markdown. Do not emit JavaScript.

Seed: {seed}
Prompt stem: {prompt.stem}
Scene analysis:
{json.dumps(analysis, indent=2)}

Validator result:
{json.dumps(validator_result, indent=2)}

Current DSL:
{json.dumps(previous_dsl, indent=2)}

Compiled source excerpt:
```js
{previous_source[:12000]}
```
"""
        result = self._client.chat_complete(
            model=self._settings.code_model,
            messages=[
                {"role": "system", "content": self._dsl_system_prompt()},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.05,
            label="dsl_repair",
        )
        attempt = _next_repair_attempt(self._paths.candidates / prompt.stem)
        self._write_usage_event(prompt=prompt, attempt=attempt, suffix="dsl_repair", usage_event=result.usage_event)
        try:
            dsl = _extract_dsl_document(result.content)
        except (json.JSONDecodeError, RuntimeError) as exc:
            dsl = self._repair_dsl_json_error(
                prompt=prompt,
                raw=result.content,
                error=str(exc),
                attempt=attempt,
                fallback=previous_dsl if isinstance(previous_dsl, dict) else _fallback_dsl(),
            )
        try:
            return self._compile_and_persist_dsl(prompt=prompt, dsl=dsl, attempt=attempt, suffix="repair")
        except RuntimeError as exc:
            fixed = self._repair_dsl_compile_error(prompt=prompt, dsl=dsl, error=str(exc), attempt=attempt)
            return self._compile_and_persist_dsl(prompt=prompt, dsl=fixed, attempt=attempt, suffix="repair_compile_fix")

    def visual_repair(
        self,
        *,
        prompt: PromptSpec,
        prompt_path: Path,
        render_path: Path,
        seed: int,
        current_source: str,
        critique: dict,
        repair_index: int,
    ) -> bytes:
        repair_plan, plan_usage = self._visual_repair_plan(
            prompt=prompt,
            prompt_path=prompt_path,
            render_path=render_path,
            critique=critique,
            repair_index=repair_index,
        )
        self._write_usage_event(
            prompt=prompt,
            attempt=repair_index,
            suffix="dsl_visual_plan",
            usage_event=plan_usage,
        )
        previous_dsl = self._latest_dsl(prompt)
        prompt_text = f"""Patch this procedural object DSL to improve rendered visual match.

Return strict JSON only. Do not include markdown. Do not emit JavaScript.

Seed: {seed}
Prompt stem: {prompt.stem}
Current critique:
{json.dumps(critique, indent=2)}

Structured visual repair plan:
{json.dumps(repair_plan, indent=2)}

Current DSL:
{json.dumps(previous_dsl, indent=2)}

Current compiled source excerpt:
```js
{current_source[:12000]}
```

Patch priorities:
- Correct object category, silhouette, proportions, orientation, and missing parts.
- Prefer low-poly primitive decomposition and procedural textures.
- Stay within the supported DSL schema and validator-safe limits.
- If the critique score is below 3, the repaired DSL must make structural edits:
  add or remove parts, change part kinds, alter profile/curve points, or change material/texture.
- Do not return an unchanged DSL unless the critique score is already 6 or higher.
"""
        result = self._client.chat_complete(
            model=self._settings.code_model,
            messages=[
                {"role": "system", "content": self._dsl_system_prompt()},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.08,
            label="dsl_visual_repair",
        )
        self._write_usage_event(
            prompt=prompt,
            attempt=repair_index,
            suffix="dsl_visual_repair",
            usage_event=result.usage_event,
        )
        try:
            dsl = _extract_dsl_document(result.content)
        except (json.JSONDecodeError, RuntimeError) as exc:
            dsl = self._repair_dsl_json_error(
                prompt=prompt,
                raw=result.content,
                error=str(exc),
                attempt=repair_index,
                fallback=previous_dsl if isinstance(previous_dsl, dict) else _fallback_dsl(),
            )
        try:
            return self._compile_and_persist_dsl(prompt=prompt, dsl=dsl, attempt=repair_index, suffix="visual_repair")
        except RuntimeError as exc:
            fixed = self._repair_dsl_compile_error(prompt=prompt, dsl=dsl, error=str(exc), attempt=repair_index)
            return self._compile_and_persist_dsl(
                prompt=prompt,
                dsl=fixed,
                attempt=repair_index,
                suffix="visual_repair_compile_fix",
            )

    def _generate_dsl(self, *, prompt: PromptSpec, analysis: dict, seed: int) -> tuple[dict, dict]:
        prompt_text = f"""Generate a procedural object DSL JSON document for this prompt analysis.

Return strict JSON only. Do not include markdown. Do not emit JavaScript.

Seed: {seed}
Generation strategy: {_generation_strategy_text()}
Prompt stem: {prompt.stem}
Scene analysis:
{json.dumps(analysis, indent=2)}

The DSL will be compiled into Three.js. It must describe a low-poly 3D object as parameterized primitives,
curves, lathes, instanced details, edge accents, materials, and procedural textures.

Hard mapping rules:
- Every high-priority analysis part and every must_have_detail must be represented by at least one DSL part.
- Multi-part objects must not collapse to a single primitive. Use at least 4 parts for objects with appendages,
  handles, straps, lids, wheels, legs, labels, or visible surface details.
- Build in this order: primary silhouette/body, defining appendages, repeated details, material/texture cues.
- Use tube parts for handles, straps, cords, rails, spouts, horns, and curved appendages.
- Use lathe parts for vessels, bottles, bulbs, bowls, pumpkins, knobs, lamps, and rotational bodies.
- Use line_edges and instanced details for seams, panels, stitching, buckles, vents, ticks, rivets, and logos.
"""
        result = self._client.chat_complete(
            model=self._settings.code_model,
            messages=[
                {"role": "system", "content": self._dsl_system_prompt()},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.12,
            label="dsl_generate",
        )
        try:
            dsl = _extract_dsl_document(result.content)
        except (json.JSONDecodeError, RuntimeError) as exc:
            dsl = self._repair_dsl_json_error(
                prompt=prompt,
                raw=result.content,
                error=str(exc),
                attempt=1,
                fallback=_fallback_dsl(),
            )
        return dsl, result.usage_event

    def _compile_and_persist_dsl(self, *, prompt: PromptSpec, dsl: dict, attempt: int, suffix: str) -> bytes:
        dsl_path = self._paths.dsl / f"{prompt.stem}.{suffix}_{attempt:03d}.json"
        dsl_path.write_text(json.dumps(dsl, indent=2), encoding="utf-8")
        try:
            source = compile_dsl_document(dsl)
        except DslError as exc:
            raise RuntimeError(f"DSL compilation failed: {exc}") from exc
        return source.encode("utf-8")

    def _repair_dsl_compile_error(self, *, prompt: PromptSpec, dsl: dict, error: str, attempt: int) -> dict:
        prompt_text = f"""Fix this procedural object DSL so it compiles.

Return strict JSON only. Do not include markdown. Do not emit JavaScript.

Prompt stem: {prompt.stem}
Compilation error:
{error}

Current DSL:
{json.dumps(dsl, indent=2)}
"""
        result = self._client.chat_complete(
            model=self._settings.code_model,
            messages=[
                {"role": "system", "content": self._dsl_system_prompt()},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.03,
            label="dsl_compile_fix",
        )
        self._write_usage_event(
            prompt=prompt,
            attempt=attempt,
            suffix="dsl_compile_fix",
            usage_event=result.usage_event,
        )
        try:
            return _extract_dsl_document(result.content)
        except (json.JSONDecodeError, RuntimeError):
            return dsl

    def _repair_dsl_json_error(
        self,
        *,
        prompt: PromptSpec,
        raw: str,
        error: str,
        attempt: int,
        fallback: dict,
    ) -> dict:
        prompt_text = f"""Convert this malformed procedural DSL response into strict JSON.

Return strict JSON only. Do not include markdown. Do not emit JavaScript.

Prompt stem: {prompt.stem}
JSON parse error:
{error}

Malformed response:
```text
{raw[:16000]}
```
"""
        result = self._client.chat_complete(
            model=self._settings.code_model,
            messages=[
                {"role": "system", "content": self._dsl_system_prompt()},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.02,
            label="dsl_json_fix",
        )
        self._write_usage_event(
            prompt=prompt,
            attempt=attempt,
            suffix="dsl_json_fix",
            usage_event=result.usage_event,
        )
        try:
            return _extract_dsl_document(result.content)
        except (json.JSONDecodeError, RuntimeError):
            return fallback

    def _latest_dsl(self, prompt: PromptSpec) -> dict:
        paths = sorted(self._paths.dsl.glob(f"{prompt.stem}.*.json"))
        if not paths:
            return {"version": 1, "materials": [{"id": "body", "type": "standard", "color": "#cccccc"}], "parts": []}
        return _read_json_if_exists(paths[-1], default={})

    def _dsl_system_prompt(self) -> str:
        return f"""You emit only JSON for a procedural object DSL that compiles to validator-compliant Three.js.

Do not emit JavaScript. Do not include markdown fences. The root object must have:
- version: 1
- materials: non-empty array
- parts: non-empty array

Supported material types: basic, standard, physical, line, points.
Material fields: id, type, color (#RRGGBB), roughness, metalness, clearcoat, clearcoat_roughness, size.
Optional material texture: kind stripe/checker/speckle, size 8-512, scale 1-128, color_a, color_b.

Supported part kinds:
- primitive: material, shape, position, rotation, scale
- lathe: material, profile [[radius,y],...], segments, position, rotation, scale
- tube: material, points [[x,y,z],...], radius, tubular_segments, radial_segments, closed, position, rotation, scale
- line_edges: material, shape, position, rotation, scale
- instanced: material, shape, count, pattern, instance_scale, position, rotation, scale

Supported primitive shape kinds:
- box: size [x,y,z]
- sphere: radius, width_segments, height_segments
- cylinder: radius_top, radius_bottom, height, radial_segments
- cone: radius, height, radial_segments
- capsule: radius, length, cap_segments, radial_segments
- torus: radius, tube, radial_segments, tubular_segments

Instanced patterns:
- line: spacing
- ring: radius, y
- grid: columns, spacing

Coordinate conventions: Y-up, +Z forward, object centered near origin. The compiler normalizes to [-0.5,0.5].
Prefer recognizable silhouette and distinctive parts over high detail. Use simple numeric values and compact arrays.
Never use a single primitive fallback for a multi-part object. If the analysis lists handles, spouts, straps,
wheels, legs, lids, doors, buttons, seams, labels, or surface markings, include separate DSL parts for them.
When repairing a low-scoring render, make explicit structural changes; unchanged DSL is invalid for score below 6.

Example DSL documents:
{self._dsl_examples}
"""


class ChutesCritic:
    """Chutes-backed prompt-vs-render visual critic."""

    def __init__(self, *, settings: ChutesSettings) -> None:
        self._settings = settings
        self._client = ChutesClient(settings)

    def critique(self, *, prompt: PromptSpec, prompt_path: Path, render_path: Path) -> dict:
        result = self._client.chat_complete(
            model=self._settings.vision_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You critique procedural Three.js renders against source prompt images. "
                        "Return compact JSON only. Do not include markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Compare the rendered 3D object to the prompt image. "
                                "Score strict visual match from 0 to 10. Return JSON with keys: "
                                "score, object_match, silhouette, geometry_errors, missing_parts, "
                                "wrong_parts, color_material_errors, orientation_errors, "
                                "recommended_fix, concise_prompt_for_code_repair. "
                                f"Prompt stem: {prompt.stem}."
                            ),
                        },
                        {"type": "text", "text": "Prompt image:"},
                        {"type": "image_url", "image_url": {"url": _image_data_url(prompt_path)}},
                        {"type": "text", "text": "Rendered Three.js output:"},
                        {"type": "image_url", "image_url": {"url": _image_data_url(render_path)}},
                    ],
                },
            ],
            temperature=0.05,
            label="critique",
        )
        parsed = _parse_jsonish_or_raw(result.content)
        critique = parsed if isinstance(parsed, dict) else {"raw": result.content}
        critique["_chutes_usage"] = result.usage_event
        return critique


class LocalRenderService:
    """Starts render-service-js and renders source files through /render/grid."""

    def __init__(
        self,
        *,
        repo_root: Path,
        port: int = DEFAULT_RENDER_PORT,
        static_port: int = DEFAULT_RENDER_STATIC_PORT,
        startup_timeout_seconds: float = 120.0,
        request_timeout_seconds: float = 60.0,
    ) -> None:
        self._service_root = repo_root / "render-service-js"
        self._port = port
        self._static_port = static_port
        self._startup_timeout_seconds = startup_timeout_seconds
        self._request_timeout_seconds = request_timeout_seconds
        self._process: subprocess.Popen[str] | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def __enter__(self) -> "LocalRenderService":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def start(self) -> None:
        if not self._service_root.exists():
            raise RuntimeError(f"render-service-js not found at {self._service_root}")
        if not (self._service_root / "node_modules").exists():
            raise RuntimeError(f"render-service-js dependencies missing. Run `npm install` in {self._service_root}")

        env = os.environ.copy()
        env.update(
            {
                "PORT": str(self._port),
                "STATIC_PORT": str(self._static_port),
                "RENDER_POOL_SIZE": "1",
                "VALIDATION_POOL_SIZE": "1",
            }
        )
        self._process = subprocess.Popen(  # noqa: S603
            ["node", "src/server.js"],
            cwd=self._service_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_until_ready()

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        self._process = None

    def render_grid(self, source: str, destination: Path) -> None:
        try:
            data = self._render_grid_bytes(source)
        except RuntimeError as exc:
            if "render-service-js request failed" not in str(exc):
                raise
            self.stop()
            self.start()
            data = self._render_grid_bytes(source)

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    def _render_grid_bytes(self, source: str) -> bytes:
        payload = json.dumps({"source": source}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/render/grid",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._request_timeout_seconds) as response:  # noqa: S310
                data = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"render-service-js HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"render-service-js request failed: {exc}") from exc

        return data

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._startup_timeout_seconds
        last_error = ""
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                output = self._read_process_output()
                raise RuntimeError(f"render-service-js exited during startup:\n{output}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/ping", timeout=2) as response:  # noqa: S310
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = str(exc)
            time.sleep(0.5)
        output = self._read_process_output()
        raise RuntimeError(f"render-service-js did not become ready: {last_error}\n{output}")

    def _read_process_output(self) -> str:
        if self._process is None or self._process.stdout is None:
            return ""
        try:
            return self._process.stdout.read()
        except OSError:
            return ""


def read_prompt_list(path: Path, limit: int | None = None) -> list[PromptSpec]:
    """Read prompt URLs from a text file."""
    prompts: list[PromptSpec] = []
    seen: set[str] = set()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        prompt = prompt_from_url(line)
        if prompt.stem in seen:
            continue
        seen.add(prompt.stem)
        prompts.append(prompt)

        if limit is not None and len(prompts) >= limit:
            break

    return prompts


def prompt_from_url(url: str) -> PromptSpec:
    """Derive the prompt stem from the URL path."""
    parsed = urlparse(url)
    filename = Path(parsed.path).name
    if not filename:
        raise ValueError(f"URL has no filename: {url}")

    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Prompt URL must end in .png/.jpg/.jpeg: {url}")

    return PromptSpec(url=url, stem=Path(filename).stem.lower(), extension=suffix)


def download_prompt(prompt: PromptSpec, destination: Path, timeout_seconds: float) -> tuple[bool, str | None]:
    """Download one prompt image unless it already exists."""
    if destination.exists() and destination.stat().st_size > 0:
        return True, None

    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        request = urllib.request.Request(prompt.url, headers={"User-Agent": "404-gen-local-harness/0.1"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            data = response.read()
        tmp.write_bytes(data)
        tmp.replace(destination)
        return True, None
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        if tmp.exists():
            tmp.unlink()
        return False, str(exc)


def run_validator(project_root: Path, js_path: Path, timeout_seconds: float) -> dict:
    """Run the miner-reference validator CLI and return its JSON output."""
    command = ["node", "tools/validate.js", "--json", str(js_path)]
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )

    if completed.stdout.strip():
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            pass

    return {
        "file": str(js_path),
        "passed": False,
        "stagesRun": [],
        "failures": [
            {
                "stage": "harness",
                "rule": "VALIDATOR_INVOCATION_FAILED",
                "detail": (completed.stderr or completed.stdout or f"exit code {completed.returncode}").strip(),
            }
        ],
        "metrics": None,
        "moduleLoadMs": None,
        "executionMs": None,
        "totalMs": 0,
    }


def run_harness(
    *,
    project_root: Path,
    prompt_list: Path,
    runs_dir: Path,
    run_id: str | None,
    limit: int | None,
    seed: int,
    download_timeout_seconds: float,
    validator_timeout_seconds: float,
    overwrite: bool,
    generator_name: str,
    max_attempts: int,
    render: bool,
    critique: bool,
    visual_repair_attempts: int,
    critique_threshold: float,
    render_port: int,
    render_static_port: int,
) -> Path:
    """Execute one local harness run and return the run directory."""
    project_root = project_root.resolve()
    prompt_list = _resolve_under_project(project_root, prompt_list)
    runs_dir = _resolve_under_project(project_root, runs_dir)
    reference_js = _resolve_under_project(project_root, REFERENCE_JS)

    prompts = read_prompt_list(prompt_list, limit=limit)
    if not prompts:
        raise RuntimeError(f"No prompt URLs found in {prompt_list}")

    run_name = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = runs_dir / run_name
    if run_root.exists() and overwrite:
        shutil.rmtree(run_root)
    elif run_root.exists():
        raise FileExistsError(f"Run directory already exists: {run_root}")

    paths = HarnessPaths.create(run_root)
    if generator_name == "stub":
        generator = StubGenerator(reference_js=reference_js)
    elif generator_name == "chutes":
        generator = ChutesGenerator(project_root=project_root, paths=paths, settings=ChutesSettings.from_env())
    elif generator_name == "chutes-dsl":
        generator = ChutesDslGenerator(project_root=project_root, paths=paths, settings=ChutesSettings.from_env())
    else:
        raise ValueError(f"Unknown generator: {generator_name}")

    critic = ChutesCritic(settings=ChutesSettings.from_env_require(vision=True, code=False)) if critique else None

    items: list[dict] = []
    failed_manifest: dict[str, str] = {}
    start = time.monotonic()
    renderer: LocalRenderService | None = None

    try:
        if render:
            renderer = LocalRenderService(
                repo_root=project_root.parent,
                port=render_port,
                static_port=render_static_port,
            )
            renderer.start()

        for index, prompt in enumerate(prompts, start=1):
            _process_prompt(
                index=index,
                total=len(prompts),
                prompt=prompt,
                paths=paths,
                run_root=run_root,
                project_root=project_root,
                generator=generator,
                renderer=renderer,
                critic=critic,
                seed=seed,
                download_timeout_seconds=download_timeout_seconds,
                validator_timeout_seconds=validator_timeout_seconds,
                max_attempts=max_attempts,
                visual_repair_attempts=visual_repair_attempts,
                critique_threshold=critique_threshold,
                items=items,
                failed_manifest=failed_manifest,
            )
    finally:
        if renderer is not None:
            renderer.stop()

    elapsed = time.monotonic() - start
    summary = _build_summary(run_root, items, failed_manifest, total=len(prompts), seed=seed)
    summary["elapsed_seconds"] = round(elapsed, 3)
    _write_summary_files(run_root, summary, failed_manifest)
    return run_root


def _process_prompt(
    *,
    index: int,
    total: int,
    prompt: PromptSpec,
    paths: HarnessPaths,
    run_root: Path,
    project_root: Path,
    generator: StubGenerator | ChutesGenerator,
    renderer: LocalRenderService | None,
    critic: ChutesCritic | None,
    seed: int,
    download_timeout_seconds: float,
    validator_timeout_seconds: float,
    max_attempts: int,
    visual_repair_attempts: int,
    critique_threshold: float,
    items: list[dict],
    failed_manifest: dict[str, str],
) -> None:
        prompt_path = paths.prompts / f"{prompt.stem}{prompt.extension}"
        item: dict = {
            "stem": prompt.stem,
            "prompt_url": prompt.url,
            "prompt_path": _relative_to_run(prompt_path, run_root),
            "status": "pending",
            "attempts": 1,
        }

        downloaded, error = download_prompt(prompt, prompt_path, timeout_seconds=download_timeout_seconds)
        if not downloaded:
            item["status"] = "failed"
            item["failure"] = f"prompt download failed: {error}"
            failed_manifest[prompt.stem] = item["failure"]
            items.append(item)
            _write_progress_summary(run_root, items, failed_manifest, total=total, seed=seed)
            return

        candidate_dir = paths.candidates / prompt.stem
        candidate_dir.mkdir(parents=True, exist_ok=True)
        attempts: list[dict] = []
        validator: dict | None = None
        candidate_path: Path | None = None
        previous_source = ""

        for attempt in range(1, max_attempts + 1):
            candidate_path = candidate_dir / f"attempt_{attempt:03d}.js"
            if attempt == 1:
                source_bytes = generator.generate(prompt, prompt_path, seed)
            else:
                source_bytes = generator.repair(prompt, prompt_path, seed, previous_source, validator or {})

            candidate_path.write_bytes(source_bytes)
            previous_source = source_bytes.decode("utf-8", errors="replace")

            validator = run_validator(project_root, candidate_path, timeout_seconds=validator_timeout_seconds)
            validator = _reject_bad_validator_metrics(validator)
            validation_path = candidate_dir / f"attempt_{attempt:03d}.validate.json"
            validation_path.write_text(json.dumps(validator, indent=2), encoding="utf-8")
            attempts.append(
                {
                    "attempt": attempt,
                    "candidate_js": _relative_to_run(candidate_path, run_root),
                    "validator_json": _relative_to_run(validation_path, run_root),
                    "validator": _summarize_validator(validator),
                }
            )
            if validator.get("passed") is True:
                break

        item["attempts"] = len(attempts)
        item["attempt_history"] = attempts
        if attempts:
            item.update(
                {
                    "candidate_js": attempts[-1]["candidate_js"],
                    "validator_json": attempts[-1]["validator_json"],
                    "validator": attempts[-1]["validator"],
                }
            )

        if validator and validator.get("passed") is True and candidate_path is not None:
            final_path = paths.results / f"{prompt.stem}.js"
            shutil.copyfile(candidate_path, final_path)
            item["status"] = "passed"
            item["final_js"] = _relative_to_run(final_path, run_root)
            if renderer is not None:
                _render_critique_and_visual_repair(
                    prompt=prompt,
                    prompt_path=prompt_path,
                    paths=paths,
                    run_root=run_root,
                    project_root=project_root,
                    generator=generator,
                    renderer=renderer,
                    critic=critic,
                    item=item,
                    final_path=final_path,
                    seed=seed,
                    validator_timeout_seconds=validator_timeout_seconds,
                    visual_repair_attempts=visual_repair_attempts,
                    critique_threshold=critique_threshold,
                )
        else:
            item["status"] = "failed"
            item["failure"] = _validator_failure_message(validator or {})
            failed_manifest[prompt.stem] = item["failure"]

        items.append(item)
        render_suffix = " render_failed" if item.get("render_error") else ""
        print(f"[{index}/{total}] {prompt.stem}: {item['status']}{render_suffix}", flush=True)
        _write_progress_summary(run_root, items, failed_manifest, total=total, seed=seed)


def _render_critique_and_visual_repair(
    *,
    prompt: PromptSpec,
    prompt_path: Path,
    paths: HarnessPaths,
    run_root: Path,
    project_root: Path,
    generator: StubGenerator | ChutesGenerator,
    renderer: LocalRenderService,
    critic: ChutesCritic | None,
    item: dict,
    final_path: Path,
    seed: int,
    validator_timeout_seconds: float,
    visual_repair_attempts: int,
    critique_threshold: float,
) -> None:
    best_source = final_path.read_text(encoding="utf-8")
    best_score = -1.0
    visual_history: list[dict] = []

    for visual_attempt in range(0, visual_repair_attempts + 1):
        suffix = "initial" if visual_attempt == 0 else f"repair_{visual_attempt:03d}"
        render_path = paths.renders / f"{prompt.stem}.{suffix}.png"
        try:
            renderer.render_grid(best_source, render_path)
            item["render_png"] = _relative_to_run(render_path, run_root)
        except Exception as exc:
            item["render_error"] = str(exc)
            return

        critique_result: dict = {}
        critique_score = 0.0
        if critic is not None:
            try:
                critique_result = critic.critique(prompt=prompt, prompt_path=prompt_path, render_path=render_path)
                critique_path = paths.critiques / f"{prompt.stem}.{suffix}.json"
                critique_path.write_text(json.dumps(critique_result, indent=2), encoding="utf-8")
                item["critique_json"] = _relative_to_run(critique_path, run_root)
                item["critique"] = _summarize_critique(critique_result)
                critique_score = _critique_score(critique_result)
            except Exception as exc:
                item["critique_error"] = str(exc)

        visual_history.append(
            {
                "visual_attempt": visual_attempt,
                "render_png": item.get("render_png"),
                "critique_json": item.get("critique_json"),
                "critique_score": critique_score,
            }
        )
        if critique_score >= best_score:
            best_score = critique_score
            final_path.write_text(best_source, encoding="utf-8")

        if critic is None or critique_score >= critique_threshold or visual_attempt >= visual_repair_attempts:
            break
        if not isinstance(generator, ChutesGenerator):
            break

        repair_index = visual_attempt + 1
        candidate_dir = paths.candidates / prompt.stem
        repair_path = candidate_dir / f"visual_repair_{repair_index:03d}.js"
        try:
            repaired = generator.visual_repair(
                prompt=prompt,
                prompt_path=prompt_path,
                render_path=render_path,
                seed=seed,
                current_source=best_source,
                critique=critique_result,
                repair_index=repair_index,
            )
        except Exception as exc:
            item["visual_repair_error"] = str(exc)
            break

        repair_path.write_bytes(repaired)
        repaired_validator = run_validator(project_root, repair_path, timeout_seconds=validator_timeout_seconds)
        repaired_validator = _reject_bad_validator_metrics(repaired_validator)
        repair_validation_path = candidate_dir / f"visual_repair_{repair_index:03d}.validate.json"
        repair_validation_path.write_text(json.dumps(repaired_validator, indent=2), encoding="utf-8")
        visual_history[-1]["repair_candidate_js"] = _relative_to_run(repair_path, run_root)
        visual_history[-1]["repair_validator_json"] = _relative_to_run(repair_validation_path, run_root)
        visual_history[-1]["repair_validator"] = _summarize_validator(repaired_validator)
        if repaired_validator.get("passed") is True:
            best_source = repaired.decode("utf-8", errors="replace")
            item["final_js"] = _relative_to_run(final_path, run_root)
        else:
            fixed = _repair_invalid_visual_candidate(
                prompt=prompt,
                prompt_path=prompt_path,
                project_root=project_root,
                paths=paths,
                run_root=run_root,
                generator=generator,
                seed=seed,
                repair_index=repair_index,
                previous_source=repaired.decode("utf-8", errors="replace"),
                validator_result=repaired_validator,
                validator_timeout_seconds=validator_timeout_seconds,
            )
            if fixed is None:
                item["visual_repair_error"] = _validator_failure_message(repaired_validator)
                break
            best_source = fixed
            item["final_js"] = _relative_to_run(final_path, run_root)

    if visual_history:
        item["visual_history"] = visual_history


def _repair_invalid_visual_candidate(
    *,
    prompt: PromptSpec,
    prompt_path: Path,
    project_root: Path,
    paths: HarnessPaths,
    run_root: Path,
    generator: ChutesGenerator,
    seed: int,
    repair_index: int,
    previous_source: str,
    validator_result: dict,
    validator_timeout_seconds: float,
) -> str | None:
    candidate_dir = paths.candidates / prompt.stem
    repaired_path = candidate_dir / f"visual_repair_{repair_index:03d}.validator_fix.js"
    try:
        fixed_bytes = generator.repair(
            prompt=prompt,
            prompt_path=prompt_path,
            seed=seed,
            previous_source=previous_source,
            validator_result=validator_result,
        )
    except Exception:
        return None
    repaired_path.write_bytes(fixed_bytes)
    fixed_validator = run_validator(project_root, repaired_path, timeout_seconds=validator_timeout_seconds)
    fixed_validator = _reject_bad_validator_metrics(fixed_validator)
    fixed_validation_path = candidate_dir / f"visual_repair_{repair_index:03d}.validator_fix.validate.json"
    fixed_validation_path.write_text(json.dumps(fixed_validator, indent=2), encoding="utf-8")
    if fixed_validator.get("passed") is not True:
        return None
    return fixed_bytes.decode("utf-8", errors="replace")


def _resolve_under_project(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _relative_to_run(path: Path, run_root: Path) -> str:
    return path.relative_to(run_root).as_posix()


def _summarize_validator(validator: dict) -> dict:
    return {
        "passed": bool(validator.get("passed")),
        "metrics": validator.get("metrics"),
        "failures": validator.get("failures", []),
        "moduleLoadMs": validator.get("moduleLoadMs"),
        "executionMs": validator.get("executionMs"),
        "totalMs": validator.get("totalMs"),
    }


def _reject_bad_validator_metrics(validator: dict) -> dict:
    if validator.get("passed") is not True:
        return validator
    metrics = validator.get("metrics") or {}
    bbox = metrics.get("bbox") or {}
    values: list[object] = []
    for side in ("min", "max"):
        point = bbox.get(side) or {}
        values.extend(point.get(axis) for axis in ("x", "y", "z"))
    if all(isinstance(value, (int, float)) for value in values):
        return validator
    fixed = dict(validator)
    fixed["passed"] = False
    failures = list(fixed.get("failures") or [])
    failures.append(
        {
            "stage": "harness",
            "rule": "NON_FINITE_VALIDATOR_METRICS",
            "detail": "Validator passed but returned a non-finite or incomplete bounding box.",
        }
    )
    fixed["failures"] = failures
    return fixed


def _summarize_critique(critique: dict) -> dict:
    return {
        "score": critique.get("score"),
        "object_match": critique.get("object_match"),
        "recommended_fix": critique.get("recommended_fix"),
        "missing_parts": critique.get("missing_parts"),
        "geometry_errors": critique.get("geometry_errors"),
        "color_material_errors": critique.get("color_material_errors"),
        "orientation_errors": critique.get("orientation_errors"),
    }


def _critique_score(critique: dict) -> float:
    return _coerce_float(critique.get("score"), default=0.0)


def _coerce_float(value: object, *, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _model_prices_from_env() -> dict[str, dict[str, float]]:
    prices = dict(DEFAULT_MODEL_PRICES_USD_PER_M_TOKENS)
    raw = os.environ.get("CHUTES_MODEL_PRICES_JSON")
    if not raw:
        return prices
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("CHUTES_MODEL_PRICES_JSON must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("CHUTES_MODEL_PRICES_JSON must be a JSON object")
    for model, value in parsed.items():
        if not isinstance(value, dict):
            continue
        input_price = value.get("input", value.get("prompt"))
        output_price = value.get("output", value.get("completion"))
        if input_price is None or output_price is None:
            continue
        prices[str(model)] = {"input": float(input_price), "output": float(output_price)}
    return prices


def _build_usage_event(*, label: str, model: str, usage: dict, prices: dict[str, dict[str, float]]) -> dict:
    prompt_tokens = _int_token_count(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _int_token_count(usage, "completion_tokens", "output_tokens")
    total_tokens = _int_token_count(usage, "total_tokens") or prompt_tokens + completion_tokens
    price = prices.get(model, {})
    input_cost = prompt_tokens * float(price.get("input", 0.0)) / 1_000_000
    output_cost = completion_tokens * float(price.get("output", 0.0)) / 1_000_000
    return {
        "label": label,
        "model": model,
        "usage": usage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(input_cost + output_cost, 8),
        "pricing_assumption": "USD per 1M input/output tokens",
    }


def _int_token_count(usage: dict, *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return 0


def _next_repair_attempt(candidate_dir: Path) -> int:
    existing = sorted(candidate_dir.glob("attempt_*.js"))
    return len(existing) + 1


def collect_usage_events(run_root: Path) -> list[dict]:
    """Collect persisted Chutes usage events for a run."""
    events: list[dict] = []
    for path in sorted(run_root.glob("candidates/**/*.usage.json")):
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(event, dict):
            event["path"] = _relative_to_run(path, run_root)
            events.append(event)
    for path in sorted(run_root.glob("critiques/*.json")):
        try:
            critique = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        event = critique.get("_chutes_usage") if isinstance(critique, dict) else None
        if isinstance(event, dict):
            event = dict(event)
            event["path"] = _relative_to_run(path, run_root)
            events.append(event)
    return events


def summarize_usage(events: list[dict]) -> dict:
    """Summarize token usage and estimated cost."""
    by_model: dict[str, dict] = {}
    for event in events:
        model = str(event.get("model", "unknown"))
        row = by_model.setdefault(
            model,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
        )
        row["calls"] += 1
        row["prompt_tokens"] += int(event.get("prompt_tokens") or 0)
        row["completion_tokens"] += int(event.get("completion_tokens") or 0)
        row["total_tokens"] += int(event.get("total_tokens") or 0)
        row["estimated_cost_usd"] += float(event.get("estimated_cost_usd") or 0.0)
    total = {
        "calls": sum(row["calls"] for row in by_model.values()),
        "prompt_tokens": sum(row["prompt_tokens"] for row in by_model.values()),
        "completion_tokens": sum(row["completion_tokens"] for row in by_model.values()),
        "total_tokens": sum(row["total_tokens"] for row in by_model.values()),
        "estimated_cost_usd": round(sum(row["estimated_cost_usd"] for row in by_model.values()), 8),
    }
    for row in by_model.values():
        row["estimated_cost_usd"] = round(row["estimated_cost_usd"], 8)
    return {"total": total, "by_model": by_model}


def _validator_failure_message(validator: dict) -> str:
    failures = validator.get("failures") or []
    if not failures:
        return "validator failed without failure details"
    first = failures[0]
    return f"{first.get('stage', 'unknown')}/{first.get('rule', 'unknown')}: {first.get('detail', '')}"


def _build_summary(
    run_root: Path,
    items: list[dict],
    failed_manifest: dict[str, str],
    *,
    total: int,
    seed: int,
) -> dict:
    passed = sum(1 for item in items if item.get("status") == "passed")
    failed = sum(1 for item in items if item.get("status") == "failed")
    return {
        "run_id": run_root.name,
        "seed": seed,
        "total": total,
        "processed": len(items),
        "passed": passed,
        "failed": failed,
        "pending": total - len(items),
        "results_dir": "results",
        "failed_manifest": "results/_failed.json" if failed_manifest else None,
        "items": items,
    }


def _write_progress_summary(
    run_root: Path,
    items: list[dict],
    failed_manifest: dict[str, str],
    *,
    total: int,
    seed: int,
) -> None:
    summary = _build_summary(run_root, items, failed_manifest, total=total, seed=seed)
    usage_events = collect_usage_events(run_root)
    if usage_events:
        summary["usage"] = summarize_usage(usage_events)
    _write_summary_files(run_root, summary, failed_manifest)


def _write_summary_files(run_root: Path, summary: dict, failed_manifest: dict[str, str]) -> None:
    usage_events = collect_usage_events(run_root)
    if usage_events:
        summary["usage"] = summarize_usage(usage_events)
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_root / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")

    failed_path = run_root / "results" / "_failed.json"
    if failed_manifest:
        failed_path.write_text(json.dumps(failed_manifest, indent=2), encoding="utf-8")
    elif failed_path.exists():
        failed_path.unlink()


def _summary_markdown(summary: dict) -> str:
    lines = [
        f"# Harness Run {summary['run_id']}",
        "",
        f"- Seed: `{summary['seed']}`",
        f"- Processed: `{summary['processed']}/{summary['total']}`",
        f"- Passed: `{summary['passed']}`",
        f"- Failed: `{summary['failed']}`",
        "",
        "| Stem | Status | Vertices | Draw Calls | Render | Score | Failure |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for item in summary["items"]:
        metrics = (item.get("validator") or {}).get("metrics") or {}
        failure = item.get("failure", "")
        render_cell = item.get("render_png") or item.get("render_error", "")
        score = (item.get("critique") or {}).get("score", "")
        lines.append(
            "| {stem} | {status} | {vertices} | {draw_calls} | {render} | {score} | {failure} |".format(
                stem=item["stem"],
                status=item["status"],
                vertices=metrics.get("vertices", ""),
                draw_calls=metrics.get("drawCalls", ""),
                render=str(render_cell).replace("|", "\\|"),
                score=score,
                failure=failure.replace("|", "\\|"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _parse_jsonish(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_fence(stripped)
    stripped = _strip_json_trailing_commas(stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(_strip_json_trailing_commas(stripped[start : end + 1]))
        raise


def _parse_jsonish_or_raw(text: str) -> object:
    try:
        return _parse_jsonish(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _extract_dsl_document(text: str) -> dict:
    parsed = _parse_jsonish(text)
    if not isinstance(parsed, dict):
        raise RuntimeError("Generated DSL response was not a JSON object")
    return parsed


def _read_json_if_exists(path: Path, *, default: object) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _load_dsl_examples(project_root: Path) -> str:
    examples_dir = project_root / "examples" / "dsl"
    snippets: list[str] = []
    for path in sorted(examples_dir.glob("*.json"))[:4]:
        try:
            snippets.append(f"{path.name}:\n{path.read_text(encoding='utf-8')[:5000]}")
        except OSError:
            continue
    return "\n\n".join(snippets)


def _extract_js_module(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_fence(stripped).strip()
    if "export default" not in stripped:
        raise RuntimeError("Generated response did not contain `export default`")
    return _trim_js_module(stripped) + "\n"


def _strip_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    for index, line in enumerate(lines):
        if line.strip().startswith("```"):
            lines = lines[:index]
            break
    return "\n".join(lines)


def _trim_js_module(text: str) -> str:
    start = text.find("export default")
    if start < 0:
        return text.strip()
    lines = text[start:].splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index > 0 and (stripped.startswith("```") or stripped.startswith("### ") or stripped.startswith("Explanation:")):
            return "\n".join(lines[:index]).strip()
    return "\n".join(lines).strip()


def _strip_json_trailing_commas(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"(^|\s)//.*?$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    text = re.sub(r"([{\[,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)", r'\1"\2"\3', text)
    return text


def _fallback_dsl() -> dict:
    return {
        "version": 1,
        "materials": [{"id": "body", "type": "standard", "color": "#cccccc", "roughness": 0.6, "metalness": 0}],
        "parts": [
            {
                "kind": "primitive",
                "material": "body",
                "shape": {"kind": "sphere", "radius": 0.35, "width_segments": 24, "height_segments": 16},
                "scale": [1, 0.8, 1],
            }
        ],
    }


def _code_system_prompt(rules: str, example: str) -> str:
    example_block = ""
    if os.environ.get("ENSEMBLE_INCLUDE_STYLE_EXAMPLE", "1").strip().lower() in {"1", "true", "yes", "on"}:
        example_block = f"""
Known-good validator fixture for syntax style only. Do not copy this object,
comments, materials, dimensions, helpers, or part names unless the prompt image
is actually this exact low-poly red car:
```js
{example}
```
"""
    return f"""You generate procedural Three.js modules for a strict validator.

Hard rules:
{rules}
{example_block}

Return only complete JavaScript modules. Do not include markdown or explanations.
The object must be derived from the current prompt image and scene analysis, not
from any fixture or placeholder."""


def _generation_strategy_text() -> str:
    strategy = os.environ.get("CHUTES_GENERATION_STRATEGY", "balanced").strip().lower()
    strategies = {
        "balanced": (
            "Balanced reconstruction. Prioritize a validator-passing object with the correct category, "
            "major silhouette, colors, and 3-6 distinctive details."
        ),
        "silhouette": (
            "Silhouette-first reconstruction. Spend geometry on the outer contour, proportions, and "
            "orientation before adding surface decoration."
        ),
        "detail": (
            "Detail-rich reconstruction. Add repeated subparts, seams, handles, rims, feet, labels, "
            "surface patterning, and material variation while staying inside validator budgets."
        ),
        "primitive_assembly": (
            "Primitive assembly reconstruction. Use many simple Box/Cylinder/Sphere/Torus/Tube parts "
            "with explicit transforms instead of risky custom math."
        ),
        "part_graph_rich_js": (
            "Generic part-graph reconstruction. First infer a category-agnostic graph: primary body, "
            "secondary volumes, appendages, loops/handles, supports, openings, repeated details, and "
            "material regions. The final JS must instantiate visible geometry for every high-priority "
            "part in that graph, using richer Three.js operators such as LatheGeometry, TubeGeometry, "
            "TorusGeometry, ExtrudeGeometry, LineSegments, InstancedMesh, and DataTexture where useful."
        ),
        "self_consistency": (
            "Self-consistency reconstruction. Identify the stable parts that multiple plausible 3D "
            "interpretations would share, then generate JS for those robust parts first. Avoid single-body "
            "collapse: if handles, straps, spouts, legs, lids, seams, buttons, labels, or repeated details "
            "are visible, they must become separate geometry or procedural texture regions."
        ),
        "critic_before_code": (
            "Critic-before-code reconstruction. Before writing geometry, anticipate likely judging failures: "
            "wrong object identity, missing defining parts, weak silhouette, orientation errors, and material "
            "mismatch. Generate JS that explicitly prevents those failures, favoring recognizable topology "
            "over generic polished shapes."
        ),
        "multi_candidate": (
            "Multi-candidate reconstruction. Internally draft at least three different procedural design plans "
            "from the image, reject generic or single-body plans, then write only the strongest final JS. Favor "
            "the candidate with the clearest silhouette, most defining parts, and safest validator-compliant "
            "geometry."
        ),
        "ranked_no_repair": (
            "Ranked first-pass reconstruction. Spend effort before code: list plausible object identity, visible "
            "parts, silhouette axes, repeated details, and materials; rank two implementation plans; then write "
            "the best one as final JS. Do not rely on later repair, so the first valid output must be complete."
        ),
        "ranked_with_repair": (
            "Ranked reconstruction with repair awareness. Build a strong first-pass object from a ranked part "
            "plan, leaving clear editable part groups and names so any visual repair can make targeted structural "
            "changes instead of returning unchanged code."
        ),
        "dense_geometry": (
            "Dense low-poly reconstruction. Use more visible primitives and procedural details than the balanced "
            "strategy: bevel-like layered boxes, rims, seams, handles, legs, straps, spokes, panels, labels, "
            "surface bands, and repeated instanced details where the prompt supports them. Stay below validator "
            "budgets but avoid coarse blocky one-piece outputs."
        ),
    }
    return strategies.get(strategy, strategies["balanced"])


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _template_source_for_analysis(analysis: dict) -> str | None:
    text = json.dumps(analysis, sort_keys=True).lower()
    source: str | None = None
    if _has_any_term(text, ("teapot", "tea pot", "kettle")):
        source = _teapot_template()
    elif _has_any_term(text, ("dumbbell", "barbell", "weight")):
        source = _dumbbell_template()
    elif _has_any_term(text, ("boat", "canoe", "rowboat", "skiff")):
        source = _boat_template()
    elif _has_any_term(text, ("candle holder", "candelabra", "candle")):
        source = _candle_template()
    elif _has_any_term(text, ("lamp", "lighting", "lantern")):
        source = _lamp_template()
    elif _has_any_term(text, ("chair", "table", "furniture", "stool", "bench")):
        source = _furniture_template()
    elif _has_any_term(text, ("spaceship", "space ship", "spacecraft", "rocket", "space")):
        source = _spacecraft_template()
    elif _has_any_term(text, ("weapon", "firearm", "sword", "blade", "tool", "fantasy weapon")):
        source = _weapon_tool_template()
    elif _has_any_term(text, ("toy", "stuffed", "animal", "character", "creature")):
        source = _toy_animal_template()
    elif _has_any_term(text, ("accessory", "accessories", "jewelry", "ornament", "ring", "pendant", "headwear", "hat")):
        source = _accessory_template()
    elif _has_any_term(text, ("door", "gate", "portal")):
        source = _door_template()
    elif _has_any_term(text, ("mechanical device", "household appliance", "appliance", "electronics", "optical device", "camera", "device")):
        source = _device_template()
    elif _has_any_term(text, ("storage", "storage container", "box", "crate", "backpack", "bag")):
        source = _storage_template()
    elif _has_any_term(text, ("vehicle", "car", "truck", "suv")):
        source = _vehicle_template()
    elif _has_any_term(text, ("lighthouse", "tower", "architecture", "building")):
        source = _tower_template()
    elif _has_any_term(text, ("vase", "bottle", "chalice", "goblet", "container", "cookware", "pot", "pan", "ceramic", "glassware")):
        source = _vessel_template()
    return _apply_color_hint(source, text) if source else None


def _has_any_term(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        pattern = r"(?<![a-z0-9])" + re.escape(term.lower()).replace(r"\ ", r"[\s_-]+") + r"(?![a-z0-9])"
        if re.search(pattern, text):
            return True
    return False


def _apply_color_hint(source: str, text: str) -> str:
    color = _dominant_color_hex(text)
    if not color:
        return source
    secondary = _darker_hex(color)
    replacements = {
        "0x2f5f91": color,
        "0x214766": secondary,
        "0x3b5f7a": color,
        "0x28475f": secondary,
        "0xf07a22": color,
        "0x6b3f24": color,
        "0x7c4a2b": secondary,
        "0xb97845": color,
        "0x8c5632": secondary,
        "0x9a5b2f": color,
        "0x8b4e2a": color,
        "0xb56a3a": color,
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def _dominant_color_hex(text: str) -> str | None:
    colors = [
        ("purple", "0x7b3fc8"),
        ("violet", "0x7b3fc8"),
        ("pink", "0xff7aa8"),
        ("red", "0xc83a32"),
        ("orange", "0xe87928"),
        ("yellow", "0xe0b72f"),
        ("gold", "0xd0a13a"),
        ("green", "0x3f8f4e"),
        ("blue", "0x2f5f91"),
        ("cyan", "0x2aa6b8"),
        ("black", "0x1f1f1f"),
        ("white", "0xe8e2d8"),
        ("silver", "0xb8b8b8"),
        ("gray", "0x777777"),
        ("grey", "0x777777"),
        ("brown", "0x7a4a2a"),
    ]
    for name, value in colors:
        if name in text:
            return value
    return None


def _darker_hex(hex_literal: str) -> str:
    value = int(hex_literal.replace("0x", ""), 16)
    r = int(((value >> 16) & 255) * 0.72)
    g = int(((value >> 8) & 255) * 0.72)
    b = int((value & 255) * 0.72)
    return f"0x{r:02x}{g:02x}{b:02x}"


def _shared_template_helpers() -> str:
    return """
function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z);
  const scale = maxDim > 0 ? 0.95 / maxDim : 1;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}

function makeStripeTexture(THREE, colorA, colorB) {
  const size = 32;
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const useA = ((x + y * 2) % 12) < 6;
      const color = useA ? colorA : colorB;
      const i = (y * size + x) * 4;
      data[i] = (color >> 16) & 255;
      data[i + 1] = (color >> 8) & 255;
      data[i + 2] = color & 255;
      data[i + 3] = 255;
    }
  }
  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  texture.needsUpdate = true;
  return texture;
}
"""


def _teapot_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const ceramicTexture = makeStripeTexture(THREE, 0x6b3f24, 0x7c4a2b);
  const bodyMat = new THREE.MeshPhysicalMaterial({{ color: 0xffffff, map: ceramicTexture, roughness: 0.32, metalness: 0.15, clearcoat: 0.45 }});
  const metalMat = new THREE.MeshStandardMaterial({{ color: 0xb8b8aa, roughness: 0.18, metalness: 0.85 }});
  const darkMat = new THREE.MeshStandardMaterial({{ color: 0x1f1b17, roughness: 0.45, metalness: 0.2 }});

  const body = new THREE.Mesh(new THREE.SphereGeometry(0.36, 48, 24), bodyMat);
  body.scale.set(1.12, 0.82, 0.92);
  body.position.y = 0.02;
  root.add(body);

  const shoulder = new THREE.Mesh(new THREE.SphereGeometry(0.28, 40, 16), bodyMat);
  shoulder.scale.set(1.0, 0.34, 0.88);
  shoulder.position.y = 0.23;
  root.add(shoulder);

  const lid = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.27, 0.07, 48), metalMat);
  lid.position.y = 0.43;
  root.add(lid);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.055, 24, 12), darkMat);
  knob.scale.y = 0.8;
  knob.position.y = 0.5;
  root.add(knob);

  const base = new THREE.Mesh(new THREE.TorusGeometry(0.27, 0.035, 16, 56), metalMat);
  base.rotation.x = Math.PI / 2;
  base.position.y = -0.28;
  root.add(base);

  const handleCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.32, 0.25, 0),
    new THREE.Vector3(0.55, 0.2, 0),
    new THREE.Vector3(0.58, -0.12, 0),
    new THREE.Vector3(0.32, -0.18, 0),
  ]);
  const handle = new THREE.Mesh(new THREE.TubeGeometry(handleCurve, 36, 0.035, 12, false), metalMat);
  root.add(handle);

  const spoutCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.3, 0.13, 0),
    new THREE.Vector3(-0.5, 0.18, 0),
    new THREE.Vector3(-0.63, 0.28, 0),
  ]);
  const spout = new THREE.Mesh(new THREE.TubeGeometry(spoutCurve, 28, 0.045, 14, false), bodyMat);
  root.add(spout);
  const spoutLip = new THREE.Mesh(new THREE.TorusGeometry(0.055, 0.012, 8, 24), metalMat);
  spoutLip.position.set(-0.64, 0.29, 0);
  spoutLip.rotation.y = Math.PI / 2;
  root.add(spoutLip);
  for (let i = 0; i < 12; i++) {{
    const angle = (i / 12) * Math.PI * 2;
    const rivet = new THREE.Mesh(new THREE.SphereGeometry(0.018, 10, 8), metalMat);
    rivet.scale.y = 0.5;
    rivet.position.set(Math.cos(angle) * 0.31, 0.27, Math.sin(angle) * 0.25);
    root.add(rivet);
  }}
  for (const y of [-0.18, 0.12, 0.34]) {{
    const band = new THREE.Mesh(new THREE.TorusGeometry(0.31, 0.01, 8, 56), metalMat);
    band.rotation.x = Math.PI / 2;
    band.scale.z = 0.82;
    band.position.y = y;
    root.add(band);
  }}

  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _dumbbell_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const rubber = new THREE.MeshStandardMaterial({{ color: 0x181818, roughness: 0.72, metalness: 0.05 }});
  const grip = new THREE.MeshStandardMaterial({{ color: 0x5c5c58, roughness: 0.35, metalness: 0.75 }});
  const accent = new THREE.MeshStandardMaterial({{ color: 0x2b2b2b, roughness: 0.8, metalness: 0.0 }});

  const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.72, 32), grip);
  bar.rotation.z = Math.PI / 2;
  root.add(bar);

  for (const side of [-1, 1]) {{
    for (let i = 0; i < 3; i++) {{
      const plate = new THREE.Mesh(new THREE.CylinderGeometry(0.18 - i * 0.018, 0.18 - i * 0.018, 0.07, 48), rubber);
      plate.rotation.z = Math.PI / 2;
      plate.position.x = side * (0.28 + i * 0.055);
      root.add(plate);
    }}
    const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.055, 32), grip);
    collar.rotation.z = Math.PI / 2;
    collar.position.x = side * 0.2;
    root.add(collar);
    const rim = new THREE.Mesh(new THREE.TorusGeometry(0.145, 0.012, 8, 48), accent);
    rim.rotation.y = Math.PI / 2;
    rim.position.x = side * 0.39;
    root.add(rim);
  }}

  for (let i = -4; i <= 4; i++) {{
    const groove = new THREE.Mesh(new THREE.TorusGeometry(0.047, 0.003, 6, 18), accent);
    groove.rotation.y = Math.PI / 2;
    groove.position.x = i * 0.026;
    root.add(groove);
  }}

  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _boat_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const hullMat = new THREE.MeshPhysicalMaterial({{ color: 0x8b4e2a, roughness: 0.5, metalness: 0.02, clearcoat: 0.2 }});
  const innerMat = new THREE.MeshStandardMaterial({{ color: 0xd0a06a, roughness: 0.65, metalness: 0.0 }});
  const darkMat = new THREE.MeshStandardMaterial({{ color: 0x3a2417, roughness: 0.7, metalness: 0.0 }});

  const hull = new THREE.Mesh(new THREE.SphereGeometry(0.46, 48, 16, 0, Math.PI * 2, Math.PI * 0.45, Math.PI * 0.5), hullMat);
  hull.scale.set(0.72, 0.28, 1.75);
  hull.rotation.x = Math.PI;
  hull.position.y = -0.03;
  root.add(hull);

  const rimLeft = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.035, 0.86), darkMat);
  rimLeft.position.set(-0.27, 0.11, 0);
  root.add(rimLeft);
  const rimRight = rimLeft.clone();
  rimRight.position.x = 0.27;
  root.add(rimRight);

  for (const z of [-0.23, 0.02, 0.27]) {{
    const seat = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.035, 0.065), innerMat);
    seat.position.set(0, 0.13, z);
    root.add(seat);
  }}

  const oarCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.45, 0.2, -0.34),
    new THREE.Vector3(0.0, 0.18, 0.02),
    new THREE.Vector3(0.48, 0.2, 0.36),
  ]);
  const oar = new THREE.Mesh(new THREE.TubeGeometry(oarCurve, 16, 0.018, 8, false), darkMat);
  root.add(oar);
  const paddle = new THREE.Mesh(new THREE.SphereGeometry(0.08, 18, 8), innerMat);
  paddle.scale.set(0.55, 0.08, 1.15);
  paddle.position.set(0.55, 0.2, 0.43);
  root.add(paddle);

  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _lamp_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const shadeTexture = makeStripeTexture(THREE, 0xf1dfb5, 0xd2b783);
  const brass = new THREE.MeshStandardMaterial({{ color: 0xb8893f, roughness: 0.32, metalness: 0.7 }});
  const shadeMat = new THREE.MeshPhysicalMaterial({{ color: 0xffffff, map: shadeTexture, roughness: 0.55, metalness: 0.0, clearcoat: 0.1, transparent: true, opacity: 0.92 }});
  const dark = new THREE.MeshStandardMaterial({{ color: 0x27211a, roughness: 0.55, metalness: 0.2 }});

  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.28, 0.06, 48), brass);
  base.position.y = -0.38;
  root.add(base);
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.045, 0.54, 32), brass);
  stem.position.y = -0.08;
  root.add(stem);
  const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.2, 0.28, 48, 1, true), shadeMat);
  shade.position.y = 0.26;
  root.add(shade);
  const topRing = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.012, 8, 48), dark);
  topRing.rotation.x = Math.PI / 2;
  topRing.position.y = 0.4;
  root.add(topRing);
  const bottomRing = new THREE.Mesh(new THREE.TorusGeometry(0.3, 0.012, 8, 48), dark);
  bottomRing.rotation.x = Math.PI / 2;
  bottomRing.position.y = 0.12;
  root.add(bottomRing);
  const finial = new THREE.Mesh(new THREE.SphereGeometry(0.04, 20, 12), brass);
  finial.position.y = 0.47;
  root.add(finial);
  for (let j = 0; j < 5; j++) {{
    const band = new THREE.Mesh(new THREE.TorusGeometry(0.205 + j * 0.024, 0.004, 6, 48), brass);
    band.rotation.x = Math.PI / 2;
    band.position.y = 0.145 + j * 0.052;
    root.add(band);
  }}
  for (let i = 0; i < 4; i++) {{
    const strut = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.3, 8), brass);
    strut.position.set(Math.cos(i * Math.PI / 2) * 0.18, 0.25, Math.sin(i * Math.PI / 2) * 0.18);
    strut.rotation.z = 0.18 * Math.cos(i * Math.PI / 2);
    strut.rotation.x = -0.18 * Math.sin(i * Math.PI / 2);
    root.add(strut);
  }}
  for (let i = 0; i < 12; i++) {{
    const bead = new THREE.Mesh(new THREE.SphereGeometry(0.018, 10, 8), brass);
    const angle = (i / 12) * Math.PI * 2;
    bead.position.set(Math.cos(angle) * 0.235, -0.36, Math.sin(angle) * 0.235);
    root.add(bead);
  }}
  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _candle_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const wax = new THREE.MeshPhysicalMaterial({{ color: 0xfff2d0, roughness: 0.6, metalness: 0.0, clearcoat: 0.08 }});
  const brass = new THREE.MeshStandardMaterial({{ color: 0xb58b42, roughness: 0.34, metalness: 0.75 }});
  const flameMat = new THREE.MeshBasicMaterial({{ color: 0xffb347 }});
  const dark = new THREE.MeshStandardMaterial({{ color: 0x2a2018, roughness: 0.65, metalness: 0.1 }});
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.34, 0.055, 56), brass);
  base.position.y = -0.38;
  root.add(base);
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.045, 0.26, 32), brass);
  stem.position.y = -0.23;
  root.add(stem);
  for (const x of [-0.22, 0, 0.22]) {{
    const cup = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.11, 0.055, 40), brass);
    cup.position.set(x, -0.08 + (x === 0 ? 0.09 : 0), 0);
    root.add(cup);
    const candle = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.05, 0.32, 32), wax);
    candle.position.set(x, 0.1 + (x === 0 ? 0.09 : 0), 0);
    root.add(candle);
    const wick = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.04, 8), dark);
    wick.position.set(x, 0.28 + (x === 0 ? 0.09 : 0), 0);
    root.add(wick);
    const flame = new THREE.Mesh(new THREE.SphereGeometry(0.04, 16, 10), flameMat);
    flame.scale.set(0.65, 1.35, 0.65);
    flame.position.set(x, 0.34 + (x === 0 ? 0.09 : 0), 0);
    root.add(flame);
  }}
  for (const x of [-0.11, 0.11]) {{
    const armCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, -0.16, 0),
      new THREE.Vector3(x, -0.08, 0),
      new THREE.Vector3(x * 2, -0.08, 0),
    ]);
    const arm = new THREE.Mesh(new THREE.TubeGeometry(armCurve, 24, 0.018, 8, false), brass);
    root.add(arm);
  }}
  for (let i = 0; i < 10; i++) {{
    const bead = new THREE.Mesh(new THREE.SphereGeometry(0.014, 8, 6), brass);
    const angle = (i / 10) * Math.PI * 2;
    bead.position.set(Math.cos(angle) * 0.23, -0.35, Math.sin(angle) * 0.23);
    root.add(bead);
  }}
  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _furniture_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const wood = new THREE.MeshStandardMaterial({{ color: 0x9a5b2f, roughness: 0.62, metalness: 0.02 }});
  const dark = new THREE.MeshStandardMaterial({{ color: 0x3a2416, roughness: 0.7, metalness: 0.0 }});

  const seat = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.08, 0.42), wood);
  seat.position.y = 0.0;
  root.add(seat);
  const back = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.48, 0.07), wood);
  back.position.set(0, 0.25, -0.2);
  back.rotation.x = -0.15;
  root.add(back);
  for (const x of [-0.19, 0.19]) {{
    for (const z of [-0.15, 0.15]) {{
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.032, 0.42, 16), dark);
      leg.position.set(x, -0.25, z);
      leg.rotation.x = z > 0 ? 0.08 : -0.08;
      leg.rotation.z = x > 0 ? -0.08 : 0.08;
      root.add(leg);
    }}
  }}
  for (const x of [-0.19, 0.19]) {{
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.55, 16), dark);
    post.position.set(x, 0.23, -0.22);
    root.add(post);
  }}
  for (const y of [0.18, 0.32]) {{
    const rail = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.035, 0.04), dark);
    rail.position.set(0, y, -0.25);
    root.add(rail);
  }}
  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _vehicle_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const paintTexture = makeStripeTexture(THREE, 0x2f5f91, 0x214766);
  const bodyMat = new THREE.MeshPhysicalMaterial({{ color: 0x2f5f91, roughness: 0.36, metalness: 0.2, clearcoat: 0.55 }});
  bodyMat.map = paintTexture;
  const glass = new THREE.MeshPhysicalMaterial({{ color: 0x7fb3c9, roughness: 0.08, metalness: 0.0, clearcoat: 0.8, transparent: true, opacity: 0.72 }});
  const tire = new THREE.MeshStandardMaterial({{ color: 0x111111, roughness: 0.85, metalness: 0.0 }});
  const metal = new THREE.MeshStandardMaterial({{ color: 0xb8b8b8, roughness: 0.28, metalness: 0.75 }});

  const chassis = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.18, 0.28), bodyMat);
  chassis.position.y = -0.05;
  root.add(chassis);
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.2, 0.25), bodyMat);
  cabin.position.set(-0.05, 0.12, 0);
  root.add(cabin);
  const windshield = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.11, 0.255), glass);
  windshield.position.set(-0.27, 0.14, 0);
  root.add(windshield);
  const rearWindow = windshield.clone();
  rearWindow.position.x = 0.14;
  root.add(rearWindow);
  const sidePanel = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.09, 0.012), new THREE.MeshStandardMaterial({{ color: 0xf6f0ff, roughness: 0.35, metalness: 0.0 }}));
  sidePanel.position.set(0.05, 0.03, 0.147);
  root.add(sidePanel);
  const sidePanelR = sidePanel.clone();
  sidePanelR.position.z = -0.147;
  root.add(sidePanelR);
  for (const z of [-0.154, 0.154]) {{
    for (let i = 0; i < 5; i++) {{
      const stripe = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.095, 0.01), new THREE.MeshStandardMaterial({{ color: [0xff3040, 0xffb000, 0xfee440, 0x42d66b, 0x3777ff][i], roughness: 0.4, metalness: 0.0 }}));
      stripe.position.set(-0.03 + i * 0.035, 0.034, z);
      root.add(stripe);
    }}
    const horn = new THREE.Mesh(new THREE.ConeGeometry(0.025, 0.11, 16), new THREE.MeshStandardMaterial({{ color: 0xfff1a8, roughness: 0.28, metalness: 0.25 }}));
    horn.rotation.z = -Math.PI / 2;
    horn.position.set(-0.42, 0.12, z * 0.55);
    root.add(horn);
    const ear = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.08, 12), new THREE.MeshStandardMaterial({{ color: 0xffa0c8, roughness: 0.48, metalness: 0.0 }}));
    ear.rotation.z = Math.PI;
    ear.position.set(-0.25, 0.27, z * 0.8);
    root.add(ear);
  }}
  for (const x of [-0.25, 0.25]) {{
    for (const z of [-0.16, 0.16]) {{
      const wheel = new THREE.Mesh(new THREE.TorusGeometry(0.085, 0.028, 14, 32), tire);
      wheel.rotation.y = Math.PI / 2;
      wheel.position.set(x, -0.16, z);
      root.add(wheel);
      const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.018, 20), metal);
      hub.rotation.z = Math.PI / 2;
      hub.position.set(x, -0.16, z);
      root.add(hub);
    }}
  }}
  const bumperA = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.06, 0.3), metal);
  bumperA.position.set(-0.4, -0.08, 0);
  root.add(bumperA);
  const bumperB = bumperA.clone();
  bumperB.position.x = 0.4;
  root.add(bumperB);
  for (const z of [-0.155, 0.155]) {{
    const rail = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.5, 10), metal);
    rail.rotation.z = Math.PI / 2;
    rail.position.set(0, 0.27, z);
    root.add(rail);
  }}
  for (const x of [-0.34, 0.34]) {{
    for (const z of [-0.08, 0.08]) {{
      const light = new THREE.Mesh(new THREE.SphereGeometry(0.025, 12, 8), new THREE.MeshStandardMaterial({{ color: x < 0 ? 0xfff1b8 : 0xaa1020, roughness: 0.2, metalness: 0.0 }}));
      light.scale.z = 0.35;
      light.position.set(x, -0.02, z);
      root.add(light);
    }}
  }}
  for (let i = 0; i < 8; i++) {{
    const vent = new THREE.Mesh(new THREE.BoxGeometry(0.006, 0.035, 0.18), metal);
    vent.position.set(-0.08 + i * 0.02, 0.235, -0.002);
    root.add(vent);
  }}
  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _spacecraft_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const hullMat = new THREE.MeshPhysicalMaterial({{ color: 0xd9dde6, roughness: 0.32, metalness: 0.45, clearcoat: 0.25 }});
  const dark = new THREE.MeshStandardMaterial({{ color: 0x202632, roughness: 0.5, metalness: 0.35 }});
  const glow = new THREE.MeshStandardMaterial({{ color: 0x4cc9ff, roughness: 0.2, metalness: 0.0 }});
  const accent = new THREE.MeshStandardMaterial({{ color: 0xff5a3d, roughness: 0.38, metalness: 0.2 }});

  const body = new THREE.Mesh(new THREE.SphereGeometry(0.23, 48, 20), hullMat);
  body.scale.set(1.9, 0.46, 0.62);
  root.add(body);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.15, 0.28, 36), hullMat);
  nose.rotation.z = -Math.PI / 2;
  nose.position.x = -0.48;
  root.add(nose);
  const engine = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.18, 0.12, 36), dark);
  engine.rotation.z = Math.PI / 2;
  engine.position.x = 0.48;
  root.add(engine);
  for (const z of [-0.19, 0.19]) {{
    const wing = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.035, 0.22), hullMat);
    wing.position.set(0.08, -0.02, z);
    wing.rotation.y = z > 0 ? -0.28 : 0.28;
    root.add(wing);
    const tip = new THREE.Mesh(new THREE.ConeGeometry(0.045, 0.14, 16), accent);
    tip.rotation.x = z > 0 ? Math.PI / 2 : -Math.PI / 2;
    tip.position.set(0.1, -0.025, z * 1.7);
    root.add(tip);
  }}
  for (let i = 0; i < 7; i++) {{
    const porthole = new THREE.Mesh(new THREE.SphereGeometry(0.028, 12, 8), glow);
    porthole.scale.y = 0.45;
    porthole.position.set(-0.22 + i * 0.07, 0.09, -0.145);
    root.add(porthole);
  }}
  for (let i = 0; i < 8; i++) {{
    const panel = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.012, 0.24), dark);
    panel.position.set(-0.28 + i * 0.075, 0.118, 0);
    root.add(panel);
  }}
  for (const y of [-0.06, 0.06]) {{
    const thruster = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.08, 18), glow);
    thruster.rotation.z = Math.PI / 2;
    thruster.position.set(0.56, y, 0);
    root.add(thruster);
  }}
  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _weapon_tool_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const metal = new THREE.MeshStandardMaterial({{ color: 0xbfc3c7, roughness: 0.22, metalness: 0.9 }});
  const dark = new THREE.MeshStandardMaterial({{ color: 0x272018, roughness: 0.68, metalness: 0.1 }});
  const accent = new THREE.MeshStandardMaterial({{ color: 0x8a1f1f, roughness: 0.42, metalness: 0.25 }});

  const blade = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.055, 0.035), metal);
  blade.position.x = -0.12;
  root.add(blade);
  const tip = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.16, 4), metal);
  tip.rotation.z = -Math.PI / 2;
  tip.position.x = -0.51;
  root.add(tip);
  const spine = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.018, 0.045), new THREE.MeshStandardMaterial({{ color: 0x777d82, roughness: 0.3, metalness: 0.8 }}));
  spine.position.set(-0.05, 0.025, 0);
  root.add(spine);
  const guard = new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.26, 0.06), accent);
  guard.position.x = 0.22;
  root.add(guard);
  const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.05, 0.32, 24), dark);
  handle.rotation.z = Math.PI / 2;
  handle.position.x = 0.42;
  root.add(handle);
  const pommel = new THREE.Mesh(new THREE.SphereGeometry(0.06, 18, 12), accent);
  pommel.position.x = 0.61;
  root.add(pommel);
  for (let i = 0; i < 7; i++) {{
    const wrap = new THREE.Mesh(new THREE.TorusGeometry(0.052, 0.004, 6, 20), accent);
    wrap.rotation.y = Math.PI / 2;
    wrap.position.x = 0.3 + i * 0.035;
    root.add(wrap);
  }}
  for (let i = 0; i < 6; i++) {{
    const notch = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.025, 0.045), dark);
    notch.position.set(-0.32 + i * 0.065, 0.047, 0);
    root.add(notch);
  }}
  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _toy_animal_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const furTex = makeStripeTexture(THREE, 0xb97845, 0x8c5632);
  const fur = new THREE.MeshPhysicalMaterial({{ color: 0xffffff, map: furTex, roughness: 0.85, metalness: 0.0, clearcoat: 0.05 }});
  const dark = new THREE.MeshStandardMaterial({{ color: 0x17120f, roughness: 0.5, metalness: 0.0 }});
  const accent = new THREE.MeshStandardMaterial({{ color: 0xff7aa8, roughness: 0.55, metalness: 0.0 }});
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.24, 40, 20), fur);
  body.scale.set(0.9, 1.18, 0.72);
  body.position.y = -0.05;
  root.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.2, 40, 18), fur);
  head.position.y = 0.28;
  root.add(head);
  for (const x of [-0.12, 0.12]) {{
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.026, 12, 8), dark);
    eye.position.set(x, 0.32, -0.17);
    root.add(eye);
    const ear = new THREE.Mesh(new THREE.ConeGeometry(0.075, 0.16, 18), fur);
    ear.position.set(x * 1.3, 0.49, 0);
    ear.rotation.z = x > 0 ? -0.28 : 0.28;
    root.add(ear);
    const arm = new THREE.Mesh(new THREE.CapsuleGeometry(0.045, 0.2, 8, 16), fur);
    arm.position.set(x * 2.25, 0.02, -0.02);
    arm.rotation.z = x > 0 ? -0.45 : 0.45;
    root.add(arm);
    const foot = new THREE.Mesh(new THREE.SphereGeometry(0.07, 18, 10), fur);
    foot.scale.set(1.2, 0.55, 0.85);
    foot.position.set(x * 1.1, -0.35, -0.06);
    root.add(foot);
  }}
  const snout = new THREE.Mesh(new THREE.SphereGeometry(0.07, 18, 10), new THREE.MeshStandardMaterial({{ color: 0xe9c19d, roughness: 0.75, metalness: 0.0 }}));
  snout.scale.set(1.1, 0.72, 0.62);
  snout.position.set(0, 0.25, -0.19);
  root.add(snout);
  const nose = new THREE.Mesh(new THREE.SphereGeometry(0.025, 12, 8), dark);
  nose.position.set(0, 0.27, -0.235);
  root.add(nose);
  const bow = new THREE.Mesh(new THREE.TorusGeometry(0.08, 0.012, 8, 24), accent);
  bow.scale.set(1.45, 0.45, 0.25);
  bow.position.set(0, 0.09, -0.22);
  root.add(bow);
  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _device_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const bodyMat = new THREE.MeshPhysicalMaterial({{ color: 0xf07a22, roughness: 0.45, metalness: 0.25, clearcoat: 0.35 }});
  const dark = new THREE.MeshStandardMaterial({{ color: 0x1f2428, roughness: 0.58, metalness: 0.2 }});
  const glass = new THREE.MeshPhysicalMaterial({{ color: 0x89c7ff, roughness: 0.08, metalness: 0.0, transparent: true, opacity: 0.72, clearcoat: 0.8 }});
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.42, 0.22), bodyMat);
  root.add(body);
  const screen = new THREE.Mesh(new THREE.BoxGeometry(0.27, 0.18, 0.014), glass);
  screen.position.set(-0.05, 0.05, -0.118);
  root.add(screen);
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 0.08, 40), dark);
  lens.rotation.x = Math.PI / 2;
  lens.position.set(0.16, 0.02, -0.15);
  root.add(lens);
  const lensGlass = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.065, 0.012, 32), glass);
  lensGlass.rotation.x = Math.PI / 2;
  lensGlass.position.set(0.16, 0.02, -0.195);
  root.add(lensGlass);
  for (let i = 0; i < 6; i++) {{
    const button = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.018, 16), dark);
    button.rotation.x = Math.PI / 2;
    button.position.set(-0.19 + i * 0.075, -0.16, -0.125);
    root.add(button);
  }}
  for (let i = 0; i < 5; i++) {{
    const vent = new THREE.Mesh(new THREE.BoxGeometry(0.025, 0.006, 0.014), dark);
    vent.position.set(-0.2 + i * 0.055, 0.18, -0.125);
    root.add(vent);
  }}
  const knob = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.045, 24), dark);
  knob.position.set(0.17, 0.235, 0);
  root.add(knob);
  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _storage_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const fabric = new THREE.MeshPhysicalMaterial({{ color: 0x3b5f7a, roughness: 0.78, metalness: 0.0, map: makeStripeTexture(THREE, 0x3b5f7a, 0x28475f) }});
  const trim = new THREE.MeshStandardMaterial({{ color: 0x222222, roughness: 0.65, metalness: 0.05 }});
  const metal = new THREE.MeshStandardMaterial({{ color: 0xb0b7ba, roughness: 0.28, metalness: 0.75 }});
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.52, 0.22), fabric);
  body.position.y = -0.02;
  root.add(body);
  const frontPocket = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.22, 0.035), fabric);
  frontPocket.position.set(0, -0.07, -0.13);
  root.add(frontPocket);
  const flap = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.055, 0.04), trim);
  flap.position.set(0, 0.075, -0.155);
  root.add(flap);
  for (const x of [-0.17, 0.17]) {{
    const strapCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(x, 0.2, 0.13),
      new THREE.Vector3(x * 1.15, -0.05, 0.2),
      new THREE.Vector3(x, -0.28, 0.13),
    ]);
    const strap = new THREE.Mesh(new THREE.TubeGeometry(strapCurve, 24, 0.014, 8, false), trim);
    root.add(strap);
    const buckle = new THREE.Mesh(new THREE.TorusGeometry(0.035, 0.006, 6, 16), metal);
    buckle.scale.y = 0.65;
    buckle.position.set(x * 0.9, -0.15, -0.165);
    root.add(buckle);
  }}
  for (let i = 0; i < 8; i++) {{
    const stitch = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.006, 0.012), metal);
    stitch.position.set(-0.16 + i * 0.046, 0.11, -0.18);
    root.add(stitch);
  }}
  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _accessory_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const metal = new THREE.MeshPhysicalMaterial({{ color: 0xd6b45a, roughness: 0.24, metalness: 0.85, clearcoat: 0.35 }});
  const gem = new THREE.MeshPhysicalMaterial({{ color: 0x7b3fc8, roughness: 0.08, metalness: 0.0, clearcoat: 0.9, transparent: true, opacity: 0.86 }});
  const dark = new THREE.MeshStandardMaterial({{ color: 0x242018, roughness: 0.6, metalness: 0.2 }});
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.22, 0.028, 20, 80), metal);
  ring.rotation.x = Math.PI / 2;
  root.add(ring);
  const crown = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.12, 0.05, 32), metal);
  crown.position.y = 0.24;
  root.add(crown);
  const stone = new THREE.Mesh(new THREE.OctahedronGeometry(0.105, 2), gem);
  stone.scale.y = 0.75;
  stone.position.y = 0.31;
  root.add(stone);
  for (let i = 0; i < 12; i++) {{
    const bead = new THREE.Mesh(new THREE.SphereGeometry(0.022, 10, 8), metal);
    const angle = (i / 12) * Math.PI * 2;
    bead.position.set(Math.cos(angle) * 0.22, Math.sin(angle) * 0.22, 0);
    root.add(bead);
  }}
  for (const x of [-0.12, 0.12]) {{
    const prong = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, 0.14, 8), dark);
    prong.position.set(x, 0.28, 0);
    prong.rotation.z = x > 0 ? -0.25 : 0.25;
    root.add(prong);
  }}
  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _door_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const wood = new THREE.MeshStandardMaterial({{ color: 0x7a4a2a, roughness: 0.72, metalness: 0.02, map: makeStripeTexture(THREE, 0x7a4a2a, 0x55331d) }});
  const metal = new THREE.MeshStandardMaterial({{ color: 0xa88a42, roughness: 0.3, metalness: 0.75 }});
  const dark = new THREE.MeshStandardMaterial({{ color: 0x1d1711, roughness: 0.65, metalness: 0.1 }});
  const slab = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.72, 0.055), wood);
  root.add(slab);
  const arch = new THREE.Mesh(new THREE.TorusGeometry(0.215, 0.018, 10, 48, Math.PI), metal);
  arch.position.y = 0.36;
  arch.rotation.z = Math.PI;
  root.add(arch);
  for (const x of [-0.12, 0, 0.12]) {{
    const plank = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.68, 0.062), dark);
    plank.position.x = x;
    root.add(plank);
  }}
  for (const y of [-0.22, 0.02, 0.25]) {{
    const band = new THREE.Mesh(new THREE.BoxGeometry(0.47, 0.035, 0.07), metal);
    band.position.y = y;
    root.add(band);
  }}
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.035, 16, 10), metal);
  knob.position.set(0.13, -0.02, -0.045);
  root.add(knob);
  for (let i = 0; i < 10; i++) {{
    const rivet = new THREE.Mesh(new THREE.SphereGeometry(0.012, 8, 6), metal);
    rivet.position.set(-0.2 + i * 0.044, 0.25, -0.048);
    root.add(rivet);
  }}
  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _tower_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const wall = new THREE.MeshStandardMaterial({{ color: 0xd8d0bd, roughness: 0.72, metalness: 0.0 }});
  const roofMat = new THREE.MeshStandardMaterial({{ color: 0x9b2f25, roughness: 0.5, metalness: 0.05 }});
  const glass = new THREE.MeshPhysicalMaterial({{ color: 0x8fd0ff, roughness: 0.12, transparent: true, opacity: 0.68 }});

  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.24, 0.72, 48), wall);
  shaft.position.y = -0.05;
  root.add(shaft);
  const gallery = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.07, 48), roofMat);
  gallery.position.y = 0.34;
  root.add(gallery);
  const lantern = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.17, 32), glass);
  lantern.position.y = 0.46;
  root.add(lantern);
  const roof = new THREE.Mesh(new THREE.ConeGeometry(0.18, 0.16, 40), roofMat);
  roof.position.y = 0.62;
  root.add(roof);
  for (let i = 0; i < 6; i++) {{
    const win = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.085, 0.012), glass);
    const angle = i * Math.PI / 3;
    win.position.set(Math.cos(angle) * 0.171, 0.02, Math.sin(angle) * 0.171);
    win.rotation.y = -angle;
    root.add(win);
  }}
  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def _vessel_template() -> str:
    return f"""export default function generate(THREE) {{
  const root = new THREE.Group();
  const ceramic = new THREE.MeshPhysicalMaterial({{ color: 0xb56a3a, roughness: 0.38, metalness: 0.05, clearcoat: 0.35 }});
  const rimMat = new THREE.MeshStandardMaterial({{ color: 0xd7b26a, roughness: 0.28, metalness: 0.65 }});
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.28, 48, 24), ceramic);
  body.scale.set(0.88, 1.25, 0.88);
  body.position.y = -0.05;
  root.add(body);
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.17, 0.28, 48), ceramic);
  neck.position.y = 0.28;
  root.add(neck);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.13, 0.02, 12, 48), rimMat);
  rim.rotation.x = Math.PI / 2;
  rim.position.y = 0.43;
  root.add(rim);
  const foot = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.22, 0.07, 48), rimMat);
  foot.position.y = -0.42;
  root.add(foot);
  for (const x of [-1, 1]) {{
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(x * 0.19, 0.16, 0),
      new THREE.Vector3(x * 0.34, 0.05, 0),
      new THREE.Vector3(x * 0.22, -0.16, 0),
    ]);
    const handle = new THREE.Mesh(new THREE.TubeGeometry(curve, 24, 0.018, 10, false), rimMat);
    root.add(handle);
  }}
  fitToUnitCube(THREE, root);
  return root;
}}
{_shared_template_helpers()}"""


def stable_run_id(prefix: str, prompt_list: Path, seed: int, limit: int | None) -> str:
    """Build a deterministic run id for tests or reproducible experiments."""
    source = f"{prompt_list.resolve()}:{seed}:{limit}".encode()
    digest = hashlib.sha256(source).hexdigest()[:10]
    return f"{prefix}-{digest}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Three.js miner validation harness.")
    parser.add_argument("--prompt-list", type=Path, default=DEFAULT_PROMPT_LIST)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N prompt URLs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-timeout", type=float, default=60.0)
    parser.add_argument("--validator-timeout", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--generator", choices=["stub", "chutes", "chutes-dsl"], default="stub")
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--render", action="store_true", help="Render passing JS files to runs/<id>/renders/*.png")
    parser.add_argument("--critique", action="store_true", help="Use Chutes VLM to score prompt image vs render")
    parser.add_argument("--visual-repair-attempts", type=int, default=0)
    parser.add_argument("--critique-threshold", type=float, default=6.0)
    parser.add_argument("--render-port", type=int, default=DEFAULT_RENDER_PORT)
    parser.add_argument("--render-static-port", type=int, default=DEFAULT_RENDER_STATIC_PORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    try:
        run_root = run_harness(
            project_root=project_root,
            prompt_list=args.prompt_list,
            runs_dir=args.runs_dir,
            run_id=args.run_id,
            limit=args.limit,
            seed=args.seed,
            download_timeout_seconds=args.download_timeout,
            validator_timeout_seconds=args.validator_timeout,
            overwrite=args.overwrite,
            generator_name=args.generator,
            max_attempts=args.max_attempts,
            render=args.render,
            critique=args.critique,
            visual_repair_attempts=args.visual_repair_attempts,
            critique_threshold=args.critique_threshold,
            render_port=args.render_port,
            render_static_port=args.render_static_port,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Run written to {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
