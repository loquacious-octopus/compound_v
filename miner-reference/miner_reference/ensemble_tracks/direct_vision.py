"""Direct vision-to-JavaScript track prompts.

This module is intentionally data-only: it defines prompt text and a small
factory for the existing ``ensemble_core.TrackSpec`` shape, but performs no
model calls and imports no runner code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


TRACK_NAME = "direct_vision_code"
CANDIDATES = 3


class TrackSpecFactory(Protocol):
    """Callable compatible with ``ensemble_core.TrackSpec``."""

    def __call__(
        self,
        *,
        name: str,
        description: str,
        candidates: int,
        analysis_prompt: str,
        code_prompt: str,
        repair_prompt: str | None = None,
        use_retrieved_js: bool = False,
        use_vision_for_code: bool = False,
    ) -> Any: ...


@dataclass(frozen=True)
class DirectVisionPromptBundle:
    """Prompt fragments for direct image-to-JS generation and repair."""

    name: str
    description: str
    candidates: int
    analysis_prompt: str
    code_prompt: str
    repair_prompt: str
    validator_repair_prompt: str
    use_retrieved_js: bool = False
    use_vision_for_code: bool = True

    def as_track_kwargs(self) -> dict[str, Any]:
        """Return kwargs accepted by ``ensemble_core.TrackSpec``."""

        return {
            "name": self.name,
            "description": self.description,
            "candidates": self.candidates,
            "analysis_prompt": self.analysis_prompt,
            "code_prompt": self.code_prompt,
            "repair_prompt": self.repair_prompt,
            "use_retrieved_js": self.use_retrieved_js,
            "use_vision_for_code": self.use_vision_for_code,
        }

    def build_track_spec(self, track_spec_factory: TrackSpecFactory) -> Any:
        """Build a runner TrackSpec without importing the runner here."""

        return track_spec_factory(**self.as_track_kwargs())


ANALYSIS_PROMPT = """
Analyze the prompt image for direct procedural Three.js generation.
Return strict JSON only.

Goal: preserve direct image-to-JS object identity. Do not describe a generic
version of the category. Identify the exact visible object type, its pose, and
the small features that distinguish it from nearby categories.

Required JSON keys:
- object_identity: specific noun phrase, category, and negative identity checks.
- camera_and_pose: view angle, front/back cues, up axis, symmetry, occlusion.
- silhouette: dominant outline in x/y/z, profile breaks, tapering, holes, rims,
  protrusions, handles, feet, supports, wheels, straps, lids, spouts, nozzles,
  buttons, seams, labels, cutouts, and repeated motifs.
- part_graph: ordered high-priority parts with parent, relative position,
  approximate scale, primitive families, material zone, and visibility priority.
- material_palette: base colors, accent colors, roughness/metalness cues, and
  where each material appears.
- validator_safe_plan: simple allowed Three.js geometry choices for each part,
  avoiding DOM, external assets, top-level THREE usage, dynamic property access,
  and unsupported APIs.
- identity_preservation_tests: 4-8 checks a rendered candidate must pass to be
  recognized as this exact object rather than a generic block assembly.

Prefer robust primitive assemblies over risky custom BufferGeometry unless a
custom contour is essential. Spend detail budget on identity-defining geometry.
""".strip()


CODE_PROMPT = """
Generate raw validator-compliant Three.js directly from the prompt image and the
JSON analysis. Return only JavaScript source with exactly:
export default function generate(THREE)

Identity target:
- The generated object must read as the same object in the image, not merely the
  same broad category.
- Use named groups for the main body, secondary volumes, appendages, openings,
  repeated motifs, and material zones.
- Build every high-priority visible part from the analysis as real geometry or a
  clear procedural material cue. Do not collapse handles, feet, rims, seams,
  spokes, labels, straps, buttons, holes, supports, lids, spouts, or wheels into
  comments or variable names.

Validator-safe coding rules:
- Put all executable code inside generate. No imports and no top-level constants
  that mention THREE or instantiate geometry/materials.
- Access Three.js APIs only as direct members like THREE.Group and
  THREE.MeshStandardMaterial, or via named destructuring inside generate.
- Do not use THREE[...], object[key], dynamic dispatch, call/apply/bind,
  eval, Function, window, document, globalThis, self, fetch, Date, performance,
  Math.random, renderer, scene, camera, texture loaders, DOM APIs, or external
  assets.
- If a helper needs Three.js, define it inside generate and either close over the
  generate parameter or declare the parameter name exactly THREE. Do not rename
  the parameter to T, three, lib, api, or ctx.
- Never reference THREE at module top level. Never store THREE in arrays,
  objects, aliases, rest/spread, or return it from a helper.
- Prefer allowed primitives: Group, Mesh, InstancedMesh, BoxGeometry,
  SphereGeometry, CylinderGeometry, ConeGeometry, TorusGeometry, TubeGeometry,
  LatheGeometry, ExtrudeGeometry, Shape, Vector2, Vector3, CatmullRomCurve3,
  MeshStandardMaterial, MeshPhysicalMaterial, MeshBasicMaterial, LineSegments,
  EdgesGeometry, LineBasicMaterial, Color, Matrix4, Quaternion, Euler, and
  DataTexture only when compact.
- Keep the final object within [-0.5, 0.5] on all axes. Use Y-up and +Z forward.
- Keep draw calls and hierarchy modest. Use repeated simple helpers or
  InstancedMesh for motifs, but cap instances well below validator limits.

Construction strategy:
1. Create a root THREE.Group.
2. Define compact material constants inside generate.
3. Define local helpers for primitives that set position, rotation, scale, name,
   and material.
4. Assemble the identity-critical silhouette first.
5. Add visible appendages/openings/repeated details second.
6. Add color bands, labels, seams, rims, edge lines, or simple procedural
   textures only when they improve recognition.
7. Before returning, mentally audit validator rules and object bounds.
""".strip()


REPAIR_PROMPT = """
Patch visual mismatches while preserving direct image-derived object identity.
Return only the complete JavaScript module.

Repair priorities:
- If identity is wrong, change geometry topology first: add/remove/reposition
  the defining parts that distinguish the image object from adjacent categories.
- If silhouette is weak, resize the main volumes and appendages before adding
  decoration.
- If missing visible details are named by the critique, add actual meshes, lines,
  repeated motifs, holes/rims, bands, labels, seams, feet, supports, handles,
  wheels, buttons, nozzles, straps, lids, or cutouts.
- Preserve any already-correct image-specific parts; do not simplify back to a
  generic safe object.
- Keep every helper and every THREE reference validator-safe.
""".strip()


VALIDATOR_REPAIR_PROMPT = """
Repair only enough to pass static/runtime validator while preserving the current
image-derived object identity and geometry. Return only complete JavaScript.

Targeted rule fixes:
- IDENTIFIER_NOT_ALLOWED: remove or rename the offending identifier to ordinary
  local names already accepted by the validator. Do not introduce browser,
  Node, renderer, loader, or undeclared globals. Replace unsupported helpers with
  simple local functions inside generate.
- FORBIDDEN_IDENTIFIER, including window/document/globalThis/self/fetch/Date:
  delete those references. Replace DOM, time, random, environment, network, and
  external asset behavior with fixed literal values and procedural geometry.
- THREE_AT_TOP_LEVEL: move every THREE.* read, new THREE.* construction,
  material, geometry, color, vector, shape, curve, and helper default using THREE
  into the generate function body.
- THREE_ALIAS_FORBIDDEN or invalid helper scope: do not alias, stash, spread, or
  return THREE. Helpers must be declared inside generate; if they accept a Three
  parameter it must be named exactly THREE, and calls must pass the generate
  parameter directly.
- COMPUTED_PROPERTY_ACCESS: replace THREE[name], THREE[`Name`], obj[prop] for
  gated objects, method dispatch, and constructor lookup tables with explicit
  branches or direct THREE.ClassName references.
- UNKNOWN_THREE_API or FORBIDDEN_THREE_API: swap the API for simple allowed
  primitives and materials. For tubes, use THREE.CatmullRomCurve3 with Vector3
  points; do not use 2D Path as a TubeGeometry curve.
- EXECUTION_THREW: initialize all arrays/constants before use, keep helper
  signatures consistent, use new THREE.ClassName(...), and remove references to
  missing variables.
- BOUNDING_BOX_OUT_OF_RANGE: uniformly scale or reposition root and long parts
  so all axes fit inside [-0.5, 0.5].
- LITERAL_BUDGET_EXCEEDED: remove large embedded arrays/textures and replace
  them with compact procedural loops.

Do not fix validator errors by deleting most of the object. Preserve the named
body, silhouette, appendages, repeated motifs, and color/material zones unless a
specific offending expression must be rewritten.
""".strip()


DIRECT_VISION_TRACK = DirectVisionPromptBundle(
    name=TRACK_NAME,
    description=(
        "Vision model writes raw Three.js directly from the prompt image with "
        "strong validator-safe coding constraints and targeted repair guidance."
    ),
    candidates=CANDIDATES,
    analysis_prompt=ANALYSIS_PROMPT,
    code_prompt=CODE_PROMPT,
    repair_prompt=REPAIR_PROMPT,
    validator_repair_prompt=VALIDATOR_REPAIR_PROMPT,
)


def track_kwargs() -> dict[str, Any]:
    """Return kwargs accepted by the current ``TrackSpec`` dataclass."""

    return DIRECT_VISION_TRACK.as_track_kwargs()


def build_track_spec(track_spec_factory: TrackSpecFactory) -> Any:
    """Build a runner-compatible TrackSpec from the standalone prompt bundle."""

    return DIRECT_VISION_TRACK.build_track_spec(track_spec_factory)


def validator_repair_prompt() -> str:
    """Return rule-specific repair instructions for validator failures."""

    return DIRECT_VISION_TRACK.validator_repair_prompt


__all__ = [
    "ANALYSIS_PROMPT",
    "CANDIDATES",
    "CODE_PROMPT",
    "DIRECT_VISION_TRACK",
    "DirectVisionPromptBundle",
    "REPAIR_PROMPT",
    "TRACK_NAME",
    "VALIDATOR_REPAIR_PROMPT",
    "build_track_spec",
    "track_kwargs",
    "validator_repair_prompt",
]
