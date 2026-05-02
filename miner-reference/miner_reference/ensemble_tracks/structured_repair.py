"""Structured-repair track prompts.

This module intentionally contains no Chutes, rendering, or validator calls. It
only defines prompt text and a TrackSpec-compatible data shape that the runner
can adapt into its local ``TrackSpec`` dataclass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


EDIT_OPERATIONS: tuple[str, ...] = (
    "ADD_PART",
    "REMOVE_PART",
    "RESIZE_PART",
    "REPOSITION_PART",
    "CHANGE_PRIMITIVE",
    "ADD_REPEAT",
    "CHANGE_SYMMETRY",
    "CHANGE_MATERIAL",
    "ADJUST_CAMERA",
)


VALIDATOR_SAFE_CONSTRAINTS = """Validator-safe constraints:
- Return exactly one raw JavaScript module with `export default function generate(THREE)`.
- Use no imports, external assets, network, DOM, renderer, scene, camera creation, eval, Function, randomness, or time.
- Never reference THREE outside `generate`; keep helpers inside `generate` or pass THREE explicitly.
- Do not create top-level `new THREE.*`, THREE constants, Vector3s, DataTextures, or materials.
- Fit the whole object inside [-0.5, 0.5], use Y-up and +Z forward, and keep geometry robust from multiple views.
- Prefer allowed simple geometry and standard materials over risky custom BufferGeometry.
"""


ANALYSIS_PROMPT = f"""Return strict JSON only: a structured repair-ready object plan for procedural Three.js reconstruction.

Required JSON keys:
- object_identity: concrete category and subtype, not a generic shape.
- view_and_camera: expected view angle, focal framing, visible sides, and any crop or silhouette emphasis.
- validator_safe_plan: short acknowledgement of these constraints: no imports/assets/randomness/time/network/DOM, all THREE usage inside generate, object fits [-0.5, 0.5], Y-up, +Z forward.
- silhouette: dominant outline, proportions on x/y/z, negative spaces, holes, rims, openings, and asymmetries.
- symmetry: one of none, bilateral_x, bilateral_z, radial, repeated_linear, or mixed, with axis/count.
- materials: named material zones with colors, roughness/metalness hints, transparency only if essential.
- part_graph: array of editable parts. Each part must include id, parent_id, visual_priority, primitive_family, dimensions, position, rotation, material_zone, attachment_rule, and why_visible.
- repeated_motifs: array for repeated teeth, slats, spokes, buttons, legs, labels, seams, ribs, handles, panels, wheels, or supports.
- missing_part_risks: visible structures likely to be forgotten by a generic reconstruction.
- repair_targets: likely future edit operations using only: {", ".join(EDIT_OPERATIONS)}.

Planning rules:
- Name every visible structural component separately enough that it can later be edited.
- Separate body volumes, appendages, supports, handles, rims, holes, panels, labels, seams, wheels, spokes, knobs, and repeated details.
- Include at least 6 meaningful parts unless the object is truly simple; avoid a single-body plan.
- Mark hidden-side assumptions explicitly instead of inventing excessive geometry.
- Keep all dimensions relative and validator-safe.

{VALIDATOR_SAFE_CONSTRAINTS}"""


CODE_PROMPT = f"""Generate a validator-safe raw Three.js module from the structured plan.

Structural coding requirements:
- Build named groups or named meshes for every high-priority `part_graph` entry.
- Use simple local constants for dimensions, positions, colors, and repeat counts so repair can visibly edit them.
- Preserve editability: avoid baking the whole object into one merged mesh.
- Use helper functions for repeated motifs, but define them inside `generate`.
- Make repeated motifs visible with actual geometry, not comments or variable names.
- Encode material zones as distinct materials or visibly distinct colors/roughness.
- If the object has symmetry, implement it with explicit mirrored/repeated geometry so later CHANGE_SYMMETRY or ADD_REPEAT edits are easy.
- Include object-defining details before decorative details; avoid generic placeholder boxes.

No-op prevention:
- Do not satisfy the prompt with only color swaps or renamed variables.
- Do not omit a visible handle, support, opening, rim, label, wheel, leg, fin, seam, or repeated motif if it appears in the plan.

{VALIDATOR_SAFE_CONSTRAINTS}"""


REPAIR_PROMPT = f"""Repair this module by first deriving an explicit edit script, then applying it to the JavaScript.

Allowed edit operations only:
{chr(10).join(f"- {operation}" for operation in EDIT_OPERATIONS)}

Internal edit-script requirements:
- Infer 3 to 8 concrete edits from the critique.
- Every edit must name a target part id or a new part id, state the visual problem, choose exactly one allowed operation, and define numeric or material parameters.
- At least two edits must be structural geometry edits from ADD_PART, REMOVE_PART, RESIZE_PART, REPOSITION_PART, CHANGE_PRIMITIVE, ADD_REPEAT, or CHANGE_SYMMETRY unless the critique only complains about camera/material.
- Use ADJUST_CAMERA only for framing/view mismatch; do not use it to hide bad geometry.
- Use CHANGE_MATERIAL only when material/color/opacity is visibly wrong; pair it with structural edits when identity is weak.
- Prefer ADD_PART for missing visible components, ADD_REPEAT for repeated motifs, CHANGE_PRIMITIVE for wrong silhouette, RESIZE_PART for proportion errors, and REPOSITION_PART for attachment/pose errors.

Apply the edit script:
- Make actual JavaScript changes that visibly alter geometry, repeats, symmetry, materials, or object framing.
- Preserve all validator-safe constraints and keep exactly `export default function generate(THREE)`.
- Keep helpers inside `generate`; no imports, assets, randomness, time, network, DOM, renderer, scene, camera creation, eval, or Function.
- Fit the repaired object inside [-0.5, 0.5] with Y-up and +Z forward.
- Return only the complete patched JavaScript module.

No-op rejection:
- Do not return the original source unchanged.
- Do not only rename variables, reformat code, change comments, or make invisible micro-adjustments.
- Do not remove distinctive parts to simplify unless REMOVE_PART fixes a clearly wrong or hallucinated component.
- If uncertain, add or adjust the most visible missing structural feature rather than making a cosmetic-only patch.

{VALIDATOR_SAFE_CONSTRAINTS}"""


@dataclass(frozen=True)
class StructuredRepairTrack:
    """TrackSpec-compatible prompt bundle without importing the runner."""

    name: str
    description: str
    candidates: int
    analysis_prompt: str
    code_prompt: str
    repair_prompt: str | None = None
    use_retrieved_js: bool = False
    use_vision_for_code: bool = False

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


class TrackSpecFactory(Protocol):
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


STRUCTURED_REPAIR_TRACK = StructuredRepairTrack(
    name="structured_repair",
    description=(
        "Structured repair track that forces visible part-graph generation and "
        "explicit edit-operation repairs instead of no-op or cosmetic patches."
    ),
    candidates=2,
    analysis_prompt=ANALYSIS_PROMPT,
    code_prompt=CODE_PROMPT,
    repair_prompt=REPAIR_PROMPT,
)


def track_kwargs() -> dict[str, Any]:
    """Return kwargs compatible with ``ensemble_core.TrackSpec``."""

    return STRUCTURED_REPAIR_TRACK.asdict()


def make_track_spec(track_spec_cls: TrackSpecFactory) -> Any:
    """Create a runner-native TrackSpec without importing the runner here."""

    return track_spec_cls(**track_kwargs())
