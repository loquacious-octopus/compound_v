"""Object-identity motif planner track.

This module is deliberately prompt-only: it defines a stronger replacement for
the weak ``motif_hybrid`` direction without making network calls or importing
the adaptation runner. Pass ``build_track(TrackSpec)`` from
``miner_reference.ensemble_core`` when wiring it into ``default_tracks``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, TypeVar


TRACK_NAME = "object_identity_motif"


class TrackSpecLike(Protocol):
    """Constructor protocol matching ``ensemble_core.TrackSpec``."""

    def __init__(
        self,
        name: str,
        description: str,
        candidates: int,
        analysis_prompt: str,
        code_prompt: str,
        repair_prompt: str | None = None,
        use_retrieved_js: bool = False,
        use_vision_for_code: bool = False,
    ) -> None: ...


T = TypeVar("T", bound=TrackSpecLike)


@dataclass(frozen=True)
class ObjectIdentityTrack:
    """Plain fallback spec with the same fields as the runner's TrackSpec."""

    name: str
    description: str
    candidates: int
    analysis_prompt: str
    code_prompt: str
    repair_prompt: str | None = None
    use_retrieved_js: bool = False
    use_vision_for_code: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VALIDATOR_SAFE_CONSTRAINTS = """
Validator-safe constraints:
- Return only a raw Three.js module with exactly `export default function generate(THREE)`.
- Use no imports, no external assets, no network/time/randomness, and no top-level `new THREE.*`.
- Keep all helpers inside `generate`; every helper that needs Three.js must receive/use the local THREE argument.
- Fit the full object in [-0.5, 0.5], Y-up, +Z forward; include camera-safe proportions and avoid giant offscreen parts.
- Prefer stable built-ins: BoxGeometry, SphereGeometry, CylinderGeometry, ConeGeometry, TorusGeometry, TubeGeometry, LatheGeometry,
  ExtrudeGeometry, simple BufferGeometry only when clearly necessary, MeshStandard/Physical/Basic materials, EdgesGeometry.
- Keep primitive/material counts modest; use loops only for bounded repeated details; no shader tricks or unsupported loaders.
""".strip()


ANALYSIS_PROMPT = f"""
Return strict JSON for an object-identity-first procedural Three.js plan.

The plan must force the target category before choosing motifs. Do not start from generic boxes/blobs and decorate them.
Infer what object this is, what it is definitely not, and which visual evidence would make a judge recognize it.

Required JSON keys:
- object_identity: concrete noun phrase with subtype, pose, view, and confidence.
- avoid_confusions: 4-8 nearby categories to explicitly avoid, each with `confusion`, `why_not`, and `geometry_guardrail`.
- silhouette_family: one of vessel_lathe, vehicle_wedge, animal_body_appendages, furniture_frame, tool_handle_head,
  appliance_block_panels, wearable_loop_strap, container_opening_lid, plant_branching, symbol_flat_extrusion, other_named.
- silhouette_plan: front/side/top profile notes, dominant axes, negative spaces, taper/waist/overhangs, and non-box contour cues.
- part_graph: array of named parts with id, parent, role, primitive_family, pose, scale, attachment_point, symmetry,
  visual_priority 1-10, and required boolean.
- material_zones: array of zones with id, color, roughness/metalness intent, surface pattern, and which part ids use it.
- motif_choices: array of selected motifs only when visible or strongly implied. Allowed motif families:
  lathe_body, rounded_body, tapered_box, bevel_panel_stack, wheel_tire_hub, spoke_ring, handle_loop, strap_band, rim_lip,
  lid_cap, spout_nozzle, leg_foot, fin_slats, button_knob, dial_gauge, grille_slots, label_panel, seam_stripe,
  cutout_hole, fork_prong, blade_head, bristle_cluster, antenna_rod, hinge_pin.
- composition_rules: how parts attach, repeat, mirror, orbit, stack, or pierce the main body.
- validator_safe_geometry_plan: concrete safe primitives/helpers for each high-priority part.
- generic_failure_tests: checks that would reject generic block/blob output.
- candidate_diversity: two implementation styles named `silhouette_locked` and `part_graph_locked`.

{VALIDATOR_SAFE_CONSTRAINTS}

Important behavior:
- If identity is uncertain, encode the uncertainty as alternate guardrails, not as vague geometry.
- Every high-priority visible part must map to at least one part_graph entry and one material zone.
- Avoid copying public JavaScript structure unless it matches the same identity; this track should work without public JS.
""".strip()


CODE_PROMPT = f"""
Generate one validator-safe raw Three.js module from the object-identity plan.

Identity rules:
- Build in this order: locked silhouette family, dominant body/profile, required part graph, material zones, then motifs/details.
- Do not emit a generic cube/sphere/blob with minor decorations. A valid candidate must pass the plan's generic_failure_tests.
- Instantiate visible geometry for every required part with visual_priority >= 7.
- Use named groups or named meshes for identity-bearing parts: body/profile, supports/appendages, openings/rims, repeated motifs,
  trim/material-zone cues, and front-facing distinctive details.
- Encode avoid_confusions directly in geometry. Example: if it could be confused with a plain box, add the non-box contour,
  correct appendages, holes/rims, handles, legs, wheels, spout, label, or asymmetry that rules the confusion out.

Motif rules:
- Choose motifs only from motif_choices and adapt them to the object, not vice versa.
- Use helper functions for repeated visible motifs such as spokes, grille slots, fins, legs, buttons, seams, or stripes.
- Favor silhouette-changing motifs over surface-only details when token or geometry budget is tight.
- Material zones must be visible as separate materials, panels, bands, caps, rubber/metal/glass/accent regions, or texture cues.

Candidate diversity:
- Candidate 1 should be `silhouette_locked`: strongest outer contour, taper/lathe/wedge/cutout/profile fidelity first.
- Candidate 2 should be `part_graph_locked`: strongest named attachments, repeated details, and material-zone separations first.
- Additional candidates should hybridize those styles but still obey avoid_confusions.

{VALIDATOR_SAFE_CONSTRAINTS}
Return only JavaScript source with exactly `export default function generate(THREE)`.
""".strip()


REPAIR_PROMPT = f"""
Patch the candidate by changing visible geometry, not comments or variable names.

Repair priority:
1. Correct object_identity and avoid_confusions.
2. Fix silhouette_family and the largest profile/proportion error.
3. Add, remove, resize, or reposition required part_graph entries with visual_priority >= 7.
4. Restore material_zones as visible separate materials, panels, bands, rims, labels, glass, rubber, metal, or accent regions.
5. Add selected motifs only where they strengthen identity and remain validator-safe.

Use concrete edit operations mentally: ADD_PART, REMOVE_PART, RESIZE_PART, REPOSITION_PART, CHANGE_PRIMITIVE,
ADD_REPEAT, CHANGE_SYMMETRY, CHANGE_MATERIAL, ADD_CUTOUT_PROXY, ADJUST_CAMERA.

{VALIDATOR_SAFE_CONSTRAINTS}
Return only the complete patched JavaScript module.
""".strip()


DESCRIPTION = (
    "Object-identity-first motif planner that forces avoid_confusions, silhouette family, "
    "part graph, material zones, and selected visible motifs while avoiding irrelevant public JS."
)


def build_track(track_cls: type[T] | None = None, *, candidates: int = 2) -> T | ObjectIdentityTrack:
    """Build a runner-compatible track spec.

    ``track_cls`` may be ``miner_reference.ensemble_core.TrackSpec``. If it
    is omitted, the function returns this module's plain dataclass fallback.
    """

    spec: dict[str, Any] = {
        "name": TRACK_NAME,
        "description": DESCRIPTION,
        "candidates": candidates,
        "analysis_prompt": ANALYSIS_PROMPT,
        "code_prompt": CODE_PROMPT,
        "repair_prompt": REPAIR_PROMPT,
        "use_retrieved_js": False,
        "use_vision_for_code": False,
    }
    if track_cls is None:
        return ObjectIdentityTrack(**spec)
    return track_cls(**spec)


def integration_notes() -> str:
    """Return the minimal runner integration hint without editing the runner."""

    return (
        "In miner_reference.ensemble_core.default_tracks(), import "
        "`build_track` from `miner_reference.ensemble_tracks.object_identity` and "
        "replace the `motif_hybrid` TrackSpec with `build_track(TrackSpec)`. "
        "Optionally add a `_variant_instruction` branch for "
        "`object_identity_motif`; the code prompt already carries candidate-specific "
        "silhouette_locked vs part_graph_locked instructions, so this is not required."
    )


__all__ = [
    "ANALYSIS_PROMPT",
    "CODE_PROMPT",
    "DESCRIPTION",
    "ObjectIdentityTrack",
    "REPAIR_PROMPT",
    "TRACK_NAME",
    "VALIDATOR_SAFE_CONSTRAINTS",
    "build_track",
    "integration_notes",
]
