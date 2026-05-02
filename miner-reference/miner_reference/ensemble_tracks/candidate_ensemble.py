"""Candidate-ensemble track prompts.

This module intentionally has no Chutes or runner dependency. The runner can
pass its local TrackSpec class into candidate_ensemble_track(TrackSpec), or use
candidate_ensemble_track_dict() when constructing a compatible object itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, TypeVar


TrackSpecT = TypeVar("TrackSpecT")

TRACK_NAME = "candidate_ensemble"
TRACK_DESCRIPTION = (
    "Diverse first-pass candidates with selector-friendly metadata, balanced "
    "between silhouette, part graph, material cues, and validator-safe repairability."
)
CANDIDATE_COUNT = 4

VALIDATOR_SAFE_CONSTRAINTS = (
    "Validator-safe constraints: raw JavaScript module only; exactly "
    "`export default function generate(THREE)`; no imports, external assets, "
    "network, randomness, time, DOM, loaders, async work, or top-level THREE "
    "usage; keep helpers inside generate or pass THREE explicitly; fit all "
    "geometry into [-0.5, 0.5] with Y-up and +Z forward; prefer standard safe "
    "Three.js primitives and simple materials."
)

ANALYSIS_PROMPT = f"""
Return strict JSON for a candidate ensemble, not prose.

Goal: build multiple implementation hypotheses that are visibly different but
all plausible for the same prompt. Do not lock onto one explicit variant wording
or a narrow recipe; preserve diversity so later validator/render/critic/pairwise
selection can choose the best output.

Required top-level keys:
- object_identity: concise noun phrase plus any visible subcategory.
- validator_safe_constraints: copy the important constraints in your own words.
- shared_scene_facts: camera/view, silhouette axes, symmetry, colors, materials,
  scale landmarks, and must-have visible parts.
- candidate_blueprints: exactly four entries with strategy_id values
  silhouette_anchor, part_graph_anchor, material_cue_anchor, negative_space_anchor.
- selector_metadata_schema: fields the code candidate should expose for judging.

For each candidate_blueprints entry include:
- strategy_id and one-line selector_label.
- intended_strength: what this candidate should win on.
- visual_differentiators: concrete geometry/material choices that make it
  different from the other candidates.
- part_graph: named parts with parent/child attachments and rough dimensions.
- contour_plan: primitive families and transforms for the external silhouette.
- material_cues: colors, roughness/metalness where useful, stripes, seams,
  labels, rims, panels, holes, repeated details, or texture-like procedural cues.
- negative_identity_checks: 3-5 wrong object readings this candidate must avoid.
- likely_judge_failures: risks the selector should consider.
- validator_risk_controls: how to stay simple, bounded, deterministic, and safe.

{VALIDATOR_SAFE_CONSTRAINTS}
""".strip()

CODE_PROMPT = f"""
Generate one raw Three.js candidate from the ensemble analysis.

Use the candidate number supplied by the runner to choose a blueprint:
1 -> silhouette_anchor, 2 -> part_graph_anchor, 3 -> material_cue_anchor,
4 -> negative_space_anchor. If the matching blueprint is missing, choose the
nearest strategy but keep the output architecturally distinct from earlier
candidate styles.

Selector-friendly requirements:
- Add a small metadata object to the returned root object, for example
  `root.userData.selectorMetadata = {{ ... }}`. Keep it plain JSON-compatible.
- Include strategy_id, selector_label, intended_strength, likely_judge_failures,
  validator_risk_controls, and prominent_parts in that metadata.
- Name major groups and meshes descriptively so rendered/debug artifacts are
  easy to inspect.
- Prioritize object identity, outer contour, visible attachments, and material
  zones over generic block assembly.

Diversity requirements:
- Candidate 1 should emphasize the recognizable outline and camera-facing
  silhouette.
- Candidate 2 should emphasize a coherent attachable part graph and proportions.
- Candidate 3 should emphasize material/color/detail cues without becoming
  surface-only.
- Candidate 4 should emphasize holes, gaps, rims, supports, handles, loops,
  spokes, cutouts, or other negative-space cues when applicable.

Do not over-specialize to a single previously successful explicit variant prompt.
Use broadly applicable procedural modeling decisions derived from the image.

{VALIDATOR_SAFE_CONSTRAINTS}
Return only JavaScript source with exactly `export default function generate(THREE)`.
""".strip()

REPAIR_PROMPT = f"""
Patch the JavaScript with visible structural changes while preserving candidate
diversity and metadata.

Use critique findings as edit operations: ADD_PART, REMOVE_PART, RESIZE_PART,
REPOSITION_PART, CHANGE_PRIMITIVE, ADD_REPEAT, CHANGE_SYMMETRY, CHANGE_MATERIAL,
ADD_NEGATIVE_SPACE_CUE, or SIMPLIFY_FOR_VALIDATOR.

Update `root.userData.selectorMetadata` when the strategy, prominent parts, or
known judge risks change. Do not merely rename variables, comments, or metadata.

{VALIDATOR_SAFE_CONSTRAINTS}
Return only the complete patched JavaScript module.
""".strip()


@dataclass(frozen=True)
class CandidateEnsembleSpec:
    """TrackSpec-equivalent data without importing the runner module."""

    name: str = TRACK_NAME
    description: str = TRACK_DESCRIPTION
    candidates: int = CANDIDATE_COUNT
    analysis_prompt: str = ANALYSIS_PROMPT
    code_prompt: str = CODE_PROMPT
    repair_prompt: str | None = REPAIR_PROMPT
    use_retrieved_js: bool = False
    use_vision_for_code: bool = False


def candidate_ensemble_spec() -> CandidateEnsembleSpec:
    """Return the standalone dataclass form of the candidate-ensemble track."""

    return CandidateEnsembleSpec()


def candidate_ensemble_track_dict() -> dict[str, Any]:
    """Return fields compatible with ensemble_core.TrackSpec."""

    return asdict(candidate_ensemble_spec())


def candidate_ensemble_track(
    track_spec_factory: Callable[..., TrackSpecT] | None = None,
) -> CandidateEnsembleSpec | TrackSpecT:
    """Build the track, optionally using the runner's TrackSpec class.

    Example integration in miner_reference.ensemble_core:

        from miner_reference.ensemble_tracks import candidate_ensemble_track

        ...
        candidate_ensemble_track(TrackSpec)

    Passing TrackSpec keeps the object type identical to the runner's inline
    tracks while this module remains importable by itself.
    """

    fields = candidate_ensemble_track_dict()
    if track_spec_factory is None:
        return CandidateEnsembleSpec(**fields)
    return track_spec_factory(**fields)
