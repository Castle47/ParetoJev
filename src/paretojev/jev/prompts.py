from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DecisionObjective:
    name: str
    question: str
    true_criteria: str
    false_criteria: str
    direction: int = 1


DEFAULT_OBJECTIVES: tuple[DecisionObjective, ...] = (
    DecisionObjective(
        "click",
        "Would this user validly play or click this candidate video in the current feed context?",
        "The candidate closely matches the user's recent interests and is likely to receive a valid play/click.",
        "The candidate is unlikely to receive a valid play/click from this user.",
    ),
    DecisionObjective(
        "long_view",
        "Would this user watch this candidate long enough to count as a long view?",
        "The candidate is likely to hold this user's attention for a long view.",
        "The candidate is unlikely to hold this user's attention for a long view.",
    ),
    DecisionObjective(
        "like",
        "Would this user positively endorse this candidate by liking it?",
        "The content strongly fits the user's preferences and is likely to be liked.",
        "A like is unlikely even if the user might briefly watch it.",
    ),
    DecisionObjective(
        "follow",
        "Would this candidate make the user follow the creator?",
        "The content and creator appear compelling enough to cause a new follow.",
        "The candidate is unlikely to cause a creator follow.",
    ),
    DecisionObjective(
        "deep_engagement",
        "Would this candidate cause a deeper engagement such as comment or forward?",
        "The candidate is likely to prompt a comment, share, or forward.",
        "The candidate is unlikely to prompt comment/share/forward behavior.",
    ),
    DecisionObjective(
        "negative_feedback",
        "Would this candidate cause explicit negative feedback such as hate/dislike?",
        "The candidate conflicts with the user's preferences or context and is likely to receive negative feedback.",
        "Explicit negative feedback is unlikely.",
        direction=-1,
    ),
)


OBJECTIVES_BY_NAME: dict[str, DecisionObjective] = {
    objective.name: objective for objective in DEFAULT_OBJECTIVES
}


def select_objectives(names: Iterable[str] | None = None) -> tuple[DecisionObjective, ...]:
    """Resolve a stable, validated subset of decision objectives by name."""

    if names is None:
        return DEFAULT_OBJECTIVES
    requested = tuple(str(name) for name in names)
    if not requested:
        raise ValueError("At least one objective is required")
    if len(set(requested)) != len(requested):
        raise ValueError(f"Duplicate objective names: {requested}")
    unknown = [name for name in requested if name not in OBJECTIVES_BY_NAME]
    if unknown:
        raise ValueError(
            f"Unknown objectives {unknown}; available: {sorted(OBJECTIVES_BY_NAME)}"
        )
    return tuple(OBJECTIVES_BY_NAME[name] for name in requested)


def _candidate_text(candidate: dict[str, Any]) -> str:
    fields = []
    for key in (
        "caption",
        "show_cover_text",
        "category",
        "first_level_category_name",
        "second_level_category_name",
        "duration_ms",
        "creator",
    ):
        value = candidate.get(key)
        if value is not None and value != "":
            fields.append(f"{key}: {value}")
    return "; ".join(fields) or f"item_id: {candidate.get('item_id', candidate.get('video_id', 'unknown'))}"


def build_state(
    history: Iterable[dict[str, Any]],
    candidates: Iterable[dict[str, Any]] = (),
    context: dict[str, Any] | None = None,
    *,
    include_candidates: bool = True,
) -> dict[str, Any]:
    """Build a serializable recommendation state.

    Pointwise Noul scoring should normally set ``include_candidates=False``:
    candidate content is then carried only by that candidate's question, so a
    score cannot depend on competing candidates. Choice scoring necessarily
    keeps the full candidate set in state.
    """

    history_rows = [
        {
            "position": idx + 1,
            "item": _candidate_text(item),
            "feedback": item.get("feedback", {}),
        }
        for idx, item in enumerate(history)
    ]
    candidate_rows = [
        {
            "candidate_id": str(item.get("candidate_id", item.get("item_id", item.get("video_id", idx)))),
            "item": _candidate_text(item),
        }
        for idx, item in enumerate(candidates)
    ]
    state: dict[str, Any] = {
        "task": "short-video recommendation reranking",
        "recent_history_oldest_to_newest": history_rows,
        "current_context": context or {},
    }
    if include_candidates:
        state["candidates"] = candidate_rows
    return state


def build_typesafe_choice_questions(
    candidates: list[dict[str, Any]],
    objectives: Iterable[DecisionObjective] = DEFAULT_OBJECTIVES,
) -> dict[str, dict[str, Any]]:
    criteria = {
        str(item.get("candidate_id", item.get("item_id", item.get("video_id", idx)))): _candidate_text(item)
        for idx, item in enumerate(candidates)
    }
    if len(criteria) < 2:
        raise ValueError("choice mode requires at least two candidates")
    return {
        objective.name: {
            "type": "choice",
            "instructions": objective.question + " Select the single best candidate relative to the alternatives.",
            "criteria": criteria,
        }
        for objective in objectives
    }


def build_typesafe_noul_questions(
    candidates: list[dict[str, Any]],
    objectives: Iterable[DecisionObjective] = DEFAULT_OBJECTIVES,
) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    for idx, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("candidate_id", candidate.get("item_id", candidate.get("video_id", idx))))
        text = _candidate_text(candidate)
        for objective in objectives:
            questions[f"{objective.name}__{candidate_id}"] = {
                "type": "noul",
                "instructions": f"Candidate {candidate_id}: {text}. {objective.question}",
                "criteria": {
                    "true": objective.true_criteria,
                    "false": objective.false_criteria,
                },
            }
    return questions


def build_open_jev_choice_requests(
    state: dict[str, Any],
    candidates: list[dict[str, Any]],
    objectives: Iterable[DecisionObjective] = DEFAULT_OBJECTIVES,
) -> list[tuple[str, Any, str, list[str] | None]]:
    options = [
        str(item.get("candidate_id", item.get("item_id", item.get("video_id", idx))))
        for idx, item in enumerate(candidates)
    ]
    return [("choice", state, objective.question, options) for objective in objectives]


def build_open_jev_noul_requests(
    state: dict[str, Any],
    candidates: list[dict[str, Any]],
    objectives: Iterable[DecisionObjective] = DEFAULT_OBJECTIVES,
) -> list[tuple[str, Any, str, list[str] | None]]:
    requests_: list[tuple[str, Any, str, list[str] | None]] = []
    for idx, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("candidate_id", candidate.get("item_id", candidate.get("video_id", idx))))
        candidate_text = _candidate_text(candidate)
        for objective in objectives:
            question = (
                f"Candidate {candidate_id}: {candidate_text}. "
                f"{objective.question}"
            )
            requests_.append(("noul", state, question, None))
    return requests_
