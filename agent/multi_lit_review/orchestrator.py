"""
Multi-agent orchestrator — coordinates Scout → Writer → Adversary loop.

Flow:
  Phase 1: Scout enriches the paper pool and writes a field briefing (1 call + tool use)
  Phase 2: Writer produces the initial draft from enriched pool + briefing (1 call)
  Phase 3: For each round (max MAX_ROUNDS_CAP):
    a. Adversary critiques, searches for missing papers (1 call + tool use)
    b. If score >= ACCEPT_THRESHOLD or last round: Writer writes final polish → done
    c. Else: Writer revises (1 call) → next round

Stopping guarantees (no forever-running):
  - Hard cap: MAX_ROUNDS_CAP rounds regardless of user input
  - Early stop: Adversary score >= ACCEPT_THRESHOLD
  - After final round: Writer always produces a final polish and exits
  Max total Claude calls: 2 + max_rounds * 2 + 1  (= 7 for 2 rounds, 9 for 3 rounds)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional
import anthropic

from agent.multi_lit_review.scout_agent import scout_enrich
from agent.multi_lit_review.writer_agent import writer_draft, writer_revise, writer_final_polish
from agent.multi_lit_review.adversary_agent import adversary_critique

ACCEPT_THRESHOLD = 7   # Adversary score at which we go straight to final polish
MAX_ROUNDS_CAP   = 3   # Absolute ceiling on debate rounds

# ── Event phases (used by UI to decide rendering) ─────────────────────────────
PHASE_STATUS   = "status"     # short status string — not full content
PHASE_BRIEFING = "briefing"   # Scout's field briefing
PHASE_DRAFT    = "draft"      # Writer's initial draft
PHASE_REVISION = "revision"   # Writer's mid-round revision
PHASE_FINAL    = "final"      # Writer's final polished output
PHASE_FEEDBACK = "feedback"   # Adversary's full critique
PHASE_TOOL     = "tool_call"  # Tool call notification (Scout or Adversary)


@dataclass
class AgentEvent:
    agent: str           # "scout" | "writer" | "adversary"
    round: int           # 0 = setup phases; 1+ = debate rounds
    phase: str
    content: str
    score: Optional[int] = None


EventCallback = Callable[[AgentEvent], None]


def run_multi_agent_review(
    client: anthropic.Anthropic,
    research_question: str,
    paper_type: str,
    papers: list,
    n_raw: int,
    n_deduped: int,
    max_rounds: int,
    on_event: EventCallback,
) -> tuple:
    """
    Run the 3-agent Scout → Writer → Adversary collaborative review.

    Returns:
        final_output (str)
        all_events (list[AgentEvent])
        final_paper_pool (list[dict])
    """
    max_rounds = min(max(max_rounds, 1), MAX_ROUNDS_CAP)
    events: List[AgentEvent] = []
    all_papers = list(papers)

    def emit(ev: AgentEvent) -> None:
        events.append(ev)
        on_event(ev)

    # ── Phase 1: Scout ─────────────────────────────────────────────────────────
    emit(AgentEvent("scout", 0, PHASE_STATUS,
                    "Enriching paper pool and mapping the field…"))

    def _scout_tool_cb(name: str, inputs: dict) -> None:
        emit(AgentEvent("scout", 0, PHASE_TOOL,
                        f"{name}: {inputs.get('query', '')}"))

    briefing, new_papers = scout_enrich(client, research_question, all_papers, _scout_tool_cb)
    all_papers.extend(new_papers)
    emit(AgentEvent("scout", 0, PHASE_BRIEFING, briefing))

    # ── Phase 2: Writer initial draft ──────────────────────────────────────────
    emit(AgentEvent("writer", 0, PHASE_STATUS, "Writing initial draft…"))
    draft = writer_draft(
        client, research_question, paper_type, all_papers,
        briefing, n_raw, n_deduped,
    )
    emit(AgentEvent("writer", 0, PHASE_DRAFT, draft))

    critique = ""
    final_score: Optional[int] = None

    # ── Phase 3: Adversarial review loop ───────────────────────────────────────
    for round_num in range(max_rounds):
        rnd = round_num + 1

        emit(AgentEvent("adversary", rnd, PHASE_STATUS,
                        "Adversarial review: searching for weaknesses and missing papers…"))

        def _adv_tool_cb(name: str, inputs: dict) -> None:
            emit(AgentEvent("adversary", rnd, PHASE_TOOL,
                            f"{name}: {inputs.get('query', '')}"))

        critique, score, adv_new_papers = adversary_critique(
            client, draft, research_question, all_papers, _adv_tool_cb,
        )
        all_papers.extend(adv_new_papers)
        final_score = score
        emit(AgentEvent("adversary", rnd, PHASE_FEEDBACK, critique, score))

        is_last       = (round_num >= max_rounds - 1)
        is_acceptable = (score >= ACCEPT_THRESHOLD)

        if is_acceptable or is_last:
            reason = (f"score {score}/10 meets threshold"
                      if is_acceptable else
                      f"max rounds ({max_rounds}) reached")
            emit(AgentEvent("writer", rnd, PHASE_STATUS,
                            f"Writing final polished version ({reason})…"))
            final = writer_final_polish(
                client, research_question, all_papers, briefing, draft, critique,
            )
            emit(AgentEvent("writer", rnd, PHASE_FINAL, final, score))
            return final, events, all_papers

        # Writer revises for next round
        emit(AgentEvent("writer", rnd, PHASE_STATUS,
                        f"Revising draft (score {score}/10 — below threshold)…"))
        draft = writer_revise(
            client, research_question, all_papers, briefing, draft, critique,
        )
        emit(AgentEvent("writer", rnd, PHASE_REVISION, draft))

    # Safety fallback (unreachable under normal conditions)
    return draft, events, all_papers
