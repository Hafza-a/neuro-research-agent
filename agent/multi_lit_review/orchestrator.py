"""
Multi-agent orchestrator — coordinates the PhD Student, Supervisor, Peer, and Journal Reviewer
agents through a bounded debate loop to produce a polished, well-reviewed literature review.

Stopping conditions (prevents forever-running):
  1. Journal Reviewer scores >= ACCEPT_THRESHOLD → early stop, student writes final polish
  2. max_rounds reached → force final polish regardless of score
  Hard upper bound: max_rounds is capped at MAX_ROUNDS_CAP
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import anthropic

from agent.multi_lit_review.student_agent import (
    student_write_initial,
    student_revise,
    student_final_polish,
)
from agent.multi_lit_review.supervisor_agent import supervisor_review
from agent.multi_lit_review.peer_agent import peer_review_and_search
from agent.multi_lit_review.reviewer_agent import reviewer_critique

# ── Constants ──────────────────────────────────────────────────────────────────
ACCEPT_THRESHOLD = 8   # Score at which we stop debating and write the final version
MAX_ROUNDS_CAP   = 3   # Absolute maximum rounds regardless of user input


# ── Event dataclass ────────────────────────────────────────────────────────────
@dataclass
class AgentEvent:
    agent: str           # "student" | "supervisor" | "peer" | "reviewer" | "system"
    round: int           # 0 = initial write; 1+ = debate rounds
    phase: str           # see constants below
    content: str
    score: Optional[int] = None  # set only by reviewer events

# Phase constants (used by UI to decide what to render)
PHASE_STATUS   = "status"      # brief status string — not full content
PHASE_DRAFT    = "draft"       # student initial draft
PHASE_REVISION = "revision"    # student mid-round revision
PHASE_FINAL    = "final"       # student final polished output
PHASE_FEEDBACK = "feedback"    # supervisor / peer / reviewer full feedback
PHASE_TOOL     = "tool_call"   # peer tool call notification

EventCallback = Callable[[AgentEvent], None]


# ── Public API ─────────────────────────────────────────────────────────────────
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
    Run the 4-agent collaborative literature review.

    Args:
        client:            Anthropic client
        research_question: the review topic
        paper_type:        one of the paper type strings defined in student_system.py
        papers:            curated paper pool (list of dicts)
        n_raw:             total papers from search before dedup (for Methods section)
        n_deduped:         papers after dedup
        max_rounds:        user-selected debate rounds (capped at MAX_ROUNDS_CAP)
        on_event:          callback fired for each event (for real-time UI updates)

    Returns:
        (final_output: str, all_events: list[AgentEvent], final_paper_pool: list[dict])
    """
    max_rounds = min(max_rounds, MAX_ROUNDS_CAP)
    events: List[AgentEvent] = []
    all_papers = list(papers)

    def emit(event: AgentEvent) -> None:
        events.append(event)
        on_event(event)

    # ── Phase 0: Student writes initial draft ──────────────────────────────────
    emit(AgentEvent("student", 0, PHASE_STATUS, "Writing initial draft…"))
    draft = student_write_initial(
        client, research_question, paper_type, all_papers, n_raw, n_deduped
    )
    emit(AgentEvent("student", 0, PHASE_DRAFT, draft))

    sup_feedback  = ""
    peer_feedback = ""
    rev_feedback  = ""
    final_score   = None

    # ── Debate rounds ──────────────────────────────────────────────────────────
    for round_num in range(max_rounds):
        rnd = round_num + 1

        # 1. Supervisor
        emit(AgentEvent("supervisor", rnd, PHASE_STATUS,
                        "Reviewing structure, argument, and scientific rigour…"))
        sup_feedback = supervisor_review(client, draft, research_question)
        emit(AgentEvent("supervisor", rnd, PHASE_FEEDBACK, sup_feedback))

        # 2. Peer (with tool use — may expand the paper pool)
        emit(AgentEvent("peer", rnd, PHASE_STATUS,
                        "Searching for papers you might have missed…"))

        def _peer_tool_cb(name: str, inputs: dict) -> None:
            q = inputs.get("query", inputs.get("doi", ""))
            emit(AgentEvent("peer", rnd, PHASE_TOOL, f"{name}: {q}"))

        peer_feedback, new_papers = peer_review_and_search(
            client, draft, research_question, all_papers, _peer_tool_cb
        )
        if new_papers:
            all_papers.extend(new_papers)
        emit(AgentEvent("peer", rnd, PHASE_FEEDBACK, peer_feedback))

        # 3. Journal Reviewer
        emit(AgentEvent("reviewer", rnd, PHASE_STATUS,
                        "Evaluating for journal submission…"))
        rev_feedback, score = reviewer_critique(client, draft)
        final_score = score
        emit(AgentEvent("reviewer", rnd, PHASE_FEEDBACK, rev_feedback, score))

        # 4. Stopping check ────────────────────────────────────────────────────
        is_last_round  = (round_num >= max_rounds - 1)
        is_acceptable  = (score >= ACCEPT_THRESHOLD)

        if is_acceptable or is_last_round:
            reason = (f"score {score}/10 ≥ {ACCEPT_THRESHOLD} — accepted"
                      if is_acceptable else
                      f"reached max rounds ({max_rounds}) — forcing final polish")
            emit(AgentEvent("student", rnd, PHASE_STATUS,
                            f"Writing final polished version ({reason})…"))
            final = student_final_polish(
                client, draft, sup_feedback, peer_feedback, rev_feedback,
                research_question, all_papers,
            )
            emit(AgentEvent("student", rnd, PHASE_FINAL, final, score))
            return final, events, all_papers

        # 5. Student revises for next round ────────────────────────────────────
        emit(AgentEvent("student", rnd, PHASE_STATUS,
                        f"Revising draft (score: {score}/10 — below threshold, continuing)…"))
        draft = student_revise(
            client, draft, sup_feedback, peer_feedback, rev_feedback,
            research_question, all_papers,
        )
        emit(AgentEvent("student", rnd, PHASE_REVISION, draft))

    # Safety fallback — should never reach here due to is_last_round check above
    return draft, events, all_papers
