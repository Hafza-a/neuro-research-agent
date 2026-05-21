"""
Multi-agent orchestrator — coordinates Scout → Writer → Adversary loop.

Flow:
  Phase 1: Scout enriches the paper pool and writes a field briefing (1 call + tool use)
  Phase 2: Writer produces the initial draft from enriched pool + briefing (1 call)
  Phase 3: For each round (max MAX_ROUNDS_CAP):
    a. Adversary critiques, searches for missing papers (1 call + tool use)
    b. If score >= ACCEPT_THRESHOLD or last round: Writer writes final polish → done
    c. Else: Writer revises (1 call) → next round

Ablation modes (run_mode param):
  "standard"     — full Scout → Writer ↔ Adversary pipeline
  "writer_only"  — no Scout, no Adversary (baseline single-agent)
  "no_scout"     — Writer gets no briefing; Adversary still runs
  "no_adversary" — Scout runs; Writer drafts once, then final polish (no debate)

Stopping guarantees (no forever-running):
  - Hard cap: MAX_ROUNDS_CAP rounds regardless of user input
  - Early stop: Adversary score >= ACCEPT_THRESHOLD
  - After final round: Writer always produces a final polish and exits
  Max total Claude calls: 2 + max_rounds * 2 + 1  (= 7 for 2 rounds, 9 for 3 rounds)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import anthropic

from agent.multi_lit_review.scout_agent import scout_enrich
from agent.multi_lit_review.writer_agent import writer_draft, writer_revise, writer_final_polish
from agent.multi_lit_review.adversary_agent import adversary_critique

ACCEPT_THRESHOLD = 7   # Adversary score at which we go straight to final polish
MAX_ROUNDS_CAP   = 3   # Absolute ceiling on debate rounds

RUN_MODES = ("standard", "writer_only", "no_scout", "no_adversary")

# ── Event phases (used by UI to decide rendering) ─────────────────────────────
PHASE_STATUS   = "status"       # short status string — not full content
PHASE_BRIEFING = "briefing"     # Scout's field briefing
PHASE_DRAFT    = "draft"        # Writer's initial draft
PHASE_REVISION = "revision"     # Writer's mid-round revision
PHASE_FINAL    = "final"        # Writer's final polished output
PHASE_FEEDBACK = "feedback"     # Adversary's full critique
PHASE_TOOL     = "tool_call"    # Tool call notification (Scout or Adversary)
PHASE_TOKENS   = "token_usage"  # Per-agent token usage summary


@dataclass
class TokenUsage:
    agent: str
    round: int
    label: str          # e.g. "scout", "writer_draft", "adversary_r1"
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class AgentEvent:
    agent: str                      # "scout" | "writer" | "adversary"
    round: int                      # 0 = setup phases; 1+ = debate rounds
    phase: str
    content: str
    score: Optional[int] = None
    token_usage: Optional[TokenUsage] = None


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
    run_mode: str = "standard",
) -> tuple:
    """
    Run the multi-agent collaborative review.

    Returns:
        final_output (str)
        all_events (list[AgentEvent])
        final_paper_pool (list[dict])
        token_summary (list[TokenUsage])
    """
    if run_mode not in RUN_MODES:
        raise ValueError(f"run_mode must be one of {RUN_MODES}, got {run_mode!r}")

    max_rounds = min(max(max_rounds, 1), MAX_ROUNDS_CAP)
    events: List[AgentEvent] = []
    token_summary: List[TokenUsage] = []
    all_papers = list(papers)

    def emit(ev: AgentEvent) -> None:
        events.append(ev)
        on_event(ev)

    def _emit_tokens(usage_buf: dict, agent: str, rnd: int, label: str) -> None:
        tu = TokenUsage(
            agent=agent, round=rnd, label=label,
            input_tokens=usage_buf.get("input_tokens", 0),
            output_tokens=usage_buf.get("output_tokens", 0),
        )
        token_summary.append(tu)
        total_k = tu.total / 1000
        emit(AgentEvent(agent, rnd, PHASE_TOKENS,
                        f"{label}: {tu.input_tokens:,} in / {tu.output_tokens:,} out "
                        f"({total_k:.1f}k total)",
                        token_usage=tu))

    # ── Phase 1: Scout (skipped in writer_only and no_scout modes) ────────────
    if run_mode in ("standard", "no_adversary"):
        emit(AgentEvent("scout", 0, PHASE_STATUS,
                        "Enriching paper pool and mapping the field…"))

        def _scout_tool_cb(name: str, inputs: dict) -> None:
            emit(AgentEvent("scout", 0, PHASE_TOOL,
                            f"{name}: {inputs.get('query', '')}"))

        scout_usage: dict = {}
        briefing, new_papers = scout_enrich(
            client, research_question, all_papers, _scout_tool_cb, usage_out=scout_usage,
        )
        all_papers.extend(new_papers)
        emit(AgentEvent("scout", 0, PHASE_BRIEFING, briefing))
        _emit_tokens(scout_usage, "scout", 0, "scout")
    else:
        briefing = ""

    # ── Phase 2: Writer initial draft ──────────────────────────────────────────
    emit(AgentEvent("writer", 0, PHASE_STATUS, "Writing initial draft…"))
    draft_usage: dict = {}
    draft = writer_draft(
        client, research_question, paper_type, all_papers,
        briefing, n_raw, n_deduped, usage_out=draft_usage,
    )
    emit(AgentEvent("writer", 0, PHASE_DRAFT, draft))
    _emit_tokens(draft_usage, "writer", 0, "writer_draft")

    critique = ""
    final_score: Optional[int] = None

    # ── Phase 3: Adversarial review loop (skipped in writer_only / no_adversary) ─
    if run_mode in ("standard", "no_scout"):
        for round_num in range(max_rounds):
            rnd = round_num + 1

            emit(AgentEvent("adversary", rnd, PHASE_STATUS,
                            "Adversarial review: searching for weaknesses and missing papers…"))

            def _adv_tool_cb(name: str, inputs: dict, _rnd=rnd) -> None:
                emit(AgentEvent("adversary", _rnd, PHASE_TOOL,
                                f"{name}: {inputs.get('query', '')}"))

            adv_usage: dict = {}
            critique, score, adv_new_papers = adversary_critique(
                client, draft, research_question, all_papers, _adv_tool_cb,
                usage_out=adv_usage,
            )
            all_papers.extend(adv_new_papers)
            final_score = score
            emit(AgentEvent("adversary", rnd, PHASE_FEEDBACK, critique, score))
            _emit_tokens(adv_usage, "adversary", rnd, f"adversary_r{rnd}")

            is_last       = (round_num >= max_rounds - 1)
            is_acceptable = (score >= ACCEPT_THRESHOLD)

            if is_acceptable or is_last:
                reason = (f"score {score}/10 meets threshold"
                          if is_acceptable else
                          f"max rounds ({max_rounds}) reached")
                emit(AgentEvent("writer", rnd, PHASE_STATUS,
                                f"Writing final polished version ({reason})…"))
                polish_usage: dict = {}
                final = writer_final_polish(
                    client, research_question, all_papers, briefing, draft, critique,
                    usage_out=polish_usage,
                )
                emit(AgentEvent("writer", rnd, PHASE_FINAL, final, score))
                _emit_tokens(polish_usage, "writer", rnd, f"writer_polish_r{rnd}")
                return final, events, all_papers, token_summary

            # Writer revises for next round
            emit(AgentEvent("writer", rnd, PHASE_STATUS,
                            f"Revising draft (score {score}/10 — below threshold)…"))
            revise_usage: dict = {}
            draft = writer_revise(
                client, research_question, all_papers, briefing, draft, critique,
                usage_out=revise_usage,
            )
            emit(AgentEvent("writer", rnd, PHASE_REVISION, draft))
            _emit_tokens(revise_usage, "writer", rnd, f"writer_revise_r{rnd}")

    else:
        # No adversary — single-pass: Writer draft → final polish with empty critique
        emit(AgentEvent("writer", 0, PHASE_STATUS,
                        "Writing final polished version (no adversarial review)…"))
        polish_usage_na: dict = {}
        final = writer_final_polish(
            client, research_question, all_papers, briefing, draft, critique="",
            usage_out=polish_usage_na,
        )
        emit(AgentEvent("writer", 0, PHASE_FINAL, final, None))
        _emit_tokens(polish_usage_na, "writer", 0, "writer_polish")
        return final, events, all_papers, token_summary

    # Safety fallback (unreachable under normal conditions)
    return draft, events, all_papers, token_summary
