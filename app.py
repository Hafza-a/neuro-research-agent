import streamlit as st
import anthropic
import pandas as pd
from typing import Optional

from agent.research_agent import run_research_turn
from agent.literature_agent import (
    plan_and_search, ai_screen_papers,
    verify_citations, build_disclaimer, LitReviewProgress,
)
from agent.contradiction_agent import detect_contradictions
from agent.multi_lit_review.orchestrator import (
    run_multi_agent_review, AgentEvent,
    PHASE_BRIEFING, PHASE_DRAFT, PHASE_REVISION, PHASE_FINAL,
    PHASE_FEEDBACK, PHASE_STATUS, PHASE_TOOL, PHASE_TOKENS,
    RUN_MODES,
)
from agent.evaluator_agent import evaluate_output, EvalScore

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroResearch",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ──────────────────────────────────────────────────────────────
# Colours that work on the dark (#0D1117) base
_C = {
    "scout":    "#38BDF8",   # sky blue
    "writer":   "#4ADE80",   # green
    "adversary":"#F87171",   # coral red
    "indigo":   "#6366F1",   # primary accent
    "muted":    "#8B949E",   # muted text
    "border":   "rgba(240,246,252,0.1)",
    "card":     "rgba(240,246,252,0.04)",
    "success":  "#238636",
}

st.markdown(f"""
<style>
  /* ── Typography ── */
  .nr-title {{
    font-size: 1.75rem; font-weight: 700; letter-spacing: -0.03em;
    color: #E6EDF3; margin-bottom: 0.15rem;
  }}
  .nr-sub {{
    font-size: 0.85rem; color: {_C['muted']}; margin-bottom: 1.4rem;
    letter-spacing: 0.01em;
  }}

  /* ── Step bar ── */
  .step-row {{ display: flex; gap: 10px; margin-bottom: 1.4rem; }}
  .step-cell {{
    flex: 1; padding: 10px 14px; border-radius: 8px;
    background: {_C['card']}; border: 1px solid {_C['border']};
  }}
  .step-cell.active  {{ border-color: {_C['indigo']}; background: rgba(99,102,241,0.10); }}
  .step-cell.done    {{ border-color: {_C['success']}; background: rgba(35,134,54,0.08); }}
  .step-num  {{ font-size: 10px; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.12em; color: {_C['muted']}; margin-bottom: 2px; }}
  .step-name {{ font-size: 13px; font-weight: 600; color: #E6EDF3; }}

  /* ── Agent pills ── */
  .pill {{
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; margin-right: 6px;
  }}
  .pill-scout    {{ background: rgba(56,189,248,0.15); color: {_C['scout']}; }}
  .pill-writer   {{ background: rgba(74,222,128,0.15);  color: {_C['writer']}; }}
  .pill-adversary{{ background: rgba(248,113,113,0.15); color: {_C['adversary']}; }}

  /* ── Status badge ── */
  .status-badge {{
    display: inline-block; padding: 6px 16px; border-radius: 6px;
    font-size: 0.82rem; background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3); color: #A5B4FC;
    font-weight: 500;
  }}

  /* ── Verdict chips ── */
  .verdict-accept  {{ color: #4ADE80; font-weight: 700; }}
  .verdict-minor   {{ color: #FCD34D; font-weight: 700; }}
  .verdict-major   {{ color: #F87171; font-weight: 700; }}
  .verdict-reject  {{ color: #EF4444; font-weight: 700; }}

  /* ── Info card ── */
  .info-card {{
    background: {_C['card']}; border: 1px solid {_C['border']};
    border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
    font-size: 0.84rem; line-height: 1.6; color: {_C['muted']};
  }}
  .info-card b {{ color: #C9D1D9; }}

  /* ── Transcript divider ── */
  .transcript-row {{
    border-left: 2px solid {_C['border']}; padding-left: 14px;
    margin-bottom: 18px;
  }}
  .transcript-row.scout    {{ border-color: {_C['scout']}; }}
  .transcript-row.writer   {{ border-color: {_C['writer']}; }}
  .transcript-row.adversary{{ border-color: {_C['adversary']}; }}

  /* ── Sidebar ── */
  .sb-section {{ font-size: 0.78rem; color: {_C['muted']}; margin-top: 6px; }}
  .sb-label   {{ font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
                 letter-spacing: 0.12em; color: {_C['muted']}; margin-bottom: 4px; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_client() -> Optional[anthropic.Anthropic]:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        key = ""
    if not key or key == "your-api-key-here":
        key = st.session_state.get("api_key_input", "")
    return anthropic.Anthropic(api_key=key) if key else None


def _auth_err():
    st.error(
        "Invalid API key. Get one at "
        "[console.anthropic.com](https://console.anthropic.com) "
        "and update the sidebar or Streamlit Cloud secrets."
    )


def _score_html(score: int) -> str:
    if score >= 8:
        cls, label = "accept", "Accept"
    elif score >= 6:
        cls, label = "minor", "Minor Revision"
    elif score >= 4:
        cls, label = "major", "Major Revision"
    else:
        cls, label = "reject", "Reject"
    return f'<span class="verdict-{cls}">{score}/10 — {label}</span>'


_DB_LABELS = {
    "search_pubmed": "PubMed",
    "search_semantic_scholar": "Semantic Scholar",
    "search_arxiv": "arXiv",
    "search_biorxiv": "bioRxiv",
    "verify_doi": "CrossRef",
}

def _tool_log_cb(log_list: list, ph):
    def _cb(name, inputs):
        label = _DB_LABELS.get(name, name)
        q = inputs.get("query", inputs.get("doi", ""))
        log_list.append(f"**{label}** — *{q}*")
        ph.markdown("\n\n".join(log_list))
    return _cb


def _generate_research_brief(client, research_question: str, papers: list) -> str:
    """Quick single-call brief (no tools) — tells the user what the papers collectively cover."""
    if not papers:
        return ""
    titles_text = "\n".join(
        f"[{i+1}] {p.get('title','?')} ({p.get('year','n.d.')}) — {p.get('source','')}"
        for i, p in enumerate(papers[:30])
    )
    abstracts_text = "\n\n".join(
        f"[{i+1}] {(p.get('abstract') or '')[:250]}"
        for i, p in enumerate(papers[:15])
        if p.get("abstract")
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                f'Research question: "{research_question}"\n\n'
                f"Papers retrieved ({len(papers)}):\n{titles_text}\n\n"
                f"Abstracts (first 15):\n{abstracts_text}\n\n"
                "Write a concise **Research Brief** (150–180 words) structured as:\n"
                "- **Core themes covered** (2–3 bullet points, specific to these papers)\n"
                "- **Coverage of the question** (1 sentence: does this set adequately address it?)\n"
                "- **Likely gaps the Scout will fill** (1 sentence: what's probably missing)\n\n"
                "Be specific to the actual papers above — no generic language."
            ),
        }],
    )
    return "".join(b.text for b in response.content if b.type == "text")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-label">NeuroResearch Agent</div>', unsafe_allow_html=True)
    st.markdown("Research infrastructure for neuroscience")
    st.divider()

    try:
        has_key = (bool(st.secrets.get("ANTHROPIC_API_KEY", ""))
                   and st.secrets.get("ANTHROPIC_API_KEY") != "your-api-key-here")
    except Exception:
        has_key = False

    if not has_key:
        st.markdown('<div class="sb-label">API Key</div>', unsafe_allow_html=True)
        st.text_input("Anthropic API Key", type="password",
                      key="api_key_input", placeholder="sk-ant-…",
                      label_visibility="collapsed")
        if not st.session_state.get("api_key_input"):
            st.warning("Enter your API key to use the agent.")
        st.divider()

    st.markdown('<div class="sb-label">Modes</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-section">'
        '<b>Literature Review</b> — 3-agent collaborative synthesis<br>'
        '<b>Research Assistant</b> — Q&amp;A, gap finding, paper analysis<br>'
        '<b>Contradiction Detector</b> — Find where evidence conflicts'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="sb-label">Agent Panel</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sb-section">'
        f'<span style="color:{_C["scout"]}">Scout</span> — enriches the paper pool<br>'
        f'<span style="color:{_C["writer"]}">Writer</span> — produces the review<br>'
        f'<span style="color:{_C["adversary"]}">Adversary</span> — adversarial critique'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<div class="sb-label">Databases</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-section">PubMed · Semantic Scholar · arXiv · bioRxiv</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        f'<div class="sb-section" style="color:{_C["muted"]}">Powered by Claude claude-sonnet-4-6</div>',
        unsafe_allow_html=True,
    )


# ── Page header ────────────────────────────────────────────────────────────────
st.markdown('<div class="nr-title">NeuroResearch Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="nr-sub">'
    'Multi-agent literature synthesis &nbsp;&middot;&nbsp; '
    'Research Q&amp;A &nbsp;&middot;&nbsp; Contradiction detection'
    '</div>',
    unsafe_allow_html=True,
)

tab_lit, tab_research, tab_contradiction = st.tabs([
    "Literature Review",
    "Research Assistant",
    "Contradiction Detector",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LITERATURE REVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_lit:

    _LR_DEFAULT: dict = {
        "step": "idle",
        "question": "",
        "paper_type": "Systematic Literature Review",
        "max_rounds": 1,
        "run_mode": "standard",
        "papers_per_db": 10,
        "all_papers": [],
        "papers": [],
        "n_raw": 0,
        "n_deduped": 0,
        "plan": {},
        "research_brief": "",
        "selected_papers": [],
        "final_output": "",
        "final_papers": [],
        "events": [],
        "verified": [],
        "final_score": None,
        "token_summary": [],
        "eval_score": None,
    }

    if "mlr" not in st.session_state:
        st.session_state["mlr"] = dict(_LR_DEFAULT)
    mlr = st.session_state["mlr"]

    # ── Step bar ──────────────────────────────────────────────────────────────
    def _step_cls(target: str) -> str:
        order = {"idle": 0, "searched": 1, "done": 2}
        cur = order.get(mlr["step"], 0)
        tgt = order.get(target, 0)
        if cur == tgt:
            return "active"
        if cur > tgt:
            return "done"
        return ""

    st.markdown(
        f'<div class="step-row">'
        f'<div class="step-cell {_step_cls("idle")}">'
        f'<div class="step-num">Step 1</div>'
        f'<div class="step-name">Configure &amp; Search</div></div>'
        f'<div class="step-cell {_step_cls("searched")}">'
        f'<div class="step-num">Step 2</div>'
        f'<div class="step-name">Select Papers</div></div>'
        f'<div class="step-cell {_step_cls("done")}">'
        f'<div class="step-num">Step 3</div>'
        f'<div class="step-name">Review &amp; Export</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1 — Configure & Search
    # ════════════════════════════════════════════════════════════════════════
    with st.expander("Step 1 — Configure & Search",
                     expanded=(mlr["step"] == "idle")):

        cfg_col, inp_col = st.columns([1, 2])

        with cfg_col:
            # How paper pool is built — explanation
            st.markdown(
                '<div class="info-card">'
                '<b>Where does the paper pool come from?</b><br>'
                'Clicking Search queries PubMed, Semantic Scholar, arXiv, and bioRxiv '
                'using Claude-generated search terms. You then select which papers to '
                'include before the agents start. The Scout agent additionally runs '
                'targeted searches to find foundational or recent papers that the '
                'initial query may have missed.'
                '</div>',
                unsafe_allow_html=True,
            )

            paper_type = st.selectbox(
                "Paper type",
                ["Systematic Literature Review", "Narrative Review / Synthesis",
                 "Research Gap Analysis", "Introduction Section"],
                key="mlr_paper_type",
            )
            max_rounds = st.radio(
                "Adversarial review rounds",
                [1, 2, 3], index=0, horizontal=True, key="mlr_rounds",
                help=(
                    "Each round: Adversary critiques → Writer revises. "
                    "Stops early if the Adversary scores ≥ 7/10. "
                    "1 round = ~5–7 API calls; 3 rounds = ~9 API calls."
                ),
            )
            run_mode = st.selectbox(
                "Agent pipeline",
                ["standard", "no_scout", "no_adversary", "writer_only"],
                index=0, key="mlr_run_mode",
                help=(
                    "**standard** — full Scout→Writer↔Adversary pipeline\n\n"
                    "**no_scout** — skip Scout briefing (Writer gets no field orientation)\n\n"
                    "**no_adversary** — Scout runs, Writer drafts once, no critique loop\n\n"
                    "**writer_only** — baseline: Writer drafts and polishes alone"
                ),
            )
            papers_per_db = st.slider("Papers per database", 5, 20, 10, key="mlr_ppdb")

        with inp_col:
            research_question = st.text_area(
                "Research question or topic",
                placeholder=(
                    "Be specific — the more precise your question, the better the review.\n\n"
                    "e.g. What is the role of Dubosiella in neurodegenerative diseases?\n"
                    "e.g. Hippocampal neurogenesis and spatial memory consolidation\n"
                    "e.g. Neural correlates of working memory in prefrontal cortex"
                ),
                height=130, key="mlr_rq",
            )

            st.markdown(
                '<div class="info-card">'
                f'<b><span style="color:{_C["scout"]}">Scout</span></b> — '
                'Enriches the paper pool with targeted searches. '
                'Writes a field briefing (key themes, essential citations, visible debates) '
                'that the Writer uses to orient the first draft.<br><br>'
                f'<b><span style="color:{_C["writer"]}">Writer</span></b> — '
                'Produces the full literature review from the enriched pool. '
                'Only cites papers in the pool — never fabricates.<br><br>'
                f'<b><span style="color:{_C["adversary"]}">Adversary</span></b> — '
                'Adversarially reviews the draft. Searches for missing papers and '
                'unsupported claims. Scores 1–10; score ≥ 7 triggers the final polish.'
                '</div>',
                unsafe_allow_html=True,
            )

            search_btn = st.button(
                "Search Databases", type="primary",
                use_container_width=True, key="mlr_search",
            )

        if search_btn:
            if not research_question.strip():
                st.error("Please enter a research question.")
            else:
                client = get_client()
                if not client:
                    st.error("Enter your Anthropic API key in the sidebar.")
                    st.stop()

                prog_ph = st.empty()

                def _lr_prog(p: LitReviewProgress):
                    prog_ph.markdown(
                        f'<div class="status-badge">{p.phase}: {p.message}</div>',
                        unsafe_allow_html=True,
                    )

                try:
                    with st.spinner("Querying 4 databases…"):
                        plan, all_papers, papers = plan_and_search(
                            client, research_question, papers_per_db, _lr_prog,
                        )
                except anthropic.AuthenticationError:
                    prog_ph.empty()
                    _auth_err()
                    st.stop()
                except Exception as e:
                    prog_ph.empty()
                    st.error(f"Search error: {e}")
                    st.stop()

                prog_ph.empty()
                # Generate the research brief immediately after search
                brief = ""
                try:
                    with st.spinner("Generating research brief…"):
                        brief = _generate_research_brief(client, research_question, papers)
                except Exception:
                    pass  # Brief is optional — don't block on failure

                st.session_state["mlr"] = dict(_LR_DEFAULT) | {
                    "step": "searched",
                    "plan": plan,
                    "all_papers": all_papers,
                    "papers": papers,
                    "n_raw": len(all_papers),
                    "n_deduped": len(papers),
                    "question": research_question,
                    "paper_type": paper_type,
                    "max_rounds": max_rounds,
                    "run_mode": run_mode,
                    "papers_per_db": papers_per_db,
                    "research_brief": brief,
                }
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2 — Paper selection + Research Brief
    # ════════════════════════════════════════════════════════════════════════
    if mlr["step"] in ("searched", "done"):
        with st.expander(
            f"Step 2 — Select Papers  "
            f"({mlr['n_deduped']} unique · {mlr['n_raw']} retrieved)",
            expanded=(mlr["step"] == "searched"),
        ):
            # ── Research Brief ────────────────────────────────────────────
            if mlr.get("research_brief"):
                st.markdown(
                    f'<div class="info-card">'
                    f'<b>Research Brief</b><br><br>'
                    f'{mlr["research_brief"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                brief_col, _ = st.columns([1, 3])
                with brief_col:
                    if st.button("Refresh Brief", key="mlr_refresh_brief", use_container_width=True):
                        client_b = get_client()
                        if client_b:
                            with st.spinner("Re-generating brief…"):
                                try:
                                    mlr["research_brief"] = _generate_research_brief(
                                        client_b, mlr["question"], mlr["papers"]
                                    )
                                except Exception:
                                    pass
                            st.rerun()
            elif mlr["step"] == "searched":
                st.info("Research brief will appear here after searching.")

            st.divider()

            if mlr["papers"]:
                df_data = []
                for i, p in enumerate(mlr["papers"]):
                    if not p.get("title"):
                        continue
                    df_data.append({
                        "Include": True,
                        "#": i,
                        "Title": p["title"][:95],
                        "Year": p.get("year", ""),
                        "Source": p.get("source", ""),
                        "Citations": p.get("citations", ""),
                    })
                df = pd.DataFrame(df_data)

                st.caption(
                    "Toggle **Include** to remove papers. "
                    "The Scout will also search for additional foundational papers you may have missed."
                )
                edited = st.data_editor(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Include": st.column_config.CheckboxColumn("Include", default=True, width="small"),
                        "#": st.column_config.NumberColumn("#", width="small"),
                        "Title": st.column_config.TextColumn("Title", width="large"),
                        "Year": st.column_config.TextColumn("Year", width="small"),
                        "Source": st.column_config.TextColumn("Source", width="small"),
                        "Citations": st.column_config.NumberColumn("Citations", width="small"),
                    },
                    disabled=["#", "Title", "Year", "Source", "Citations"],
                    key="mlr_editor",
                )

                selected_idx = edited[edited["Include"] == True]["#"].tolist()
                n_sel = len(selected_idx)
                st.caption(f"{n_sel} of {len(df_data)} papers selected.")

                b1, b2, b3 = st.columns([3, 2, 1])
                with b1:
                    run_btn = st.button(
                        f"Generate Literature Review  →  ({n_sel} papers)",
                        type="primary", use_container_width=True,
                        disabled=(n_sel == 0), key="mlr_run",
                    )
                with b2:
                    auto_btn = st.button(
                        "Auto-select best papers",
                        use_container_width=True, key="mlr_auto",
                    )
                with b3:
                    if st.button("Reset", use_container_width=True, key="mlr_reset"):
                        st.session_state["mlr"] = dict(_LR_DEFAULT)
                        st.rerun()

                # ── Orchestrator runner ───────────────────────────────────────
                def _run_agents(client, selected_papers):
                    verbose = True  # always capture transcript; display toggle is in Step 3
                    cur_mode = mlr.get("run_mode", "standard")
                    has_scout = cur_mode in ("standard", "no_adversary")
                    has_adv   = cur_mode in ("standard", "no_scout")
                    total_steps = (
                        (1 if has_scout else 0)
                        + 1  # writer draft
                        + (mlr["max_rounds"] * 2 if has_adv else 1)
                    )
                    step_ctr = [0]

                    status_ph = st.empty()
                    prog_bar  = st.progress(0)

                    if verbose:
                        st.markdown("---")
                        st.markdown("**Agent Collaboration Transcript**")
                        ev_container = st.container()

                    def on_event(ev: AgentEvent):
                        agent_color = _C.get(ev.agent, _C["muted"])

                        if ev.phase == PHASE_STATUS:
                            step_ctr[0] = min(step_ctr[0] + 1, total_steps - 1)
                            prog_bar.progress(step_ctr[0] / total_steps)
                            status_ph.markdown(
                                f'<div class="status-badge">'
                                f'<span style="color:{agent_color}">{ev.agent.upper()}</span>'
                                f' &nbsp;— {ev.content}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            return

                        if ev.phase == PHASE_TOKENS:
                            # Always show token usage as a quiet caption
                            if verbose:
                                with ev_container:
                                    st.caption(f"Tokens — {ev.content}")
                            return

                        if not verbose:
                            return

                        if ev.phase == PHASE_BRIEFING:
                            with ev_container:
                                with st.expander("Scout — Field Briefing", expanded=False):
                                    st.markdown(ev.content)

                        elif ev.phase == PHASE_DRAFT:
                            with ev_container:
                                with st.expander("Writer — Initial Draft", expanded=False):
                                    st.markdown(ev.content)

                        elif ev.phase == PHASE_REVISION:
                            with ev_container:
                                with st.expander(f"Writer — Revision (Round {ev.round})", expanded=False):
                                    st.markdown(ev.content)

                        elif ev.phase == PHASE_FINAL:
                            with ev_container:
                                with st.expander(
                                    f"Writer — Final Version  |  {ev.score}/10" if ev.score else "Writer — Final Version",
                                    expanded=False,
                                ):
                                    st.markdown(ev.content)

                        elif ev.phase == PHASE_FEEDBACK:
                            badge = f"  ({ev.score}/10)" if ev.score else ""
                            with ev_container:
                                with st.expander(
                                    f"Adversary — Critique Round {ev.round}{badge}",
                                    expanded=False,
                                ):
                                    st.markdown(ev.content)

                        elif ev.phase == PHASE_TOOL:
                            with ev_container:
                                st.caption(f"{ev.agent.title()} searched: {ev.content}")

                    try:
                        final_output, all_events, final_papers, token_summary = run_multi_agent_review(
                            client=client,
                            research_question=mlr["question"],
                            paper_type=mlr["paper_type"],
                            papers=selected_papers,
                            n_raw=mlr["n_raw"],
                            n_deduped=mlr["n_deduped"],
                            max_rounds=mlr["max_rounds"],
                            on_event=on_event,
                            run_mode=cur_mode,
                        )
                    except anthropic.AuthenticationError:
                        status_ph.empty()
                        prog_bar.empty()
                        _auth_err()
                        st.stop()
                    except Exception as e:
                        status_ph.empty()
                        prog_bar.empty()
                        st.error(f"Agent error: {e}")
                        st.stop()

                    prog_bar.progress(1.0)
                    status_ph.markdown(
                        '<div class="status-badge">Review complete</div>',
                        unsafe_allow_html=True,
                    )

                    with st.spinner("Verifying citations via CrossRef…"):
                        verified = verify_citations(final_papers)

                    disclaimer = build_disclaimer(mlr["n_raw"], mlr["n_deduped"], len(final_papers))
                    final_output += disclaimer

                    final_score = next(
                        (e.score for e in reversed(all_events)
                         if e.phase == PHASE_FEEDBACK and e.score),
                        None,
                    )

                    eval_score = evaluate_output(
                        final_text=final_output,
                        papers=final_papers,
                        adversary_score=final_score or 0,
                    )

                    mlr.update({
                        "step": "done",
                        "selected_papers": selected_papers,
                        "final_output": final_output,
                        "final_papers": final_papers,
                        "events": [
                            {"agent": e.agent, "round": e.round, "phase": e.phase,
                             "content": e.content, "score": e.score}
                            for e in all_events
                        ],
                        "verified": verified,
                        "final_score": final_score,
                        "token_summary": [
                            {"label": t.label, "agent": t.agent,
                             "input_tokens": t.input_tokens, "output_tokens": t.output_tokens}
                            for t in token_summary
                        ],
                        "eval_score": eval_score.as_dict(),
                    })
                    st.rerun()

                if run_btn and selected_idx:
                    client = get_client()
                    if not client:
                        st.error("Enter your Anthropic API key.")
                        st.stop()
                    chosen = [mlr["papers"][i] for i in selected_idx if i < len(mlr["papers"])]
                    _run_agents(client, chosen)

                if auto_btn:
                    client = get_client()
                    if not client:
                        st.error("Enter your Anthropic API key.")
                        st.stop()
                    try:
                        with st.spinner("AI selecting best papers…"):
                            chosen = ai_screen_papers(client, mlr["question"], mlr["papers"])
                    except anthropic.AuthenticationError:
                        _auth_err()
                        st.stop()
                    except Exception as e:
                        st.error(f"Screening error: {e}")
                        st.stop()
                    _run_agents(client, chosen)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3 — Final output
    # ════════════════════════════════════════════════════════════════════════
    if mlr["step"] == "done" and mlr["final_output"]:
        st.divider()

        # Header row
        score_html = _score_html(mlr["final_score"]) if mlr["final_score"] else ""
        mode_label = mlr.get("run_mode", "standard")
        summary_md = (
            f"{len(mlr['final_papers'])} papers synthesised  ·  Mode: **{mode_label}**"
            + (f"   ·   Final verdict: " if mlr["final_score"] else "")
        )
        st.success(summary_md)
        if score_html:
            st.markdown(score_html, unsafe_allow_html=True)

        # ── EvalScore panel ──────────────────────────────────────────────────
        es = mlr.get("eval_score")
        if es:
            with st.expander("Automatic Quality Scores (EvalScore)", expanded=True):
                ec1, ec2, ec3, ec4, ec5 = st.columns(5)
                ec1.metric("Citation Coverage", f"{es['citation_coverage']*100:.0f}%",
                           help="% of pool papers cited inline in the final text")
                ec2.metric("Synthesis Depth", f"{es['synthesis_depth']}/10",
                           help="Thematic synthesis vs. one-by-one paper summaries")
                ec3.metric("Gap Specificity", f"{es['gap_specificity']}/10",
                           help="Mechanism-level gap language vs. vague 'more research needed'")
                ec4.metric("Adversary Score", f"{es['adversary_score']}/10",
                           help="Final adversary verdict (0 if no adversary used)")
                ec5.metric("Overall", f"{es['overall']:.1f}/100",
                           help="Weighted composite: coverage 25%, synthesis 35%, gaps 25%, adversary 15%")

        # ── Token usage breakdown ────────────────────────────────────────────
        tokens = mlr.get("token_summary", [])
        if tokens:
            total_in  = sum(t["input_tokens"]  for t in tokens)
            total_out = sum(t["output_tokens"] for t in tokens)
            with st.expander(
                f"Token Usage  ({(total_in+total_out)/1000:.1f}k total)", expanded=False
            ):
                rows = [{
                    "Phase": t["label"],
                    "Agent": t["agent"].title(),
                    "Input": f"{t['input_tokens']:,}",
                    "Output": f"{t['output_tokens']:,}",
                    "Total": f"{t['input_tokens']+t['output_tokens']:,}",
                } for t in tokens]
                rows.append({
                    "Phase": "**TOTAL**", "Agent": "",
                    "Input": f"{total_in:,}", "Output": f"{total_out:,}",
                    "Total": f"{total_in+total_out:,}",
                })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption(
                    f"Estimated cost (Claude Sonnet 4.6): "
                    f"~${(total_in*3 + total_out*15) / 1_000_000:.4f} USD"
                )

        # Download / reset
        dl_col, clr_col = st.columns([4, 1])
        with dl_col:
            safe = mlr["question"][:50].replace(" ", "_").replace("/", "-")
            st.download_button(
                "Download Review (.md)",
                data=mlr["final_output"],
                file_name=f"litreview_{safe}.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True,
                key="mlr_dl",
            )
        with clr_col:
            if st.button("New Review", use_container_width=True, key="mlr_new"):
                st.session_state["mlr"] = dict(_LR_DEFAULT)
                st.rerun()

        # Agent transcript (always captured; collapsed by default)
        content_events = [e for e in mlr["events"]
                          if e["phase"] in (PHASE_BRIEFING, PHASE_DRAFT, PHASE_REVISION,
                                            PHASE_FINAL, PHASE_FEEDBACK)]
        if content_events:
            with st.expander(
                f"Show agent work ({len(content_events)} exchanges — Scout briefing, drafts, critiques)",
                expanded=False,
            ):
                for e in mlr["events"]:
                    agent = e["agent"]
                    rnd   = e["round"]
                    phase = e["phase"]

                    if phase == PHASE_STATUS:
                        continue

                    color = _C.get(agent, _C["muted"])
                    label = agent.title()

                    if phase == PHASE_BRIEFING:
                        st.markdown(f'<div class="transcript-row scout">'
                                    f'<strong style="color:{color}">Scout — Field Briefing</strong>'
                                    f'</div>', unsafe_allow_html=True)
                        with st.expander("View briefing", expanded=False):
                            st.markdown(e["content"])

                    elif phase == PHASE_DRAFT:
                        st.markdown(f'<div class="transcript-row writer">'
                                    f'<strong style="color:{color}">Writer — Initial Draft</strong>'
                                    f'</div>', unsafe_allow_html=True)
                        with st.expander("View initial draft", expanded=False):
                            st.markdown(e["content"])

                    elif phase == PHASE_REVISION:
                        st.markdown(f'<div class="transcript-row writer">'
                                    f'<strong style="color:{color}">Writer — Revision (Round {rnd})</strong>'
                                    f'</div>', unsafe_allow_html=True)
                        with st.expander(f"View revision {rnd}", expanded=False):
                            st.markdown(e["content"])

                    elif phase == PHASE_FINAL:
                        badge = _score_html(e["score"]) if e["score"] else ""
                        st.markdown(
                            f'<div class="transcript-row writer">'
                            f'<strong style="color:{color}">Writer — Final Version</strong>'
                            + (f'&nbsp;&nbsp;{badge}' if badge else '')
                            + f'</div>',
                            unsafe_allow_html=True,
                        )
                        with st.expander("View final draft (pre-formatted output)", expanded=False):
                            st.markdown(e["content"])

                    elif phase == PHASE_FEEDBACK:
                        badge = _score_html(e["score"]) if e["score"] else ""
                        st.markdown(
                            f'<div class="transcript-row adversary">'
                            f'<strong style="color:{color}">Adversary — Critique Round {rnd}</strong>'
                            + (f'&nbsp;&nbsp;{badge}' if badge else '')
                            + f'</div>',
                            unsafe_allow_html=True,
                        )
                        with st.expander(f"View adversary critique (Round {rnd})", expanded=False):
                            st.markdown(e["content"])

                    elif phase == PHASE_TOOL:
                        st.caption(f"{label} searched: {e['content']}")

        # Citation verification
        if mlr["verified"]:
            with st.expander("Citation Verification Report", expanded=True):
                status_icon = {
                    "verified":  "Verified",
                    "preprint":  "Preprint",
                    "not_found": "Not found",
                    "no_doi":    "No DOI",
                }
                rows = [{
                    "Ref":     f"[{v['index']}]",
                    "Status":  status_icon.get(v["status"], v["status"]),
                    "Title":   v["title"],
                    "Journal": v.get("journal", ""),
                    "DOI/URL": v.get("doi") or v.get("url", ""),
                } for v in mlr["verified"]]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                n_ok  = sum(1 for v in mlr["verified"] if v["status"] == "verified")
                n_pre = sum(1 for v in mlr["verified"] if v["status"] == "preprint")
                n_bad = sum(1 for v in mlr["verified"]
                            if v["status"] in ("not_found", "no_doi"))
                st.caption(
                    f"{n_ok} verified via CrossRef · "
                    f"{n_pre} preprints (not peer-reviewed) · "
                    f"{n_bad} unverified — check manually before citing"
                )

        # Final review body — presented as a clean academic document
        st.markdown("---")
        st.markdown(
            f'<div style="max-width:860px; margin:0 auto;">'
            f'<div style="font-size:0.7rem; font-weight:700; text-transform:uppercase; '
            f'letter-spacing:0.12em; color:{_C["muted"]}; margin-bottom:0.5rem;">'
            f'Generated Literature Review · {mlr.get("paper_type","Review")} · '
            f'{len(mlr["final_papers"])} papers</div>',
            unsafe_allow_html=True,
        )
        st.markdown(mlr["final_output"])
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESEARCH ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
with tab_research:
    col_main, col_help = st.columns([3, 1])

    with col_help:
        with st.expander("What can I ask?", expanded=True):
            st.markdown(
                "**Research Q&A**\n"
                "> *What is the role of astrocytes in synaptic pruning?*\n\n"
                "**Gap Finder**\n"
                "> *What are open questions in hippocampal place cells?*\n\n"
                "**Paper Deep-Dive**\n"
                "> *Critique the default mode network in Alzheimer's*"
            )

    with col_main:
        if "research_messages" not in st.session_state:
            st.session_state.research_messages = []

        b1, b2, _ = st.columns([1, 1, 2])
        with b1:
            if st.button("Clear", key="clear_research", use_container_width=True):
                st.session_state.research_messages = []
                st.rerun()
        with b2:
            if st.session_state.research_messages:
                def _export():
                    lines = ["# NeuroResearch — Research Session\n"]
                    for m in st.session_state.research_messages:
                        role = "**You**" if m["role"] == "user" else "**Agent**"
                        lines.append(f"### {role}\n\n{m['content']}\n\n---\n")
                    return "\n".join(lines)
                st.download_button(
                    "Export session", data=_export(),
                    file_name="research_session.md", mime="text/markdown",
                    use_container_width=True, key="export_research",
                )

        for msg in st.session_state.research_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a neuroscience question, find gaps, or analyse a paper…"):
            client = get_client()
            if not client:
                st.error("Enter your Anthropic API key in the sidebar.")
                st.stop()

            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.research_messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                tool_ph = st.empty()
                resp_ph = st.empty()
                log = []
                try:
                    with st.spinner("Searching…"):
                        api_msgs = [{"role": m["role"], "content": m["content"]}
                                    for m in st.session_state.research_messages]
                        response = run_research_turn(
                            client, api_msgs,
                            on_tool_call=_tool_log_cb(log, tool_ph),
                        )
                    tool_ph.empty()
                    if log:
                        with st.expander(f"Searched {len(log)} database(s)", expanded=False):
                            for l in log:
                                st.markdown(l)
                    resp_ph.markdown(response)
                    st.session_state.research_messages.append(
                        {"role": "assistant", "content": response})
                except anthropic.AuthenticationError:
                    tool_ph.empty()
                    st.session_state.research_messages.pop()
                    _auth_err()
                except Exception as e:
                    tool_ph.empty()
                    st.session_state.research_messages.pop()
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONTRADICTION DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_contradiction:
    st.markdown("### Contradiction Detector")
    st.markdown(
        "Enter a specific neuroscience claim. "
        "The agent searches for papers that support and contradict it, "
        "then explains why the evidence conflicts."
    )

    col_in, col_ex = st.columns([2, 1])
    with col_ex:
        with st.expander("Example claims", expanded=True):
            st.markdown(
                "- *SSRIs increase adult hippocampal neurogenesis*\n"
                "- *The gut microbiome influences anxiety behaviour in mice*\n"
                "- *Amyloid plaques cause Alzheimer's disease*\n"
                "- *Sleep deprivation impairs long-term potentiation*\n"
                "- *Microglia are the primary mediators of synaptic pruning*"
            )

    with col_in:
        claim = st.text_area(
            "Claim to investigate",
            placeholder="e.g. SSRIs increase adult hippocampal neurogenesis",
            height=80, key="cd_claim",
        )

        if "cd_result" not in st.session_state:
            st.session_state["cd_result"] = ""
            st.session_state["cd_claim_used"] = ""

        cd_btn = st.button("Analyse Contradictions", type="primary",
                           use_container_width=True, key="cd_run")

        if cd_btn:
            if not claim.strip():
                st.error("Please enter a claim to investigate.")
            else:
                client = get_client()
                if not client:
                    st.error("Enter your Anthropic API key.")
                    st.stop()
                tool_ph = st.empty()
                log = []
                try:
                    with st.spinner("Searching for supporting and contradicting evidence…"):
                        result = detect_contradictions(
                            client, claim,
                            on_tool_call=_tool_log_cb(log, tool_ph),
                        )
                except anthropic.AuthenticationError:
                    tool_ph.empty()
                    _auth_err()
                    st.stop()
                except Exception as e:
                    tool_ph.empty()
                    st.error(f"Error: {e}")
                    st.stop()
                tool_ph.empty()
                st.session_state["cd_result"] = result
                st.session_state["cd_claim_used"] = claim
                st.rerun()

    if st.session_state.get("cd_result"):
        st.divider()
        st.success(f"Report for: *{st.session_state['cd_claim_used'][:80]}*")
        dl_col, clr_col = st.columns([4, 1])
        with dl_col:
            safe = st.session_state["cd_claim_used"][:40].replace(" ", "_")
            st.download_button(
                "Download Report (.md)",
                data=st.session_state["cd_result"],
                file_name=f"contradiction_{safe}.md",
                mime="text/markdown",
                use_container_width=True, key="cd_dl",
            )
        with clr_col:
            if st.button("Clear", use_container_width=True, key="cd_clr"):
                st.session_state["cd_result"] = ""
                st.rerun()
        st.markdown(st.session_state["cd_result"])
