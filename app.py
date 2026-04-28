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
    PHASE_DRAFT, PHASE_REVISION, PHASE_FINAL, PHASE_FEEDBACK,
    PHASE_STATUS, PHASE_TOOL,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroResearch Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size:2.1rem; font-weight:700; color:#1a1a2e; margin-bottom:.1rem; }
    .sub-header  { font-size:.95rem; color:#555; margin-bottom:1.2rem; }
    .phase-badge { background:#f0f7ff; padding:8px 16px; border-radius:20px;
                   font-size:.9rem; color:#1a73e8; font-weight:500; display:inline-block; }
    .agent-badge-student  { background:#e8f5e9; padding:5px 12px; border-radius:14px;
                            color:#1b7c3d; font-weight:600; font-size:.85rem; }
    .agent-badge-supervisor { background:#fff3e0; padding:5px 12px; border-radius:14px;
                              color:#e65100; font-weight:600; font-size:.85rem; }
    .agent-badge-peer     { background:#e3f2fd; padding:5px 12px; border-radius:14px;
                            color:#1565c0; font-weight:600; font-size:.85rem; }
    .agent-badge-reviewer { background:#fce4ec; padding:5px 12px; border-radius:14px;
                            color:#880e4f; font-weight:600; font-size:.85rem; }
    .step-label  { font-size:.8rem; font-weight:700; color:#6b7280;
                   text-transform:uppercase; letter-spacing:.05em; margin-bottom:4px; }
    .step-active { border-left:3px solid #1a73e8; padding-left:10px; }
    .step-done   { border-left:3px solid #1b7c3d; padding-left:10px; color:#888; }
</style>
""", unsafe_allow_html=True)


# ── API client ─────────────────────────────────────────────────────────────────
def get_client() -> Optional[anthropic.Anthropic]:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        key = ""
    if not key or key == "your-api-key-here":
        key = st.session_state.get("api_key_input", "")
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 NeuroResearch Agent")
    st.markdown("*AI research infrastructure for neuroscience*")
    st.divider()

    try:
        has_key = (bool(st.secrets.get("ANTHROPIC_API_KEY", ""))
                   and st.secrets.get("ANTHROPIC_API_KEY") != "your-api-key-here")
    except Exception:
        has_key = False

    if not has_key:
        st.markdown("**API Key**")
        st.text_input("Anthropic API Key", type="password",
                      key="api_key_input", placeholder="sk-ant-…")
        if not st.session_state.get("api_key_input"):
            st.warning("Enter your API key to use the agent.")
        st.divider()

    st.markdown("**Three Modes**")
    st.markdown("📚 **Literature Review** — 4-agent collaborative review")
    st.markdown("🔬 **Research Assistant** — Q&A, gap finding, paper analysis")
    st.markdown("⚡ **Contradiction Detector** — Find where evidence conflicts")
    st.divider()
    st.markdown("**4-Agent Panel**")
    st.markdown("👨‍🎓 PhD Student · 👨‍🏫 Supervisor")
    st.markdown("🧑‍💻 Peer Researcher · 🔬 Journal Reviewer")
    st.divider()
    st.markdown("**Databases**")
    st.markdown("PubMed · Semantic Scholar · arXiv · bioRxiv")
    st.divider()
    st.markdown("*Powered by Claude claude-sonnet-4-6*")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 NeuroResearch Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Multi-Agent Literature Reviews &nbsp;·&nbsp; Research Q&A &nbsp;·&nbsp; '
    'Contradiction Detection'
    '</div>',
    unsafe_allow_html=True,
)

tab_litreview, tab_research, tab_contradiction = st.tabs([
    "📚 Literature Review",
    "🔬 Research Assistant",
    "⚡ Contradiction Detector",
])


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════
_DB_LABELS = {
    "search_pubmed": "PubMed",
    "search_semantic_scholar": "Semantic Scholar",
    "search_arxiv": "arXiv",
    "search_biorxiv": "bioRxiv",
    "verify_doi": "CrossRef",
}

def _tool_log_callback(log_list: list, placeholder):
    def _cb(name, inputs):
        label = _DB_LABELS.get(name, name)
        q = inputs.get("query", inputs.get("doi", ""))
        log_list.append(f"🔍 **{label}**: *{q}*")
        placeholder.markdown("\n".join(log_list))
    return _cb

_AGENT_EMOJI = {
    "student": "👨‍🎓",
    "supervisor": "👨‍🏫",
    "peer": "🧑‍💻",
    "reviewer": "🔬",
    "system": "⚙️",
}
_AGENT_LABEL = {
    "student": "PhD Student",
    "supervisor": "Supervisor",
    "peer": "Peer Researcher",
    "reviewer": "Journal Reviewer",
    "system": "System",
}

def _score_badge(score: int) -> str:
    if score >= 9:
        return f"✅ {score}/10 — Accept"
    if score >= 7:
        return f"🔄 {score}/10 — Minor Revision"
    if score >= 5:
        return f"⚠️ {score}/10 — Major Revision"
    return f"❌ {score}/10 — Reject"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LITERATURE REVIEW (4-agent multi-agent)
# ══════════════════════════════════════════════════════════════════════════════
with tab_litreview:

    # ── State init ────────────────────────────────────────────────────────────
    _LR_DEFAULT = {
        "step": "idle",          # idle | searched | done
        "question": "",
        "paper_type": "Systematic Literature Review",
        "max_rounds": 2,
        "view_mode": "final",    # "verbose" | "final"
        "papers_per_db": 10,
        "all_papers": [],
        "papers": [],
        "n_raw": 0,
        "n_deduped": 0,
        "plan": {},
        "selected_papers": [],
        "final_output": "",
        "final_papers": [],
        "events": [],            # list of dicts (serialised AgentEvents)
        "verified": [],
        "final_score": None,
    }
    if "mlr" not in st.session_state:
        st.session_state["mlr"] = dict(_LR_DEFAULT)
    mlr = st.session_state["mlr"]

    # ── Step indicator ────────────────────────────────────────────────────────
    s1, s2, s3 = st.columns(3)
    with s1:
        cls = "step-active" if mlr["step"] == "idle" else "step-done"
        st.markdown(f'<div class="{cls}"><div class="step-label">Step 1</div>Search & Configure</div>',
                    unsafe_allow_html=True)
    with s2:
        cls = "step-active" if mlr["step"] == "searched" else (
              "step-done" if mlr["step"] == "done" else "")
        st.markdown(f'<div class="{cls}"><div class="step-label">Step 2</div>Select Papers</div>',
                    unsafe_allow_html=True)
    with s3:
        cls = "step-active" if mlr["step"] == "done" else ""
        st.markdown(f'<div class="{cls}"><div class="step-label">Step 3</div>Review & Export</div>',
                    unsafe_allow_html=True)
    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1 — Configure & Search
    # ════════════════════════════════════════════════════════════════════════
    with st.expander("📝 Step 1 — Define Research Question & Options",
                     expanded=(mlr["step"] == "idle")):

        top_col, inp_col = st.columns([1, 2])

        with top_col:
            paper_type = st.selectbox(
                "Paper type",
                ["Systematic Literature Review", "Narrative Review / Synthesis",
                 "Research Gap Analysis", "Introduction Section"],
                key="mlr_paper_type",
            )
            max_rounds = st.radio(
                "Agent debate rounds",
                [1, 2, 3],
                index=1,
                horizontal=True,
                key="mlr_rounds",
                help="1 = fast, 3 = highest quality. Each round: Supervisor → Peer → Reviewer → Student revises.",
            )
            view_mode = st.radio(
                "Output mode",
                ["Final output only", "Verbose (show all agent dialogue)"],
                index=0,
                key="mlr_view_mode",
                help=(
                    "Verbose shows every agent's full output as it happens — "
                    "great for transparency. Final-only is faster to read."
                ),
            )
            papers_per_db = st.slider("Papers per database", 5, 20, 10, key="mlr_ppdb")

        with inp_col:
            examples = [
                "Synaptic plasticity mechanisms in hippocampal memory consolidation",
                "Role of the gut-brain axis in anxiety disorders",
                "Neuroinflammation and major depressive disorder",
                "Optogenetic dissection of basal ganglia circuits",
                "Adult neurogenesis in the human hippocampus",
                "Glial cells in Alzheimer's disease pathology",
                "Dopamine signalling in reward learning",
            ]
            ex = st.selectbox("Quick example:", [""] + examples, key="mlr_example")
            research_question = st.text_area(
                "Research question / topic",
                value=ex if ex else "",
                placeholder="e.g. What is the role of neuroinflammation in major depressive disorder?",
                height=110, key="mlr_rq",
            )

            st.markdown("""
**How the 4-agent panel works:**
> 👨‍🎓 **Student** writes the initial draft from your paper pool
> 👨‍🏫 **Supervisor** critiques structure, rigour, and argument
> 🧑‍💻 **Peer** searches for missing papers and adds perspectives
> 🔬 **Reviewer** scores 1–10 (≥ 8 = accepted, else another round)
> 👨‍🎓 **Student** revises and writes the final polished version
""")

            search_btn = st.button("🔍 Search Databases", type="primary",
                                   use_container_width=True, key="mlr_search")

        if search_btn:
            if not research_question.strip():
                st.error("Please enter a research question.")
            else:
                client = get_client()
                if not client:
                    st.error("Please enter your Anthropic API key.")
                    st.stop()

                prog_ph = st.empty()
                def _lr_prog(p: LitReviewProgress):
                    prog_ph.markdown(
                        f'<div class="phase-badge">📍 {p.phase}: {p.message}</div>',
                        unsafe_allow_html=True)

                try:
                    with st.spinner("Planning search strategy and querying 4 databases…"):
                        plan, all_papers, papers = plan_and_search(
                            client, research_question, papers_per_db, _lr_prog)
                except anthropic.AuthenticationError:
                    prog_ph.empty()
                    st.error("❌ **Invalid API key.** Go to [console.anthropic.com](https://console.anthropic.com) to get a valid key.")
                    st.stop()
                except Exception as e:
                    prog_ph.empty()
                    st.error(f"❌ Search error: {e}")
                    st.stop()

                prog_ph.empty()
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
                    "view_mode": "verbose" if "Verbose" in view_mode else "final",
                    "papers_per_db": papers_per_db,
                }
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2 — Paper Selection
    # ════════════════════════════════════════════════════════════════════════
    if mlr["step"] in ("searched", "done"):
        with st.expander(
            f"📄 Step 2 — Select Papers  "
            f"({mlr['n_deduped']} unique · {mlr['n_raw']} retrieved)",
            expanded=(mlr["step"] == "searched"),
        ):
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
                st.markdown("Toggle papers to **exclude**. All included by default.")
                edited = st.data_editor(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Include": st.column_config.CheckboxColumn("✅", default=True, width="small"),
                        "#": st.column_config.NumberColumn("#", width="small"),
                        "Title": st.column_config.TextColumn("Title", width="large"),
                        "Year": st.column_config.TextColumn("Year", width="small"),
                        "Source": st.column_config.TextColumn("Source", width="small"),
                        "Citations": st.column_config.NumberColumn("Citations", width="small"),
                    },
                    disabled=["#", "Title", "Year", "Source", "Citations"],
                    key="mlr_paper_editor",
                )

                selected_indices = edited[edited["Include"] == True]["#"].tolist()
                n_sel = len(selected_indices)
                st.caption(f"{n_sel} papers selected for the agent panel.")

                btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 1])
                with btn_col1:
                    run_btn = st.button(
                        f"🚀 Run 4-Agent Review ({n_sel} papers)",
                        type="primary", use_container_width=True,
                        disabled=(n_sel == 0), key="mlr_run",
                    )
                with btn_col2:
                    auto_btn = st.button(
                        "🤖 Auto-select Best Papers, then Run",
                        use_container_width=True, key="mlr_auto",
                    )
                with btn_col3:
                    if st.button("🗑️ Reset", use_container_width=True, key="mlr_reset"):
                        st.session_state["mlr"] = dict(_LR_DEFAULT)
                        st.rerun()

                # ── Run the multi-agent pipeline ──────────────────────────────
                def _run_agents(client, selected_papers):
                    """Launch the orchestrator with real-time event rendering."""

                    verbose = (mlr["view_mode"] == "verbose")
                    total_steps = 1 + mlr["max_rounds"] * 4
                    step_counter = [0]

                    status_ph = st.empty()
                    prog_bar  = st.progress(0)
                    if verbose:
                        st.markdown("---")
                        st.markdown("### 🎭 Agent Collaboration Transcript")
                        events_container = st.container()

                    def on_event(ev: AgentEvent):
                        emoji = _AGENT_EMOJI.get(ev.agent, "🤖")
                        label = _AGENT_LABEL.get(ev.agent, ev.agent)

                        # Advance progress on status events (one per major action)
                        if ev.phase == PHASE_STATUS:
                            step_counter[0] = min(step_counter[0] + 1, total_steps - 1)
                            prog_bar.progress(step_counter[0] / total_steps)
                            status_ph.markdown(
                                f'<div class="phase-badge">'
                                f'{emoji} **{label}** (Round {ev.round}): {ev.content}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        if not verbose:
                            return  # Final-only mode: no transcript rendering

                        # Verbose: render full content in expanders
                        if ev.phase == PHASE_DRAFT:
                            with events_container:
                                with st.expander(
                                    f"{emoji} PhD Student — Initial Draft (Round 0)",
                                    expanded=False,
                                ):
                                    st.markdown(ev.content)

                        elif ev.phase == PHASE_REVISION:
                            with events_container:
                                with st.expander(
                                    f"{emoji} PhD Student — Revised Draft (Round {ev.round})",
                                    expanded=False,
                                ):
                                    st.markdown(ev.content)

                        elif ev.phase == PHASE_FINAL:
                            score_txt = _score_badge(ev.score) if ev.score else ""
                            with events_container:
                                with st.expander(
                                    f"{emoji} PhD Student — Final Polished Version  |  {score_txt}",
                                    expanded=False,
                                ):
                                    st.markdown(ev.content)

                        elif ev.phase == PHASE_FEEDBACK and ev.agent == "supervisor":
                            with events_container:
                                with st.expander(
                                    f"👨‍🏫 Supervisor Feedback — Round {ev.round}",
                                    expanded=False,
                                ):
                                    st.markdown(ev.content)

                        elif ev.phase == PHASE_FEEDBACK and ev.agent == "peer":
                            with events_container:
                                with st.expander(
                                    f"🧑‍💻 Peer Feedback — Round {ev.round}",
                                    expanded=False,
                                ):
                                    st.markdown(ev.content)

                        elif ev.phase == PHASE_FEEDBACK and ev.agent == "reviewer":
                            badge = _score_badge(ev.score) if ev.score else ""
                            with events_container:
                                with st.expander(
                                    f"🔬 Journal Reviewer — Round {ev.round}  |  {badge}",
                                    expanded=False,
                                ):
                                    st.markdown(ev.content)

                        elif ev.phase == PHASE_TOOL:
                            with events_container:
                                st.caption(f"🧑‍💻 Peer searching: *{ev.content}*")

                    # ── Call orchestrator ─────────────────────────────────────
                    try:
                        final_output, all_events, final_papers = run_multi_agent_review(
                            client=client,
                            research_question=mlr["question"],
                            paper_type=mlr["paper_type"],
                            papers=selected_papers,
                            n_raw=mlr["n_raw"],
                            n_deduped=mlr["n_deduped"],
                            max_rounds=mlr["max_rounds"],
                            on_event=on_event,
                        )
                    except anthropic.AuthenticationError:
                        status_ph.empty()
                        prog_bar.empty()
                        st.error("❌ **Invalid API key.** Go to [console.anthropic.com](https://console.anthropic.com) to get a valid key.")
                        st.stop()
                    except Exception as e:
                        status_ph.empty()
                        prog_bar.empty()
                        st.error(f"❌ Multi-agent error: {e}")
                        st.stop()

                    prog_bar.progress(1.0)
                    status_ph.markdown(
                        '<div class="phase-badge">✅ Multi-agent review complete!</div>',
                        unsafe_allow_html=True,
                    )

                    # Verify citations + build disclaimer
                    with st.spinner("Verifying citations via CrossRef…"):
                        verified = verify_citations(final_papers)
                    n_included = len(final_papers)
                    disclaimer = build_disclaimer(mlr["n_raw"], mlr["n_deduped"], n_included)
                    final_output += disclaimer

                    # Find final reviewer score
                    final_score = next(
                        (e.score for e in reversed(all_events)
                         if e.phase == PHASE_FEEDBACK and e.agent == "reviewer" and e.score),
                        None,
                    )

                    # Serialise events for replay in done state
                    serialised = [
                        {"agent": e.agent, "round": e.round, "phase": e.phase,
                         "content": e.content, "score": e.score}
                        for e in all_events
                    ]

                    mlr.update({
                        "step": "done",
                        "selected_papers": selected_papers,
                        "final_output": final_output,
                        "final_papers": final_papers,
                        "events": serialised,
                        "verified": verified,
                        "final_score": final_score,
                    })
                    st.rerun()

                if run_btn and selected_indices:
                    client = get_client()
                    if not client:
                        st.error("Please enter your Anthropic API key.")
                        st.stop()
                    chosen = [mlr["papers"][i] for i in selected_indices if i < len(mlr["papers"])]
                    _run_agents(client, chosen)

                if auto_btn:
                    client = get_client()
                    if not client:
                        st.error("Please enter your Anthropic API key.")
                        st.stop()
                    try:
                        with st.spinner("AI screening papers…"):
                            chosen = ai_screen_papers(client, mlr["question"], mlr["papers"])
                    except anthropic.AuthenticationError:
                        st.error("❌ **Invalid API key.** Go to [console.anthropic.com](https://console.anthropic.com) to get a valid key.")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Screening error: {e}")
                        st.stop()
                    _run_agents(client, chosen)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3 — Final output
    # ════════════════════════════════════════════════════════════════════════
    if mlr["step"] == "done" and mlr["final_output"]:
        st.divider()

        score_txt = _score_badge(mlr["final_score"]) if mlr["final_score"] else ""
        st.success(
            f"✅ Review ready — {len(mlr['final_papers'])} papers synthesised by 4-agent panel  "
            f"{'· Final score: **' + score_txt + '**' if score_txt else ''}"
        )

        # Download / reset row
        dl_col, clr_col = st.columns([4, 1])
        with dl_col:
            safe = mlr["question"][:50].replace(" ", "_").replace("/", "-")
            st.download_button(
                "⬇️ Download Review (.md)",
                data=mlr["final_output"],
                file_name=f"litreview_{safe}.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True,
                key="mlr_dl",
            )
        with clr_col:
            if st.button("🗑️ New Review", use_container_width=True, key="mlr_new"):
                st.session_state["mlr"] = dict(_LR_DEFAULT)
                st.rerun()

        # ── Agent transcript (always available, collapsed by default) ────────
        if mlr["events"]:
            event_count = sum(
                1 for e in mlr["events"]
                if e["phase"] in (PHASE_DRAFT, PHASE_REVISION, PHASE_FINAL, PHASE_FEEDBACK)
            )
            with st.expander(
                f"🎭 Agent Collaboration Transcript ({event_count} exchanges)",
                expanded=(mlr["view_mode"] == "verbose"),
            ):
                for e in mlr["events"]:
                    if e["phase"] == PHASE_STATUS:
                        continue  # skip status noise
                    emoji = _AGENT_EMOJI.get(e["agent"], "🤖")
                    label = _AGENT_LABEL.get(e["agent"], e["agent"])
                    rnd   = e["round"]

                    if e["phase"] == PHASE_DRAFT:
                        st.markdown(f"**{emoji} {label} — Initial Draft**")
                        with st.expander("Show initial draft", expanded=False):
                            st.markdown(e["content"])
                    elif e["phase"] == PHASE_REVISION:
                        st.markdown(f"**{emoji} {label} — Revised Draft (Round {rnd})**")
                        with st.expander(f"Show revision {rnd}", expanded=False):
                            st.markdown(e["content"])
                    elif e["phase"] == PHASE_FINAL:
                        badge = _score_badge(e["score"]) if e["score"] else ""
                        st.markdown(f"**{emoji} {label} — Final Version  |  {badge}**")
                        with st.expander("Show final draft (pre-formatted)", expanded=False):
                            st.markdown(e["content"])
                    elif e["phase"] == PHASE_FEEDBACK:
                        if e["agent"] == "reviewer":
                            badge = _score_badge(e["score"]) if e["score"] else ""
                            st.markdown(f"**🔬 Journal Reviewer — Round {rnd}  |  {badge}**")
                        else:
                            st.markdown(f"**{emoji} {label} — Round {rnd} Feedback**")
                        with st.expander(f"Show {label} feedback (Round {rnd})", expanded=False):
                            st.markdown(e["content"])
                    elif e["phase"] == PHASE_TOOL:
                        st.caption(f"🧑‍💻 Peer searched: *{e['content']}*")
                    st.divider()

        # ── Citation verification ────────────────────────────────────────────
        if mlr["verified"]:
            with st.expander("🔍 Citation Verification Report", expanded=True):
                status_icon = {
                    "verified": "✅ Verified",
                    "preprint": "⚠️ Preprint",
                    "not_found": "❌ Not found",
                    "no_doi": "❓ No DOI",
                }
                rows = []
                for v in mlr["verified"]:
                    rows.append({
                        "Ref": f"[{v['index']}]",
                        "Status": status_icon.get(v["status"], v["status"]),
                        "Title": v["title"],
                        "Journal": v.get("journal", ""),
                        "DOI/URL": v.get("doi") or v.get("url", ""),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                n_ok  = sum(1 for v in mlr["verified"] if v["status"] == "verified")
                n_pre = sum(1 for v in mlr["verified"] if v["status"] == "preprint")
                n_bad = sum(1 for v in mlr["verified"]
                            if v["status"] in ("not_found", "no_doi"))
                st.caption(
                    f"✅ {n_ok} verified via CrossRef · "
                    f"⚠️ {n_pre} preprints · "
                    f"❓ {n_bad} unverified (check manually before citing)"
                )

        # ── Final review output ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📄 Final Literature Review")
        st.markdown(mlr["final_output"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESEARCH ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
with tab_research:
    col_main, col_help = st.columns([3, 1])

    with col_help:
        with st.expander("💡 What can I ask?", expanded=True):
            st.markdown("""
**Research Q&A**
> *"What is the role of astrocytes in synaptic pruning?"*
> *"Explain the complement cascade in neurodegeneration"*

**Gap Finder**
> *"What are open questions in hippocampal place cells?"*
> *"What gaps exist in microglia's role in depression?"*

**Paper Deep-Dive**
> *"Critique the default mode network in Alzheimer's"*
> *"Analyse: Attractor dynamics in neural circuits"*
            """)

    with col_main:
        if "research_messages" not in st.session_state:
            st.session_state.research_messages = []

        b1, b2, _ = st.columns([1, 1, 2])
        with b1:
            if st.button("🗑️ Clear", key="clear_research", use_container_width=True):
                st.session_state.research_messages = []
                st.rerun()
        with b2:
            if st.session_state.research_messages:
                def _export_chat():
                    lines = ["# NeuroResearch Agent — Research Session\n"]
                    for m in st.session_state.research_messages:
                        role = "**You**" if m["role"] == "user" else "**Agent**"
                        lines.append(f"### {role}\n\n{m['content']}\n\n---\n")
                    return "\n".join(lines)
                st.download_button(
                    "⬇️ Export", data=_export_chat(),
                    file_name="research_session.md", mime="text/markdown",
                    use_container_width=True, key="export_research",
                )

        for msg in st.session_state.research_messages:
            with st.chat_message(msg["role"],
                                 avatar="🧠" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a neuroscience question, find gaps, or analyse a paper…"):
            client = get_client()
            if not client:
                st.error("Please enter your Anthropic API key in the sidebar.")
                st.stop()

            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            st.session_state.research_messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant", avatar="🧠"):
                tool_ph = st.empty()
                resp_ph = st.empty()
                log = []
                try:
                    with st.spinner("Searching databases…"):
                        api_msgs = [{"role": m["role"], "content": m["content"]}
                                    for m in st.session_state.research_messages]
                        response = run_research_turn(
                            client, api_msgs,
                            on_tool_call=_tool_log_callback(log, tool_ph),
                        )
                    tool_ph.empty()
                    if log:
                        with st.expander(f"📡 Searched {len(log)} database(s)", expanded=False):
                            for l in log:
                                st.markdown(l)
                    resp_ph.markdown(response)
                    st.session_state.research_messages.append(
                        {"role": "assistant", "content": response})
                except anthropic.AuthenticationError:
                    tool_ph.empty()
                    st.session_state.research_messages.pop()
                    st.error("❌ **Invalid API key.** Go to [console.anthropic.com](https://console.anthropic.com) to get a valid key.")
                except Exception as e:
                    tool_ph.empty()
                    st.session_state.research_messages.pop()
                    st.error(f"❌ Unexpected error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONTRADICTION DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_contradiction:
    st.markdown("### ⚡ Contradiction Detector")
    st.markdown(
        "Enter a specific neuroscience claim or finding. "
        "The agent searches for papers that **support it AND contradict it**, "
        "then explains *why* the evidence conflicts — methods, populations, replication status."
    )

    col_in, col_ex = st.columns([2, 1])
    with col_ex:
        with st.expander("📝 Example claims", expanded=True):
            st.markdown("""
- *"SSRIs increase adult hippocampal neurogenesis"*
- *"The gut microbiome influences anxiety behaviour in mice"*
- *"Amyloid plaques cause Alzheimer's disease"*
- *"Sleep deprivation impairs long-term potentiation"*
- *"Microglia are the primary mediators of synaptic pruning"*
            """)

    with col_in:
        claim = st.text_area(
            "Claim to investigate",
            placeholder="e.g. SSRIs increase adult hippocampal neurogenesis",
            height=80, key="cd_claim",
        )

        if "cd_result" not in st.session_state:
            st.session_state["cd_result"] = ""
            st.session_state["cd_claim_used"] = ""

        cd_btn = st.button("🔍 Analyse Contradictions", type="primary",
                           use_container_width=True, key="cd_run")

        if cd_btn:
            if not claim.strip():
                st.error("Please enter a claim to investigate.")
            else:
                client = get_client()
                if not client:
                    st.error("Please enter your Anthropic API key.")
                    st.stop()
                tool_ph = st.empty()
                log = []
                try:
                    with st.spinner("Searching for supporting and contradicting evidence…"):
                        result = detect_contradictions(
                            client, claim,
                            on_tool_call=_tool_log_callback(log, tool_ph),
                        )
                except anthropic.AuthenticationError:
                    tool_ph.empty()
                    st.error("❌ **Invalid API key.** Go to [console.anthropic.com](https://console.anthropic.com) to get a valid key.")
                    st.stop()
                except Exception as e:
                    tool_ph.empty()
                    st.error(f"❌ Unexpected error: {e}")
                    st.stop()
                tool_ph.empty()
                st.session_state["cd_result"] = result
                st.session_state["cd_claim_used"] = claim
                st.rerun()

    if st.session_state.get("cd_result"):
        st.divider()
        st.success(f"Contradiction report for: *{st.session_state['cd_claim_used'][:80]}*")
        dl2, clr2 = st.columns([4, 1])
        with dl2:
            safe2 = st.session_state["cd_claim_used"][:40].replace(" ", "_")
            st.download_button(
                "⬇️ Download Report (.md)",
                data=st.session_state["cd_result"],
                file_name=f"contradiction_{safe2}.md",
                mime="text/markdown",
                use_container_width=True,
                key="cd_dl",
            )
        with clr2:
            if st.button("🗑️ Clear", use_container_width=True, key="cd_clr"):
                st.session_state["cd_result"] = ""
                st.rerun()
        st.markdown(st.session_state["cd_result"])
