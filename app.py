import streamlit as st
import anthropic
import pandas as pd
from typing import Optional

from agent.research_agent import run_research_turn
from agent.literature_agent import (
    plan_and_search, ai_screen_papers, synthesize_review,
    verify_citations, build_disclaimer, LitReviewProgress,
)
from agent.hypothesis_agent import generate_hypotheses
from agent.contradiction_agent import detect_contradictions
from agent.position_agent import position_paper

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
    .main-header { font-size:2.1rem; font-weight:700; color:#1a1a2e; margin-bottom:0.1rem; }
    .sub-header  { font-size:0.95rem; color:#555; margin-bottom:1.2rem; }
    .phase-badge { background:#f0f7ff; padding:8px 16px; border-radius:20px;
                   font-size:0.9rem; color:#1a73e8; font-weight:500; display:inline-block; }
    .verify-ok   { color:#1b7c3d; font-weight:600; }
    .verify-pre  { color:#b45309; font-weight:600; }
    .verify-bad  { color:#b91c1c; font-weight:600; }
    .step-label  { font-size:0.8rem; font-weight:700; color:#6b7280;
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

    st.markdown("**Four Modes**")
    st.markdown("🔬 **Research Assistant** — Q&A, gap finding, paper analysis")
    st.markdown("📚 **Literature Review** — Systematic PRISMA review with human screening")
    st.markdown("⚡ **Contradiction Detector** — Find where the evidence conflicts")
    st.markdown("📍 **Position My Paper** — Where your work fits & where to submit")
    st.divider()
    st.markdown("**Databases**")
    st.markdown("PubMed · Semantic Scholar · arXiv · bioRxiv")
    st.divider()
    st.markdown("*Powered by Claude claude-sonnet-4-6*")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 NeuroResearch Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Research Q&A · Systematic Literature Reviews · Contradiction Detection · Paper Positioning</div>', unsafe_allow_html=True)

tab_research, tab_litreview, tab_contradiction, tab_position = st.tabs([
    "🔬 Research Assistant",
    "📚 Literature Review",
    "⚡ Contradiction Detector",
    "📍 Position My Paper",
])


# ══════════════════════════════════════════════════════════════════════════════
# SHARED TOOL-CALL RENDERER
# ══════════════════════════════════════════════════════════════════════════════
_DB_LABELS = {
    "search_pubmed": "PubMed",
    "search_semantic_scholar": "Semantic Scholar",
    "search_arxiv": "arXiv",
    "search_biorxiv": "bioRxiv",
    "verify_doi": "CrossRef DOI",
}

def _tool_log_callback(log_list: list, placeholder):
    def _cb(name, inputs):
        label = _DB_LABELS.get(name, name)
        q = inputs.get("query", inputs.get("doi", ""))
        log_list.append(f"🔍 Searching **{label}**: *{q}*")
        placeholder.markdown("\n".join(log_list))
    return _cb


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESEARCH ASSISTANT
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
> *"Analyze: Attractor dynamics in neural circuits"*
> *"Critique the default mode network in Alzheimer's"*
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
                st.download_button("⬇️ Export", data=_export_chat(),
                                   file_name="research_session.md", mime="text/markdown",
                                   use_container_width=True, key="export_research")

        for msg in st.session_state.research_messages:
            with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a neuroscience question, find research gaps, or analyze a paper…"):
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
                        response = run_research_turn(client, api_msgs,
                                                     on_tool_call=_tool_log_callback(log, tool_ph))
                    tool_ph.empty()
                    if log:
                        with st.expander(f"📡 Searched {len(log)} database(s)", expanded=False):
                            for l in log:
                                st.markdown(l)
                    resp_ph.markdown(response)
                    st.session_state.research_messages.append({"role": "assistant", "content": response})
                except anthropic.AuthenticationError:
                    tool_ph.empty()
                    st.session_state.research_messages.pop()  # remove the user msg we just added
                    st.error("❌ **Invalid API key.** Go to [console.anthropic.com](https://console.anthropic.com) to get a valid key, then update it in the sidebar or in Streamlit Cloud secrets.")
                except Exception as e:
                    tool_ph.empty()
                    st.session_state.research_messages.pop()
                    st.error(f"❌ Unexpected error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LITERATURE REVIEW  (3-step human-in-the-loop flow)
# ══════════════════════════════════════════════════════════════════════════════
with tab_litreview:
    # ── State init ────────────────────────────────────────────────────────────
    if "lr" not in st.session_state:
        st.session_state["lr"] = {
            "step": "idle",       # idle | searched | done
            "plan": {},
            "all_papers": [],
            "papers": [],
            "selected_papers": [],
            "review": "",
            "verified": [],
            "hypotheses": "",
            "n_raw": 0,
            "n_deduped": 0,
            "question": "",
            "citation_style": "APA",
        }
    lr = st.session_state["lr"]

    # ── Step indicator ────────────────────────────────────────────────────────
    s1, s2, s3 = st.columns(3)
    with s1:
        cls = "step-active" if lr["step"] == "idle" else "step-done"
        st.markdown(f'<div class="{cls}"><div class="step-label">Step 1</div>Search Papers</div>',
                    unsafe_allow_html=True)
    with s2:
        cls = "step-active" if lr["step"] == "searched" else ("step-done" if lr["step"] == "done" else "")
        st.markdown(f'<div class="{cls}"><div class="step-label">Step 2</div>Select & Review</div>',
                    unsafe_allow_html=True)
    with s3:
        cls = "step-active" if lr["step"] == "done" else ""
        st.markdown(f'<div class="{cls}"><div class="step-label">Step 3</div>Hypotheses & Export</div>',
                    unsafe_allow_html=True)

    st.divider()

    # ── STEP 1: Input form ────────────────────────────────────────────────────
    with st.expander("📝 Step 1 — Define Your Research Question",
                     expanded=(lr["step"] == "idle")):
        cfg_col, inp_col = st.columns([1, 2])
        with cfg_col:
            papers_per_db = st.slider("Papers per database", 5, 20, 10, key="lr_ppdb")
            citation_style = st.selectbox("Citation style", ["APA", "Vancouver", "Nature", "Chicago"],
                                          key="lr_cstyle")
        with inp_col:
            examples = [
                "Synaptic plasticity mechanisms in hippocampal memory consolidation",
                "Role of the gut-brain axis in anxiety disorders",
                "Neuroinflammation and major depressive disorder",
                "Optogenetic dissection of basal ganglia circuits",
                "Adult neurogenesis in the human hippocampus",
            ]
            ex = st.selectbox("Example topics:", [""] + examples, key="lr_example")
            rq_default = ex if ex else ""
            research_question = st.text_area(
                "Research question / topic",
                value=rq_default,
                placeholder="e.g. What is the role of neuroinflammation in major depressive disorder?",
                height=100, key="lr_rq",
            )
            search_btn = st.button("🔍 Search Papers", type="primary",
                                   use_container_width=True, key="lr_search")

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
                    with st.spinner("Planning and searching databases…"):
                        plan, all_papers, papers = plan_and_search(
                            client, research_question, papers_per_db, _lr_prog)
                except anthropic.AuthenticationError:
                    prog_ph.empty()
                    st.error("❌ **Invalid API key.** Go to [console.anthropic.com](https://console.anthropic.com) to get a valid key, then update it in the sidebar or in Streamlit Cloud secrets.")
                    st.stop()
                except Exception as e:
                    prog_ph.empty()
                    st.error(f"❌ Unexpected error during search: {e}")
                    st.stop()

                prog_ph.empty()
                lr.update({
                    "step": "searched",
                    "plan": plan,
                    "all_papers": all_papers,
                    "papers": papers,
                    "n_raw": len(all_papers),
                    "n_deduped": len(papers),
                    "question": research_question,
                    "citation_style": citation_style,
                    "review": "",
                    "verified": [],
                    "hypotheses": "",
                    "selected_papers": [],
                })
                st.rerun()

    # ── STEP 2: Paper selection ───────────────────────────────────────────────
    if lr["step"] in ("searched", "done"):
        with st.expander(
            f"📄 Step 2 — Select Papers  ({lr['n_deduped']} found, {lr['n_raw']} raw)",
            expanded=(lr["step"] == "searched"),
        ):
            if lr["papers"]:
                # Build an editable dataframe
                df_data = []
                for i, p in enumerate(lr["papers"]):
                    if not p.get("title"):
                        continue
                    df_data.append({
                        "Include": True,
                        "#": i,
                        "Title": p["title"][:90],
                        "Year": p.get("year", ""),
                        "Source": p.get("source", ""),
                        "Citations": p.get("citations", ""),
                    })
                df = pd.DataFrame(df_data)

                st.markdown("Toggle papers you want to **exclude**. All are included by default.")
                edited = st.data_editor(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Include": st.column_config.CheckboxColumn("✅ Include", default=True, width="small"),
                        "#": st.column_config.NumberColumn("#", width="small"),
                        "Title": st.column_config.TextColumn("Title", width="large"),
                        "Year": st.column_config.TextColumn("Year", width="small"),
                        "Source": st.column_config.TextColumn("Source", width="small"),
                        "Citations": st.column_config.NumberColumn("Citations", width="small"),
                    },
                    disabled=["#", "Title", "Year", "Source", "Citations"],
                    key="lr_paper_editor",
                )

                selected_indices = edited[edited["Include"] == True]["#"].tolist()
                n_selected = len(selected_indices)
                st.caption(f"{n_selected} papers selected for synthesis.")

                gen_col, auto_col, reset_col = st.columns([2, 2, 1])
                with gen_col:
                    gen_btn = st.button(
                        f"📝 Generate Review ({n_selected} papers)",
                        type="primary", use_container_width=True,
                        disabled=(n_selected == 0), key="lr_gen",
                    )
                with auto_col:
                    auto_btn = st.button(
                        "🤖 Auto-select Best Papers & Generate",
                        use_container_width=True, key="lr_auto",
                    )
                with reset_col:
                    if st.button("🗑️ Reset", use_container_width=True, key="lr_reset"):
                        lr.update({"step": "idle", "papers": [], "review": "",
                                   "hypotheses": "", "verified": []})
                        st.rerun()

                def _run_synthesis(client, selected_papers):
                    prog_ph2 = st.empty()
                    bar = st.progress(0)
                    phase_pct = {"Synthesizing": 0.3, "Complete": 1.0}

                    def _sp(p: LitReviewProgress):
                        bar.progress(phase_pct.get(p.phase, 0.5))
                        prog_ph2.markdown(
                            f'<div class="phase-badge">📍 {p.phase}: {p.message}</div>',
                            unsafe_allow_html=True)

                    try:
                        with st.spinner("Writing literature review…"):
                            review = synthesize_review(
                                client, lr["question"], selected_papers,
                                lr["plan"], lr["n_raw"], lr["n_deduped"],
                                lr["citation_style"], _sp,
                            )
                    except anthropic.AuthenticationError:
                        prog_ph2.empty()
                        bar.empty()
                        st.error("❌ **Invalid API key.** Go to [console.anthropic.com](https://console.anthropic.com) to get a valid key, then update it in the sidebar or in Streamlit Cloud secrets.")
                        st.stop()
                    except Exception as e:
                        prog_ph2.empty()
                        bar.empty()
                        st.error(f"❌ Unexpected error during synthesis: {e}")
                        st.stop()
                    prog_ph2.empty()
                    bar.progress(1.0)

                    with st.spinner("Verifying citations…"):
                        verified = verify_citations(selected_papers)

                    disclaimer = build_disclaimer(lr["n_raw"], lr["n_deduped"], len(selected_papers))
                    review += disclaimer

                    lr.update({
                        "step": "done",
                        "selected_papers": selected_papers,
                        "review": review,
                        "verified": verified,
                    })
                    st.rerun()

                if gen_btn and selected_indices:
                    client = get_client()
                    if not client:
                        st.error("Please enter your Anthropic API key.")
                        st.stop()
                    selected_papers = [lr["papers"][i] for i in selected_indices
                                       if i < len(lr["papers"])]
                    _run_synthesis(client, selected_papers)

                if auto_btn:
                    client = get_client()
                    if not client:
                        st.error("Please enter your Anthropic API key.")
                        st.stop()
                    try:
                        with st.spinner("AI screening papers…"):
                            selected_papers = ai_screen_papers(client, lr["question"], lr["papers"])
                    except anthropic.AuthenticationError:
                        st.error("❌ **Invalid API key.** Go to [console.anthropic.com](https://console.anthropic.com) to get a valid key, then update it in the sidebar or in Streamlit Cloud secrets.")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Unexpected error during screening: {e}")
                        st.stop()
                    _run_synthesis(client, selected_papers)

    # ── STEP 3: Review output ─────────────────────────────────────────────────
    if lr["step"] == "done" and lr["review"]:
        st.divider()
        st.success(f"✅ Review ready — {len(lr['selected_papers'])} papers synthesised")

        # Download + clear buttons
        dl_col, clr_col = st.columns([4, 1])
        with dl_col:
            safe = lr["question"][:50].replace(" ", "_").replace("/", "-")
            st.download_button(
                "⬇️ Download Review (.md)", data=lr["review"],
                file_name=f"litreview_{safe}.md", mime="text/markdown",
                type="primary", use_container_width=True, key="lr_dl",
            )
        with clr_col:
            if st.button("🗑️ New Review", use_container_width=True, key="lr_new"):
                st.session_state["lr"] = {
                    "step": "idle", "plan": {}, "all_papers": [], "papers": [],
                    "selected_papers": [], "review": "", "verified": [],
                    "hypotheses": "", "n_raw": 0, "n_deduped": 0,
                    "question": "", "citation_style": "APA",
                }
                st.rerun()

        # Citation verification table
        if lr["verified"]:
            with st.expander("🔍 Citation Verification Report", expanded=True):
                status_icon = {"verified": "✅ Verified", "preprint": "⚠️ Preprint",
                               "not_found": "❌ Not found", "no_doi": "❓ No DOI"}
                rows = []
                for v in lr["verified"]:
                    rows.append({
                        "Ref": f"[{v['index']}]",
                        "Status": status_icon.get(v["status"], v["status"]),
                        "Title": v["title"],
                        "Journal": v.get("journal", ""),
                        "DOI/URL": v.get("doi") or v.get("url", ""),
                    })
                vdf = pd.DataFrame(rows)
                st.dataframe(vdf, use_container_width=True, hide_index=True)

                n_ok = sum(1 for v in lr["verified"] if v["status"] == "verified")
                n_pre = sum(1 for v in lr["verified"] if v["status"] == "preprint")
                n_bad = sum(1 for v in lr["verified"]
                            if v["status"] in ("not_found", "no_doi"))
                st.caption(
                    f"✅ {n_ok} verified via CrossRef · "
                    f"⚠️ {n_pre} preprints (not peer-reviewed) · "
                    f"❓ {n_bad} unverified (verify manually before citing)"
                )

        # The review itself
        st.markdown("---")
        st.markdown(lr["review"])

        # Hypothesis generator
        st.divider()
        st.markdown("### 💡 Step 3 — Generate Research Hypotheses")
        st.markdown(
            "Turn the review's identified gaps into **concrete, fundable research hypotheses** "
            "with experimental designs, feasibility assessments, and funding fit."
        )

        if lr["hypotheses"]:
            st.success("Hypotheses already generated — shown below.")
            hyp_dl_col, hyp_regen_col = st.columns([3, 1])
            with hyp_dl_col:
                st.download_button(
                    "⬇️ Download Hypotheses (.md)", data=lr["hypotheses"],
                    file_name=f"hypotheses_{safe}.md", mime="text/markdown",
                    use_container_width=True, key="hyp_dl",
                )
            with hyp_regen_col:
                regen = st.button("🔄 Regenerate", use_container_width=True, key="hyp_regen")
            if not regen:
                st.markdown(lr["hypotheses"])
            else:
                lr["hypotheses"] = ""
                st.rerun()
        else:
            if st.button("🚀 Generate Hypotheses from This Review",
                         type="primary", use_container_width=True, key="hyp_gen"):
                client = get_client()
                if not client:
                    st.error("Please enter your Anthropic API key.")
                    st.stop()
                try:
                    with st.spinner("Generating hypotheses and experimental designs…"):
                        hyp = generate_hypotheses(client, lr["review"], lr["question"])
                except anthropic.AuthenticationError:
                    st.error("❌ **Invalid API key.** Go to [console.anthropic.com](https://console.anthropic.com) to get a valid key, then update it in the sidebar or in Streamlit Cloud secrets.")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Unexpected error generating hypotheses: {e}")
                    st.stop()
                lr["hypotheses"] = hyp
                st.rerun()


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
                    st.error("❌ **Invalid API key.** Go to [console.anthropic.com](https://console.anthropic.com) to get a valid key, then update it in the sidebar or in Streamlit Cloud secrets.")
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
                "⬇️ Download Report (.md)", data=st.session_state["cd_result"],
                file_name=f"contradiction_{safe2}.md", mime="text/markdown",
                use_container_width=True, key="cd_dl",
            )
        with clr2:
            if st.button("🗑️ Clear", use_container_width=True, key="cd_clr"):
                st.session_state["cd_result"] = ""
                st.rerun()
        st.markdown(st.session_state["cd_result"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — POSITION MY PAPER
# ══════════════════════════════════════════════════════════════════════════════
with tab_position:
    st.markdown("### 📍 Position My Paper")
    st.markdown(
        "Paste your paper's **abstract** (or a detailed description). "
        "The agent searches the literature for the most similar existing work, "
        "identifies your genuine novelty, lists must-cite papers, recommends journals, "
        "and flags potential reviewer objections."
    )

    col_abs, col_tip = st.columns([2, 1])
    with col_tip:
        with st.expander("💡 How to use this", expanded=True):
            st.markdown("""
**Paste your abstract** before submission to:
- Check if similar work exists you might have missed
- Understand exactly what's novel vs. what's already published
- Get a list of papers reviewers will expect you to cite
- Identify the best journal fit
- Pre-empt likely reviewer objections

Works best with a full abstract (150-300 words).
            """)

    with col_abs:
        abstract = st.text_area(
            "Your paper's abstract",
            placeholder="Paste your abstract here (150-300 words recommended)…",
            height=200, key="pos_abstract",
        )

        if "pos_result" not in st.session_state:
            st.session_state["pos_result"] = ""

        pos_btn = st.button("🔍 Analyse My Paper's Position",
                            type="primary", use_container_width=True, key="pos_run")

        if pos_btn:
            if not abstract.strip() or len(abstract.split()) < 30:
                st.error("Please paste a more complete abstract (at least 30 words).")
            else:
                client = get_client()
                if not client:
                    st.error("Please enter your Anthropic API key.")
                    st.stop()
                tool_ph3 = st.empty()
                log3 = []
                try:
                    with st.spinner("Searching for related work and building positioning report…"):
                        result = position_paper(
                            client, abstract,
                            on_tool_call=_tool_log_callback(log3, tool_ph3),
                        )
                except anthropic.AuthenticationError:
                    tool_ph3.empty()
                    st.error("❌ **Invalid API key.** Go to [console.anthropic.com](https://console.anthropic.com) to get a valid key, then update it in the sidebar or in Streamlit Cloud secrets.")
                    st.stop()
                except Exception as e:
                    tool_ph3.empty()
                    st.error(f"❌ Unexpected error: {e}")
                    st.stop()
                tool_ph3.empty()
                if log3:
                    with st.expander(f"📡 Searched {len(log3)} database(s)", expanded=False):
                        for l in log3:
                            st.markdown(l)
                st.session_state["pos_result"] = result
                st.rerun()

    if st.session_state.get("pos_result"):
        st.divider()
        st.success("Positioning report ready!")
        dl3, clr3 = st.columns([4, 1])
        with dl3:
            st.download_button(
                "⬇️ Download Report (.md)", data=st.session_state["pos_result"],
                file_name="paper_positioning_report.md", mime="text/markdown",
                use_container_width=True, key="pos_dl",
            )
        with clr3:
            if st.button("🗑️ Clear", use_container_width=True, key="pos_clr"):
                st.session_state["pos_result"] = ""
                st.rerun()
        st.markdown(st.session_state["pos_result"])
