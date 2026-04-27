# 🧠 NeuroResearch Agent

An AI-powered research assistant for neuroscience researchers, built with Claude claude-sonnet-4-6 and Streamlit.

## Features

**Research Assistant tab**
- Ask niche neuroscience questions — get cited answers backed by live database searches
- Find research gaps: identify open questions and underexplored directions in any topic
- Paper deep-dives: structured critique (contribution, methods, limitations, field context)

**Literature Review tab**
- Systematic, PRISMA 2020-aligned literature reviews
- Searches PubMed, Semantic Scholar, arXiv, and bioRxiv in parallel
- Thematic synthesis with gap analysis
- Downloadable Markdown output

---

## Local Setup

### 1. Clone & install
```bash
git clone <your-repo-url>
cd neuro-research-agent
pip install -r requirements.txt
```

### 2. Add your API key
Edit `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

### 3. Run
```bash
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud (free shareable link)

1. Push this repo to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** → select your repo → set main file to `app.py`
4. In **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
5. Click **Deploy** — you'll get a `*.streamlit.app` URL to share

> **Note:** `.streamlit/secrets.toml` is gitignored — your key is never committed to GitHub.

---

## Project Structure

```
neuro-research-agent/
├── app.py                     # Streamlit UI — two-tab layout
├── agent/
│   ├── research_agent.py      # Research assistant (Q&A, gap finder, paper analysis)
│   ├── literature_agent.py    # Systematic literature review pipeline
│   └── tools/
│       ├── pubmed.py          # PubMed E-utilities
│       ├── semantic_scholar.py # Semantic Scholar Graph API
│       ├── arxiv_search.py    # arXiv API
│       ├── biorxiv.py         # bioRxiv/medRxiv
│       └── crossref.py        # DOI verification
├── prompts/
│   ├── research_system.py     # System prompt for research mode
│   └── litreview_system.py    # System prompt for literature reviews
├── requirements.txt
└── .streamlit/
    └── secrets.toml           # API key (gitignored)
```

---

## Example Queries

**Research Assistant**
- *"What is the role of astrocytes in synaptic pruning, and which complement pathway components are involved?"*
- *"What are the open research questions in hippocampal sharp-wave ripples?"*
- *"Analyze the paper: Replay of cortical spiking sequences during human memory retrieval"*

**Literature Review**
- *"Neuroinflammation and major depressive disorder"*
- *"Synaptic plasticity mechanisms in hippocampal memory consolidation"*
- *"Role of microglia in Alzheimer's disease pathogenesis"*

---

## Requirements

- Python 3.11+
- Anthropic API key (Claude claude-sonnet-4-6 access)
- All academic database APIs are **free and require no authentication**
