"""
Automatic evaluation of final literature review output.
Pure heuristic scoring — no LLM calls required.
Used for ablation studies and the NeurIPS paper evaluation section.
"""
import re
from dataclasses import dataclass


@dataclass
class EvalScore:
    citation_coverage: float   # 0.0–1.0: fraction of pool papers cited inline
    synthesis_depth: int       # 1–10: thematic synthesis vs. one-by-one summaries
    gap_specificity: int       # 1–10: mechanism-level gaps vs. vague "more research needed"
    adversary_score: int       # 1–10: final adversary verdict (0 if no adversary)

    @property
    def overall(self) -> float:
        """Weighted composite 0–100: coverage 25%, synthesis 35%, gaps 25%, adversary 15%."""
        return (
            self.citation_coverage * 25
            + (self.synthesis_depth / 10) * 35
            + (self.gap_specificity / 10) * 25
            + (self.adversary_score / 10) * 15
        )

    def as_dict(self) -> dict:
        return {
            "citation_coverage": round(self.citation_coverage, 3),
            "synthesis_depth": self.synthesis_depth,
            "gap_specificity": self.gap_specificity,
            "adversary_score": self.adversary_score,
            "overall": round(self.overall, 1),
        }


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _score_citation_coverage(text: str, papers: list) -> float:
    """Fraction of papers in the pool that appear as inline citations."""
    if not papers:
        return 0.0
    cited = 0
    for p in papers:
        authors = p.get("authors", "") or ""
        year = str(p.get("year", "") or "")
        if not year:
            continue
        # Extract first author surname (last token of first comma-separated entry)
        first_entry = authors.split(",")[0].strip()
        surname = first_entry.split()[-1] if first_entry else ""
        if not surname:
            continue
        pattern = re.compile(
            rf"\({re.escape(surname)}"
            rf"(?:\s+et\s+al\.?)?"
            rf"[,\s]*{re.escape(year)}\)",
            re.IGNORECASE,
        )
        if pattern.search(text):
            cited += 1
    return cited / len(papers)


_THEMATIC_MARKERS = [
    "collectively", "together these", "taken together", "converging evidence",
    "across studies", "across the literature", "the evidence suggests",
    "consistent with", "in contrast to", "however", "although", "whereas",
    "mechanisms underlying", "building on", "extending this work",
    "these findings suggest", "synthesizing", "emerging consensus",
    "paradigm", "framework", "theoretical", "integration of",
]

_ONE_BY_ONE_PATTERNS = [
    r"\b\w+ et al\.\s*\(\d{4}\)\s+(?:found|showed|reported|demonstrated|observed|noted|argued|suggested|concluded)",
]


def _score_synthesis_depth(text: str) -> int:
    """1–10: how thematic the synthesis is vs. a walk through papers one-by-one."""
    text_lower = text.lower()
    thematic = sum(1 for m in _THEMATIC_MARKERS if m in text_lower)
    one_by_one = sum(
        len(re.findall(p, text, re.IGNORECASE)) for p in _ONE_BY_ONE_PATTERNS
    )
    total = thematic + one_by_one
    if total == 0:
        return 5
    ratio = thematic / total
    return max(1, min(10, round(ratio * 9) + 1))


_SPECIFIC_MARKERS = [
    "mechanism", "pathway", "circuit", "receptor", "molecular", "cellular",
    "dose-response", "longitudinal", "randomized controlled", "causal",
    "biomarker", "neural correlate", "precisely", "quantif", "synap",
    "neurotransmitter", "cortical", "subcortical", "hippocampal",
    "electrophysiolog", "fmri", "eeg", "optogenetic",
]

_VAGUE_MARKERS = [
    "more research is needed", "further studies are needed",
    "remains poorly understood", "remains unclear", "better understanding",
    "more data are needed", "not well understood", "limited evidence",
    "future work should", "additional investigation",
]


def _score_gap_specificity(text: str) -> int:
    """1–10: how specific the identified research gaps are."""
    text_lower = text.lower()
    specific = sum(1 for m in _SPECIFIC_MARKERS if m in text_lower)
    vague = sum(1 for m in _VAGUE_MARKERS if m in text_lower)
    if specific == 0 and vague == 0:
        return 5
    if vague == 0:
        return min(10, 6 + specific // 3)
    ratio = specific / (specific + vague * 2)
    return max(1, min(10, round(ratio * 9) + 1))


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate_output(
    final_text: str,
    papers: list,
    adversary_score: int = 0,
) -> EvalScore:
    """
    Score a final literature review output automatically.

    Args:
        final_text:      the complete text of the review
        papers:          the paper pool used for writing
        adversary_score: final adversary verdict score (0 if run in no-adversary mode)

    Returns:
        EvalScore with citation_coverage, synthesis_depth, gap_specificity, adversary_score
    """
    return EvalScore(
        citation_coverage=_score_citation_coverage(final_text, papers),
        synthesis_depth=_score_synthesis_depth(final_text),
        gap_specificity=_score_gap_specificity(final_text),
        adversary_score=adversary_score,
    )
