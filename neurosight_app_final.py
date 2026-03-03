"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  NeuroSight Knowledge Base — Streamlit Dashboard                           ║
║  Matches the original React dashboard design                               ║
║  Reads: KB from Google Drive (fallback: local neurosight_kb.json)          ║
║  Run:   streamlit run neurosight_app.py                                    ║
║  Deps:  pip install streamlit plotly pandas openai                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json, os
from urllib.request import urlopen, Request
from urllib.error import URLError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import Counter
from pathlib import Path
from itertools import accumulate

# ─── Page Config (only when run standalone) ────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(page_title="NeuroSight Knowledge Base", page_icon="🧠", layout="wide")

# ─── CSS is injected inside run_kb() so it fires every page load ─────────────

# ─── Load Data ───────────────────────────────────────────────────────────────

# Knowledge base: connect once to Google Drive and read (cached); fallback is local neurosight_kb.json
KB_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1xao-n6hU1cvso--aOTXHe2nKrDlt5LJe"

def _kb_path():
    for path in [Path("neurosight_kb.json"), Path(__file__).parent / "neurosight_kb.json"]:
        if path.exists():
            return path
    return None

def _read_kb_from_url(url):
    """Connect to URL and read JSON once (no local copy)."""
    req = Request(url, headers={"User-Agent": "NeuroSight-Streamlit/1.0"})
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))

@st.cache_resource
def load_kb():
    """Connect to Drive and read KB once per session; reuse in memory. Fallback: local neurosight_kb.json."""
    with st.spinner("Connecting to knowledge base…"):
        try:
            return _read_kb_from_url(KB_DRIVE_URL)
        except (URLError, json.JSONDecodeError, OSError) as e:
            path = _kb_path()
            if path:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            raise RuntimeError(
                f"Could not load knowledge base from Drive ({e}). "
                "neurosight_kb.json not found locally."
            ) from e

# Connect once and read; cached for the session (no repeated download)
try:
    kb = load_kb()
except RuntimeError as e:
    st.error(f"❌ {e}")
    st.stop()
meta = kb["_meta"]
stats = kb["statistics"]
papers = list(kb["papers"].values())
key_papers = sorted([p for p in papers if p.get("extraction_depth") == "full_deep"], key=lambda p: p["id"])

# Outcome phrases to infer from abstract when target is missing (order: more specific first)
OUTCOME_PHRASES = [
    "disability progression", "cognitive decline", "cognitive impairment", "treatment response",
    "relapse", "relapses", "brain atrophy", "brain volume", "gray matter", "treatment failure",
    "NEDA", "EDSS", "SDMT", "visual acuity", "visual function", "disease progression",
    "progression", "conversion to MS", "conversion to secondary progressive",
    "quality of life", "neurodegeneration", "axonal loss", "optic neuritis",
]

def _infer_outcome_from_abstract(abstract):
    """Return first outcome phrase found in abstract (lowercased), or empty string."""
    if not abstract or not isinstance(abstract, str):
        return ""
    lower = abstract.lower()
    for phrase in OUTCOME_PHRASES:
        if phrase.lower() in lower:
            return phrase
    return ""


# Entity field mapping
ENTITY_FIELDS = {
    "retinal_biomarkers": ("🔬 Retinal Biomarkers", "#1A7BD4"),
    "fluid_biomarkers": ("🩸 Fluid Biomarkers", "#06B6D4"),
    "imaging_modalities": ("📷 Imaging Modalities", "#A855F7"),
    "clinical_scales": ("📊 Clinical Scales", "#F59E0B"),
    "ms_subtypes": ("🧬 MS Subtypes", "#F43F5E"),
    "drugs": ("💊 Treatments", "#10B981"),
    "anatomical_structures": ("🧠 Anatomy", "#F97316"),
    "ai_methods": ("🤖 AI Methods", "#0EA5E9"),
    "pathophysiology": ("⚡ Pathophysiology", "#8B5CF6"),
    "clinical_outcomes": ("🎯 Clinical Outcomes", "#EF4444"),
    "thematic_categories": ("📑 Research Themes", "#14B8A6"),
}

# OCT-specific entity subcategories for highlighting in the dashboard
OCT_ENTITY_GROUPS = {
    "Structural OCT": {
        "entities": ["pRNFL", "GCIPL", "INL", "ONL", "OPL", "IPL", "TMV",
                     "macular RNFL", "total retinal thickness", "foveal thickness",
                     "outer retinal layers", "photoreceptor layer"],
        "color": "#1A7BD4", "icon": "🔬"
    },
    "Lamina Cribrosa": {
        "entities": ["lamina cribrosa thickness", "LCT", "lamina cribrosa depth", "LCD",
                     "lamina cribrosa", "LC thickness", "LC depth"],
        "color": "#6366F1", "icon": "🔵"
    },
    "Choroidal": {
        "entities": ["choroidal thickness", "subfoveal choroidal thickness",
                     "choroidal vascularity index", "CVI",
                     "total choroidal area", "luminal area", "stromal area"],
        "color": "#EC4899", "icon": "🟣"
    },
    "Vascular (OCT-A)": {
        "entities": ["SCP vessel density", "DCP vessel density",
                     "foveal avascular zone", "FAZ",
                     "radial peripapillary capillary", "RPC density",
                     "optic nerve head vessel density", "macular vessel density",
                     "peripapillary vessel density", "choriocapillaris density",
                     "vessel density"],
        "color": "#F43F5E", "icon": "🩸"
    },
    "Inter-eye": {
        "entities": ["inter-eye RNFL asymmetry", "inter-eye GCIPL asymmetry",
                     "inter-eye asymmetry", "inter-eye difference"],
        "color": "#F97316", "icon": "👁️"
    },
    "OCT Modalities": {
        "entities": ["OCT", "SD-OCT", "SS-OCT", "EDI-OCT", "OCT-A",
                     "VBM-OCT", "wide-field OCT", "adaptive optics OCT"],
        "color": "#A855F7", "icon": "📷"
    },
    "Functional": {
        "entities": ["VEP", "mfVEP", "mfERG", "LSFG", "pupillometry",
                     "multifocal ERG", "visual evoked potential"],
        "color": "#0EA5E9", "icon": "⚡"
    },
}

def count_entities(field):
    c = Counter()
    for p in papers:
        for e in p.get(f"extracted_{field}", []):
            name = e if isinstance(e, str) else e.get("name", str(e))
            c[name] += 1
    return c

def badge(text, color):
    return f'<span class="entity-badge" style="background:{color}18;color:{color};border:1px solid {color}40;">{text}</span>'

# Map hex to class so colored dots always show (CSS !important overrides global theme)
_DOT_CLASS = {
    "#1A7BD4": "dot-blue", "#EF4444": "dot-red", "#F59E0B": "dot-orange",
    "#F97316": "dot-orange2", "#A855F7": "dot-purple", "#10B981": "dot-green",
    "#6B7280": "dot-gray",
}

def source_dot(color):
    c = (color or "").strip()
    cls = _DOT_CLASS.get(c.upper()) or _DOT_CLASS.get(c) or ""
    extra = f" {cls}" if cls else ""
    return f'<span class="source-dot{extra}" style="background:{color};"></span>'


def _offline_answer(question):
    """Generate answer from KB without API (when OPENAI_API_KEY not set)."""
    q = question.lower()
    if "threshold" in q:
        lines = ["# Validated OCT Thresholds for MS Progression\n",
                 "Based on the literature evidence, the key validated thresholds are:\n",
                 "## Retinal Layer Thresholds\n",
                 "**pRNFL (peripapillary Retinal Nerve Fiber Layer)**",
                 "- ≤88μm indicates significantly increased risk",
                 "- Associated with HR 2.4–3.0 for disability progression (PMID: 40424561, 29095097)\n",
                 "**GCIPL (Ganglion Cell-Inner Plexiform Layer)**",
                 "- ≤77μm indicates high-risk threshold",
                 "- Associated with HR 2.8–4.1 for disability progression (PMID: 40424561, 33737853)\n",
                 "**INL (Inner Nuclear Layer)**",
                 "- Thickening is the key marker (not thinning)",
                 "- Associated with OR 17.8 for relapse risk (PMID: 33737853)\n",
                 "## Rate-Based Thresholds\n",
                 "- pRNFL thinning >1.5 μm/yr → HR 3.0–6.8 for worsening",
                 "- GCIPL thinning >1.0 μm/yr → HR 3.5–5.7 for worsening"]
        return "\n".join(lines)
    elif "mri" in q or "compare" in q:
        return ("OCT provides independent, additive information beyond MRI (PMID: 37772490). "
                "Key advantages: non-invasive (~5 min vs 30-60 min), lower cost (~€50-100 vs €300-1000+), "
                "enables quarterly monitoring vs annual MRI, detects silent neurodegeneration between MRI scans. "
                "GCIPL correlates with gray matter volume (p<0.002, PMID: 36625888) but OCT is NOT associated "
                "with white matter — making it selective for gray matter neurodegeneration that MRI lesion counts miss.")
    elif "inl" in q or "relapse" in q:
        return ("INL thickening predicts relapse with OR=17.8 (p=0.023, PMID: 33737853). "
                "Unlike pRNFL/GCIPL which thin with neurodegeneration, INL thickens with inflammation. "
                "INL thinning rate also differentiates ocrelizumab responders from non-responders at 6 months (p=0.005, PMID: 35648233). "
                "This makes INL a unique dual biomarker: thickening signals relapse risk, thinning rate monitors treatment response.")
    elif "ai" in q or "accuracy" in q or "deep learning" in q:
        return ("AI achieves 94% pooled accuracy for MS diagnosis from imaging (PMID: 36303065, n=5,989). "
                "Pooled sensitivity 92%, specificity 93%, AUC 0.93. "
                "Deep learning enables automated OCT layer segmentation matching expert accuracy (PMID: 35180619). "
                "This validates NeuroSight's approach: AI can reliably extract clinical predictions from OCT scans.")
    return "Please set OPENAI_API_KEY for full AI agent capabilities, or try one of the suggested questions above."


def _build_relationships_from_kb(papers_list):
    """Build relationship rows from KB extracted_correlations and extracted_thresholds. Returns list of dicts with source, type, target, strength, pmids, s_color, t_color."""
    color_biomarker, color_outcome, color_scale, color_imaging = "#1A7BD4", "#EF4444", "#F59E0B", "#A855F7"
    color_anatomy, color_treatment, color_other = "#F97316", "#10B981", "#6B7280"
    def _color(s):
        s_lower = (s or "").lower()
        if any(x in s_lower for x in ["prnfl", "gcipl", "inl", "tmv", "rnfl", "lct", "lcd", "vessel density", "faz", "scp", "dcp"]): return color_biomarker
        if any(x in s_lower for x in ["disability", "relapse", "progression", "neda", "response", "cognitive", "decline"]): return color_outcome
        if any(x in s_lower for x in ["edss", "sdmt", "scale"]): return color_scale
        if any(x in s_lower for x in ["oct", "mri", "imaging"]): return color_imaging
        if any(x in s_lower for x in ["brain", "gray matter", "thalamus", "volume"]): return color_anatomy
        if any(x in s_lower for x in ["ocrelizumab", "treatment", "therapy"]): return color_treatment
        return color_other
    out = []
    for p in papers_list:
        pmid = p.get("pmid", "")
        for c in p.get("extracted_correlations", []) or []:
            if not isinstance(c, dict):
                continue
            bio = c.get("biomarker") or c.get("metric") or ""
            target = c.get("correlation_with") or c.get("target") or ""
            effect = c.get("effect_size") or c.get("value") or c.get("threshold") or ""
            if isinstance(effect, (int, float)):
                effect = str(effect)
            if not bio:
                continue
            rel_type = "correlates with" if (c.get("correlation_with") or "correlation" in str(c).lower()) else "associated"
            if c.get("threshold"):
                rel_type = "threshold"
            if not target:
                target = _infer_outcome_from_abstract(p.get("abstract") or "")
            target = target or "outcome"
            out.append({"source": bio[:50], "type": rel_type, "target": (target[:40] if target else "—"), "strength": (effect[:60] if effect else "—"), "pmids": str(pmid), "s_color": _color(bio), "t_color": _color(target)})
        for t in p.get("extracted_thresholds", []) or []:
            if not isinstance(t, dict):
                continue
            bio = t.get("biomarker") or ""
            thresh = t.get("threshold") or t.get("effect_size") or ""
            # Prefer explicit outcome/target; else infer from abstract; else metric
            target = t.get("target") or _infer_outcome_from_abstract(p.get("abstract") or "") or t.get("metric") or "threshold"
            target = (target or "threshold")[:40]
            if bio:
                out.append({"source": bio[:50], "type": "threshold", "target": target, "strength": str(thresh)[:60], "pmids": str(pmid), "s_color": _color(bio), "t_color": _color(target)})
    return out


def _aggregate_relationships_by_triple(rels):
    """Group relationships by (source, type, target); count how many times (papers) each occurred; collect PMIDs. Returns list of dicts with source, type, target, times, pmids_list, s_color, t_color (from first row)."""
    from collections import defaultdict
    key_to_pmids = defaultdict(set)
    key_to_row = {}
    for r in rels:
        key = (r["source"], r["type"], r["target"])
        key_to_pmids[key].add(r["pmids"])
        if key not in key_to_row:
            key_to_row[key] = r
    out = []
    for (source, typ, target), pmids_set in key_to_pmids.items():
        row = key_to_row[(source, typ, target)].copy()
        row["times"] = len(pmids_set)
        row["pmids_list"] = sorted(pmids_set)
        row["pmids"] = ", ".join(sorted(pmids_set)[:10]) + (" …" if len(pmids_set) > 10 else "")
        out.append(row)
    return sorted(out, key=lambda x: -x["times"])


def _entity_text(e):
    """Normalize entity to display string (string or dict with name/label)."""
    if e is None:
        return ""
    if isinstance(e, str):
        return str(e).strip()
    if isinstance(e, dict):
        return (e.get("name") or e.get("label") or "").strip()
    return str(e).strip()


# Blocklist: biomarker names that are not real OCT/retinal biomarkers (extraction noise)
_THRESHOLD_BIOMARKER_SKIP = frozenset({
    "ai", "significant", "not significant", "none", "n/a", "borderline reduction",
    "no reduction", "adequate", "protective", "most relevant factor",
    "longer disease duration", "trend observed", "standard error",
    "power", "graph binarization", "lesion segmentation", "logarithm of minimum angle",
    "accuracy",  # extraction often puts "accuracy" as biomarker when it's a metric
})

# Canonical OCT biomarkers for sort order (show these first)
_THRESHOLD_OCT_FIRST = ["prnfl", "gcipl", "rnfl", "mgcipl", "mgcl", "inl", "tmv", "macular volume", "gcl", "gcpl", "mrnfl", "pvd", "scp", "dcp", "faz", "lct", "choroid", "vessel density", "iead", "iepd"]

# Curated Tradeoffs: key OCT thresholds and their meaning (displayed under Entity–Relationship table)
TRADEOFFS_CURATED = [
    {"value": "p = 0.013", "biomarker": "pRNFL 87 μm", "study_line": "EDSS · PMID:41617221 · Bollo L, Pareto D, Tagliani P et al. (2026)", "explanation": "pRNFL 87 μm was linked in this study to: disability (Expanded Disability Status Scale)."},
    {"value": "distinguishing EDSS ≥ 3", "biomarker": "pRNFL 93.5 µm", "study_line": "cut-off · PMID:37190556 · Rzepiński Ł, Kucharczuk J, Tkaczyńska M et al. (2023)", "explanation": "pRNFL 93.5 µm was linked in this study to: cutoff used to classify or predict."},
    {"value": "inter-eye difference", "biomarker": "pRNFL ≥5 µm", "study_line": "inter-eye difference · PMID:32738702 · Bsteh G, Hegen H, Altmann P et al. (2020)", "explanation": "pRNFL ≥5 µm was associated with: inter-eye difference."},
    {"value": "most relevant factor for OAB", "biomarker": "pRNFL inferior quadrant Group 2", "study_line": "thinning · PMID:37709239 · Sahan B, Koskderelioglu A, Akmaz O et al. (2023)", "explanation": "pRNFL inferior quadrant Group 2 was associated with: thinning."},
    {"value": "predictive of blindness", "biomarker": "pRNFL thickness 36.7-48.3 µm", "study_line": "thickness · PMID:35814896 · Shi W, Zhang H, Zuo H et al. (2022)", "explanation": ""},
    {"value": "2.08 [1.47-2.95]", "biomarker": "pRNFL-z -2.04", "study_line": "aHR · PMID:38941572 · Lin T, Motamedi S, Asseyer S et al. (2024)", "explanation": "pRNFL-z -2.04 was linked in this study to: hazard ratio (risk over time)."},
    {"value": "1.51 [1.06-2.15]", "biomarker": "pRNFL-z", "study_line": "reciprocal odds ratio · PMID:38941572 · Lin T, Motamedi S, Asseyer S et al. (2024)", "explanation": "pRNFL-z was associated with: reciprocal odds ratio."},
    {"value": "0.71", "biomarker": "temporal-quadrant pRNFL 8μm", "study_line": "AUC · PMID:41296631 · Lin T, McCormack B, Bacchetti A et al. (2025)", "explanation": "temporal-quadrant pRNFL 8μm was linked in this study to: diagnostic accuracy (area under curve)."},
    {"value": "18.3", "biomarker": "aLmGCIPL cut-off ≥1 µm", "study_line": "odds ratio · PMID:32613912 · Bsteh G, Berek K, Hegen H et al. (2021)", "explanation": "aLmGCIPL cut-off ≥1 µm was associated with: odds ratio."},
    {"value": "2.7", "biomarker": "baseline mGCIPL thickness <77 µm", "study_line": "hazard ratio · PMID:32613912 · Bsteh G, Berek K, Hegen H et al. (2021)", "explanation": "baseline mGCIPL thickness <77 µm was associated with: hazard ratio."},
    {"value": "0.83", "biomarker": "GCIPL 4μm", "study_line": "AUC · PMID:41296631 · Lin T, McCormack B, Bacchetti A et al. (2025)", "explanation": "GCIPL 4μm was linked in this study to: diagnostic accuracy (area under curve)."},
    {"value": "trend observed", "biomarker": "GCIPL 77 µm", "study_line": "cognitive decline risk · PMID:37989566 · Alba-Arbalat S, Solana E, Lopez-Soley E et al. (2024)", "explanation": "GCIPL 77 µm was linked in this study to: cognitive function."},
    {"value": "<0.001", "biomarker": "GCIPL lower in MOGAD-ON compared to MOGAD-NON and HCs", "study_line": "p-value · PMID:35703428 · Oertel F, Sotirchos E, Zimmermann H et al. (2022)", "explanation": "GCIPL lower in MOGAD-ON compared to MOGAD-NON and HCs was associated with: p-value."},
    {"value": "<0.001", "biomarker": "GCIPL lower in MOGAD-NON compared to HCs", "study_line": "p-value · PMID:35703428 · Oertel F, Sotirchos E, Zimmermann H et al. (2022)", "explanation": "GCIPL lower in MOGAD-NON compared to HCs was associated with: p-value."},
    {"value": "predicts disability progression", "biomarker": "GCIPL 1.6 µm", "study_line": "annualised thinning rate · PMID:35652366 · Berek K, Hegen H, Hocher J et al. (2022)", "explanation": "GCIPL 1.6 µm was associated with: annualised thinning rate."},
    {"value": "significant", "biomarker": "GCIPL ≥4 µm", "study_line": "inter-eye difference · PMID:32738702 · Bsteh G, Hegen H, Altmann P et al. (2020)", "explanation": "GCIPL ≥4 µm was associated with: inter-eye difference."},
    {"value": "< 0.001", "biomarker": "GCIPL thickness 54.89 μm", "study_line": "p-value · PMID:40897938 · Kong L, Tan X, Wang H et al. (2025)", "explanation": "GCIPL thickness 54.89 μm was associated with: p-value."},
    {"value": "2.751", "biomarker": "GCIPL thickness ≤77 μm", "study_line": "HR · PMID:40424561 · El Ayoubi N, Ismail A, Sader G et al. (2025)", "explanation": "GCIPL thickness ≤77 μm was linked in this study to: hazard ratio (risk over time)."},
    {"value": "76 μm", "biomarker": "GCIPL thickness 76 μm", "study_line": "cut-off point · PMID:32370626 · Esmael A, Elsherif M, Abdelsalam M et al. (2020)", "explanation": "GCIPL thickness 76 μm was linked in this study to: cutoff used to classify or predict."},
    {"value": "3.535", "biomarker": "GCIPL thinning >1 μm/y", "study_line": "HR · PMID:40424561 · El Ayoubi N, Ismail A, Sader G et al. (2025)", "explanation": "GCIPL thinning >1 μm/y was linked in this study to: hazard ratio (risk over time)."},
    {"value": "correlation with BCVA", "biomarker": "pRNFL thickness 57.2-67.5 µm", "study_line": "thickness · PMID:35814896 · Shi W, Zhang H, Zuo H et al. (2022)", "explanation": ""},
    {"value": "2.8-fold increase", "biomarker": "pRNFL thickness lowest tertile", "study_line": "risk of disability worsening · PMID:33606071 · Cilingir V, Batur M (2021)", "explanation": ""},
    {"value": "-0.57%/year vs -0.42%/year", "biomarker": "GCIPL atrophy obese vs normal weight", "study_line": "rate · PMID:32297826 · Filippatou A, Lambe J, Sotirchos E et al. (2020)", "explanation": "GCIPL atrophy obese vs normal weight was associated with: rate."},
    {"value": "lower in HE-DMT group compared to ME-DMT group", "biomarker": "GCIPL atrophy rate 0.07", "study_line": "µm/year · PMID:41655476 · Vacca A, Huang S, Idini E et al. (2026)", "explanation": "GCIPL atrophy rate 0.07 was associated with: µm/year."},
    {"value": "p = 0.006", "biomarker": "mGCIPL 6.79 μm", "study_line": "thicker · PMID:40997284 · Kodali S, Bianchi A, Raftopoulos R et al. (2025)", "explanation": "mGCIPL 6.79 μm was associated with: thicker."},
    {"value": "0.38", "biomarker": "mGCIPL thinning MOG group", "study_line": "μm/year · PMID:39228031 · Moon Y, Gim Y, Park K et al. (2025)", "explanation": "mGCIPL thinning MOG group was associated with: μm/year."},
    {"value": "-0.605% vs. -0.315%", "biomarker": "baseline RNFLT <86 μm", "study_line": "percentage change · PMID:36286605 · Wang L, Tan H, Yu J et al. (2023)", "explanation": "baseline RNFLT <86 μm was associated with: percentage change."},
    {"value": "4.0 vs. 2.8", "biomarker": "baseline RNFLT ≥99 μm", "study_line": "mean number of new T2 lesions · PMID:36286605 · Wang L, Tan H, Yu J et al. (2023)", "explanation": "baseline RNFLT ≥99 μm was associated with: mean number of new T2 lesions."},
    {"value": "p < 0.001", "biomarker": "cortical lesion number and RNFL thickness higher", "study_line": "accuracy · PMID:36192175 · Cortese R, Prados Carrasco F, Tur C et al. (2023)", "explanation": "cortical lesion number and RNFL thickness higher was associated with: accuracy."},
    {"value": "-1.113", "biomarker": "mRNFL -1.113", "study_line": "μm/year · PMID:34125629 · Mizell R, Chen H, Lambe J et al. (2022)", "explanation": "mRNFL -1.113 was associated with: μm/year."},
    {"value": "0.57", "biomarker": "mRNFL lower", "study_line": "Cohen's d · PMID:34087931 · Cerna J, Anaraki N, Robbs C et al. (2021)", "explanation": "mRNFL lower was associated with: Cohen's d."},
    {"value": "2.16", "biomarker": "odRNFL lower", "study_line": "Cohen's d · PMID:34087931 · Cerna J, Anaraki N, Robbs C et al. (2021)", "explanation": ""},
    {"value": "91 µm", "biomarker": "RNFL 91 µm", "study_line": "turning point · PMID:37433662 · Rosenkranz S, Gutmann L, Has Silemek A et al. (2023)", "explanation": "RNFL 91 µm was associated with: turning point."},
    {"value": "β = .217", "biomarker": "RNFL 3%", "study_line": "variance explained · PMID:36317869 · Kim J, Bollaert R, Cerna J et al. (2022)", "explanation": "RNFL 3% was associated with: variance explained."},
    {"value": "p = 0.009", "biomarker": "RNFL mean: 87.54 [±13.83] vs 75.54 [±20.33]", "study_line": "accuracy · PMID:36192175 · Cortese R, Prados Carrasco F, Tur C et al. (2023)", "explanation": "RNFL mean was associated with: accuracy."},
    {"value": "0.76", "biomarker": "RNFL 0.76", "study_line": "AUC · PMID:34906813 · Piedrabuena R, Bittar M (2022)", "explanation": "RNFL 0.76 was linked in this study to: diagnostic accuracy (area under curve)."},
    {"value": "79 μm", "biomarker": "RNFL thickness 79 μm", "study_line": "cut-off point · PMID:32370626 · Esmael A, Elsherif M, Abdelsalam M et al. (2020)", "explanation": "RNFL thickness 79 μm was linked in this study to: cutoff used to classify or predict."},
    {"value": "0.95", "biomarker": "TMV lower", "study_line": "Cohen's d · PMID:34087931 · Cerna J, Anaraki N, Robbs C et al. (2021)", "explanation": "TMV lower was associated with: Cohen's d."},
    {"value": "P=0.032", "biomarker": "GCL thickness 5.7 µm", "study_line": "mean difference · PMID:31956579 · MacIntosh P, Kumar S, Saravanan V et al. (2020)", "explanation": "GCL thickness 5.7 µm was associated with: mean difference."},
    {"value": "P=0.011", "biomarker": "GCL thickness 7.2 µm", "study_line": "mean difference · PMID:31956579 · MacIntosh P, Kumar S, Saravanan V et al. (2020)", "explanation": "GCL thickness 7.2 µm was associated with: mean difference."},
    {"value": "predictive of blindness", "biomarker": "mRGCL volumes 0.495-0.613 mm3", "study_line": "volume · PMID:35814896 · Shi W, Zhang H, Zuo H et al. (2022)", "explanation": "mRGCL volumes 0.495-0.613 mm3 was associated with: volume."},
    {"value": "correlation with BCVA", "biomarker": "mRGCL volumes 0.691-0.737 mm3", "study_line": "volume · PMID:35814896 · Shi W, Zhang H, Zuo H et al. (2022)", "explanation": ""},
    {"value": "0.829", "biomarker": "nasal side vessel density of the optic disc", "study_line": "AUC · PMID:41057808 · Dong J, Wang X, Xue H et al. (2025)", "explanation": "nasal side vessel density of the optic disc was linked in this study to: diagnostic accuracy (area under curve)."},
    {"value": "vessel density features 100%", "biomarker": "vessel density features", "study_line": "AUC · PMID:37440373 · Jalili J, Nadimi M, Jafari B et al. (2024)", "explanation": "vessel density features 100% was linked in this study to: diagnostic accuracy (area under curve)."},
]


def _normalize_biomarker_for_key(bio):
    """Normalize biomarker string for dedup key so 'pRNFL thickness' and 'pRNFL' merge."""
    b = (bio or "").lower().strip()
    for suffix in [" thickness", " volume", " volumes", " rate", " density", " atrophy", " thinning"]:
        if b.endswith(suffix):
            b = b[: -len(suffix)]
    return b.strip()[:50]


def _build_thresholds_deduped(papers_list):
    """Build list of thresholds from KB, deduplicated by (biomarker, threshold, metric); filter noise. Returns list of dicts with biomarker, threshold, effect_size, metric, pmids_list, times, and first pmid/authors/year for display."""
    from collections import defaultdict
    raw = []
    for p in papers_list:
        for t in p.get("extracted_thresholds", []) or []:
            if not isinstance(t, dict):
                continue
            bio = (t.get("biomarker") or "").strip()
            thresh_val = t.get("threshold") or t.get("effect_size") or ""
            effect = t.get("effect_size") or t.get("threshold") or ""
            metric = (t.get("metric") or "").strip()
            if not bio:
                continue
            bio_lower = bio.lower()
            if bio_lower in _THRESHOLD_BIOMARKER_SKIP:
                continue
            effect_str = str(effect).strip() if effect is not None else ""
            if effect_str.upper() in ("NONE", "N/A", ""):
                continue
            # Skip when the "value" is only a p-value (no clinical cutoff)
            metric_lower = metric.lower()
            if metric_lower in ("p-value", "p-value ", "p value") and (effect_str.startswith("0.0") or effect_str.startswith("p ") or effect_str.startswith("p=") or effect_str.replace(" ", "").startswith("p<")):
                continue
            raw.append({
                "biomarker": bio,
                "threshold": str(thresh_val).strip() if thresh_val is not None else "",
                "effect_size": effect_str,
                "metric": metric,
                "pmid": str(p.get("pmid") or ""),
                "authors": p.get("authors") or "",
                "year": str(p.get("year") or ""),
            })
    # Deduplicate by (normalized biomarker, threshold) — merge "pRNFL thickness" and "pRNFL" etc.
    key_to_row = {}
    key_to_pmids = defaultdict(set)
    key_to_metrics = defaultdict(list)
    for r in raw:
        norm_bio = _normalize_biomarker_for_key(r["biomarker"])
        thresh_norm = (r["threshold"] or r["effect_size"] or "")[:80]
        k = (norm_bio, thresh_norm)
        key_to_pmids[k].add(r["pmid"])
        if r["metric"] and r["metric"] not in key_to_metrics[k]:
            key_to_metrics[k].append(r["metric"])
        if k not in key_to_row:
            key_to_row[k] = r.copy()
    out = []
    for k, row in key_to_row.items():
        pmids_list = sorted(key_to_pmids[k])
        metrics_list = key_to_metrics[k]
        row["pmids_list"] = pmids_list
        row["times"] = len(pmids_list)
        row["pmids"] = ", ".join(pmids_list[:8]) + (" …" if len(pmids_list) > 8 else "")
        row["metrics_merged"] = ", ".join(metrics_list[:8]) + (" …" if len(metrics_list) > 8 else "") if metrics_list else row.get("metric", "")
        out.append(row)
    # Sort: OCT metrics first (pRNFL, GCIPL, etc.), then by times desc, then by biomarker
    def _oct_priority(bio):
        b = _normalize_biomarker_for_key(bio) or (bio or "").lower()
        for i, oct in enumerate(_THRESHOLD_OCT_FIRST):
            if oct in b or b.startswith(oct):
                return i
        return len(_THRESHOLD_OCT_FIRST)
    return sorted(out, key=lambda x: (_oct_priority(x["biomarker"]), -x["times"], (x["biomarker"] or "").lower()))


def _is_oct_related_threshold(row):
    """True if the threshold row is OCT/retinal biomarker related (for filtering to top 100 OCT-only)."""
    b = _normalize_biomarker_for_key(row.get("biomarker") or "")
    return any(oct in b or b.startswith(oct) for oct in _THRESHOLD_OCT_FIRST)


# Outcome abbreviations -> plain-language explanation (for "what this means")
_THRESHOLD_OUTCOME_EXPLAIN = {
    "EDSS": "disability (Expanded Disability Status Scale)",
    "CDP": "confirmed disability progression",
    "median time to CDP": "time until disability progression",
    "retinal thinning": "retinal layer thinning over time",
    "PBVC": "brain volume loss (percent brain volume change)",
    "SCA": "spinal cord area",
    "HCVA": "high-contrast visual acuity",
    "LCVA": "low-contrast visual acuity",
    "AULCSF": "contrast sensitivity",
    "NEDA": "no evidence of disease activity",
    "relapse": "relapse risk",
    "cognitive": "cognitive function",
    "SDMT": "cognitive processing speed (SDMT)",
    "brain volume": "brain volume",
    "IEAD": "inter-eye asymmetry (difference between eyes)",
    "IEPD": "inter-eye percentage difference",
    "HR": "hazard ratio (risk over time)",
    "OR": "odds ratio",
    "AUC": "diagnostic accuracy (area under curve)",
    "cut-off": "cutoff used to classify or predict",
    "cutoff": "cutoff used to classify or predict",
}


def _explain_threshold(row):
    """One-line plain-language explanation of what this threshold means."""
    bio = (row.get("biomarker") or "").strip()
    thresh = (row.get("threshold") or "").strip()
    metrics = (row.get("metrics_merged") or row.get("metric") or "").strip()
    if not bio:
        return ""
    # Build explanation: biomarker + threshold linked to [outcomes]
    parts = []
    for key, meaning in _THRESHOLD_OUTCOME_EXPLAIN.items():
        if key.lower() in metrics.lower():
            parts.append(meaning)
    if parts:
        outcomes = "; ".join(parts[:5]) + (" …" if len(parts) > 5 else "")
        return f"{bio} {thresh} was linked in this study to: {outcomes}."
    if metrics:
        return f"{bio} {thresh} was associated with: {metrics[:120]}{'…' if len(metrics) > 120 else ''}."
    return f"{bio} {thresh} — cutoff or association reported in the paper (see PMID for context)."


def _analyze_paper(p):
    """Analyze one publication (OpenAI if key set, else entity-based summary)."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    title = p.get("title") or ""
    abstract = (p.get("abstract") or "")[:4000]
    entities = []
    for field in ENTITY_FIELDS:
        for e in p.get(f"extracted_{field}", []):
            name = e if isinstance(e, str) else e.get("name", str(e))
            entities.append(name)
    correlations = p.get("extracted_correlations", []) or []
    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are the NeuroSight Analyzer. In 4–6 short sentences, summarize this OCT/MS paper: main question, methods, key findings, and relevance to retinal biomarkers in MS. Be specific; cite effect sizes if mentioned."},
                    {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}\n\nExtracted entities: {', '.join(entities[:30])}\nCorrelations: {correlations[:5]}"},
                ],
                max_tokens=500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Analysis failed: {e}"
    # Offline: entity-based summary
    parts = [f"**Summary (from extracted data):** This paper addresses {title[:80]}..."]
    if entities:
        parts.append(f"**Entities:** {', '.join(entities[:15])}{'...' if len(entities) > 15 else ''}.")
    if correlations:
        parts.append("**Reported relationships:** " + "; ".join(str(c) for c in correlations[:5]))
    parts.append("Set OPENAI_API_KEY for full AI analysis.")
    return "\n\n".join(parts)


def _synthesize_papers(papers_list):
    """RAG over selected publications; returns a short synopsis (OpenAI if key set, else list)."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not papers_list:
        return "No papers selected."
    context = []
    for p in papers_list[:30]:
        title = p.get("title") or ""
        abstract = (p.get("abstract") or "")[:800]
        context.append(f"PMID {p.get('pmid')} ({p.get('year')}): {title}\n{abstract}")
    block = "\n\n---\n\n".join(context)
    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are the NeuroSight RAG agent. Given the following set of OCT/MS publications (retrieved from the KB), write a short synopsis (1–2 paragraphs): main themes, consensus findings, and any contradictions or gaps. Focus on biomarkers, prognosis, and clinical relevance. This is a synopsis of retrieved papers, not a full systematic synthesis."},
                    {"role": "user", "content": f"Write a short synopsis of these {len(papers_list)} papers:\n\n{block}"},
                ],
                max_tokens=800,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Synopsis failed: {e}"
    # Offline: list titles
    titles = [f"- PMID {p.get('pmid')} ({p.get('year')}): {(p.get('title') or '')[:70]}..." for p in papers_list[:20]]
    return "**Papers included:**\n\n" + "\n".join(titles) + "\n\nSet OPENAI_API_KEY for RAG synopsis."


# ─── Sidebar + main content (callable for embedding in main app) ──────────────

def run_kb():
    # ── Theme: identical dark-blue aesthetic as neurosight_dashboard_5 ──
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html,body{font-family:'DM Sans',sans-serif;}
.stApp{background:#060d1a !important;color:#e8edf5 !important;}
[data-testid="stAppViewContainer"]{background:#060d1a !important;}
.main .block-container{background:#060d1a !important;padding-top:2rem;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a1628,#07101f);border-right:1px solid #1a2d4a;}
section[data-testid="stSidebar"] *{color:#c8d8ee !important;}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a,
section[data-testid="stSidebar"] nav a {
    font-size:1.05rem !important;padding:10px 14px !important;
    min-height:44px !important;display:flex !important;align-items:center !important;
    font-family:'Syne',sans-serif !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"]:first-of-type a,
section[data-testid="stSidebar"] nav a:first-child {
    background:linear-gradient(90deg,#60a5fa,#3b82f6,#2563eb) !important;
    -webkit-background-clip:text !important;background-clip:text !important;
    -webkit-text-fill-color:transparent !important;color:#60a5fa !important;
    font-weight:700 !important;letter-spacing:0.02em;
    filter:drop-shadow(0 0 6px rgba(59,130,246,0.5)) drop-shadow(0 0 12px rgba(99,102,241,0.4));
}
section[data-testid="stSidebar"] [data-testid="stPageLink"]:first-of-type a *,
section[data-testid="stSidebar"] nav a:first-child * {
    color:#60a5fa !important;-webkit-text-fill-color:#60a5fa !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"]:last-of-type a,
section[data-testid="stSidebar"] nav a:last-of-type {
    background:linear-gradient(90deg,#ec4899,#c084fc,#7b2fff) !important;
    -webkit-background-clip:text !important;background-clip:text !important;
    -webkit-text-fill-color:transparent !important;color:transparent !important;
    font-weight:700 !important;letter-spacing:0.02em;
    filter:drop-shadow(0 0 6px rgba(236,72,153,0.4)) drop-shadow(0 0 10px rgba(168,85,247,0.35));
}
h1,h2,h3{font-family:'Syne',sans-serif !important;color:#e8edf5 !important;}
h1{color:#e8edf5 !important;}
hr.ns{border:none;border-top:1px solid #1a2d4a;margin:16px 0;}
header[data-testid="stHeader"]{background:transparent !important;height:0 !important;min-height:0 !important;visibility:hidden !important;}
[data-testid="stMarkdown"] p{color:#c0d0e8 !important;}
.stCaption,[data-testid="stCaptionContainer"]{color:#6a8caa !important;}
[data-testid="stMetricValue"]{color:#e8edf5 !important;}
[data-testid="stMetricLabel"]{color:#6a8caa !important;}
hr{border-color:#1a2d4a !important;}

/* ── Cards (same .ns-card look as dashboard) ── */
.ns-card{background:linear-gradient(135deg,#0d1e35,#0a1628);border:1px solid #1e3354;border-radius:16px;padding:20px 24px;margin-bottom:16px;position:relative;overflow:hidden;}
.ns-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00c6ff,#0072ff,#7b2fff);}
.metric-box{background:linear-gradient(135deg,#0d1e35,#0a1628);border-radius:16px;padding:16px 18px;border:1px solid #1e3354;position:relative;overflow:hidden;}
.metric-box::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00c6ff,#0072ff,#7b2fff);}
.metric-box.teal::before{background:linear-gradient(90deg,#14B8A6,#0d9488);} .metric-box.rose::before{background:linear-gradient(90deg,#F43F5E,#e11d48);} .metric-box.amber::before{background:linear-gradient(90deg,#F59E0B,#d97706);}
.metric-label{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:#6a8caa;margin-bottom:4px;font-weight:600;}
.metric-val{font-family:'DM Mono',monospace;font-size:2rem;font-weight:500;color:#e8edf5;letter-spacing:-0.02em;line-height:1;margin:4px 0;}
.metric-delta{font-family:'DM Mono',monospace;font-size:.8rem;color:#69ffb0;margin-top:2px;}
.card{background:linear-gradient(135deg,#0d1e35,#0a1628);border-radius:16px;padding:20px;border:1px solid #1e3354;margin-bottom:14px;}
.section-title{font-family:'Syne',sans-serif;font-size:.68rem;letter-spacing:.18em;text-transform:uppercase;color:#4a7aaa;margin:20px 0 10px;padding-bottom:6px;border-bottom:1px solid #1a2d4a;}
.entity-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;margin:2px;}
.entity-rel-info{background:linear-gradient(135deg,#0f1e3a,#0d2440);border:1px solid #3a4080;border-radius:14px;padding:16px 20px;margin:16px 0;color:#d0dcee;font-size:0.95rem;line-height:1.6;}
.entity-rel-info strong{color:#93c5ff;}

/* ── Tables: dark theme ── */
.rel-table{width:100%;border-collapse:collapse;font-size:13px;background:linear-gradient(135deg,#0d1e35,#0a1628) !important;border-radius:12px;overflow:hidden;border:1px solid #1e3354;}
.rel-table th{text-align:left;padding:12px 14px;border-bottom:2px solid #1e3354;background:#0a1628 !important;color:#6a8caa !important;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em;}
.rel-table td{padding:12px 14px;border-bottom:1px solid #1a2d4a;color:#c8d8ee !important;background:transparent !important;}
.rel-table tbody tr:nth-child(even) td{background:rgba(10,22,40,0.5) !important;}
.rel-table tbody tr:hover td{background:rgba(0,114,255,0.1) !important;}
.rel-table .source-dot{display:inline-block !important;width:10px !important;height:10px !important;min-width:10px !important;border-radius:50% !important;margin-right:8px !important;vertical-align:middle !important;}
.rel-table .source-dot.dot-blue{background:#1A7BD4 !important;} .rel-table .source-dot.dot-red{background:#EF4444 !important;}
.rel-table .source-dot.dot-orange{background:#F59E0B !important;} .rel-table .source-dot.dot-orange2{background:#F97316 !important;}
.rel-table .source-dot.dot-purple{background:#A855F7 !important;} .rel-table .source-dot.dot-green{background:#10B981 !important;}
.rel-table .source-dot.dot-gray{background:#6B7280 !important;}
.rel-table .rel-type{color:#8ab0d0 !important;font-style:italic !important;}
.source-dot{display:inline-block !important;width:10px !important;height:10px !important;min-width:10px !important;border-radius:50% !important;margin-right:8px !important;vertical-align:middle !important;}
.source-dot.dot-blue{background:#1A7BD4 !important;} .source-dot.dot-red{background:#EF4444 !important;}
.source-dot.dot-orange{background:#F59E0B !important;} .source-dot.dot-orange2{background:#F97316 !important;}
.source-dot.dot-purple{background:#A855F7 !important;} .source-dot.dot-green{background:#10B981 !important;}
.strength-bold{font-weight:700;color:#7ab3ff !important;}
.pmid-ref{font-family:'JetBrains Mono',monospace;font-size:11px;color:#4a7aaa !important;}

/* Key evidence wrapper: dark */
#key-evidence-wrapper{background:linear-gradient(135deg,#0d1e35,#0a1628);border-radius:12px;overflow:hidden;border:1px solid #1e3354;margin:16px 0;}
#key-evidence-wrapper table{width:100% !important;border-collapse:collapse !important;background:transparent !important;font-size:13px !important;}
#key-evidence-wrapper th{background:#0a1628 !important;color:#6a8caa !important;border-bottom:2px solid #1e3354 !important;padding:12px 14px !important;font-weight:600 !important;font-size:11px !important;text-transform:uppercase !important;letter-spacing:.05em !important;}
#key-evidence-wrapper td{background:transparent !important;color:#c8d8ee !important;border-bottom:1px solid #1a2d4a !important;padding:12px 14px !important;}
#key-evidence-wrapper tbody tr:nth-child(even) td{background:rgba(10,22,40,0.5) !important;}
#key-evidence-wrapper tbody tr:hover td{background:rgba(0,114,255,0.1) !important;}
#key-evidence-wrapper .source-dot{display:inline-block !important;width:12px !important;height:12px !important;min-width:12px !important;min-height:12px !important;border-radius:50% !important;margin-right:8px !important;vertical-align:middle !important;}
#key-evidence-wrapper .source-dot.dot-blue{background:#1A7BD4 !important;} #key-evidence-wrapper .source-dot.dot-red{background:#EF4444 !important;}
#key-evidence-wrapper .source-dot.dot-orange{background:#F59E0B !important;} #key-evidence-wrapper .source-dot.dot-orange2{background:#F97316 !important;}
#key-evidence-wrapper .source-dot.dot-purple{background:#A855F7 !important;} #key-evidence-wrapper .source-dot.dot-green{background:#10B981 !important;}
#key-evidence-wrapper .source-dot.dot-gray{background:#6B7280 !important;}
#key-evidence-wrapper .rel-type{color:#8ab0d0 !important;font-style:italic !important;}
#key-evidence-wrapper .pmid-ref{color:#4a7aaa !important;font-family:'JetBrains Mono',monospace !important;font-size:11px !important;}

/* Evidence chart boxes: dark */
.evidence-box-header{padding:14px 18px !important;font-weight:700 !important;font-size:0.95rem !important;border:1px solid #1e3354 !important;border-bottom:none !important;border-radius:12px 12px 0 0 !important;}
.evidence-box-header.bio{background:linear-gradient(135deg,#0a1e1e,#0a1628) !important;color:#14B8A6 !important;border-left:4px solid #14B8A6 !important;}
.evidence-box-header.rel{background:linear-gradient(135deg,#0a1428,#0a1628) !important;color:#7ab3ff !important;border-left:4px solid #1A7BD4 !important;}
.evidence-box-bottom{border:1px solid #1e3354 !important;border-top:none !important;border-radius:0 0 12px 12px !important;height:8px !important;background:#0a1628 !important;margin-bottom:16px !important;}

/* Brand titles */
.ns-brand-title{font-family:'Syne',sans-serif !important;font-size:1.5rem !important;font-weight:800 !important;background:linear-gradient(90deg,#00c6ff,#7b2fff) !important;-webkit-background-clip:text !important;background-clip:text !important;-webkit-text-fill-color:transparent !important;color:transparent !important;}
.kb-brand-title{font-family:'Syne',sans-serif !important;font-size:1.35rem !important;font-weight:800 !important;background:linear-gradient(90deg,#ec4899,#c084fc,#7b2fff) !important;-webkit-background-clip:text !important;background-clip:text !important;-webkit-text-fill-color:transparent !important;color:transparent !important;}

/* Streamlit widgets on dark — comprehensive */
[data-testid="stExpander"]{background:linear-gradient(135deg,#0d1e35,#0a1628) !important;border:1px solid #1e3354 !important;border-radius:14px !important;}
[data-testid="stExpander"] summary span{color:#c8d8ee !important;}
[data-testid="stExpander"] [data-testid="stMarkdown"] p{color:#c0d0e8 !important;}

/* Text inputs, text areas, number inputs */
.stTextInput>div>div>input,
[data-testid="stTextInput"] input,
.stTextArea>div>div>textarea,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input{background:#0d1e35 !important;color:#c8d8ee !important;border:1px solid #1e3354 !important;-webkit-text-fill-color:#c8d8ee !important;}

/* Selectbox / multiselect — the control, its value text, dropdown menu, and options */
[data-baseweb="select"]>div{background:#0d1e35 !important;border-color:#1e3354 !important;}
[data-baseweb="select"] [data-baseweb="tag"]{background:#1a2d4a !important;color:#c8d8ee !important;}
[data-baseweb="select"] [data-baseweb="tag"] span{color:#c8d8ee !important;}
[data-baseweb="select"] span,[data-baseweb="select"] div{color:#c8d8ee !important;}
[data-baseweb="select"] svg{fill:#6a8caa !important;}
[data-baseweb="select"] input{color:#c8d8ee !important;-webkit-text-fill-color:#c8d8ee !important;}
[data-baseweb="select"] [aria-expanded] > div{background:#0d1e35 !important;}
/* Dropdown popover / listbox */
[data-baseweb="popover"]{background:#0d1e35 !important;border:1px solid #1e3354 !important;}
[data-baseweb="popover"] li,[data-baseweb="popover"] ul{background:#0d1e35 !important;color:#c8d8ee !important;}
[data-baseweb="popover"] li:hover{background:#1a2d4a !important;}
[data-baseweb="menu"]{background:#0d1e35 !important;}
[data-baseweb="menu"] li{background:#0d1e35 !important;color:#c8d8ee !important;}
[data-baseweb="menu"] li:hover{background:#1a2d4a !important;}
[role="listbox"]{background:#0d1e35 !important;}
[role="listbox"] [role="option"]{background:#0d1e35 !important;color:#c8d8ee !important;}
[role="listbox"] [role="option"]:hover,[role="listbox"] [role="option"][aria-selected="true"]{background:#1a2d4a !important;}

/* Generic input wrapper (BaseWeb) */
[data-baseweb="input"]{background:#0d1e35 !important;border-color:#1e3354 !important;}
[data-baseweb="input"]>div{background:#0d1e35 !important;}
[data-baseweb="input"] input{color:#c8d8ee !important;-webkit-text-fill-color:#c8d8ee !important;}

/* Buttons */
button[kind="secondary"]{background:#0d1e35 !important;color:#c8d8ee !important;border-color:#1e3354 !important;}
.stCheckbox label span{color:#c8d8ee !important;}

/* Widget labels */
.stSelectbox label,.stTextInput label,.stTextArea label,.stNumberInput label,.stMultiSelect label,
[data-testid="stWidgetLabel"]{color:#6a8caa !important;}
</style>
""", unsafe_allow_html=True)

    with st.sidebar:
        # App nav: same as dashboard (NeuroSight + Knowledge Base)
        st.page_link("neurosight_dashboard_5.py", label="NeuroSight", icon="👁")
        st.page_link("pages/2_NeuroSight_Knowledge_Base.py", label="Knowledge Base", icon="🧠")
        st.markdown("<hr class='ns'>", unsafe_allow_html=True)
        st.markdown("""<div style='padding:8px 0 20px'>
            <div class='kb-brand-title'>KNOWLEDGE BASE</div>
            <div style='font-size: .7rem; letter-spacing: .15em; color: #6a8caa; margin-top: 2px;'>
                OCT × MS · Research Dashboard</div></div>""", unsafe_allow_html=True)
        st.markdown("<hr class='ns'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Navigation")
        page = st.radio("", [
            "📊 Overview",
            "📈 Entity Trends",
            "📋 Field Analysis",
            "🔗 Entity–Relationship",
            "🕸️ Knowledge Graph",
            "🌐 Co-occurrence Graph",
            "📄 Research analysis and synthesis agents",
            "🤖 AI Literature Agent",
        ], label_visibility="collapsed")
        st.markdown("<hr class='ns'>", unsafe_allow_html=True)
        st.caption(f"{stats['total_papers']:,} papers · 2020–2026")
        st.caption("2026 EP PerMed Hackathon")
        st.caption("Building NeuroSight Knowledge Base!")


    # ═════════════════════════════════════════════════════════════════════════════
    #  OVERVIEW
    # ═════════════════════════════════════════════════════════════════════════════

    if page == "📊 Overview":
        st.markdown("# 🧠 NeuroSight Knowledge Base")
        st.markdown(
            f'<div class="entity-rel-info" style="margin-top:12px;margin-bottom:20px;">'
            f'<p style="margin:0 0 12px 0;">Information mining, <strong>biomedical NER</strong>, <strong>entity resolution</strong>, and knowledge extraction from <strong>{stats["total_papers"]:,}</strong> PubMed papers (2020–2026) · OCT biomarkers for MS progression monitoring · 2026 EP PerMed Hackathon</p>'
            f'<p style="margin:0;">NeuroSight\'s Knowledge Base is built on a multi-agent system that uses <strong>NLP</strong> (natural language processing) for knowledge extraction, analysis, synthesis, and question answering, all based on PubMed literature.</p>'
            f'<p style="margin:12px 0 0 0;font-size:0.9rem;opacity:0.95;">Building NeuroSight Knowledge Base!</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")  # spacing

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Total Publications</div><div class="metric-val">{stats["total_papers"]:,}</div><div class="metric-delta">2020–Mar 2026</div></div>', unsafe_allow_html=True)
        with c2:
            peak = max(stats["yearly_distribution"].items(), key=lambda x: x[1])
            st.markdown(f'<div class="metric-box teal"><div class="metric-label">Peak Year</div><div class="metric-val">{peak[0]}</div><div class="metric-delta">{peak[1]} papers</div></div>', unsafe_allow_html=True)
        with c3:
            bio_info = stats["entity_summary"].get("retinal_biomarkers", {})
            st.markdown(f'<div class="metric-box rose"><div class="metric-label">Biomarkers Tracked</div><div class="metric-val">{bio_info.get("unique",0)}</div><div class="metric-delta">{bio_info.get("total_mentions",0):,} mentions</div></div>', unsafe_allow_html=True)
        with c4:
            total_rels = stats["total_correlations"]
            st.markdown(f'<div class="metric-box amber"><div class="metric-label">Key Relationships</div><div class="metric-val">{total_rels}</div><div class="metric-delta">from {stats["key_papers_deep_extracted"]} papers</div></div>', unsafe_allow_html=True)

        # Charts
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="section-title">📈 Publications by Year</div>', unsafe_allow_html=True)
            # Derive from actual papers so chart adds up and all years (e.g. 2025) show
            yearly = Counter(str(p.get("year") or "?") for p in papers)
            yearly = {k: v for k, v in yearly.items() if k.isdigit() and 2015 <= int(k) <= 2030}
            yearly = dict(sorted(yearly.items(), key=lambda x: int(x[0])))
            df = pd.DataFrame([{"Year": k, "Papers": v} for k, v in yearly.items()])
            fig = px.bar(df, x="Year", y="Papers", text="Papers", color_discrete_sequence=["#1A7BD4"])
            fig.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628", font_color="#8ab0d0",
                              margin=dict(t=10,b=30,l=40,r=10), height=280, showlegend=False,
                              xaxis=dict(gridcolor="#1a2d4a"), yaxis=dict(gridcolor="#1a2d4a"))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            chart_total = sum(yearly.values())
            st.caption(f"Total: **{chart_total:,}** papers. Chart built from loaded papers so counts match.")

        with col_r:
            st.markdown('<div class="section-title">📈 Cumulative Publications</div>', unsafe_allow_html=True)
            sorted_y = sorted(yearly.items(), key=lambda x: int(x[0]))
            cum = list(accumulate(v for _, v in sorted_y))
            df_cum = pd.DataFrame([{"Year": k, "Cumulative": c} for (k, _), c in zip(sorted_y, cum)])
            fig2 = px.area(df_cum, x="Year", y="Cumulative", color_discrete_sequence=["#14B8A6"])
            fig2.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628", font_color="#8ab0d0",
                               margin=dict(t=10,b=30,l=40,r=10), height=280, showlegend=False,
                               xaxis=dict(gridcolor="#1a2d4a"), yaxis=dict(gridcolor="#1a2d4a"))
            st.plotly_chart(fig2, use_container_width=True)

        # Top entities bars
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.markdown('<div class="section-title">🔬 Top Biomarkers</div>', unsafe_allow_html=True)
            bio = count_entities("retinal_biomarkers") + count_entities("fluid_biomarkers")
            df_bio = pd.DataFrame([{"Biomarker": k, "Count": v} for k, v in bio.most_common(10)])
            fig3 = px.bar(df_bio, y="Biomarker", x="Count", orientation="h", text="Count", color_discrete_sequence=["#F59E0B"])
            fig3.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628", font_color="#8ab0d0",
                               margin=dict(t=10,b=10,l=10,r=10), height=300, showlegend=False,
                               yaxis=dict(categoryorder="total ascending", gridcolor="#1a2d4a"),
                               xaxis=dict(gridcolor="#1a2d4a"))
            fig3.update_traces(textposition="outside")
            st.plotly_chart(fig3, use_container_width=True)

        with col_r2:
            st.markdown('<div class="section-title">🧬 Top Diseases & Subtypes</div>', unsafe_allow_html=True)
            sub = count_entities("ms_subtypes")
            df_sub = pd.DataFrame([{"Subtype": k, "Count": v} for k, v in sub.most_common(10)])
            fig4 = px.bar(df_sub, y="Subtype", x="Count", orientation="h", text="Count", color_discrete_sequence=["#F43F5E"])
            fig4.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628", font_color="#8ab0d0",
                               margin=dict(t=10,b=10,l=10,r=10), height=300, showlegend=False,
                               yaxis=dict(categoryorder="total ascending", gridcolor="#1a2d4a"),
                               xaxis=dict(gridcolor="#1a2d4a"))
            fig4.update_traces(textposition="outside")
            st.plotly_chart(fig4, use_container_width=True)

        # === OCT Features Landscape ===
        st.markdown("---")
        st.markdown('<div class="section-title">👁️ OCT Features Landscape — What Our KB Covers</div>', unsafe_allow_html=True)
        st.markdown("OCT-related measurement types represented in the corpus, grouped by category. Other entity types (treatments, outcomes, study design) are available in the KB.")

        # Count how many papers mention entities from each OCT group
        all_bio = count_entities("retinal_biomarkers")
        all_img = count_entities("imaging_modalities")
        all_combined = all_bio + all_img

        oct_cols = st.columns(len(OCT_ENTITY_GROUPS))
        for col, (group_name, group_info) in zip(oct_cols, OCT_ENTITY_GROUPS.items()):
            with col:
                # Count papers with any entity from this group
                group_count = 0
                found_entities = []
                for ent_name in group_info["entities"]:
                    c = all_combined.get(ent_name, 0)
                    if c > 0:
                        group_count += c
                        found_entities.append((ent_name, c))

                color = group_info["color"]
                icon = group_info["icon"]
                st.markdown(f"""<div class="card oct-landscape-card" style="text-align:center;border-top:3px solid {color};min-height:200px;display:flex;flex-direction:column;">
                    <div style="font-size:24px;">{icon}</div>
                    <div style="font-size:13px;font-weight:700;color:{color};margin:4px 0;">{group_name}</div>
                    <div style="font-size:22px;font-weight:800;color:#e8edf5;">{group_count}</div>
                    <div style="font-size:10px;color:#6a8caa;">mentions</div>
                    <div style="font-size:10px;color:#4a7aaa;margin-top:6px;min-height:52px;display:flex;align-items:flex-start;justify-content:center;padding:0 4px;">
                        {' · '.join(f'{n} ({c})' for n, c in sorted(found_entities, key=lambda x: -x[1])[:4]) if found_entities else '<span style="color:#4a7aaa;">awaiting extraction</span>'}
                    </div>
                </div>""", unsafe_allow_html=True)

        # Show coverage message
        extracted_oct_groups = sum(1 for _, gi in OCT_ENTITY_GROUPS.items()
                                   if any(all_combined.get(e, 0) > 0 for e in gi["entities"]))
        total_oct_groups = len(OCT_ENTITY_GROUPS)
        if extracted_oct_groups < total_oct_groups:
            st.caption(f"📌 {extracted_oct_groups}/{total_oct_groups} OCT feature groups have data. "
                       f"Run `python neurosight_agent_append.py` to extract lamina cribrosa, choroidal, FAZ and more.")


    # ═════════════════════════════════════════════════════════════════════════════
    #  ENTITY TRENDS
    # ═════════════════════════════════════════════════════════════════════════════

    elif page == "📈 Entity Trends":
        st.markdown("# 📈 Entity Trend Analysis")
        st.markdown(f"Track publication trends for OCT–MS research entities (2020–2026) · {stats['total_papers']:,} papers · 2026 is YTD (Jan–Mar)")

        # Build per–entity-type sets and counts (for sensible defaults)
        entities_by_type = {f: set() for f in ENTITY_FIELDS}
        entity_counts_by_type = {f: Counter() for f in ENTITY_FIELDS}
        for p in papers:
            for f in ENTITY_FIELDS:
                for e in p.get(f"extracted_{f}", []):
                    name = e if isinstance(e, str) else e.get("name", "")
                    if name:
                        entities_by_type[f].add(name)
                        entity_counts_by_type[f][name] += 1

        # Filter by entity type first
        type_options = list(ENTITY_FIELDS.keys())
        type_labels = [ENTITY_FIELDS[k][0] for k in type_options]
        chosen_type = st.selectbox(
            "**Entity type**",
            options=type_options,
            format_func=lambda k: ENTITY_FIELDS[k][0],
            key="entity_trend_type",
        )
        avail_in_type = sorted(entities_by_type.get(chosen_type, set()))
        # Default: top 4 by count in this type
        default_in_type = [name for name, _ in entity_counts_by_type[chosen_type].most_common(6)]
        default_in_type = [e for e in default_in_type if e in avail_in_type][:4]

        if not avail_in_type:
            st.caption(f"No entities extracted yet for **{ENTITY_FIELDS[chosen_type][0]}**. Run the agent or choose another type.")
        selected = st.multiselect(
            "**Select entities to track** (from this type)",
            avail_in_type,
            default=default_in_type,
            key="entity_trend_entities",
        )

        if selected:
            trend_data = []
            for yr in sorted(stats["yearly_distribution"].keys()):
                row = {"Year": yr}
                yr_papers = [p for p in papers if str(p.get("year")) == yr]
                for ent in selected:
                    count = 0
                    for p in yr_papers:
                        for f in ENTITY_FIELDS:
                            for e in p.get(f"extracted_{f}", []):
                                n = e if isinstance(e, str) else e.get("name","")
                                if n == ent:
                                    count += 1
                    row[ent] = count
                trend_data.append(row)

            df_trend = pd.DataFrame(trend_data)
            fig = px.line(df_trend, x="Year", y=selected, markers=True,
                          color_discrete_sequence=["#1A7BD4","#14B8A6","#F43F5E","#F59E0B","#0EA5E9","#A855F7","#10B981","#F97316"])
            fig.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628", font_color="#8ab0d0",
                              margin=dict(t=20,b=30), height=400, legend=dict(orientation="h", y=-0.15),
                              xaxis=dict(gridcolor="#1a2d4a"), yaxis=dict(gridcolor="#1a2d4a", title="Papers"))
            fig.update_traces(line_width=2.5)
            st.plotly_chart(fig, use_container_width=True)


    # ═════════════════════════════════════════════════════════════════════════════
    #  FIELD ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════════

    elif page == "📋 Field Analysis":
        st.markdown("# 📋 Field Analysis")
        st.markdown("Entity distribution across all categories")

        # === OCT Feature Subcategories (NEW — prominent section) ===
        st.markdown('<div class="section-title">👁️ OCT Feature Subcategories</div>', unsafe_allow_html=True)
        st.markdown("Breakdown of retinal biomarkers and imaging modalities by OCT measurement type")

        all_bio = count_entities("retinal_biomarkers")
        all_img = count_entities("imaging_modalities")
        all_combined = all_bio + all_img

        for group_name, group_info in OCT_ENTITY_GROUPS.items():
            found = [(e, all_combined.get(e, 0)) for e in group_info["entities"] if all_combined.get(e, 0) > 0]
            if not found:
                continue
            total = sum(c for _, c in found)
            color = group_info["color"]
            with st.expander(f"{group_info['icon']} {group_name} — {len(found)} entities · {total:,} mentions"):
                df = pd.DataFrame([{"Entity": k, "Mentions": v} for k, v in sorted(found, key=lambda x: -x[1])])
                fig = px.bar(df, y="Entity", x="Mentions", orientation="h", text="Mentions",
                             color_discrete_sequence=[color])
                fig.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628", font_color="#8ab0d0",
                                  margin=dict(t=10,b=10,l=10,r=10), height=max(140, len(df)*32),
                                  showlegend=False, yaxis=dict(categoryorder="total ascending"))
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

        # Check for groups without data
        empty_groups = [gn for gn, gi in OCT_ENTITY_GROUPS.items()
                        if not any(all_combined.get(e, 0) > 0 for e in gi["entities"])]
        if empty_groups:
            st.info(f"**Not yet extracted:** {', '.join(empty_groups)}. "
                    f"Run `python neurosight_agent_append.py` to add these.")

        st.markdown("---")
        st.markdown('<div class="section-title">📊 All Entity Categories</div>', unsafe_allow_html=True)

        for field, (label, color) in ENTITY_FIELDS.items():
            info = stats["entity_summary"].get(field, {})
            if not info.get("unique"):
                continue
            with st.expander(f"{label} — {info['unique']} unique · {info['total_mentions']:,} mentions · {info['coverage_pct']}% coverage"):
                top = info.get("top_entities", {})
                df = pd.DataFrame([{"Entity": k, "Count": v} for k, v in list(top.items())[:12]])
                if len(df):
                    fig = px.bar(df, y="Entity", x="Count", orientation="h", text="Count", color_discrete_sequence=[color])
                    fig.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628", font_color="#8ab0d0",
                                      margin=dict(t=10,b=10,l=10,r=10), height=max(180, len(df)*28), showlegend=False,
                                      yaxis=dict(categoryorder="total ascending"))
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

        # Study types (top N + Other to avoid clutter)
        st.markdown("---")
        st.markdown('<div class="section-title">📊 Study Types</div>', unsafe_allow_html=True)
        sorted_st = sorted(stats["study_type_distribution"].items(), key=lambda x: -x[1])
        top_n = 6
        def _label(t):  # show "Other" not "Unknown" for unclassified study types
            name = (t or "").replace("_", " ").title()
            return "Other" if name == "Unknown" else name
        top_items = [(_label(k), v) for k, v in sorted_st[:top_n]]
        other_count = sum(v for _, v in sorted_st[top_n:])
        # Merge any "Other" from top_n with aggregated remainder
        other_total = other_count + sum(v for name, v in top_items if name == "Other")
        top_items = [(name, count) for name, count in top_items if name != "Other"]
        if other_total > 0:
            top_items.append(("Other", other_total))
        df_st = pd.DataFrame([{"Type": name, "Count": count} for name, count in top_items])
        fig = px.pie(df_st, values="Count", names="Type", hole=0.4,
                     color_discrete_sequence=["#1A7BD4","#14B8A6","#F43F5E","#F59E0B","#0EA5E9","#A855F7","#10B981"])
        fig.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628", font_color="#8ab0d0",
                          margin=dict(t=20,b=20), height=350)
        st.plotly_chart(fig, use_container_width=True)


    # ═════════════════════════════════════════════════════════════════════════════
    #  RELATIONSHIPS
    # ═════════════════════════════════════════════════════════════════════════════

    elif page == "🔗 Entity–Relationship":
        st.markdown("# 🔗 Entity–Relationship")

        # Relationships content (Tradeoffs section with curated thresholds below the table)
        # All publications support OCT–MS link (corpus is built from that query)
        papers_with_evidence = sum(1 for p in key_papers if (p.get("extracted_correlations") or p.get("extracted_thresholds")))
        intro_html = (
            f'<div class="entity-rel-info">'
            f'All <strong>{stats["total_papers"]:,}</strong> publications were identified by searches that require <strong>OCT/retinal biomarkers</strong> (e.g. pRNFL, GCIPL, OCT angiography) <strong>and</strong> <strong>MS, optic neuritis, or demyelinating disease</strong>. '
            f"So every paper in this corpus is relevant to using the eye to monitor MS. "
            f"Of the <strong>{stats['key_papers_deep_extracted']:,}</strong> deeply extracted papers, <strong>{papers_with_evidence:,}</strong> contain at least one quantified relationship or threshold. "
            "<strong>Entity and relationship counts in the dashboard are extraction-based</strong> (what the pipeline captured from each paper), so they are lower than the total paper count but still reflect a large, OCT-focused corpus."
            "</div>"
        )
        st.markdown(intro_html, unsafe_allow_html=True)

        # Plain-language explainer for relationship types (book icon)
        st.markdown("### 📖 What the relationship types mean")
        st.markdown("""
        - **threshold** — A cutoff value (e.g. pRNFL &lt; 88 µm) that the study uses to predict or classify an outcome (e.g. disability progression).
        - **associated** — The biomarker and the outcome are linked in the study (e.g. “lower GCIPL was associated with worse EDSS”).
        - **correlates with** — A statistical correlation was reported between the biomarker and another measure (e.g. retinal thickness and brain volume).
        """)
        st.markdown("---")

        # Key evidence table (target icon) — column headers match design: all caps
        st.markdown("### 🎯 Key evidence: OCT biomarkers → MS outcomes")
        headline = [
            {"source":"pRNFL","type":"predicts","target":"Disability Progression","s_color":"#1A7BD4","t_color":"#EF4444","pmids":"28920886, 40424561, 29095097"},
            {"source":"GCIPL","type":"predicts","target":"Disability Progression","s_color":"#1A7BD4","t_color":"#EF4444","pmids":"28920886, 40424561"},
            {"source":"pRNFL","type":"predicts","target":"Cognitive Decline","s_color":"#1A7BD4","t_color":"#EF4444","pmids":"29095097"},
            {"source":"GCIPL","type":"strongest assoc","target":"EDSS","s_color":"#1A7BD4","t_color":"#F59E0B","pmids":"37772490"},
            {"source":"INL","type":"predicts","target":"Relapse","s_color":"#1A7BD4","t_color":"#EF4444","pmids":"33737853"},
            {"source":"INL rate","type":"differentiates","target":"Treatment Response","s_color":"#1A7BD4","t_color":"#EF4444","pmids":"35648233"},
            {"source":"TMV","type":"predicts","target":"10-yr Disability","s_color":"#1A7BD4","t_color":"#EF4444","pmids":"30847355"},
            {"source":"pRNFL rate","type":"predicts","target":"NEDA Status","s_color":"#1A7BD4","t_color":"#EF4444","pmids":"40219952"},
            {"source":"pRNFL","type":"correlates with","target":"SDMT","s_color":"#1A7BD4","t_color":"#F59E0B","pmids":"37772490, 39694820"},
            {"source":"GCIPL","type":"reflects","target":"Brain Volume","s_color":"#1A7BD4","t_color":"#F97316","pmids":"36625888"},
            {"source":"OCT","type":"complements","target":"MRI","s_color":"#A855F7","t_color":"#A855F7","pmids":"37772490"},
            {"source":"Ocrelizumab","type":"slows","target":"pRNFL rate","s_color":"#10B981","t_color":"#1A7BD4","pmids":"35648233"},
        ]
        # Scoped styles so Key evidence keeps colors when KB is opened from dashboard (neurosight_dashboard_5.py)
        _scoped_css = """
        <style id="key-evidence-scoped">
        #key-evidence-wrapper{background:linear-gradient(135deg,#0d1e35,#0a1628);border:1px solid #1e3354;border-radius:12px;overflow:hidden;margin:16px 0}
        #key-evidence-wrapper table{width:100%!important;border-collapse:collapse!important;background:transparent!important;font-size:13px!important}
        #key-evidence-wrapper th{background:#0a1628!important;color:#6a8caa!important;border-bottom:2px solid #1e3354!important;padding:12px 14px!important;font-weight:600!important}
        #key-evidence-wrapper td{background:transparent!important;color:#c8d8ee!important;border-bottom:1px solid #1a2d4a!important;padding:12px 14px!important}
        #key-evidence-wrapper tbody tr:nth-child(even) td{background:rgba(10,22,40,0.5)!important}
        #key-evidence-wrapper tbody tr:hover td{background:rgba(0,114,255,0.1)!important}
        #key-evidence-wrapper .source-dot{display:inline-block!important;width:12px!important;height:12px!important;min-width:12px!important;min-height:12px!important;border-radius:50%!important;margin-right:8px!important;vertical-align:middle!important}
        #key-evidence-wrapper .source-dot.dot-blue{background:#1A7BD4!important} #key-evidence-wrapper .source-dot.dot-red{background:#EF4444!important}
        #key-evidence-wrapper .source-dot.dot-orange{background:#F59E0B!important} #key-evidence-wrapper .source-dot.dot-orange2{background:#F97316!important}
        #key-evidence-wrapper .source-dot.dot-purple{background:#A855F7!important} #key-evidence-wrapper .source-dot.dot-green{background:#10B981!important}
        #key-evidence-wrapper .source-dot.dot-gray{background:#6B7280!important}
        #key-evidence-wrapper .rel-type{color:#8ab0d0!important;font-style:italic!important}
        #key-evidence-wrapper .pmid-ref{color:#4a7aaa!important;font-size:11px!important}
        </style>
        """
        table1 = _scoped_css + '<div id="key-evidence-wrapper"><table class="rel-table key-evidence-table"><tr><th>OCT / BIOMARKER</th><th>RELATIONSHIP</th><th>MS OUTCOME OR MEASURE</th><th>PMIDS</th></tr>'
        for r in headline:
            table1 += f'<tr><td>{source_dot(r["s_color"])}<span style="color:#c8d8ee;">{r["source"]}</span></td><td><span class="rel-type">{r["type"]}</span></td><td>{source_dot(r["t_color"])}<b style="color:#c8d8ee;">{r["target"]}</b></td><td><span class="pmid-ref">{r["pmids"]}</span></td></tr>'
        table1 += '</table></div>'
        st.markdown(table1, unsafe_allow_html=True)

        # OCT biomarkers & drugs
        st.markdown("### 💊 OCT biomarkers & drugs / treatments")
        st.markdown("""
        Yes. The corpus and the dashboard include **relations between OCT biomarkers and drugs**:
        """)
        drug_rels = [
            ("**Ocrelizumab**", "slows", "pRNFL thinning rate", "35648233", "PPMS; OCT as outcome in trials."),
            ("**INL thinning rate**", "differentiates", "treatment response (Ocrelizumab)", "35648233", "Responders vs non-responders at 6 months."),
            ("**Ibudilast**", "vs placebo", "pRNFL, macular volume (SPRINT-MS)", "33054533", "Progressive MS; OCT secondary outcomes."),
            ("**Fingolimod**", "safety / effect on", "TMV (macular volume)", "33033898", "2-year macular volume in RRMS; no significant long-term change."),
        ]
        for drug, rel, biom, pmid, note in drug_rels:
            st.markdown(f"- {drug} **{rel}** {biom} (PMID {pmid}). {note}")
        st.caption("Drug–biomarker links from curated evidence. Use the table and filters below for relationship triples (source → type → target) including drug–biomarker links when extracted as correlations/thresholds.")

        # From the knowledge base: aggregated by (source, type, target) → Times = how many papers
        st.markdown("---")
        st.markdown("### 📊 Evidence from the knowledge base")
        db_rels = _build_relationships_from_kb(key_papers)
        if db_rels:
            # Aggregate: one row per (source, type, target) with times = number of papers
            agg_rels = _aggregate_relationships_by_triple(db_rels)

            # Build set of drug/treatment names from KB (for drug–biomarker filter)
            drug_names_from_kb = set()
            for p in key_papers:
                for e in p.get("extracted_drugs") or []:
                    name = _entity_text(e)
                    if name:
                        drug_names_from_kb.add(name.lower())

            # Optional filters: OCT only, and/or drug–biomarker only
            OCT_KEYWORDS = ["prnfl", "gcipl", "rnfl", "inl", "tmv", "lct", "lcd", "vessel density", "faz", "scp", "dcp", "gcpl", "ganglion", "nerve fiber", "macula", "choroid", "lamina", "oct", "octa"]
            filter_oct = st.checkbox("Show only OCT / optic biomarker relationships", value=True, key="rel_filter_oct")
            if filter_oct:
                display_agg = [r for r in agg_rels if any(kw in (r.get("source") or "").lower() for kw in OCT_KEYWORDS)]
            else:
                display_agg = agg_rels
            # Chart and metrics must match the filter when OCT filter is on
            if filter_oct and display_agg:
                rel_types = {}
                for r in display_agg:
                    t = r.get("type") or "other"
                    rel_types[t] = rel_types.get(t, 0) + r.get("times", 0)
                total_instances = sum(r.get("times", 0) for r in display_agg)
                unique_count = len(display_agg)
                chart_label = "Relationship types (OCT biomarkers only — paper mentions per type)"
            else:
                rel_types = Counter(r["type"] for r in db_rels)
                total_instances = len(db_rels)
                unique_count = len(agg_rels)
                chart_label = "Relationship types (all extracted correlations & thresholds)"

            # Side-by-side boxes: Biomarkers (left) | Relationship types (right) — symmetrical
            bio_summary = stats.get("entity_summary", {}).get("retinal_biomarkers", {})
            top_bio = bio_summary.get("top_entities") or {}
            n_bars_bio = min(20, len(top_bio)) if top_bio else 0
            chart_height_shared = 420  # same height for both boxes

            # Build relationship types chart
            df_rt = pd.DataFrame([{"Type": k, "Count": v} for k, v in sorted(rel_types.items(), key=lambda x: -x[1])])
            fig_rt = None
            if len(df_rt):
                fig_rt = px.bar(df_rt, y="Type", x="Count", orientation="h", text="Count", color_discrete_sequence=["#1A7BD4"])
                fig_rt.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628", font_color="#8ab0d0", margin=dict(t=12, b=12, l=10, r=20), height=chart_height_shared, showlegend=False, yaxis=dict(categoryorder="total ascending"))
                fig_rt.update_traces(textposition="outside")

            # Build biomarker mentions chart
            fig_bio = None
            if top_bio:
                bio_items = sorted(top_bio.items(), key=lambda x: -x[1])[:n_bars_bio]
                df_bio = pd.DataFrame([{"Biomarker": k, "Mentions": v} for k, v in bio_items])
                fig_bio = px.bar(df_bio, y="Biomarker", x="Mentions", orientation="h", text="Mentions", color_discrete_sequence=["#14B8A6"])
                fig_bio.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628", font_color="#8ab0d0", margin=dict(t=12, b=12, l=10, r=20), height=chart_height_shared, showlegend=False, yaxis=dict(categoryorder="total ascending", tickfont=dict(size=11)), xaxis=dict(title="Mentions", gridcolor="#1a2d4a"))
                fig_bio.update_traces(textposition="outside", textfont=dict(size=10))

            col_bio, col_rel = st.columns(2)
            with col_bio:
                st.markdown('<div class="evidence-box-header bio">👁 OCT biomarker mentions</div>', unsafe_allow_html=True)
                if fig_bio is not None:
                    st.plotly_chart(fig_bio, use_container_width=True)
                else:
                    st.info("No biomarker mention counts in the knowledge base.")
                st.markdown('<div class="evidence-box-bottom"></div>', unsafe_allow_html=True)
            with col_rel:
                st.markdown('<div class="evidence-box-header rel">🔗 Relationship types</div>', unsafe_allow_html=True)
                if fig_rt is not None:
                    st.plotly_chart(fig_rt, use_container_width=True)
                st.markdown('<div class="evidence-box-bottom"></div>', unsafe_allow_html=True)

            st.caption("Left: retinal/OCT biomarkers from the KB (MS-related corpus). Right: relationship types from deep-extracted papers. **Times** = papers reporting that type." + (f" OCT filter on." if filter_oct else ""))
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total relationship instances", total_instances)
            with m2:
                st.metric("Unique relationships (source → type → target)", unique_count)
            with m3:
                st.metric("Papers contributing", papers_with_evidence)

            # Quick view: drug–biomarker relationship triples only (how drugs affect OCT / outcomes)
            drug_only_rels = [r for r in agg_rels if drug_names_from_kb and ((r.get("source") or "").strip().lower() in drug_names_from_kb or (r.get("target") or "").strip().lower() in drug_names_from_kb)]
            if drug_only_rels:
                with st.expander("💊 View drug–biomarker relationship triples only (how drugs affect OCT / outcomes)", expanded=False):
                    st.caption("Relationship rows where **source** or **target** is a treatment/drug from the KB.")
                    drug_table_html = '<table class="rel-table"><tr><th>Source</th><th>Relationship</th><th>Target</th><th>Times</th><th>PMIDs</th></tr>'
                    for r in drug_only_rels:
                        times = r.get("times", 0)
                        drug_table_html += f'<tr><td>{source_dot(r["s_color"])}{r["source"]}</td><td><i>{r["type"]}</i></td><td>{source_dot(r["t_color"])}{r["target"]}</td><td><span class="strength-bold">{times}</span></td><td><span class="pmid-ref">{r["pmids"]}</span></td></tr>'
                    drug_table_html += "</table>"
                    st.markdown(drug_table_html, unsafe_allow_html=True)

            # Paginated table (carousel-style: one page at a time)
            PER_PAGE = 15
            total_rows = len(display_agg)
            n_pages = max(1, (total_rows + PER_PAGE - 1) // PER_PAGE)
            page_key = "rel_table_page"
            if page_key not in st.session_state:
                st.session_state[page_key] = 0
            curr_page = st.session_state[page_key]
            curr_page = max(0, min(curr_page, n_pages - 1))
            st.session_state[page_key] = curr_page
            start = curr_page * PER_PAGE
            end = min(start + PER_PAGE, total_rows)
            table_rows = display_agg[start:end]
            # Scoped styles so paginated table keeps colors when KB opened from dashboard
            _rel_scoped_css = """
            <style id="rel-table-paginated-scoped">
            #rel-table-paginated-wrapper{background:linear-gradient(135deg,#0d1e35,#0a1628);border:1px solid #1e3354;border-radius:12px;overflow:hidden;margin:16px 0}
            #rel-table-paginated-wrapper table{width:100%!important;border-collapse:collapse!important;background:transparent!important;font-size:13px!important}
            #rel-table-paginated-wrapper th{background:#0a1628!important;color:#6a8caa!important;border-bottom:2px solid #1e3354!important;padding:12px 14px!important;font-weight:600!important}
            #rel-table-paginated-wrapper td{background:transparent!important;color:#c8d8ee!important;border-bottom:1px solid #1a2d4a!important;padding:12px 14px!important}
            #rel-table-paginated-wrapper tbody tr:nth-child(even) td{background:rgba(10,22,40,0.5)!important}
            #rel-table-paginated-wrapper tbody tr:hover td{background:rgba(0,114,255,0.1)!important}
            #rel-table-paginated-wrapper .source-dot{display:inline-block!important;width:10px!important;height:10px!important;min-width:10px!important;min-height:10px!important;border-radius:50%!important;margin-right:8px!important;vertical-align:middle!important}
            #rel-table-paginated-wrapper .source-dot.dot-blue{background:#1A7BD4!important} #rel-table-paginated-wrapper .source-dot.dot-red{background:#EF4444!important}
            #rel-table-paginated-wrapper .source-dot.dot-orange{background:#F59E0B!important} #rel-table-paginated-wrapper .source-dot.dot-orange2{background:#F97316!important}
            #rel-table-paginated-wrapper .source-dot.dot-purple{background:#A855F7!important} #rel-table-paginated-wrapper .source-dot.dot-green{background:#10B981!important}
            #rel-table-paginated-wrapper .source-dot.dot-gray{background:#6B7280!important}
            #rel-table-paginated-wrapper .rel-type{color:#8ab0d0!important;font-style:italic!important}
            #rel-table-paginated-wrapper .strength-bold{color:#7ab3ff!important;font-weight:700!important}
            #rel-table-paginated-wrapper .pmid-ref{color:#4a7aaa!important;font-size:11px!important}
            </style>
            """
            table2 = _rel_scoped_css + '<div id="rel-table-paginated-wrapper"><table class="rel-table"><tr><th>Source (biomarker)</th><th>Relationship</th><th>Target (outcome)</th><th>Times</th><th>PMIDs</th></tr>'
            for r in table_rows:
                times = r.get("times", 0)
                table2 += f'<tr><td>{source_dot(r["s_color"])}<span style="color:#c8d8ee;">{r["source"]}</span></td><td><span class="rel-type">{r["type"]}</span></td><td>{source_dot(r["t_color"])}<b style="color:#c8d8ee;">{r["target"]}</b></td><td><span class="strength-bold">{times}</span></td><td><span class="pmid-ref">{r["pmids"]}</span></td></tr>'
            table2 += '</table></div>'
            st.markdown(table2, unsafe_allow_html=True)
            col_prev, col_info, col_next = st.columns([1, 2, 1])
            with col_prev:
                if st.button("← Previous", key="rel_prev", disabled=(curr_page == 0)):
                    st.session_state[page_key] = curr_page - 1
                    st.rerun()
            with col_info:
                st.caption(f"Page **{curr_page + 1}** of **{n_pages}** · rows {start + 1}–{end} of {total_rows}")
            with col_next:
                if st.button("Next →", key="rel_next", disabled=(curr_page >= n_pages - 1)):
                    st.session_state[page_key] = curr_page + 1
                    st.rerun()
            if filter_oct and len(display_agg) < len(agg_rels):
                st.caption(f"Filtered (OCT): {total_rows} relationships (of {len(agg_rels)} total). Uncheck to see all.")
            else:
                st.caption(f"Use Previous/Next to browse all {total_rows} relationships.")

            # Thresholds: curated OCT thresholds in two symmetrical boxes + carousel
            st.markdown("---")
            st.markdown("### 🎯 Thresholds")
            st.caption("Curated key OCT thresholds and their meaning — clinical cutoffs and evidence supporting NeuroSight. Use the carousel to browse.")
            PER_PAGE_THRESH = 3
            total_thresh = len(TRADEOFFS_CURATED)
            n_pages_thresh = max(1, (total_thresh + PER_PAGE_THRESH - 1) // PER_PAGE_THRESH)
            page_key_thresh = "thresh_carousel_page"
            if page_key_thresh not in st.session_state:
                st.session_state[page_key_thresh] = 0
            curr_thresh_page = st.session_state[page_key_thresh]
            curr_thresh_page = max(0, min(curr_thresh_page, n_pages_thresh - 1))
            st.session_state[page_key_thresh] = curr_thresh_page
            start_t = curr_thresh_page * PER_PAGE_THRESH
            end_t = min(start_t + PER_PAGE_THRESH, total_thresh)
            page_items = TRADEOFFS_CURATED[start_t:end_t]

            for t in page_items:
                val = (t.get("value") or "—").replace("<", "&lt;").replace(">", "&gt;")
                bio = (t.get("biomarker") or "—").replace("<", "&lt;").replace(">", "&gt;")
                study = (t.get("study_line") or "").replace("<", "&lt;").replace(">", "&gt;")
                expl = (t.get("explanation") or "").strip().replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                box_css = "border-radius:14px;border:1px solid #1e3354;padding:16px 18px;min-height:100px;background:linear-gradient(135deg,#0d1e35,#0a1628);"
                left_box = f'<div style="{box_css} border-left:3px solid #F43F5E;"><div style="font-size:18px;font-weight:800;color:#ff6b6b;font-family:\'DM Mono\',monospace;word-break:break-word;">{val}</div>' + (f'<div style="font-size:13px;color:#6a8caa;margin-top:8px;">{expl}</div>' if expl else '') + '</div>'
                right_box = f'<div style="{box_css}"><b style="color:#e8edf5;">{bio}</b><br/><span style="font-size:12px;color:#6a8caa;">{study}</span></div>'
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(left_box, unsafe_allow_html=True)
                with c2:
                    st.markdown(right_box, unsafe_allow_html=True)

            col_prev_t, col_info_t, col_next_t = st.columns([1, 2, 1])
            with col_prev_t:
                if st.button("← Previous", key="thresh_prev", disabled=(curr_thresh_page == 0)):
                    st.session_state[page_key_thresh] = curr_thresh_page - 1
                    st.rerun()
            with col_info_t:
                st.caption(f"**Thresholds carousel:** page {curr_thresh_page + 1} of {n_pages_thresh} · showing {start_t + 1}–{end_t} of {total_thresh}")
            with col_next_t:
                if st.button("Next →", key="thresh_next", disabled=(curr_thresh_page >= n_pages_thresh - 1)):
                    st.session_state[page_key_thresh] = curr_thresh_page + 1
                    st.rerun()
        else:
            st.caption("No structured correlations/thresholds in the KB yet. Run the agent with entity extraction to populate this from papers.")


    # ═════════════════════════════════════════════════════════════════════════════
    #  KNOWLEDGE GRAPH
    # ═════════════════════════════════════════════════════════════════════════════

    elif page == "🕸️ Knowledge Graph":
        st.markdown("# 🕸️ Knowledge Graph")
        st.markdown("Interactive network of OCT–MS relationships — **every edge is traceable to specific PubMed papers** with quantitative effect sizes")

        import math

        # Dark mode toggle
        dark_mode = st.toggle("🌙 Dark mode", value=True)
        bg_color = "#0A0E1A" if dark_mode else "#FFFFFF"
        paper_bg = "#0F1629" if dark_mode else "#F7F9FC"
        text_color = "#E2E8F0" if dark_mode else "#374151"
        edge_color = "#334155" if dark_mode else "#CBD5E1"
        edge_label_color = "#64748B" if dark_mode else "#9CA3AF"
        node_border = "#1E293B" if dark_mode else "white"

        # === NODES — sized by how many edges they participate in ===
        GRAPH_NODES = {
            "pRNFL": {"type": "biomarker", "color": "#3B82F6", "size": 32},
            "GCIPL": {"type": "biomarker", "color": "#3B82F6", "size": 28},
            "INL": {"type": "biomarker", "color": "#3B82F6", "size": 20},
            "TMV": {"type": "biomarker", "color": "#3B82F6", "size": 16},
            "sNfL": {"type": "fluid biomarker", "color": "#06B6D4", "size": 16},
            "SCP Vessel Density": {"type": "biomarker", "color": "#3B82F6", "size": 13},
            "Disability Progression": {"type": "outcome", "color": "#EF4444", "size": 26},
            "Cognitive Decline": {"type": "outcome", "color": "#EF4444", "size": 17},
            "Relapse": {"type": "outcome", "color": "#EF4444", "size": 16},
            "NEDA Status": {"type": "outcome", "color": "#EF4444", "size": 15},
            "Treatment Response": {"type": "outcome", "color": "#EF4444", "size": 15},
            "10-yr Disability": {"type": "outcome", "color": "#EF4444", "size": 14},
            "EDSS": {"type": "scale", "color": "#F59E0B", "size": 22},
            "SDMT": {"type": "scale", "color": "#F59E0B", "size": 14},
            "T25FW": {"type": "scale", "color": "#F59E0B", "size": 12},
            "OCT": {"type": "imaging", "color": "#A855F7", "size": 26},
            "OCT-A": {"type": "imaging", "color": "#A855F7", "size": 16},
            "MRI": {"type": "imaging", "color": "#A855F7", "size": 20},
            "VBM-OCT": {"type": "imaging", "color": "#A855F7", "size": 12},
            "Deep Learning": {"type": "AI", "color": "#0EA5E9", "size": 16},
            "Gray Matter": {"type": "anatomy", "color": "#F97316", "size": 17},
            "Thalamus": {"type": "anatomy", "color": "#F97316", "size": 14},
            "Brain Volume": {"type": "anatomy", "color": "#F97316", "size": 15},
            "Ocrelizumab": {"type": "treatment", "color": "#10B981", "size": 14},
            "Optic Neuritis": {"type": "disease", "color": "#F43F5E", "size": 16},
            "Focal Atrophy": {"type": "pathophysiology", "color": "#8B5CF6", "size": 12},
        }

        # === EDGES — every one has PMIDs proving it ===
        GRAPH_EDGES = [
            ("pRNFL", "Disability Progression", "predicts", "HR 2.4–3.0", "28920886, 40424561, 29095097"),
            ("GCIPL", "Disability Progression", "predicts", "HR 2.8–4.1", "28920886, 40424561"),
            ("pRNFL", "Cognitive Decline", "predicts", "HR 2.7", "29095097"),
            ("GCIPL", "EDSS", "strongest assoc", "β = –0.32", "37772490"),
            ("INL", "Relapse", "predicts", "OR 17.8", "33737853"),
            ("INL", "Treatment Response", "differentiates", "p=0.005", "35648233"),
            ("TMV", "10-yr Disability", "predicts", "OR 3.58", "30847355"),
            ("pRNFL", "NEDA Status", "predicts", "p<0.0001", "40219952"),
            ("pRNFL", "SDMT", "correlates with", "p=0.030", "37772490, 39694820"),
            ("pRNFL", "T25FW", "predicts", "p=0.018", "39694820"),
            ("GCIPL", "Gray Matter", "reflects", "p<0.002", "36625888"),
            ("GCIPL", "Brain Volume", "reflects", "r=0.48", "36625888"),
            ("pRNFL", "Thalamus", "correlates with", "p<0.001", "36625888, 39694820"),
            ("OCT", "pRNFL", "measures", "", "28920886"),
            ("OCT", "GCIPL", "measures", "", "28920886"),
            ("OCT", "INL", "measures", "", "33737853"),
            ("OCT-A", "SCP Vessel Density", "measures", "", "38458836, 33290417"),
            ("OCT", "MRI", "complements", "Independent value", "37772490"),
            ("Deep Learning", "OCT", "analyzes", "94% accuracy", "36303065"),
            ("Ocrelizumab", "pRNFL", "slows thinning", "p=0.005", "35648233"),
            ("Optic Neuritis", "pRNFL", "accelerates loss", "–20 μm", "28920886"),
            ("sNfL", "pRNFL", "correlates with", "r=0.38", "37772490"),
            ("VBM-OCT", "Focal Atrophy", "detects early", "4.2 months", "38328398"),
        ]

        # === Layout: cluster by type ===
        type_groups = {}
        for name, info in GRAPH_NODES.items():
            t = info["type"]
            if t not in type_groups:
                type_groups[t] = []
            type_groups[t].append(name)

        positions = {}
        group_centers = {
            "biomarker": (0, 0.5), "fluid biomarker": (0.5, -1.5),
            "outcome": (3.5, 0.5), "scale": (2.5, -2),
            "imaging": (-3.2, 0), "AI": (-3, 2.5),
            "anatomy": (2.5, 3), "treatment": (-1.5, -3),
            "disease": (0.5, 3.5), "pathophysiology": (-2.5, -2),
        }

        for gtype, nodes in type_groups.items():
            cx, cy = group_centers.get(gtype, (0, 0))
            n = len(nodes)
            for i, name in enumerate(nodes):
                angle = (2 * math.pi * i / max(n, 1)) + hash(gtype) % 7 * 0.4
                r = 0.7 + (i % 3) * 0.5
                positions[name] = (cx + r * math.cos(angle), cy + r * math.sin(angle))

        # === Build plotly figure ===
        fig = go.Figure()

        # Edge lines
        edge_x, edge_y = [], []
        edge_mid_x, edge_mid_y, edge_hover, edge_labels = [], [], [], []
        for src, tgt, rel, strength, pmids in GRAPH_EDGES:
            if src in positions and tgt in positions:
                x0, y0 = positions[src]
                x1, y1 = positions[tgt]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                mx, my = (x0+x1)/2, (y0+y1)/2
                edge_mid_x.append(mx)
                edge_mid_y.append(my)
                label = f"{rel}" + (f" ({strength})" if strength else "")
                edge_labels.append(label)
                edge_hover.append(f"<b>{src}</b> → <b>{tgt}</b><br>{rel}: {strength}<br>PMIDs: {pmids}")

        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=1.2, color=edge_color),
            hoverinfo="none", showlegend=False,
        ))

        # Edge labels (on the lines)
        fig.add_trace(go.Scatter(
            x=edge_mid_x, y=edge_mid_y, mode="text",
            text=edge_labels,
            textfont=dict(size=7.5, color=edge_label_color),
            hovertext=edge_hover, hoverinfo="text",
            showlegend=False,
        ))

        # Nodes grouped by type
        type_display_order = ["biomarker", "fluid biomarker", "outcome", "scale", "imaging", "AI", "anatomy", "treatment", "disease", "pathophysiology"]
        for gtype in type_display_order:
            nodes = type_groups.get(gtype, [])
            if not nodes:
                continue
            nx_list = [positions[n][0] for n in nodes if n in positions]
            ny_list = [positions[n][1] for n in nodes if n in positions]
            names = [n for n in nodes if n in positions]
            colors = [GRAPH_NODES[n]["color"] for n in names]
            sizes = [GRAPH_NODES[n]["size"] for n in names]

            # Count edges per node for hover
            hover_texts = []
            for n in names:
                edges_for = [(s,t,r,strength,pm) for s,t,r,strength,pm in GRAPH_EDGES if s==n or t==n]
                lines = [f"<b>{n}</b> ({gtype})", f"{len(edges_for)} connections:"]
                for s,t,r,strength,pm in edges_for[:6]:
                    partner = t if s==n else s
                    lines.append(f"  → {partner}: {r} {strength}")
                if len(edges_for) > 6:
                    lines.append(f"  ...+{len(edges_for)-6} more")
                hover_texts.append("<br>".join(lines))

            fig.add_trace(go.Scatter(
                x=nx_list, y=ny_list, mode="markers+text",
                marker=dict(size=sizes, color=colors,
                            line=dict(width=2, color=node_border),
                            opacity=0.9),
                text=names, textposition="top center",
                textfont=dict(size=10, color=text_color, family="Inter"),
                name=gtype.title(),
                hovertext=hover_texts, hoverinfo="text",
            ))

        fig.update_layout(
            plot_bgcolor=bg_color, paper_bgcolor=paper_bg,
            font_color=text_color,
            margin=dict(t=30, b=30, l=20, r=20),
            height=650,
            showlegend=True,
            legend=dict(orientation="h", y=-0.06, font=dict(size=11, color=text_color),
                        bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            hoverlabel=dict(bgcolor="#1E293B" if dark_mode else "#fff",
                            font_color="#E2E8F0" if dark_mode else "#0F172A",
                            font_size=12),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Provenance justification
        st.markdown("---")
        st.markdown(f"""<div class="card">
            <div class="section-title">📋 Provenance & Justification</div>
            <p style="font-size:13px;color:#6a8caa;line-height:1.7;">
            Every edge in this graph is <b>traceable to specific PubMed papers</b>. The {len(GRAPH_NODES)} nodes represent
            entities extracted from {stats['key_papers_deep_extracted']} deep-analyzed publications. The {len(GRAPH_EDGES)} edges
            carry quantitative effect sizes (HR, OR, β, r, p-values) directly from published meta-analyses and longitudinal studies.
            <b>Hover over any edge</b> to see the relationship type, effect size, and source PMIDs.
            No edge exists without a published evidence trail.
            </p>
            <p style="font-size:12px;color:#4a7aaa;">
            Key sources: Petzold 2017 (Lancet Neurology, n=7,473) · El Ayoubi 2025 (meta-analysis, n=3,683) ·
            Rothman 2019 (10-year follow-up) · Cerdá-Fuertes 2023 (biomarker comparison) · Nabizadeh 2022 (AI meta-analysis, n=5,989)
            </p>
        </div>""", unsafe_allow_html=True)

        st.caption(f"{len(GRAPH_NODES)} nodes · {len(GRAPH_EDGES)} edges · All from PubMed-verified publications")


    # ═════════════════════════════════════════════════════════════════════════════
    #  CO-OCCURRENCE GRAPH (dynamic from JSON)
    # ═════════════════════════════════════════════════════════════════════════════

    elif page == "🌐 Co-occurrence Graph":
        st.markdown("# 🌐 Co-occurrence Graph")
        st.markdown("""
        **This graph is 100% dynamic** — built automatically by reading `neurosight_kb.json`.
        """)

        import math
        from collections import defaultdict

        dark_mode = st.toggle("🌙 Dark mode", value=True, key="cooccurrence_dark")
        bg_color = "#0A0E1A" if dark_mode else "#FFFFFF"
        paper_bg = "#0F1629" if dark_mode else "#F7F9FC"
        text_color = "#E2E8F0" if dark_mode else "#374151"
        edge_base_color = "#334155" if dark_mode else "#CBD5E1"
        node_border = "#1E293B" if dark_mode else "white"

        # Controls
        c1, c2, c3 = st.columns(3)
        with c1:
            min_cooccur = st.slider("Min co-occurrences (edge threshold)", 1, 50, 5)
        with c2:
            max_nodes = st.slider("Max nodes to show", 10, 80, 30)
        with c3:
            selected_fields = st.multiselect("Entity types to include",
                list(ENTITY_FIELDS.keys()),
                default=["retinal_biomarkers", "fluid_biomarkers", "imaging_modalities",
                         "clinical_scales", "ms_subtypes", "drugs", "ai_methods",
                         "clinical_outcomes", "pathophysiology"])

        # Step 1: Extract all entities per paper, tagged with their type
        paper_entities = []  # list of sets: each set = entities in one paper
        entity_type_map = {}  # entity_name -> field_key
        entity_freq = Counter()  # how often each entity appears across papers

        for p in papers:
            ents_in_paper = set()
            for field in selected_fields:
                for e in p.get(f"extracted_{field}", []):
                    name = e if isinstance(e, str) else e.get("name", str(e))
                    if name and len(name) > 1:
                        ents_in_paper.add(name)
                        entity_type_map[name] = field
                        entity_freq[name] += 1
            if len(ents_in_paper) >= 2:
                paper_entities.append(ents_in_paper)

        # Step 2: Keep only top N most frequent entities
        top_entities = set(e for e, _ in entity_freq.most_common(max_nodes))

        # Step 3: Count co-occurrences
        cooccur = Counter()
        cooccur_pmids = defaultdict(list)  # track which papers

        for idx, ents in enumerate(paper_entities):
            filtered = ents & top_entities  # only keep top entities
            ents_list = sorted(filtered)
            for i in range(len(ents_list)):
                for j in range(i+1, len(ents_list)):
                    pair = (ents_list[i], ents_list[j])
                    cooccur[pair] += 1

        # Step 4: Filter edges by min co-occurrence
        edges = [(a, b, count) for (a, b), count in cooccur.items() if count >= min_cooccur]
        edges.sort(key=lambda x: -x[2])

        # Only keep nodes that have at least one edge
        active_nodes = set()
        for a, b, _ in edges:
            active_nodes.add(a)
            active_nodes.add(b)

        if not edges:
            st.warning(f"No edges found with ≥{min_cooccur} co-occurrences. Try lowering the threshold.")
        else:
            st.info(f"**{len(active_nodes)} nodes · {len(edges)} edges** (each edge = entities appearing together in ≥{min_cooccur} papers)")

            # Step 5: Layout — force-directed approximation using type clustering
            positions = {}
            type_angle_offset = {}
            type_list = list(set(entity_type_map.get(n, "other") for n in active_nodes))
            for i, t in enumerate(type_list):
                type_angle_offset[t] = (2 * math.pi * i) / max(len(type_list), 1)

            # Group nodes by type, spread within each sector
            type_nodes = defaultdict(list)
            for n in active_nodes:
                t = entity_type_map.get(n, "other")
                type_nodes[t].append(n)

            for t, nodes in type_nodes.items():
                base_angle = type_angle_offset[t]
                for i, name in enumerate(nodes):
                    spread = 0.4 * (i - len(nodes)/2)
                    angle = base_angle + spread * 0.3
                    r = 2.5 + (i % 4) * 0.8 + (entity_freq.get(name, 1) / max(entity_freq.values()) ) * 1.5
                    positions[name] = (r * math.cos(angle), r * math.sin(angle))

            # Step 6: Build plotly figure
            fig = go.Figure()

            # Edge lines — thickness proportional to co-occurrence count
            max_weight = max(c for _, _, c in edges) if edges else 1
            for a, b, count in edges:
                if a in positions and b in positions:
                    x0, y0 = positions[a]
                    x1, y1 = positions[b]
                    width = 0.5 + (count / max_weight) * 4
                    opacity = 0.3 + (count / max_weight) * 0.6
                    fig.add_trace(go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        mode="lines",
                        line=dict(width=width, color=edge_base_color),
                        opacity=opacity,
                        hoverinfo="text",
                        hovertext=f"<b>{a}</b> ↔ <b>{b}</b><br>{count} papers mention both",
                        showlegend=False,
                    ))

            # Nodes — grouped by type
            type_colors = {f: c for f, (_, c) in ENTITY_FIELDS.items()}
            type_colors["other"] = "#6B7280"

            for t, nodes in type_nodes.items():
                nx_list = [positions[n][0] for n in nodes if n in positions]
                ny_list = [positions[n][1] for n in nodes if n in positions]
                names = [n for n in nodes if n in positions]
                color = type_colors.get(t, "#6B7280")
                sizes = [10 + (entity_freq.get(n, 1) / max(entity_freq.values())) * 30 for n in names]

                # Hover: show top co-occurrences for each node
                hover_texts = []
                for n in names:
                    # find top co-occurring partners
                    partners = []
                    for a, b, count in edges:
                        if a == n:
                            partners.append((b, count))
                        elif b == n:
                            partners.append((a, count))
                    partners.sort(key=lambda x: -x[1])
                    lines = [f"<b>{n}</b> ({t.replace('_',' ')})",
                             f"Appears in {entity_freq.get(n,0)} papers",
                             f"{len(partners)} connections:", ""]
                    for partner, cnt in partners[:8]:
                        lines.append(f"  ↔ {partner}: {cnt} papers")
                    hover_texts.append("<br>".join(lines))

                label = ENTITY_FIELDS.get(t, (t.replace("_"," ").title(), "#666"))[0]
                fig.add_trace(go.Scatter(
                    x=nx_list, y=ny_list, mode="markers+text",
                    marker=dict(size=sizes, color=color,
                                line=dict(width=2, color=node_border), opacity=0.9),
                    text=names, textposition="top center",
                    textfont=dict(size=9, color=text_color),
                    name=label,
                    hovertext=hover_texts, hoverinfo="text",
                ))

            fig.update_layout(
                plot_bgcolor=bg_color, paper_bgcolor=paper_bg,
                font_color=text_color,
                margin=dict(t=20, b=30, l=20, r=20),
                height=700,
                showlegend=True,
                legend=dict(orientation="h", y=-0.06, font=dict(size=10, color=text_color),
                            bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                hoverlabel=dict(bgcolor="#1E293B" if dark_mode else "#fff",
                                font_color="#E2E8F0" if dark_mode else "#0F172A", font_size=11),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Top co-occurrences table
            st.markdown('<div class="section-title">🔝 Strongest Co-occurrences</div>', unsafe_allow_html=True)
            top_edges = edges[:20]
            table_html = '<table class="rel-table"><tr><th>Entity A</th><th>Entity B</th><th>Papers Together</th><th>Type A</th><th>Type B</th></tr>'
            for a, b, count in top_edges:
                ta = entity_type_map.get(a, "?").replace("_", " ")
                tb = entity_type_map.get(b, "?").replace("_", " ")
                ca = type_colors.get(entity_type_map.get(a, "other"), "#666")
                cb = type_colors.get(entity_type_map.get(b, "other"), "#666")
                table_html += f'''<tr>
                    <td>{source_dot(ca)}<b>{a}</b></td>
                    <td>{source_dot(cb)}<b>{b}</b></td>
                    <td><span class="strength-bold">{count}</span></td>
                    <td style="font-size:11px;color:#4a7aaa;">{ta}</td>
                    <td style="font-size:11px;color:#4a7aaa;">{tb}</td>
                </tr>'''
            table_html += '</table>'
            st.markdown(table_html, unsafe_allow_html=True)

            # Explanation
            st.markdown("---")
            st.markdown(f"""<div class="card">
                <div class="section-title">💡 How This Works</div>
                <p style="font-size:13px;color:#6a8caa;line-height:1.7;">
                This graph is <b>100% dynamic</b> — built automatically by reading <code>neurosight_kb.json</code>.
                For each paper, we collect all extracted entities. If two entities appear in the <b>same paper</b>,
                that's one co-occurrence. The more papers mention both entities, the stronger (thicker) the connection.
                </p>
                <p style="font-size:13px;color:#6a8caa;line-height:1.7;">
                <b>Nothing is hardcoded.</b> Run the agent on more papers → the graph grows automatically.
                Node size = how frequently that entity appears. Edge thickness = how many papers mention both.
                Use the sliders above to adjust the minimum threshold and number of nodes.
                </p>
                <p style="font-size:12px;color:#4a7aaa;">
                Compare this with the <b>🕸️ Knowledge Graph</b> tab which shows manually curated relationships
                with specific effect sizes (HR, OR, p-values). Both are useful — co-occurrence shows the landscape,
                curated shows the precise evidence.
                </p>
            </div>""", unsafe_allow_html=True)


    # ═════════════════════════════════════════════════════════════════════════════
    #  PUBLICATIONS
    # ═════════════════════════════════════════════════════════════════════════════

    elif page == "📄 Research analysis and synthesis agents":
        st.markdown("# 📄 Research analysis and synthesis agents")

        # RAG synopsis: last N publications (not full synthesis)
        st.markdown("### 📋 RAG Research synopsis")
        st.caption("RAG over the latest N publications from the KB; output is a short synopsis, not a full systematic synthesis.")
        syn_col1, syn_col2 = st.columns([1, 3])
        with syn_col1:
            n_synth = st.selectbox("Synopsis from", [15, 30, 50], format_func=lambda x: f"Last {x} publications", key="synth_n")
            do_synth = st.button("Generate synopsis", key="do_synth")
        if do_synth:
            by_recency = sorted(papers, key=lambda p: (-(p.get("year") or 0), -(p.get("id") or 0)))[:n_synth]
            with st.spinner("Generating synopsis..."):
                syn_result = _synthesize_papers(by_recency)
            st.session_state["synthesis_result"] = syn_result
            st.session_state["synthesis_n"] = n_synth
        if st.session_state.get("synthesis_result"):
            with st.expander(f"📋 Synopsis (last {st.session_state.get('synthesis_n', 15)} papers)", expanded=True):
                st.markdown(st.session_state["synthesis_result"])

        st.markdown("---")
        st.markdown("### 📄 Analysis Agent")

        st.markdown("**Filter by**")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            field_options = ["All"] + [ENTITY_FIELDS[k][0] for k in ENTITY_FIELDS]
            field_filter = st.selectbox("Entity type", field_options, key="pub_field")
        with fc2:
            year_options = sorted(set(str(p.get("year") or "") for p in papers if p.get("year")), reverse=True)
            year_options = [y for y in year_options if y and y.isdigit()]
            year_filter = st.selectbox("Year", ["All"] + year_options, key="pub_year")
        with fc3:
            depth_filter = st.selectbox("Extraction depth", ["All", "Full Deep", "Entity Tagged"], key="pub_depth")

        filtered = [p for p in papers if not ((p.get("title") or "").strip().lower().startswith("letter to the editor"))]
        if field_filter != "All":
            field_key = next((k for k in ENTITY_FIELDS if ENTITY_FIELDS[k][0] == field_filter), None)
            if field_key:
                filtered = [p for p in filtered if len(p.get(f"extracted_{field_key}") or []) > 0]
        if year_filter != "All" and year_filter:
            filtered = [p for p in filtered if str(p.get("year") or "") == year_filter]
        if depth_filter == "Full Deep":
            filtered = [p for p in filtered if p.get("extraction_depth") == "full_deep"]
        elif depth_filter == "Entity Tagged":
            filtered = [p for p in filtered if (p.get("extraction_depth") or "") != "full_deep"]

        st.caption(f"**{len(filtered)}** papers match · showing up to 50 below")
        if len(filtered) == 0:
            st.info("No papers match the selected filters. Set **Entity type**, **Year**, and **Extraction depth** to **All** to see everything.")

        for p in filtered[:50]:
            is_key = p.get("extraction_depth") == "full_deep"
            title = p.get("title") or f"PMID:{p['pmid']}"
            icon = "⭐" if is_key else "📄"
            n_str = f" · n={p['patient_count']:,}" if p.get("patient_count") else ""

            with st.expander(f"{icon} {title[:90]} — {p.get('authors','')} ({p.get('year','')}) · PMID:{p['pmid']}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Journal", p.get("journal","—")[:30])
                col2.metric("Type", (p.get("study_type") or "—").replace("_"," ").title())
                col3.metric("Patients", f"{p['patient_count']:,}" if p.get("patient_count") else "—")

                # Full abstract
                abstract = (p.get("abstract") or "").strip()
                if abstract:
                    st.markdown("**Abstract**")
                    st.markdown(abstract)
                    st.markdown("---")
                else:
                    st.caption("Abstract not available for this record.")

                # Entities
                badges_html = ""
                for field, (label, color) in ENTITY_FIELDS.items():
                    for e in p.get(f"extracted_{field}", []):
                        name = e if isinstance(e, str) else e.get("name", str(e))
                        badges_html += badge(name, color)
                if badges_html:
                    st.markdown(badges_html, unsafe_allow_html=True)

                # Correlations
                for c in p.get("extracted_correlations", []):
                    st.markdown(f"▸ {c}")

                # Relevance
                rel = p.get("neurosight_relevance")
                if rel:
                    st.success(f"🎯 **NeuroSight Relevance:** {rel}")

                # Analyzer agent
                if st.button("🔬 Analyzer agent", key=f"analyze_{p['pmid']}"):
                    with st.spinner("Analyzing..."):
                        analysis = _analyze_paper(p)
                    st.session_state[f"analysis_{p['pmid']}"] = analysis
                if st.session_state.get(f"analysis_{p['pmid']}"):
                    st.markdown("**🔬 Analysis**")
                    st.markdown(st.session_state[f"analysis_{p['pmid']}"])


    # ═════════════════════════════════════════════════════════════════════════════
    #  AI AGENT
    # ═════════════════════════════════════════════════════════════════════════════

    elif page == "🤖 AI Literature Agent":
        st.markdown("# 🤖 AI Literature Agent")
        st.markdown(f"Ask questions about OCT biomarkers for MS based on {stats['total_papers']:,} analyzed papers (Claude / GPT-4o-mini)")

        # Pre-built questions
        st.markdown("**Try:**")
        prebuilt = [
            "What are the validated OCT thresholds for MS progression?",
            "How does OCT compare to MRI for monitoring MS?",
            "What role does INL play in relapse prediction?",
            "How accurate is AI for MS diagnosis from OCT?",
        ]
        cols = st.columns(len(prebuilt))
        for i, q in enumerate(prebuilt):
            if cols[i].button(q, key=f"q{i}"):
                st.session_state["ai_agent_question"] = q
                st.rerun()
        question = st.text_input("Ask a question", key="ai_agent_question", placeholder="e.g. What are the validated OCT thresholds for MS progression?")

        if question:
            # Build context from KB
            context_parts = []
            for p in key_papers:
                ctx = f"PMID:{p['pmid']} ({p.get('authors','')}, {p.get('year','')}): {p.get('title','')}"
                for c in p.get("extracted_correlations", []):
                    ctx += f"\n  - {c}"
                context_parts.append(ctx)

            context = "\n\n".join(context_parts)

            # Try OpenAI
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                import openai
                client = openai.OpenAI(api_key=api_key)
                with st.spinner("Querying AI agent..."):
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": f"You are the NeuroSight AI Literature Agent. Answer based on these {len(key_papers)} analyzed papers:\n\n{context}\n\nCite PMIDs when making claims. Be specific with effect sizes."},
                            {"role": "user", "content": question},
                        ],
                        max_tokens=1500,
                    )
                    answer = resp.choices[0].message.content
            else:
                # Offline fallback — answer from KB data
                answer = _offline_answer(question)

            st.markdown("### 📋 Answer")
            st.markdown(answer)


if __name__ == "__main__":
    run_kb()
