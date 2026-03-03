"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  NeuroSight Knowledge Base Explorer                                        ║
║  Streamlit Demo for EP PerMed Hackathon Judges                             ║
║                                                                            ║
║  Run: streamlit run neurosight_app.py                                      ║
║  Requirements: pip install streamlit plotly pandas                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import Counter
from pathlib import Path

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NeuroSight · Knowledge Base Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
    
    .stApp { background-color: #0A0E1A; }
    
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
        padding: 2rem 2.5rem 1.5rem;
        border-radius: 16px;
        border: 1px solid #1E293B;
        margin-bottom: 1.5rem;
    }
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #E2E8F0;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-title span { color: #3B82F6; }
    .main-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.25rem;
    }
    .thesis-box {
        background: linear-gradient(135deg, #3B82F611, #8B5CF611);
        border: 1px solid #3B82F633;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #94A3B8;
        line-height: 1.6;
    }
    
    .metric-card {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
    }
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        color: #64748B;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .paper-card {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s;
    }
    .paper-card:hover { border-color: #3B82F6; }
    .paper-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: #E2E8F0;
        line-height: 1.4;
    }
    .paper-meta {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.25rem;
    }
    
    .entity-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 2px;
    }
    
    .correlation-item {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #94A3B8;
        padding: 0.4rem 0;
        border-bottom: 1px solid #1E293B;
        line-height: 1.5;
    }
    
    .threshold-card {
        background: #111827;
        border: 1px solid #EF444433;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
    }
    .threshold-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 900;
        color: #EF4444;
    }
    .threshold-biomarker {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #E2E8F0;
    }
    
    .pipeline-stage {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .stage-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
    }
    .stage-name {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #E2E8F0;
    }
    
    .relevance-tag {
        background: linear-gradient(135deg, #10B98111, #10B98122);
        border: 1px solid #10B98144;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #10B981;
        margin-top: 0.5rem;
    }
    
    /* Streamlit overrides */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #111827;
        border-radius: 8px;
        padding: 8px 18px;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #3B82F6 !important;
        color: white !important;
    }
    div[data-testid="stExpander"] {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
    }
    .stSelectbox > div > div { background: #111827; }
    .stMultiSelect > div > div { background: #111827; }
    .stTextInput > div > div > input { background: #111827; color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)


# ─── Load Data ───────────────────────────────────────────────────────────────

@st.cache_data
def load_kb():
    """Load the knowledge base JSON."""
    # Try multiple paths
    for path in [
        Path("neurosight_kb.json"),
        Path("../neurosight_kb.json"),
        Path(__file__).parent / "neurosight_kb.json",
    ]:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    st.error("❌ Could not find neurosight_kb.json. Place it in the same directory as this script.")
    st.stop()


kb = load_kb()
meta = kb["_meta"]
stats = kb["statistics"]
papers = kb["papers"]

# Convert to list for iteration
paper_list = list(papers.values())
key_papers = [p for p in paper_list if p.get("extraction_depth") == "full_deep"]
key_papers.sort(key=lambda p: p["id"])

ENTITY_COLORS = {
    "retinal_biomarkers": "#10B981",
    "fluid_biomarkers": "#06B6D4",
    "imaging_modalities": "#3B82F6",
    "clinical_scales": "#8B5CF6",
    "ms_subtypes": "#EC4899",
    "drugs": "#F97316",
    "anatomical_structures": "#64748B",
    "ai_methods": "#F59E0B",
    "pathophysiology": "#EF4444",
    "clinical_outcomes": "#818CF8",
    "thematic_categories": "#14B8A6",
}

ENTITY_LABELS = {
    "retinal_biomarkers": "🔬 Retinal Biomarkers",
    "fluid_biomarkers": "🩸 Fluid Biomarkers",
    "imaging_modalities": "📷 Imaging Modalities",
    "clinical_scales": "📊 Clinical Scales",
    "ms_subtypes": "🧬 MS Subtypes",
    "drugs": "💊 Drugs",
    "anatomical_structures": "🧠 Anatomy",
    "ai_methods": "🤖 AI Methods",
    "pathophysiology": "⚡ Pathophysiology",
    "clinical_outcomes": "🎯 Clinical Outcomes",
    "thematic_categories": "📑 Themes",
}


def count_entity(field_name):
    """Count entities across all papers."""
    counter = Counter()
    for p in paper_list:
        for e in p.get(f"extracted_{field_name}", []):
            name = e if isinstance(e, str) else e.get("name", str(e))
            counter[name] += 1
    return counter


def make_badge_html(text, color):
    return f'<span class="entity-badge" style="background:{color}22;color:{color};border:1px solid {color}44;">{text}</span>'


# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="main-header">
    <div style="display:flex;align-items:center;gap:14px;">
        <div style="width:48px;height:48px;border-radius:14px;background:linear-gradient(135deg,#3B82F6,#8B5CF6);display:flex;align-items:center;justify-content:center;font-size:24px;">🧠</div>
        <div>
            <p class="main-title">NeuroSight <span>Knowledge Base</span> Explorer</p>
            <p class="main-subtitle">Multi-Agent AI Literature Mining · Claude Opus 4.6 · {stats['total_papers']:,} PubMed Papers</p>
        </div>
    </div>
    <div class="thesis-box">
        <strong style="color:#3B82F6;">NeuroSight Thesis:</strong> {meta['neurosight_thesis']}
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_overview, tab_pipeline, tab_papers, tab_entities, tab_thresholds, tab_correlations = st.tabs([
    "📊 Overview",
    "⚙️ Agent Pipeline",
    "📄 Papers Explorer",
    "🧬 Entity Analytics",
    "🎯 Thresholds & Evidence",
    "🔗 Correlations",
])

# ═════════════════════════════════════════════════════════════════════════════
#  TAB: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

with tab_overview:
    # Top metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        (c1, str(f"{stats['total_papers']:,}"), "Total Papers", "#3B82F6"),
        (c2, str(stats['key_papers_deep_extracted']), "Deep Extracted", "#8B5CF6"),
        (c3, str(stats['total_correlations']), "Correlations", "#F59E0B"),
        (c4, str(stats['total_thresholds']), "Thresholds", "#EF4444"),
        (c5, str(sum(s["unique"] for s in stats["entity_summary"].values())), "Unique Entities", "#10B981"),
        (c6, str(sum(s["total_mentions"] for s in stats["entity_summary"].values())), "Total Mentions", "#06B6D4"),
    ]
    for col, val, label, color in metrics:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color};">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("📈 Publication Trend")
        yearly = stats["yearly_distribution"]
        df_yearly = pd.DataFrame([
            {"Year": str(y), "Papers": c} 
            for y, c in sorted(yearly.items())
        ])
        fig = px.bar(
            df_yearly, x="Year", y="Papers",
            color_discrete_sequence=["#3B82F6"],
            text="Papers",
        )
        fig.update_layout(
            plot_bgcolor="#111827", paper_bgcolor="#0A0E1A",
            font_color="#94A3B8", font_family="Inter",
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
            margin=dict(t=20, b=40),
            height=320,
        )
        fig.update_traces(textposition="outside", textfont_size=12)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("📋 Study Types")
        study_types = stats["study_type_distribution"]
        df_types = pd.DataFrame([
            {"Type": k.replace("_", " ").title(), "Count": v}
            for k, v in sorted(study_types.items(), key=lambda x: -x[1])
        ])
        fig2 = px.pie(
            df_types, values="Count", names="Type",
            color_discrete_sequence=["#3B82F6","#8B5CF6","#EC4899","#06B6D4","#10B981","#F59E0B","#EF4444"],
            hole=0.4,
        )
        fig2.update_layout(
            plot_bgcolor="#111827", paper_bgcolor="#0A0E1A",
            font_color="#94A3B8", font_family="Inter",
            margin=dict(t=20, b=20),
            height=320,
            showlegend=True,
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Entity coverage summary
    st.subheader("🧬 Entity Coverage Across Corpus")
    entity_data = []
    for cat, info in stats["entity_summary"].items():
        entity_data.append({
            "Category": ENTITY_LABELS.get(cat, cat),
            "Unique": info["unique"],
            "Mentions": info["total_mentions"],
            "Coverage %": info["coverage_pct"],
        })
    df_ent = pd.DataFrame(entity_data)
    
    fig3 = px.bar(
        df_ent, x="Category", y="Mentions",
        color="Coverage %",
        color_continuous_scale=["#1E293B", "#3B82F6", "#10B981"],
        text="Unique",
    )
    fig3.update_layout(
        plot_bgcolor="#111827", paper_bgcolor="#0A0E1A",
        font_color="#94A3B8", font_family="Inter",
        xaxis=dict(gridcolor="#1E293B", tickangle=-30),
        yaxis=dict(gridcolor="#1E293B"),
        margin=dict(t=20, b=80),
        height=350,
    )
    fig3.update_traces(texttemplate="%{text} unique", textposition="outside", textfont_size=10)
    st.plotly_chart(fig3, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB: AGENT PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

with tab_pipeline:
    st.subheader("⚙️ Multi-Agent Extraction Pipeline")
    st.caption("5 specialized AI agents process the corpus sequentially. Claude Opus 4.6 powers entity extraction and relationship mining.")
    
    stages = [
        ("🔍", "Literature Discovery Agent", "#3B82F6",
         "Systematic PubMed search via NCBI E-utilities API. Boolean queries with date filtering, deduplication, yearly breakdown.",
         "PubMed E-utilities, esearch/efetch", f"{stats['total_papers']:,} unique PMIDs"),
        ("📋", "Metadata Extraction Agent", "#8B5CF6",
         "Batch XML fetch of title, abstract, authors, journal, DOI, MeSH terms for every PMID. 200 papers per API call.",
         "NCBI efetch XML, ElementTree", "Full bibliographic records"),
        ("🧬", "Entity Extraction Agent", "#EC4899",
         "Claude Opus 4.6 extracts 11 domain-specific entity types per paper: retinal biomarkers, fluid biomarkers, imaging modalities, clinical scales, MS subtypes, drugs, anatomy, AI methods, pathophysiology, outcomes, themes.",
         "Anthropic Messages API", f"{sum(s['total_mentions'] for s in stats['entity_summary'].values()):,} entity mentions"),
        ("🔗", "Relationship Mining Agent", "#F59E0B",
         "Claude Opus 4.6 extracts typed correlations with quantitative effect sizes (HR, OR, β, r, p-values) and validates clinical thresholds.",
         "Anthropic Messages API", f"{stats['total_correlations']} correlations, {stats['total_thresholds']} thresholds"),
        ("🏗️", "Knowledge Assembly Agent", "#10B981",
         "Aggregates all extractions into structured JSON knowledge base. Computes entity catalog, cross-references thresholds, generates statistics.",
         "Python, JSON serialization", "neurosight_kb.json"),
    ]
    
    for i, (icon, name, color, desc, tools, output) in enumerate(stages):
        st.markdown(f"""
        <div class="pipeline-stage" style="border-left: 4px solid {color};">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                <span style="font-size:24px;">{icon}</span>
                <div>
                    <div class="stage-num" style="color:{color};">STAGE {i+1}</div>
                    <div class="stage-name">{name}</div>
                </div>
            </div>
            <div style="font-size:0.85rem;color:#94A3B8;line-height:1.5;margin-bottom:8px;">{desc}</div>
            <div style="display:flex;gap:16px;font-size:0.75rem;">
                <span style="color:#64748B;">🔧 <code>{tools}</code></span>
                <span style="color:{color};">→ {output}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🐍 Agent Code Preview")
    st.caption("neurosight_agent.py — the full executable pipeline")
    
    with st.expander("View Entity Extraction Agent (Stage 3)", expanded=True):
        st.code('''class EntityExtractionAgent:
    """Uses Claude Opus 4.6 to extract domain-specific entities from each paper."""

    SYSTEM_PROMPT = """You are the NeuroSight Entity Extraction Agent.
    Given a paper's title, abstract, and MeSH terms, extract ALL relevant entities:
    
    1. retinal_biomarkers   (pRNFL, GCIPL, INL, TMV, SCP vessel density...)
    2. fluid_biomarkers     (sNfL, GFAP, NfH, CSF oligoclonal bands...)
    3. imaging_modalities   (SD-OCT, OCT-A, SS-OCT, MRI, VBM-OCT...)
    4. clinical_scales      (EDSS, SDMT, T25FW, LCLA, MSFC...)
    5. ms_subtypes          (RRMS, SPMS, PPMS, CIS, NMOSD, MOGAD...)
    6. drugs                [{name, category}]
    7. anatomical_structures (retina, thalamus, gray matter, optic nerve...)
    8. ai_methods           (CNN, Deep Learning, U-Net, Random Forest...)
    9. pathophysiology      (neurodegeneration, demyelination, axonal loss...)
    10. clinical_outcomes   (disability progression, relapse, NEDA status...)
    11. thematic_categories (PROGNOSTIC_BIOMARKERS, AI_METHODS, ...)
    
    Respond ONLY with valid JSON."""

    def extract(self, article: dict) -> dict:
        resp = requests.post("https://api.anthropic.com/v1/messages",
            json={"model": "claude-opus-4-6",
                  "system": self.SYSTEM_PROMPT,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 2000})
        return json.loads(resp.json()["content"][0]["text"])''', language="python")
    
    with st.expander("View Relationship Mining Agent (Stage 4)"):
        st.code('''class RelationshipMiningAgent:
    """Extracts typed correlations and clinical thresholds."""

    SYSTEM_PROMPT = """Extract from this paper:
    1. correlations: Array of strings with quantitative findings.
       Include HR, OR, beta, r, p-values when available.
       Example: "pRNFL <=88 um predicts progression (HR=2.4, p<0.001)"
    
    2. thresholds: Array of validated clinical cutoffs:
       [{"biomarker": "pRNFL", "threshold": "<=88 um", 
         "metric": "baseline thickness", "effect_size": "HR=2.376"}]
    
    Respond ONLY with JSON: {"correlations": [...], "thresholds": [...]}"""''', language="python")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB: PAPERS EXPLORER
# ═════════════════════════════════════════════════════════════════════════════

with tab_papers:
    st.subheader(f"📄 Papers Explorer ({stats['total_papers']:,} papers)")
    
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        search = st.text_input("🔍 Search by title, author, or PMID", placeholder="e.g. pRNFL, Petzold, 28920886...")
    with col_f2:
        depth_filter = st.selectbox("Extraction Depth", ["All", "Full Deep (18 key)", "Entity Tagged (bulk)"])
    with col_f3:
        type_filter = st.selectbox("Study Type", ["All"] + list(stats["study_type_distribution"].keys()))
    
    # Filter
    filtered = paper_list
    if search:
        q = search.lower()
        filtered = [p for p in filtered if 
            q in (p.get("title") or "").lower() or 
            q in (p.get("authors") or "").lower() or
            q in p.get("pmid", "")]
    if depth_filter == "Full Deep (18 key)":
        filtered = [p for p in filtered if p.get("extraction_depth") == "full_deep"]
    elif depth_filter == "Entity Tagged (bulk)":
        filtered = [p for p in filtered if p.get("extraction_depth") != "full_deep"]
    if type_filter != "All":
        filtered = [p for p in filtered if p.get("study_type") == type_filter]
    
    st.caption(f"Showing {len(filtered)} papers")
    
    # Show papers
    for p in filtered[:50]:  # limit display for performance
        is_key = p.get("extraction_depth") == "full_deep"
        title = p.get("title") or f"Paper PMID:{p['pmid']}"
        
        with st.expander(
            f"{'⭐' if is_key else '📄'} **{title[:100]}{'...' if len(title)>100 else ''}** — {p.get('authors','Unknown')} ({p.get('year','')}) · PMID:{p['pmid']}",
            expanded=False
        ):
            # Meta row
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Journal", p.get("journal", "—"))
            mc2.metric("Study Type", (p.get("study_type") or "—").replace("_", " ").title())
            mc3.metric("Patients", f"{p['patient_count']:,}" if p.get("patient_count") else "—")
            mc4.metric("Depth", "🔬 Full Deep" if is_key else "📋 Entity Tagged")
            
            # Entities
            st.markdown("**Extracted Entities:**")
            badges_html = ""
            for field_key, label in ENTITY_LABELS.items():
                entities = p.get(f"extracted_{field_key}", [])
                if entities:
                    color = ENTITY_COLORS[field_key]
                    for e in entities:
                        name = e if isinstance(e, str) else e.get("name", str(e))
                        badges_html += make_badge_html(name, color)
            if badges_html:
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.caption("No entities extracted")
            
            # Correlations (only key papers)
            corrs = p.get("extracted_correlations", [])
            if corrs:
                st.markdown("**Extracted Correlations:**")
                for c in corrs:
                    st.markdown(f"""<div class="correlation-item">▸ {c}</div>""", unsafe_allow_html=True)
            
            # Thresholds
            thresholds = p.get("extracted_thresholds", [])
            if thresholds:
                st.markdown("**Clinical Thresholds:**")
                for t in thresholds:
                    st.markdown(f"""
                    <div class="threshold-card">
                        <span class="threshold-biomarker">{t['biomarker']}</span> 
                        <span class="threshold-value">{t['threshold']}</span>
                        <span style="color:#64748B;font-size:0.8rem;"> · {t['metric']} · {t['effect_size']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # NeuroSight relevance
            rel = p.get("neurosight_relevance")
            if rel:
                st.markdown(f"""<div class="relevance-tag">🎯 <strong>NeuroSight Relevance:</strong> {rel}</div>""", unsafe_allow_html=True)
            
            # DOI link
            doi = p.get("doi")
            if doi:
                st.markdown(f"[🔗 DOI: {doi}](https://doi.org/{doi})")
    
    if len(filtered) > 50:
        st.info(f"Showing first 50 of {len(filtered)} papers. Use filters to narrow down.")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB: ENTITY ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

with tab_entities:
    st.subheader("🧬 Entity Analytics")
    
    # Category selector
    selected_cats = st.multiselect(
        "Select entity categories to analyze",
        list(ENTITY_LABELS.keys()),
        default=["retinal_biomarkers", "imaging_modalities", "clinical_scales", "pathophysiology"],
        format_func=lambda x: ENTITY_LABELS[x]
    )
    
    cols = st.columns(min(len(selected_cats), 2) if selected_cats else 1)
    
    for i, cat in enumerate(selected_cats):
        with cols[i % 2]:
            info = stats["entity_summary"].get(cat, {})
            color = ENTITY_COLORS[cat]
            
            st.markdown(f"### {ENTITY_LABELS[cat]}")
            st.caption(f"{info.get('unique', 0)} unique · {info.get('total_mentions', 0):,} mentions · {info.get('coverage_pct', 0)}% coverage")
            
            top = info.get("top_entities", {})
            if top:
                df = pd.DataFrame([{"Entity": k, "Mentions": v} for k, v in top.items()])
                fig = px.bar(
                    df, y="Entity", x="Mentions", orientation="h",
                    color_discrete_sequence=[color],
                    text="Mentions",
                )
                fig.update_layout(
                    plot_bgcolor="#111827", paper_bgcolor="#0A0E1A",
                    font_color="#94A3B8", font_family="Inter",
                    xaxis=dict(gridcolor="#1E293B"),
                    yaxis=dict(gridcolor="#1E293B", categoryorder="total ascending"),
                    margin=dict(t=10, b=20, l=10, r=10),
                    height=max(200, len(top) * 28),
                    showlegend=False,
                )
                fig.update_traces(textposition="outside", textfont_size=10)
                st.plotly_chart(fig, use_container_width=True)
    
    # Co-occurrence heatmap
    st.markdown("---")
    st.subheader("🔥 Entity Co-occurrence (Key Papers)")
    
    entity_fields = ["retinal_biomarkers", "fluid_biomarkers", "imaging_modalities", 
                     "clinical_scales", "ms_subtypes", "pathophysiology"]
    all_top_entities = []
    for field in entity_fields:
        counter = count_entity(field)
        all_top_entities.extend(list(counter.keys())[:5])
    
    if all_top_entities:
        # Build co-occurrence matrix from key papers
        matrix = {}
        for e1 in all_top_entities[:20]:
            matrix[e1] = {}
            for e2 in all_top_entities[:20]:
                count = 0
                for p in key_papers:
                    all_ents = []
                    for f in entity_fields:
                        all_ents.extend([
                            x if isinstance(x, str) else x.get("name","")
                            for x in p.get(f"extracted_{f}", [])
                        ])
                    if e1 in all_ents and e2 in all_ents:
                        count += 1
                matrix[e1][e2] = count
        
        df_matrix = pd.DataFrame(matrix)
        fig_heat = px.imshow(
            df_matrix, color_continuous_scale=["#0A0E1A", "#3B82F6", "#EC4899"],
            labels=dict(color="Co-occurrences"),
        )
        fig_heat.update_layout(
            plot_bgcolor="#111827", paper_bgcolor="#0A0E1A",
            font_color="#94A3B8", font_family="Inter",
            margin=dict(t=20, b=20),
            height=500,
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB: THRESHOLDS & EVIDENCE
# ═════════════════════════════════════════════════════════════════════════════

with tab_thresholds:
    st.subheader("🎯 Validated Clinical Thresholds")
    st.caption("These are the actionable cutoffs that power NeuroSight's risk stratification — each backed by meta-analyses and longitudinal studies.")
    
    # Collect all thresholds
    all_thresholds = []
    for p in paper_list:
        for t in p.get("extracted_thresholds", []):
            all_thresholds.append({
                **t,
                "pmid": p["pmid"],
                "authors": p.get("authors", ""),
                "year": p.get("year", ""),
                "title": p.get("title", ""),
            })
    
    for t in all_thresholds:
        effect = t.get("effect_size", "")
        effect_num = ""
        for part in effect.replace("=", " ").split():
            try:
                float(part)
                effect_num = part
                break
            except ValueError:
                continue
        
        st.markdown(f"""
        <div class="threshold-card" style="display:flex;align-items:center;gap:20px;">
            <div style="min-width:80px;text-align:center;">
                <div class="threshold-value">{effect_num or effect}</div>
                <div style="font-size:0.7rem;color:#64748B;">{effect.split('=')[0] if '=' in effect else ''}</div>
            </div>
            <div style="width:1px;height:50px;background:#1E293B;"></div>
            <div style="flex:1;">
                <div><span class="threshold-biomarker">{t['biomarker']}</span> <span style="color:#F59E0B;font-weight:700;">{t['threshold']}</span></div>
                <div style="font-size:0.8rem;color:#94A3B8;margin-top:4px;">{t['metric']} · PMID:{t['pmid']} · {t['authors']} ({t['year']})</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Evidence pyramid
    st.markdown("---")
    st.subheader("📐 Evidence Pyramid")
    evidence_levels = {
        "Meta-analyses & Systematic Reviews": len([p for p in paper_list if p.get("study_type") in ["meta_analysis", "systematic_review"]]),
        "Randomized Controlled Trials": len([p for p in paper_list if p.get("study_type") == "randomized_controlled_trial"]),
        "Longitudinal Cohort Studies": len([p for p in paper_list if p.get("study_type") == "longitudinal_cohort"]),
        "Cross-sectional Studies": len([p for p in paper_list if p.get("study_type") == "cross_sectional"]),
        "Reviews": len([p for p in paper_list if p.get("study_type") == "review"]),
    }
    df_ev = pd.DataFrame([{"Level": k, "Papers": v} for k, v in evidence_levels.items()])
    fig_ev = px.funnel(df_ev, x="Papers", y="Level", color_discrete_sequence=["#3B82F6"])
    fig_ev.update_layout(
        plot_bgcolor="#111827", paper_bgcolor="#0A0E1A",
        font_color="#94A3B8", font_family="Inter",
        margin=dict(t=20, b=20),
        height=300,
    )
    st.plotly_chart(fig_ev, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB: CORRELATIONS
# ═════════════════════════════════════════════════════════════════════════════

with tab_correlations:
    st.subheader("🔗 Extracted Correlations & Relationships")
    st.caption("Quantitative findings with effect sizes, extracted by Claude Opus 4.6 from 18 key papers.")
    
    for p in key_papers:
        corrs = p.get("extracted_correlations", [])
        if not corrs:
            continue
        
        rel = p.get("neurosight_relevance", "")
        title = p.get("title", f"PMID:{p['pmid']}")
        
        with st.expander(f"📄 **{title[:80]}** — {p.get('authors','')} ({p.get('year','')}) · {len(corrs)} correlations", expanded=False):
            if rel:
                st.markdown(f"""<div class="relevance-tag">🎯 {rel}</div>""", unsafe_allow_html=True)
            
            st.markdown("")
            for c in corrs:
                # Highlight effect sizes
                highlighted = c
                for marker in ["HR=", "OR=", "beta=", "r=", "p<", "p="]:
                    if marker in highlighted:
                        idx = highlighted.index(marker)
                        # Find end of number
                        end = idx + len(marker)
                        while end < len(highlighted) and (highlighted[end].isdigit() or highlighted[end] in ".-"):
                            end += 1
                        val = highlighted[idx:end]
                        highlighted = highlighted.replace(val, f"**{val}**")
                
                st.markdown(f"▸ {highlighted}")
    
    # Summary statistics
    st.markdown("---")
    st.subheader("📊 Correlation Summary")
    
    # Count correlation types
    all_corrs = []
    for p in key_papers:
        all_corrs.extend(p.get("extracted_correlations", []))
    
    hr_count = sum(1 for c in all_corrs if "HR=" in c or "HR " in c)
    or_count = sum(1 for c in all_corrs if "OR=" in c or "OR " in c)
    p_count = sum(1 for c in all_corrs if "p<" in c or "p=" in c)
    r_count = sum(1 for c in all_corrs if "r=" in c or "r " in c)
    beta_count = sum(1 for c in all_corrs if "beta=" in c or "β=" in c or "B=" in c)
    
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Hazard Ratios", hr_count)
    mc2.metric("Odds Ratios", or_count)
    mc3.metric("P-values", p_count)
    mc4.metric("Correlations (r)", r_count)
    mc5.metric("Beta Coefficients", beta_count)


# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:1rem;font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#475569;">
    NeuroSight Knowledge Base Explorer · Claude Opus 4.6 · {stats['total_papers']:,} PubMed Papers (2017–2026) · EP PerMed Hackathon 2025
</div>
""", unsafe_allow_html=True)
