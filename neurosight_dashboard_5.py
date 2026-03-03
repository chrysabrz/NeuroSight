import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="NEUROSIGHT | MS OCT Monitoring", page_icon="👁", layout="wide", initial_sidebar_state="expanded")

if "show_dmt" not in st.session_state:
    st.session_state.show_dmt = False
if "prescriptions" not in st.session_state:
    st.session_state.prescriptions = []
if "selected_dmt_view" not in st.session_state:
    st.session_state.selected_dmt_view = None
if "sos_active" not in st.session_state:
    st.session_state.sos_active = False
if "sos_searching" not in st.session_state:
    st.session_state.sos_searching = False

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#060d1a;color:#e8edf5;}
.stApp{background:#060d1a;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a1628,#07101f);border-right:1px solid #1a2d4a;}
section[data-testid="stSidebar"] *{color:#c8d8ee !important;}
/* App nav links: larger, prominent (NeuroSight + Knowledge Base) */
section[data-testid="stSidebar"] [data-testid="stPageLink"] a,
section[data-testid="stSidebar"] nav a {
    font-size: 1.05rem !important;
    padding: 10px 14px !important;
    min-height: 44px !important;
    display: flex !important;
    align-items: center !important;
    font-family: 'Syne', sans-serif !important;
}
/* NeuroSight link — blue gradient + glow */
section[data-testid="stSidebar"] [data-testid="stPageLink"]:first-of-type a,
section[data-testid="stSidebar"] nav a:first-child {
    background: linear-gradient(90deg, #60a5fa, #3b82f6, #2563eb) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    color: #60a5fa !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    filter: drop-shadow(0 0 6px rgba(59,130,246,0.5)) drop-shadow(0 0 12px rgba(99,102,241,0.4));
}
section[data-testid="stSidebar"] [data-testid="stPageLink"]:first-of-type a *,
section[data-testid="stSidebar"] nav a:first-child * {
    color: #60a5fa !important;
    -webkit-text-fill-color: #60a5fa !important;
}
/* Knowledge Base link — pink-to-purple (brain) gradient, same prominence */
section[data-testid="stSidebar"] [data-testid="stPageLink"]:last-of-type a,
section[data-testid="stSidebar"] nav a:last-of-type {
    background: linear-gradient(90deg, #ec4899, #c084fc, #7b2fff) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    color: transparent !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    filter: drop-shadow(0 0 6px rgba(236,72,153,0.4)) drop-shadow(0 0 10px rgba(168,85,247,0.35));
}
h1,h2,h3{font-family:'Syne',sans-serif !important;}
.ns-card{background:linear-gradient(135deg,#0d1e35,#0a1628);border:1px solid #1e3354;border-radius:16px;padding:20px 24px;margin-bottom:16px;position:relative;overflow:hidden;}
.ns-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00c6ff,#0072ff,#7b2fff);}
.ns-card-alert{background:linear-gradient(135deg,#1e0d0d,#180a0a) !important;border-color:#ff4444 !important;}
.ns-card-alert::before{background:linear-gradient(90deg,#ff4444,#ff6b6b) !important;}
.ns-card-warn{background:linear-gradient(135deg,#1e1700,#181200) !important;border-color:#ffaa00 !important;}
.ns-card-warn::before{background:linear-gradient(90deg,#ffaa00,#ffd060) !important;}
.ns-card-ok{background:linear-gradient(135deg,#001e0d,#00180a) !important;border-color:#00e676 !important;}
.ns-card-ok::before{background:linear-gradient(90deg,#00e676,#69ffb0) !important;}
.ns-metric-val{font-family:'DM Mono',monospace;font-size:2.2rem;font-weight:500;line-height:1;margin:4px 0;}
.ns-metric-label{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:#6a8caa;margin-bottom:4px;}
.ns-metric-delta{font-family:'DM Mono',monospace;font-size:.8rem;}
.delta-neg{color:#ff6b6b;} .delta-pos{color:#69ffb0;}
.section-title{font-family:'Syne',sans-serif;font-size:.68rem;letter-spacing:.18em;text-transform:uppercase;color:#4a7aaa;margin:20px 0 10px;padding-bottom:6px;border-bottom:1px solid #1a2d4a;}
.ai-insight{background:linear-gradient(135deg,#070f22,#0a0d20);border:1px solid #2a2060;border-radius:14px;padding:16px 20px;margin-top:8px;}
.ai-tag{font-family:'DM Mono',monospace;font-size:.65rem;color:#7b6fff;letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px;}
.ai-text{font-size:.86rem;color:#c0d0e8;line-height:1.6;}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.7rem;font-weight:500;letter-spacing:.06em;margin-left:6px;}
.badge-rr{background:#1a2d60;color:#7ab3ff;border:1px solid #2a4080;}
.badge-active{background:#1a3020;color:#69ffb0;border:1px solid #2a5030;}
.badge-alert{background:#3a0d0d;color:#ff9090;border:1px solid #5a1d1d;}
.tl-item{display:flex;gap:12px;margin-bottom:10px;font-size:.82rem;}
.tl-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;margin-top:4px;}
hr.ns{border:none;border-top:1px solid #1a2d4a;margin:16px 0;}
header[data-testid="stHeader"]{background:transparent !important;height:0 !important;min-height:0 !important;visibility:hidden !important;}
.ns-brand-title{font-family:'Syne',sans-serif !important;font-size:1.5rem !important;font-weight:800 !important;background:linear-gradient(90deg,#00c6ff,#7b2fff) !important;-webkit-background-clip:text !important;background-clip:text !important;-webkit-text-fill-color:transparent !important;color:transparent !important;}
</style>
""", unsafe_allow_html=True)

# ── PATIENT DATA ──────────────────────────────────────────────────────────────
PATIENTS = {
    "MS-2401 · Elena Marchetti": {
        "age":34,"sex":"F","type":"RRMS","onset":"2019","dmt":"Natalizumab",
        "dmt_start":"2020-03","edss":2.0,"status":"alert","risk":78,"seed":10,"initials":"EM",
        "last_mri":"Jan 2024 (2 new T2 lesions)","last_relapse":"Mar 2024",
        "sectors_od":[120,89,58,69,110,176,99,147],"sectors_os":[118,80,63,69,111,93,195,159],
        "tsnit_avg_od":85,"tsnit_avg_os":82,"tsnit_sd_od":44.1,"tsnit_sd_os":48.2,
    },
    "MS-2388 · Jonas Weber": {
        "age":41,"sex":"M","type":"RRMS","onset":"2016","dmt":"Ocrelizumab",
        "dmt_start":"2021-09","edss":2.5,"status":"warn","risk":51,"seed":22,"initials":"JW",
        "last_mri":"Jan 2024 (stable)","last_relapse":"Jul 2023",
        "sectors_od":[132,78,58,81,153,110,97,147],"sectors_os":[135,100,75,68,88,112,165,132],
        "tsnit_avg_od":106,"tsnit_avg_os":103,"tsnit_sd_od":32.8,"tsnit_sd_os":40.0,
    },
    "MS-2412 · Sophie Dubois": {
        "age":28,"sex":"F","type":"RRMS","onset":"2022","dmt":"Cladribine",
        "dmt_start":"2022-11","edss":1.0,"status":"ok","risk":22,"seed":37,"initials":"SD",
        "last_mri":"Jan 2024 (stable)","last_relapse":"None",
        "sectors_od":[143,93,67,120,176,99,85,135],"sectors_os":[129,65,88,133,93,111,195,159],
        "tsnit_avg_od":108,"tsnit_avg_os":103,"tsnit_sd_od":40.2,"tsnit_sd_os":45.9,
    },
    "MS-2399 · Pieter van Dijk": {
        "age":55,"sex":"M","type":"PPMS","onset":"2014","dmt":"Ocrelizumab",
        "dmt_start":"2019-06","edss":4.5,"status":"alert","risk":85,"seed":55,"initials":"PD",
        "last_mri":"Jan 2024 (cortical atrophy ↑)","last_relapse":"N/A (PPMS)",
        "sectors_od":[95,62,44,58,88,72,80,101],"sectors_os":[98,60,48,55,82,68,75,95],
        "tsnit_avg_od":71,"tsnit_avg_os":68,"tsnit_sd_od":28.1,"tsnit_sd_os":30.4,
    },
}

# ── DMT DATABASE ─────────────────────────────────────────────────────────────
DMT_INFO = {
    "Natalizumab": {
        "brand": "Tysabri®",
        "class": "Anti-α4-integrin monoclonal antibody",
        "route": "IV infusion · every 4 weeks",
        "efficacy_rrr": "68%",
        "approved": "RRMS",
        "color": "#0072ff",
        "mechanism": "Blocks α4-integrin on lymphocytes, preventing their migration across the blood-brain barrier into the CNS. Reduces CNS inflammation by sequestering autoreactive lymphocytes in the bloodstream.",
        "ms_ocular_note": "Natalizumab does not directly affect retinal structure. However, treatment failure (breakthrough disease) is associated with accelerating RNFL thinning. Monitoring RNFL rate of change provides an early signal of inadequate CNS protection.",
        "key_risks": [
            ("🔴 PML (Progressive Multifocal Leukoencephalopathy)", "Risk increases with JCV antibody index >0.9, prior immunosuppression, and treatment duration >24 months. Mandatory JCV serology every 6 months."),
            ("🟡 Hypersensitivity reactions", "Infusion reactions in ~4% of patients. Monitor 1 hour post-infusion."),
            ("🟡 Rebound syndrome", "Severe disease reactivation can occur 3–6 months after discontinuation. Transition planning essential."),
            ("🟢 Hepatotoxicity", "Rare. Monitor LFTs periodically."),
        ],
        "monitoring": [
            "JCV antibody index: every 6 months",
            "MRI brain: every 12 months (high JCV risk: every 6 months)",
            "LFTs: baseline + periodically",
            "NEUROSIGHT OCT: every 3 months — RNFL thinning rate is a sensitive early marker of breakthrough activity",
        ],
        "octrims_grade": "High efficacy",
        "rrms_rank": "1st–2nd line (high efficacy)",
        "interactions": "Avoid live vaccines. Do not combine with other immunosuppressants.",
        "riziv": {
            "status": "✅ Reimbursed",
            "code": "Hoofdstuk IV – Speciale geneesmiddelen",
            "chapter": "Hoofdstuk IV",
            "conditions": "Active RRMS with ≥2 relapses in past 2 years OR 1 severe relapse + MRI activity. JCV antibody testing required before initiation.",
            "patient_contribution": "€ 0.00 (100% reimbursed)",
            "prior_auth": "Required — neurologist prescription + RIZIV form",
            "url": "https://www.riziv.fgov.be",
        },
    },
    "Ocrelizumab": {
        "brand": "Ocrevus®",
        "class": "Anti-CD20 monoclonal antibody",
        "route": "IV infusion · every 6 months",
        "efficacy_rrr": "46–47% vs interferon",
        "approved": "RRMS + PPMS",
        "color": "#7b2fff",
        "mechanism": "Selectively depletes CD20-expressing B cells via antibody-dependent cytotoxicity and complement-dependent cytotoxicity. Reduces B-cell-mediated CNS inflammation and partially reduces plasmablasts involved in MS pathology.",
        "ms_ocular_note": "In PPMS patients on ocrelizumab, slower RNFL thinning rate may serve as a surrogate marker of neuroprotective effect. Baseline retinal imaging before first infusion is strongly recommended to establish individual trajectory. Vascular density (OCT-A) changes may precede structural RNFL loss.",
        "key_risks": [
            ("🔴 Infusion reactions", "Common (~34–40%); most mild-moderate. Premedicate with methylprednisolone + antihistamine + antipyretic."),
            ("🟡 Infections", "Increased risk of respiratory and urinary infections. Pneumocystis risk in severely lymphopenic patients."),
            ("🟡 Hypogammaglobulinaemia", "IgG levels decline over time with extended treatment. Monitor annually."),
            ("🟡 Hepatitis B reactivation", "Screen all patients before treatment initiation."),
            ("🟢 Malignancy", "Small observed increase; unclear if causal. Standard cancer screening recommended."),
        ],
        "monitoring": [
            "Full blood count + immunoglobulins: before each infusion",
            "Hepatitis B serology: before initiation",
            "MRI brain/spine: every 12–24 months",
            "Vaccination status: update before starting (live vaccines contraindicated)",
            "NEUROSIGHT OCT: every 3–4 months — especially valuable in PPMS where clinical endpoints are insensitive",
        ],
        "octrims_grade": "High efficacy",
        "rrms_rank": "1st–2nd line (high efficacy); only approved DMT for PPMS",
        "interactions": "Avoid live vaccines. Caution with other immunosuppressants.",
        "riziv": {
            "status": "✅ Reimbursed",
            "code": "Hoofdstuk IV – Speciale geneesmiddelen",
            "chapter": "Hoofdstuk IV",
            "conditions": "RRMS: active disease despite prior DMT OR high disease activity at start. PPMS: imaging-confirmed active disease (gadolinium-enhancing lesions or new T2 lesions). Neurologist-supervised initiation.",
            "patient_contribution": "€ 0.00 (100% reimbursed for approved indications)",
            "prior_auth": "Required — neurologist prescription + RIZIV form FA-B",
            "url": "https://www.riziv.fgov.be",
        },
    },
    "Cladribine": {
        "brand": "Mavenclad®",
        "class": "Purine nucleoside analogue (selective lymphocyte depletion)",
        "route": "Oral · 2 short treatment courses over 2 years",
        "efficacy_rrr": "58%",
        "approved": "Active RRMS / active SPMS",
        "color": "#00e676",
        "mechanism": "Accumulates selectively in lymphocytes, causing DNA strand breaks and apoptosis of rapidly dividing T and B cells. Results in sustained lymphocyte depletion with immune reconstitution over 2–4 years. Unique 'immune reset' pharmacology.",
        "ms_ocular_note": "Following cladribine-induced immune reset, RNFL thinning rate should stabilise or slow in responding patients. NEUROSIGHT can be particularly informative from 6–12 months post-treatment to capture the OCT fingerprint of immunological remission. Stable or improving vascular density is a positive treatment response signal.",
        "key_risks": [
            ("🔴 Lymphopenia", "Expected and dose-dependent. Grade 3/4 lymphopenia in ~25–26% of patients. Monitor lymphocyte counts closely."),
            ("🔴 Teratogenicity", "Strictly contraindicated in pregnancy. Effective contraception required during treatment and 6 months after."),
            ("🟡 Herpes zoster reactivation", "Risk elevated during lymphopenic phase. VZV prophylaxis recommended in seropositive patients."),
            ("🟡 Malignancy", "Small theoretical risk; exclude active malignancy before treatment. Annual skin checks."),
            ("🟢 Infections", "Increased risk during lymphopenic nadir (typically months 2–6 post-dose)."),
        ],
        "monitoring": [
            "Lymphocyte count: before treatment, months 2, 6, 12 in each treatment year",
            "Full blood count: baseline and monitoring schedule per label",
            "MRI: before initiation + annual",
            "Pregnancy test: before each treatment course",
            "NEUROSIGHT OCT: baseline + 3, 6, 12 months post-treatment — tracks immune-reset response trajectory",
        ],
        "octrims_grade": "High efficacy",
        "rrms_rank": "1st–2nd line (high efficacy); suitable for patients preferring oral pulse therapy",
        "interactions": "Avoid live vaccines during and 4–6 weeks after treatment. Do not combine with other immunosuppressants during lymphopenic phase.",
        "riziv": {
            "status": "✅ Reimbursed",
            "code": "Hoofdstuk IV – Speciale geneesmiddelen",
            "chapter": "Hoofdstuk IV",
            "conditions": "Active RRMS with high disease activity: ≥2 disabling relapses in 1 year + ≥1 Gd+ lesion OR significant T2 lesion increase. Failure or intolerance to ≥1 prior DMT.",
            "patient_contribution": "€ 0.00 (100% reimbursed)",
            "prior_auth": "Required — neurologist prescription + RIZIV chapter IV form",
            "url": "https://www.riziv.fgov.be",
        },
    },
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def generate_longitudinal(seed, status, n):
    rng = np.random.default_rng(seed)
    d = {"alert":1.4,"warn":0.7,"ok":0.25}[status]
    rnfl = [rng.uniform(95,105)-i*d+rng.normal(0,1.2) for i in range(n)]
    gcl  = [rng.uniform(78,88)-i*d*0.8+rng.normal(0,1.0) for i in range(n)]
    vd   = [rng.uniform(46,52)-i*d*0.4+rng.normal(0,0.7) for i in range(n)]
    faz  = [rng.uniform(0.27,0.33)+i*d*0.004+rng.normal(0,0.003) for i in range(n)]
    inl  = [rng.uniform(36,42)-i*d*0.3+rng.normal(0,0.8) for i in range(n)]
    return rnfl,gcl,vd,faz,inl

def delta_html(val, unit="μm", invert=False):
    bad = (val<0) if not invert else (val>0)
    cls = "delta-neg" if bad else "delta-pos"
    sign = "▼" if val<0 else "▲"
    return f'<span class="{cls}">{sign} {abs(val):.1f}{unit}</span>'

OCT_CS = [
    [0.00,'#000080'],[0.12,'#0000ff'],[0.25,'#00aaff'],
    [0.40,'#00ffaa'],[0.55,'#aaff00'],[0.68,'#ffee00'],
    [0.80,'#ff8800'],[0.90,'#ff2200'],[1.00,'#ff0066'],
]
DEV_CS = [
    [0.00,'#cc0000'],[0.25,'#ff6600'],[0.45,'#ffcc00'],
    [0.50,'#ffffff'],[0.55,'#aaddff'],[0.75,'#0077ff'],[1.00,'#0000aa'],
]
SIG_CS = [
    [0.00,'#006600'],[0.33,'#009900'],
    [0.34,'#cc9900'],[0.66,'#ffcc00'],
    [0.67,'#cc0000'],[1.00,'#ff0000'],
]

def _base_map_layout(title, size):
    return dict(
        title=dict(text=title,font=dict(size=10,color='#8ab0d0',family='DM Mono'),x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#000000',
        margin=dict(l=0,r=35,t=28,b=0),height=220,
        xaxis=dict(showticklabels=False,showgrid=False,zeroline=False,range=[0,size]),
        yaxis=dict(showticklabels=False,showgrid=False,zeroline=False,range=[0,size],scaleanchor='x'),
    )

def _add_disc_and_labels(fig, size):
    t = np.linspace(0,2*np.pi,60)
    fig.add_trace(go.Scatter(x=0.14*np.cos(t)*size/2+size/2,y=0.14*np.sin(t)*size/2+size/2,
        mode='lines',line=dict(color='#888888',width=1.5),showlegend=False,hoverinfo='skip'))
    for ang,lbl in [(90,'S'),(270,'I'),(180,'N'),(0,'T')]:
        px_=np.cos(np.radians(ang))*0.76*size/2+size/2
        py_=np.sin(np.radians(ang))*0.76*size/2+size/2
        fig.add_annotation(x=px_,y=py_,text=lbl,showarrow=False,
            font=dict(size=9,color='white',family='DM Mono'))

def make_rnfl_map(pt, eye, title, thin):
    rng=np.random.default_rng(pt["seed"]+(0 if eye=="OD" else 1))
    S=80; x=np.linspace(-1,1,S); X,Y=np.meshgrid(x,x)
    R=np.sqrt(X**2+Y**2); Th=np.arctan2(Y,X)
    angular=(
        80*np.exp(-((Th-np.radians(90))**2)/0.4)+
        75*np.exp(-((Th-np.radians(270))**2)/0.45)+
        40*np.exp(-((Th-np.radians(180))**2)/0.6)+
        25*np.exp(-((Th-0)**2)/0.5)
    )
    radial=np.exp(-((R-0.38)**2)/0.025)
    z=angular*radial*thin+rng.normal(0,4,(S,S))
    if thin<0.92:
        tm=(Th>np.radians(-45))&(Th<np.radians(45))
        z[tm]*=(thin-0.06)
    z[R>0.92]=np.nan; z[R<0.14]=np.nan
    z=np.clip(z,0,250)
    fig=go.Figure(go.Heatmap(z=z,zmin=0,zmax=220,colorscale=OCT_CS,showscale=True,
        colorbar=dict(thickness=8,len=0.7,tickfont=dict(size=8,color='#8ab0d0',family='DM Mono'),
            tickvals=[0,50,100,150,200],ticktext=['0','50','100','150','200μm'],
            outlinecolor='#1a2d4a',outlinewidth=1,bgcolor='rgba(0,0,0,0)')))
    t=np.linspace(0,2*np.pi,60)
    fig.add_trace(go.Scatter(x=0.52*np.cos(t)*S/2+S/2,y=0.52*np.sin(t)*S/2+S/2,
        mode='lines',line=dict(color='#ffff00',width=1,dash='dot'),showlegend=False,hoverinfo='skip'))
    _add_disc_and_labels(fig,S)
    fig.update_layout(**_base_map_layout(title,S))
    return fig

def make_dev_map(pt, eye, title, thin):
    rng=np.random.default_rng(pt["seed"]+(10 if eye=="OD" else 11))
    S=80; x=np.linspace(-1,1,S); X,Y=np.meshgrid(x,x)
    R=np.sqrt(X**2+Y**2); Th=np.arctan2(Y,X)
    dev=rng.normal(0,8,(S,S))
    ring=np.exp(-((R-0.38)**2)/0.025)
    if thin<1.0:
        deficit=(1.0-thin)*60
        tm=(Th>np.radians(-50))&(Th<np.radians(50))
        dev[tm]-=deficit*ring[tm]*1.5
        sm=(Th>np.radians(60))&(Th<np.radians(130))
        dev[sm]-=deficit*0.7*ring[sm]
    dev[R>0.92]=np.nan; dev[R<0.14]=np.nan
    dev=np.clip(dev,-50,50)
    fig=go.Figure(go.Heatmap(z=dev,zmin=-50,zmax=50,colorscale=DEV_CS,showscale=True,
        colorbar=dict(thickness=8,len=0.7,tickfont=dict(size=8,color='#8ab0d0',family='DM Mono'),
            tickvals=[-50,-25,0,25,50],ticktext=['-50','-25','0','+25','+50'],
            outlinecolor='#1a2d4a',outlinewidth=1,bgcolor='rgba(0,0,0,0)')))
    _add_disc_and_labels(fig,S)
    fig.update_layout(**_base_map_layout(title,S))
    return fig

def make_sig_map(pt, eye, title, thin):
    rng=np.random.default_rng(pt["seed"]+(20 if eye=="OD" else 21))
    S=80; x=np.linspace(-1,1,S); X,Y=np.meshgrid(x,x)
    R=np.sqrt(X**2+Y**2); Th=np.arctan2(Y,X)
    sig=np.zeros((S,S)); ring=np.exp(-((R-0.38)**2)/0.025)
    if thin<0.97:
        deficit=(1.0-thin)*3
        tm=(Th>np.radians(-55))&(Th<np.radians(55))
        n1=rng.uniform(0,1,(S,S))
        sig[(tm)&(ring>0.3)&(n1<deficit)]=2
        sig[(tm)&(ring>0.15)&(n1<deficit*0.5)&(sig==0)]=1
    if thin<0.88:
        sm=(Th>np.radians(55))&(Th<np.radians(135))
        n2=rng.uniform(0,1,(S,S))
        sig[(sm)&(ring>0.25)&(n2<(1-thin)*2)]=2
    sig[R>0.92]=np.nan; sig[R<0.14]=np.nan
    fig=go.Figure(go.Heatmap(z=sig,zmin=0,zmax=2,colorscale=SIG_CS,showscale=True,
        colorbar=dict(thickness=8,len=0.7,tickfont=dict(size=8,color='#8ab0d0',family='DM Mono'),
            tickvals=[0.33,1.0,1.66],ticktext=['p>5%','1-5%','p<1%'],
            outlinecolor='#1a2d4a',outlinewidth=1,bgcolor='rgba(0,0,0,0)')))
    _add_disc_and_labels(fig,S)
    fig.update_layout(**_base_map_layout(title,S))
    return fig

def make_tsnit(pt, show_od, show_os, show_norm):
    n=100; t=np.linspace(0,1,n)
    thin={"alert":0.80,"warn":0.92,"ok":1.00}[pt["status"]]
    def curve(avg,soff,th):
        r=np.random.default_rng(pt["seed"]+soff)
        b=(avg*0.55+20*np.sin(2*np.pi*(t-0.12))**2+
           35*np.exp(-((t-0.25)**2)/0.012)*th+
           28*np.exp(-((t-0.75)**2)/0.015)*th+
           10*np.exp(-((t-0.5)**2)/0.04))
        b+=r.normal(0,3,n)
        temp=(t<0.08)|(t>0.92); b[temp]*=th
        return np.clip(b,20,240)
    ref=curve(115,99,1.0)
    nu=ref+25; nl=ref-22; np5=ref-38
    xlbl=['TEMP','','','SUP','','','NAS','','','INF','','','TEMP']
    xtpos=np.linspace(0,n-1,len(xlbl))
    fig=go.Figure()
    if show_norm:
        fig.add_trace(go.Scatter(x=list(range(n))+list(range(n-1,-1,-1)),y=list(nu)+list(nl),
            fill='toself',fillcolor='rgba(0,200,80,0.18)',line=dict(color='rgba(0,0,0,0)'),
            name='95% normal',showlegend=True,hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=list(range(n))+list(range(n-1,-1,-1)),y=list(nl)+list(np5),
            fill='toself',fillcolor='rgba(255,200,0,0.18)',line=dict(color='rgba(0,0,0,0)'),
            name='5-1%',showlegend=True,hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=list(range(n)),y=np5,fill='tozeroy',
            fillcolor='rgba(200,0,0,0.12)',line=dict(color='rgba(0,0,0,0)'),
            name='<1%',showlegend=True,hoverinfo='skip'))
        fig.add_hline(y=160,line_dash='dash',line_color='rgba(150,150,150,0.35)',line_width=1)
        fig.add_hline(y=80,line_dash='dash',line_color='rgba(150,150,150,0.35)',line_width=1)
    if show_od:
        fig.add_trace(go.Scatter(x=list(range(n)),y=curve(pt["tsnit_avg_od"],0,thin),
            mode='lines',name=f'OD ({pt["tsnit_avg_od"]}μm)',
            line=dict(color='#3388ff',width=2.5)))
    if show_os:
        fig.add_trace(go.Scatter(x=list(range(n)),y=curve(pt["tsnit_avg_os"],5,thin*0.97),
            mode='lines',name=f'OS ({pt["tsnit_avg_os"]}μm)',
            line=dict(color='#cc8800',width=2.5)))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(5,15,30,0.8)',
        height=230,margin=dict(l=10,r=10,t=10,b=30),
        legend=dict(font=dict(size=9,color='#8ab0d0'),bgcolor='rgba(0,0,0,0)',orientation='h',y=1.06),
        yaxis=dict(range=[0,240],showgrid=True,gridcolor='#1a2d4a',
            tickfont=dict(size=9,color='#4a7aaa'),title='μm',
            title_font=dict(size=9,color='#4a7aaa'),zeroline=False,linecolor='#1a2d4a'),
        xaxis=dict(showgrid=False,tickvals=list(xtpos),ticktext=xlbl,
            tickfont=dict(size=9,color='#4a7aaa'),linecolor='#1a2d4a',zeroline=False),
        hoverlabel=dict(bgcolor='#0d1e35',bordercolor='#1e3354',font_color='#e8edf5'),
    )
    return fig

def make_clock(sectors, avg, label):
    sector_angles=[90,45,0,315,270,225,180,135]
    sector_names=['S','ST','T','IT','I','IN','N','SN']
    fig=go.Figure()
    for val,ang,name in zip(sectors,sector_angles,sector_names):
        col=('#cc0000' if val<75 else '#ffcc00' if val<88 else '#aacc00' if val<95 else '#009933')
        a0,a1=np.radians(ang-22.5),np.radians(ang+22.5)
        ro=min(val/120,0.83); ri=0.12
        op=[(ro*np.cos(a),ro*np.sin(a)) for a in np.linspace(a0,a1,12)]
        ip=[(ri*np.cos(a),ri*np.sin(a)) for a in np.linspace(a1,a0,5)]
        pts=op+ip
        xs=[p[0] for p in pts]+[op[0][0]]; ys=[p[1] for p in pts]+[op[0][1]]
        fig.add_trace(go.Scatter(x=xs,y=ys,fill='toself',fillcolor=col,
            line=dict(color='#060d1a',width=1.5),mode='lines',showlegend=False,
            hovertemplate=f'<b>{name}</b>: {val}μm<extra></extra>'))
        mid=np.radians(ang)
        fig.add_annotation(x=0.65*np.cos(mid),y=0.65*np.sin(mid),text=str(val),
            showarrow=False,font=dict(size=8,color='white',family='DM Mono'))
        fig.add_annotation(x=0.93*np.cos(mid),y=0.93*np.sin(mid),text=name,
            showarrow=False,font=dict(size=7,color='#6a8caa',family='DM Mono'))
    t=np.linspace(0,2*np.pi,80)
    fig.add_trace(go.Scatter(x=np.cos(t),y=np.sin(t),mode='lines',
        line=dict(color='#1a3a6a',width=1),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=0.12*np.cos(t),y=0.12*np.sin(t),mode='lines',
        line=dict(color='#333355',width=1),showlegend=False,hoverinfo='skip'))
    fig.add_annotation(x=0,y=0.04,text=str(avg),showarrow=False,
        font=dict(size=13,color='white',family='DM Mono'))
    fig.add_annotation(x=0,y=-0.07,text='avg μm',showarrow=False,
        font=dict(size=7,color='#6a8caa',family='DM Mono'))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(5,15,30,0.6)',
        margin=dict(l=5,r=5,t=25,b=5),height=178,
        title=dict(text=label,font=dict(size=10,color='#8ab0d0',family='DM Mono'),x=0.5),
        xaxis=dict(range=[-1.12,1.12],showticklabels=False,showgrid=False,zeroline=False),
        yaxis=dict(range=[-1.12,1.12],showticklabels=False,showgrid=False,zeroline=False,scaleanchor='x'),
    )
    return fig

def make_bscan(seed, status="ok"):
    rng=np.random.default_rng(seed); W,H=300,65
    img=np.zeros((H,W))
    layers={'ILM':0.30,'RNFL':0.36,'GCL':0.41,'IPL':0.46,'INL':0.51,
            'OPL':0.55,'ONL':0.63,'ELM':0.69,'IS/OS':0.73,'RPE':0.79}
    bright={'ILM':0.88,'RNFL':0.68,'GCL':0.38,'IPL':0.52,'INL':0.33,
            'OPL':0.58,'ONL':0.24,'ELM':0.78,'IS/OS':0.88,'RPE':0.84}
    thick={'ILM':1,'RNFL':5,'GCL':4,'IPL':4,'INL':3,
           'OPL':3,'ONL':7,'ELM':1,'IS/OS':2,'RPE':3}
    img[:int(H*0.28),:]=rng.uniform(0,0.06,(int(H*0.28),W))
    img[int(H*0.85):,:]=rng.uniform(0,0.1,(H-int(H*0.85),W))
    xv=np.linspace(-1,1,W)
    curv=(0.035*(1-np.exp(-(xv*2.5)**2))*H).astype(int)
    for layer,bp in layers.items():
        br=bright[layer]; th=thick[layer]
        for xi in range(W):
            off=curv[xi]
            yc=int(bp*H)+off
            if status=="alert" and layer in ('RNFL','GCL') and xi>W//3:
                yc+=2
            for tk in range(th):
                yi=yc+tk-th//2
                if 0<=yi<H:
                    img[yi,xi]=np.clip(br*rng.uniform(0.85,1.15)+img[yi,xi],0,1)
    img+=rng.uniform(0,0.07,(H,W))*(img>0.05)
    fx=W//2
    for xi in range(fx-18,fx+18):
        if 0<=xi<W:
            dip=int(3*np.exp(-((xi-fx)/7)**2))
            img[:,xi]=np.roll(img[:,xi],dip)
    z=np.clip(img*255,0,255).astype(np.uint8)
    fig=go.Figure(go.Heatmap(z=z,colorscale='gray',showscale=False,zmin=0,zmax=255))
    bw=int(W*0.15)
    fig.add_shape(type='line',x0=8,x1=8+bw,y0=H-5,y1=H-5,
        line=dict(color='white',width=2))
    fig.add_annotation(x=8+bw/2,y=H-1.5,text='400μm',showarrow=False,
        font=dict(size=7,color='white',family='DM Mono'),yref='y')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#000000',
        margin=dict(l=0,r=0,t=0,b=0),height=85,
        xaxis=dict(showticklabels=False,showgrid=False,zeroline=False,range=[0,W]),
        yaxis=dict(showticklabels=False,showgrid=False,zeroline=False,range=[0,H],scaleanchor='x'),
    )
    return fig

def make_longitudinal(dates, series, show_norm, layers):
    cfg={
        "RNFL":            {"color":"#00c6ff","norm":(88,108),"unit":"μm","key":"rnfl"},
        "GCL+IPL":         {"color":"#7b2fff","norm":(70,88), "unit":"μm","key":"gcl"},
        "Vascular Density":{"color":"#00e676","norm":(42,54), "unit":"%", "key":"vd"},
        "INL":             {"color":"#ffaa00","norm":(34,44), "unit":"μm","key":"inl"},
    }
    if not layers: layers=["RNFL"]
    nr=len(layers)
    fig=make_subplots(rows=nr,cols=1,shared_xaxes=True,vertical_spacing=0.07,
        subplot_titles=[f"{l} ({cfg[l]['unit']})" for l in layers])
    ds=[d.strftime("%b %Y") for d in dates]
    for idx,layer in enumerate(layers):
        c=cfg[layer]; vals=series[c["key"]]; row=idx+1
        if show_norm:
            fig.add_trace(go.Scatter(
                x=ds+ds[::-1],y=[c["norm"][1]]*len(dates)+[c["norm"][0]]*len(dates),
                fill='toself',fillcolor='rgba(255,255,255,0.05)',
                line=dict(color='rgba(255,255,255,0)'),showlegend=False,hoverinfo='skip'),row=row,col=1)
        fig.add_trace(go.Scatter(x=ds,y=vals,mode='lines+markers',name=layer,
            line=dict(color=c["color"],width=2.5,shape='spline'),
            marker=dict(size=7,color=c["color"],line=dict(width=2,color='#060d1a')),
            showlegend=True),row=row,col=1)
        xn=list(range(len(vals))); z=np.polyfit(xn,vals,1)
        fig.add_trace(go.Scatter(x=ds,y=np.poly1d(z)(xn),mode='lines',
            line=dict(color=c["color"],width=1,dash='dot'),showlegend=False,hoverinfo='skip'),row=row,col=1)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(13,30,53,0.4)',
        font=dict(family='DM Sans',color='#8ab0d0',size=10),
        margin=dict(l=10,r=10,t=30,b=10),height=max(160,nr*145),
        legend=dict(font=dict(size=9,color='#8ab0d0'),bgcolor='rgba(0,0,0,0)',orientation='h',y=1.05),
        hoverlabel=dict(bgcolor='#0d1e35',bordercolor='#1e3354',font_color='#e8edf5'),
    )
    fig.update_xaxes(showgrid=True,gridcolor='#1a2d4a',tickfont=dict(size=9,color='#4a7aaa'),linecolor='#1a2d4a')
    fig.update_yaxes(showgrid=True,gridcolor='#1a2d4a',tickfont=dict(size=9,color='#4a7aaa'),linecolor='#1a2d4a')
    for ann in fig.layout.annotations: ann.font.color='#8ab0d0'; ann.font.size=9
    return fig

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Custom app nav: "NeuroSight" (main) and "Knowledge Base" (no default "neurosight dashboard 5" label)
    st.page_link("neurosight_dashboard_5.py", label="NeuroSight", icon="👁")
    st.page_link("pages/2_NeuroSight_Knowledge_Base.py", label="Knowledge Base", icon="🧠")
    st.markdown("<hr class='ns'>", unsafe_allow_html=True)
    st.markdown("""<div style='padding:8px 0 20px'>
        <div class='ns-brand-title'>NEUROSIGHT</div>
        <div style='font-size:.7rem;letter-spacing:.15em;color:#4a7aaa;margin-top:2px;'>
            OCT · MS MONITORING PLATFORM</div></div>""", unsafe_allow_html=True)

    st.markdown("#### 👤 Patient")
    selected_patient = st.selectbox("", list(PATIENTS.keys()), label_visibility="collapsed")
    pt = PATIENTS[selected_patient]

    st.markdown("<hr class='ns'>", unsafe_allow_html=True)
    st.markdown("#### 👁 Eye Selection")
    eye_side = st.radio("", ["Both Eyes","Right Eye (OD)","Left Eye (OS)"], label_visibility="collapsed")
    show_od = eye_side in ["Both Eyes","Right Eye (OD)"]
    show_os = eye_side in ["Both Eyes","Left Eye (OS)"]

    st.markdown("<hr class='ns'>", unsafe_allow_html=True)
    st.markdown("#### 🔬 OCT Layers")
    active_layers = st.multiselect("",["RNFL","GCL+IPL","Vascular Density","INL"],
        default=["RNFL","GCL+IPL","Vascular Density"],label_visibility="collapsed")

    st.markdown("<hr class='ns'>", unsafe_allow_html=True)
    st.markdown("#### 🗺 Map Type")
    map_type = st.selectbox("",["RNFL Thickness","Deviation Map","Significance Map"],
        label_visibility="collapsed")

    st.markdown("<hr class='ns'>", unsafe_allow_html=True)
    st.markdown("#### 📅 Time Range")
    timerange = st.select_slider("",options=["3 mo","6 mo","12 mo","18 mo"],value="18 mo",
        label_visibility="collapsed")

    show_normative = st.toggle("Normative Band", value=True)
    show_bscan     = st.toggle("Show B-Scan", value=True)
    show_ai        = st.toggle("AI Annotations", value=True)

    st.markdown("<hr class='ns'>", unsafe_allow_html=True)
    st.caption("NEUROSIGHT v0.9.2-beta · EP PerMed 2025")
    st.caption("⚠️ Research use only.")

# ── DATA ──────────────────────────────────────────────────────────────────────
n_pts = {"3 mo":2,"6 mo":3,"12 mo":5,"18 mo":7}[timerange]
all_dates = [datetime(2023,1,1)+timedelta(days=90*i) for i in range(7)]
dates = all_dates[-n_pts:]
rnfl,gcl,vd,faz,inl = generate_longitudinal(pt["seed"],pt["status"],7)
rnfl=rnfl[-n_pts:]; gcl=gcl[-n_pts:]; vd=vd[-n_pts:]; faz=faz[-n_pts:]; inl=inl[-n_pts:]
series={"rnfl":rnfl,"gcl":gcl,"vd":vd,"faz":faz,"inl":inl}
status=pt["status"]; risk=pt["risk"]
thin={"alert":0.80,"warn":0.92,"ok":1.00}[status]
map_fn={"RNFL Thickness":make_rnfl_map,"Deviation Map":make_dev_map,"Significance Map":make_sig_map}[map_type]

# ── HEADER ────────────────────────────────────────────────────────────────────
status_badge={"alert":'<span class="badge badge-alert">⚠ HIGH RISK</span>',
    "warn":'<span class="badge" style="background:#2a2200;color:#ffd060;border:1px solid #4a3a00;">◆ MONITORING</span>',
    "ok":'<span class="badge badge-active">✓ STABLE</span>'}[status]

dmt_name = pt["dmt"]
dmt = DMT_INFO.get(dmt_name, None)
dmt_color = dmt["color"] if dmt else "#4a7aaa"

# Sync selected_dmt_view to current patient's dmt when panel opens
if st.session_state.show_dmt and st.session_state.selected_dmt_view is None:
    st.session_state.selected_dmt_view = dmt_name

st.markdown(f"""<div class="ns-card" style="margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
    <div style="width:52px;height:52px;border-radius:50%;background:linear-gradient(135deg,#0072ff,#7b2fff);
      display:flex;align-items:center;justify-content:center;font-family:Syne,sans-serif;
      font-weight:700;font-size:1.2rem;flex-shrink:0;">{pt["initials"]}</div>
    <div style="flex:1;min-width:180px;">
      <div style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;">
        {selected_patient.split('·')[1].strip()}
        <span class="badge badge-rr">{pt["type"]}</span>
        <span class="badge badge-active">ON DMT</span>{status_badge}</div>
      <div style="font-size:.8rem;color:#6a8caa;margin-top:5px;">
        Age {pt["age"]} · {pt["sex"]} · MS onset {pt["onset"]} · EDSS {pt["edss"]}</div>
      <div style="font-size:.8rem;margin-top:5px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        <span style="color:#6a8caa;">Current DMT:</span>
        <span style="font-family:'DM Mono',monospace;font-size:.85rem;color:#6a8caa;"> since {pt["dmt_start"]}</span>
      </div>
      <div style="font-size:.8rem;color:#6a8caa;margin-top:3px;">
        Last MRI: {pt["last_mri"]} &nbsp;|&nbsp; Last relapse: {pt["last_relapse"]}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:.65rem;letter-spacing:.1em;color:#4a7aaa;text-transform:uppercase;">Scan date</div>
      <div style="font-family:'DM Mono',monospace;font-size:.9rem;">{dates[-1].strftime('%d %b %Y')}</div>
      <div style="font-size:.65rem;letter-spacing:.1em;color:#4a7aaa;text-transform:uppercase;margin-top:6px;">Device</div>
      <div style="font-family:'DM Mono',monospace;font-size:.85rem;color:#8ab0d0;">Canon OCT-A1</div>
    </div>
  </div></div>""", unsafe_allow_html=True)

# Clickable DMT pill sits below header as a proper button row
btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 4])
with btn_col1:
    open_label = f"{'🔽' if st.session_state.show_dmt else '💊'} {dmt_name}"
    if st.button(open_label, use_container_width=True,
                 type="primary" if st.session_state.show_dmt else "secondary"):
        st.session_state.show_dmt = not st.session_state.show_dmt
        st.session_state.selected_dmt_view = dmt_name if not st.session_state.show_dmt == False else dmt_name
        st.rerun()
with btn_col2:
    if st.session_state.show_dmt:
        if st.button("✕ Close", use_container_width=True):
            st.session_state.show_dmt = False
            st.session_state.selected_dmt_view = None
            st.rerun()
with btn_col3:
    st.markdown("<div style='height:1px'></div>", unsafe_allow_html=True)

# ── DMT INFO PANEL ────────────────────────────────────────────────────────────
if st.session_state.show_dmt:
    all_dmts = list(DMT_INFO.keys())
    current_view = st.session_state.selected_dmt_view or dmt_name

    # Medication selector row
    st.markdown('<div class="section-title" style="margin-top:4px;">Select Medication to View</div>', unsafe_allow_html=True)
    tab_cols = st.columns(len(all_dmts))
    for i, d_name in enumerate(all_dmts):
        d_info = DMT_INFO[d_name]
        is_active  = (d_name == current_view)
        is_current = (d_name == dmt_name)
        border = d_info["color"] if is_active else "#1e3354"
        r,g,b = int(d_info["color"][1:3],16), int(d_info["color"][3:5],16), int(d_info["color"][5:],16)
        bg = f"rgba({r},{g},{b},0.15)" if is_active else "rgba(255,255,255,0.03)"
        with tab_cols[i]:
            st.markdown(
                f'<div style="background:{bg};border:2px solid {border};border-radius:12px;'
                f'padding:10px 14px;text-align:center;margin-bottom:6px;">'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:.82rem;'
                f'color:{d_info["color"] if is_active else "#8ab0d0"};font-weight:{"600" if is_active else "400"};">'
                f'💊 {d_name}</div>'
                f'<div style="font-size:.68rem;color:#4a7aaa;margin-top:3px;">{d_info["brand"]}</div>'
                + ('<div style="font-size:.65rem;color:#69ffb0;margin-top:2px;">✓ current</div>' if is_current else '')
                + '</div>',
                unsafe_allow_html=True)
            if st.button(f"View {d_name}", key=f"dmt_tab_{d_name}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.selected_dmt_view = d_name
                st.rerun()

    view_dmt_name = current_view
    view_dmt      = DMT_INFO.get(view_dmt_name)
    view_color    = view_dmt["color"] if view_dmt else "#4a7aaa"

    if view_dmt:
        rz               = view_dmt["riziv"]
        rz_col           = "#69ffb0" if "Reimbursed" in rz["status"] else "#ffaa00"
        already_prescribed = view_dmt_name in st.session_state.prescriptions
        is_current_dmt   = (view_dmt_name == dmt_name)

        # Title bar
        current_tag = ('<span style="font-size:.72rem;color:#69ffb0;background:rgba(0,230,118,0.1);'
                       'padding:2px 10px;border-radius:20px;border:1px solid #00e67644;margin-left:8px;">'
                       '✓ Currently prescribed</span>') if is_current_dmt else ""
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#0a1628,#070d1e);'
            f'border:1px solid {view_color}55;border-radius:14px;padding:14px 20px 10px;'
            f'margin-bottom:6px;position:relative;overflow:hidden;">'
            f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
            f'background:linear-gradient(90deg,{view_color},{view_color}60,transparent);"></div>'
            f'<div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;">'
            f'<span style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:800;color:{view_color};">{view_dmt_name}</span>'
            f'<span style="font-size:.8rem;color:#6a8caa;">{view_dmt["brand"]}</span>'
            f'<span style="font-size:.72rem;color:#8ab0d0;background:rgba(255,255,255,0.05);'
            f'padding:2px 10px;border-radius:20px;border:1px solid #1e3354;">{view_dmt["class"]}</span>'
            + current_tag + '</div></div>',
            unsafe_allow_html=True)

        # Stat chips
        ch1,ch2,ch3,ch4,ch5 = st.columns(5)
        chip = "background:rgba(255,255,255,0.04);border:1px solid #1e3354;border-radius:12px;padding:10px 14px;text-align:center;margin-bottom:4px;"
        lbl  = "font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:#4a7aaa;"
        val  = "font-family:'DM Mono',monospace;font-size:.84rem;color:#c0d0e8;margin-top:3px;"
        with ch1:
            st.markdown(f'<div style="{chip}"><div style="{lbl}">Route</div><div style="{val}">{view_dmt["route"]}</div></div>', unsafe_allow_html=True)
        with ch2:
            st.markdown(f'<div style="{chip}"><div style="{lbl}">Relapse Reduction</div>' +
                f'<div style="font-family:\'DM Mono\',monospace;font-size:1.1rem;color:{view_color};font-weight:600;margin-top:3px;">{view_dmt["efficacy_rrr"]}</div></div>', unsafe_allow_html=True)
        with ch3:
            st.markdown(f'<div style="{chip}"><div style="{lbl}">Indication</div><div style="{val}">{view_dmt["approved"]}</div></div>', unsafe_allow_html=True)
        with ch4:
            st.markdown(f'<div style="{chip}"><div style="{lbl}">Efficacy Grade</div>' +
                f'<div style="font-family:\'DM Mono\',monospace;font-size:.84rem;color:#69ffb0;margin-top:3px;">{view_dmt["octrims_grade"]}</div></div>', unsafe_allow_html=True)
        with ch5:
            st.markdown(f'<div style="{chip.replace("#1e3354", rz_col+"44")}"><div style="{lbl}">RIZIV/INAMI</div>' +
                f'<div style="font-family:\'DM Mono\',monospace;font-size:.82rem;color:{rz_col};margin-top:3px;">{rz["status"]}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        col_mech, col_risk, col_admin = st.columns([1.1, 1.1, 1])

        with col_mech:
            st.markdown(f'<div style="font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:#4a7aaa;padding-bottom:5px;border-bottom:1px solid #1a2d4a;margin-bottom:10px;">Mechanism of Action</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:.82rem;color:#c0d0e8;line-height:1.7;margin-bottom:14px;">{view_dmt["mechanism"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:{view_color};padding-bottom:5px;border-bottom:1px solid {view_color}33;margin-bottom:10px;">👁 OCT / NEUROSIGHT Relevance</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:.82rem;color:#c0d0e8;line-height:1.7;background:rgba(255,255,255,0.03);border-radius:10px;padding:11px;border-left:3px solid {view_color};">{view_dmt["ms_ocular_note"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:#4a7aaa;padding-bottom:5px;border-bottom:1px solid #1a2d4a;margin:14px 0 8px;">Treatment Position</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:.8rem;color:#c0d0e8;margin-bottom:6px;">{view_dmt["rrms_rank"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:.76rem;color:#6a8caa;font-style:italic;">⚠ {view_dmt["interactions"]}</div>', unsafe_allow_html=True)

        with col_risk:
            st.markdown('<div style="font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:#4a7aaa;padding-bottom:5px;border-bottom:1px solid #1a2d4a;margin-bottom:10px;">Key Risks &amp; Safety</div>', unsafe_allow_html=True)
            for rname, rdesc in view_dmt["key_risks"]:
                st.markdown(
                    f'<div style="margin-bottom:11px;">' +
                    f'<div style="font-size:.81rem;font-weight:600;color:#c0d0e8;margin-bottom:3px;">{rname}</div>' +
                    f'<div style="font-size:.77rem;color:#8ab0d0;line-height:1.6;padding-left:10px;border-left:2px solid #1a2d4a;">{rdesc}</div></div>',
                    unsafe_allow_html=True)
            st.markdown('<div style="font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:#4a7aaa;padding-bottom:5px;border-bottom:1px solid #1a2d4a;margin:14px 0 8px;">🗓 Monitoring Schedule</div>', unsafe_allow_html=True)
            for m in view_dmt["monitoring"]:
                st.markdown(
                    f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:7px;">' +
                    f'<div style="width:6px;height:6px;border-radius:50%;background:{view_color};flex-shrink:0;margin-top:5px;"></div>' +
                    f'<div style="font-size:.79rem;color:#c0d0e8;line-height:1.5;">{m}</div></div>',
                    unsafe_allow_html=True)

        with col_admin:
            st.markdown(f'<div style="font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:{rz_col};padding-bottom:5px;border-bottom:1px solid {rz_col}44;margin-bottom:10px;">🇧🇪 RIZIV / INAMI Reimbursement</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid {rz_col}33;border-radius:12px;padding:14px;margin-bottom:12px;">' +
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">' +
                f'<span style="font-size:.8rem;font-weight:600;color:{rz_col};">{rz["status"]}</span>' +
                f'<span style="font-family:\"DM Mono\",monospace;font-size:.7rem;color:#4a7aaa;">{rz["chapter"]}</span></div>' +
                f'<div style="font-size:.62rem;color:#4a7aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px;">Conditions</div>' +
                f'<div style="font-size:.77rem;color:#c0d0e8;line-height:1.6;margin-bottom:8px;">{rz["conditions"]}</div>' +
                f'<div style="font-size:.62rem;color:#4a7aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px;">Patient Contribution</div>' +
                f'<div style="font-family:\"DM Mono\",monospace;font-size:.82rem;color:#69ffb0;margin-bottom:8px;">{rz["patient_contribution"]}</div>' +
                f'<div style="font-size:.62rem;color:#4a7aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px;">Prior Authorisation</div>' +
                f'<div style="font-size:.77rem;color:#c0d0e8;">{rz["prior_auth"]}</div></div>',
                unsafe_allow_html=True)

            st.markdown('<div style="font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:#4a7aaa;padding-bottom:5px;border-bottom:1px solid #1a2d4a;margin-bottom:10px;">📋 Prescription</div>', unsafe_allow_html=True)
            if already_prescribed:
                st.markdown(
                    f'<div style="background:rgba(0,230,118,0.08);border:1px solid #00e67644;border-radius:12px;padding:12px 16px;text-align:center;margin-bottom:8px;">' +
                    f'<div style="font-size:.85rem;color:#69ffb0;font-weight:600;">✓ On Prescription</div>' +
                    f'<div style="font-size:.74rem;color:#6a8caa;margin-top:3px;">{view_dmt["route"]}</div>' +
                    f'<div style="font-family:\"DM Mono\",monospace;font-size:.7rem;color:#4a7aaa;margin-top:3px;">Added: {datetime.now().strftime("%d %b %Y")}</div></div>',
                    unsafe_allow_html=True)
                if st.button("✕ Remove from Prescription", key="rx_remove", use_container_width=True):
                    st.session_state.prescriptions.remove(view_dmt_name)
                    st.rerun()
            else:
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.03);border:1px dashed #1e3354;border-radius:12px;padding:12px 16px;text-align:center;margin-bottom:8px;">' +
                    f'<div style="font-size:.78rem;color:#6a8caa;">Not prescribed</div>' +
                    f'<div style="font-size:.72rem;color:#4a7aaa;margin-top:2px;">{view_dmt["route"]}</div></div>',
                    unsafe_allow_html=True)
                if st.button(f"➕ Prescribe {view_dmt_name}", key="rx_add", use_container_width=True, type="primary"):
                    st.session_state.prescriptions.append(view_dmt_name)
                    st.rerun()

            if st.session_state.prescriptions:
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                st.markdown('<div style="font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:#4a7aaa;margin-bottom:6px;">Active Prescriptions</div>', unsafe_allow_html=True)
                for rx in st.session_state.prescriptions:
                    rx_d = DMT_INFO.get(rx, {})
                    rx_c = rx_d.get("color","#4a7aaa")
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;background:rgba(255,255,255,0.03);border-radius:8px;margin-bottom:4px;">' +
                        f'<div style="width:7px;height:7px;border-radius:50%;background:{rx_c};flex-shrink:0;"></div>' +
                        f'<div style="font-size:.8rem;color:#c0d0e8;flex:1;">{rx}</div>' +
                        f'<div style="font-family:\"DM Mono\",monospace;font-size:.68rem;color:#4a7aaa;">{rx_d.get("route","").split("·")[0].strip()}</div></div>',
                        unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ── KPIs ──────────────────────────────────────────────────────────────────────
kpi_cls={"alert":"ns-card ns-card-alert","warn":"ns-card ns-card-warn","ok":"ns-card ns-card-ok"}[status]
rnfl_d=rnfl[-1]-rnfl[0]; gcl_d=gcl[-1]-gcl[0]; vd_d=vd[-1]-vd[0]; faz_d=faz[-1]-faz[0]
c1,c2,c3,c4,c5=st.columns([1.2,1,1,1,1])
rc={'alert':'#ff6b6b','warn':'#ffd060','ok':'#69ffb0'}[status]
with c1:
    st.markdown(f"""<div class="{kpi_cls}"><div class="ns-metric-label">Progression Risk</div>
      <div class="ns-metric-val" style="color:{rc};">{risk}%</div>
      <div style="font-size:.75rem;color:#6a8caa;margin-top:3px;">AI fingerprint score</div></div>""",unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="ns-card"><div class="ns-metric-label">RNFL (avg)</div>
      <div class="ns-metric-val">{rnfl[-1]:.0f}<span style="font-size:.9rem;color:#6a8caa;">μm</span></div>
      <div class="ns-metric-delta">{delta_html(rnfl_d)} Δ baseline</div></div>""",unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="ns-card"><div class="ns-metric-label">GCL+IPL</div>
      <div class="ns-metric-val">{gcl[-1]:.0f}<span style="font-size:.9rem;color:#6a8caa;">μm</span></div>
      <div class="ns-metric-delta">{delta_html(gcl_d)} Δ baseline</div></div>""",unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="ns-card"><div class="ns-metric-label">Vasc. Density</div>
      <div class="ns-metric-val">{vd[-1]:.1f}<span style="font-size:.9rem;color:#6a8caa;">%</span></div>
      <div class="ns-metric-delta">{delta_html(vd_d,unit='%')} Δ baseline</div></div>""",unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="ns-card"><div class="ns-metric-label">FAZ Area</div>
      <div class="ns-metric-val">{faz[-1]:.3f}<span style="font-size:.9rem;color:#6a8caa;">mm²</span></div>
      <div class="ns-metric-delta">{delta_html(faz_d,unit='mm²',invert=True)} Δ baseline</div></div>""",unsafe_allow_html=True)

# ── OCT MAPS ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">OCT Retinal Maps · Canon OCT-A1 Style</div>',unsafe_allow_html=True)

# Layout: show only selected eyes
eyes_to_show=[]
if show_od: eyes_to_show.append(("OD","Right Eye (OD)",pt["seed"],pt["sectors_od"],pt["tsnit_avg_od"],thin))
if show_os: eyes_to_show.append(("OS","Left Eye (OS)",pt["seed"]+1,pt["sectors_os"],pt["tsnit_avg_os"],thin*0.97))

n_eyes=len(eyes_to_show)
if n_eyes==0:
    st.info("Select at least one eye in the sidebar.")
else:
    # Columns: one map column per eye + clocks column + TSNIT column
    col_widths=[1]*n_eyes+[0.85,1.5]
    cols=st.columns(col_widths)

    for i,(eye,eye_label,seed,sectors,avg,tf) in enumerate(eyes_to_show):
        with cols[i]:
            st.plotly_chart(map_fn(pt,eye,f"{map_type} — {eye_label}",tf),use_container_width=True)
            if show_bscan:
                st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:.68rem;color:#4a7aaa;text-align:center;margin-bottom:2px;">B-Scan · {eye_label}</div>',unsafe_allow_html=True)
                st.plotly_chart(make_bscan(seed,status),use_container_width=True)

    # Sector clocks
    with cols[n_eyes]:
        for eye,eye_label,seed,sectors,avg,tf in eyes_to_show:
            st.plotly_chart(make_clock(sectors,avg,f"RNFL Sectors {eye}"),use_container_width=True)
        # ONH table
        rng2=np.random.default_rng(pt["seed"]+90)
        da=round(rng2.uniform(1.15,1.75),2); ra=round(da*rng2.uniform(0.78,0.90),2)
        cd=round(rng2.uniform(0.12,0.22),2)
        st.markdown(f"""<div class="ns-card" style="padding:12px 14px;margin-top:0;">
          <div class="ns-metric-label" style="margin-bottom:6px;">ONH Parameters</div>
          <table style="width:100%;font-size:.75rem;font-family:'DM Mono',monospace;border-collapse:collapse;">
            <tr><td style="color:#6a8caa;padding:2px 0;">Disc Area</td><td style="color:#c0d0e8;text-align:right;">{da} mm²</td></tr>
            <tr><td style="color:#6a8caa;">Rim Area</td><td style="color:#c0d0e8;text-align:right;">{ra} mm²</td></tr>
            <tr><td style="color:#6a8caa;">C/D Area</td><td style="color:#{'ff6b6b' if cd>0.18 else 'c0d0e8'};text-align:right;">{cd}</td></tr>
            <tr><td style="color:#6a8caa;">TSNIT OD</td><td style="color:#{'ff6b6b' if pt['tsnit_avg_od']<85 else 'c0d0e8'};text-align:right;">{pt['tsnit_avg_od']} μm</td></tr>
            <tr><td style="color:#6a8caa;">TSNIT OS</td><td style="color:#{'ff6b6b' if pt['tsnit_avg_os']<85 else 'c0d0e8'};text-align:right;">{pt['tsnit_avg_os']} μm</td></tr>
            <tr><td style="color:#6a8caa;">Symmetry</td><td style="color:#c0d0e8;text-align:right;">0.90</td></tr>
          </table></div>""",unsafe_allow_html=True)

    # TSNIT
    with cols[n_eyes+1]:
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:.68rem;color:#4a7aaa;letter-spacing:.12em;text-transform:uppercase;margin-bottom:4px;">TSNIT Profile (3.45mm Circle)</div>',unsafe_allow_html=True)
        st.plotly_chart(make_tsnit(pt,show_od,show_os,show_normative),use_container_width=True)
        st.markdown("""<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:2px;">
          <div style="display:flex;align-items:center;gap:4px;font-size:.72rem;color:#6a8caa;">
            <div style="width:10px;height:10px;background:#009933;border-radius:2px;"></div>p&gt;5%</div>
          <div style="display:flex;align-items:center;gap:4px;font-size:.72rem;color:#6a8caa;">
            <div style="width:10px;height:10px;background:#cccc00;border-radius:2px;"></div>1–5%</div>
          <div style="display:flex;align-items:center;gap:4px;font-size:.72rem;color:#6a8caa;">
            <div style="width:10px;height:10px;background:#cc0000;border-radius:2px;"></div>p&lt;1%</div>
        </div>""",unsafe_allow_html=True)

        # Significance legend for deviation
        if map_type=="Deviation Map":
            st.markdown("""<div style="margin-top:10px;font-size:.72rem;color:#6a8caa;font-family:'DM Mono',monospace;">
              Warm = above norm &nbsp;|&nbsp; Cool = below norm<br>White = at norm (0μm deviation)</div>""",unsafe_allow_html=True)

# ── TRENDS + AI ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Longitudinal Biomarker Trends</div>',unsafe_allow_html=True)
col_trend,col_ai=st.columns([3,2])

with col_trend:
    layers_to_show=active_layers if active_layers else ["RNFL"]
    st.plotly_chart(make_longitudinal(dates,series,show_normative,layers_to_show),use_container_width=True)

with col_ai:
    if show_ai:
        ins={
            "alert":{"text":"Accelerated RNFL thinning at −2.1μm/quarter exceeds the progression threshold. Temporal sector shows p<1% significance on the deviation map. Vascular density decline in the superficial capillary plexus correlates with disease activity. Pattern consistent with subclinical relapse or early treatment failure.",
                "recs":["🔴 Unscheduled MRI within 4–6 weeks","🔴 Review DMT adherence & JCV status","🟡 Repeat OCT in 6 weeks","🟡 Cross-reference with serum NfL"],"conf":87},
            "warn":{"text":"Moderate RNFL and GCL thinning within expected range. Vascular dropout in the deep capillary plexus is 1.4 SD above normative cohort mean. Attention maps highlight parafoveal temporal sector as primary region of subclinical change.",
                "recs":["🟡 Maintain current DMT","🟡 Increase scan frequency to 8 weeks","🟢 Next MRI on schedule","🟢 Patient-reported outcomes recommended"],"conf":72},
            "ok":{"text":"Retinal biomarker trajectory stable across all measured layers. RNFL and GCL thinning rates within physiological ageing range. Vascular density preserved. No structural signal of subclinical disease activity detected in this scanning window.",
                "recs":["🟢 Continue current DMT — good response","🟢 Maintain quarterly OCT schedule","🟢 Next MRI as planned","🟢 Consider extended monitoring interval"],"conf":91},
        }[status]
        st.markdown(f"""<div class="ai-insight">
          <div class="ai-tag">▶ NEUROSIGHT AI · Temporal CNN + Attention · Confidence {ins['conf']}%</div>
          <div class="ai-text">{ins['text']}</div></div>""",unsafe_allow_html=True)

        recs_html = "".join(
            f'<div style="display:flex;align-items:flex-start;gap:10px;padding:7px 0;'
            f'border-bottom:1px solid rgba(0,114,255,0.12);">'
            f'<div style="font-size:.86rem;color:#c0d0e8;line-height:1.5;">{r}</div></div>'
            for r in ins["recs"]
        )
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#061428,#080f22);'
            f'border:1px solid #0a2a5a;border-radius:14px;padding:14px 18px;margin-top:10px;">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:.65rem;color:#2a6acc;'
            f'letter-spacing:.12em;text-transform:uppercase;margin-bottom:8px;">📋 Clinical Recommendations</div>'
            + recs_html + '</div>',
            unsafe_allow_html=True)
    else:
        st.markdown('<div class="ns-card"><div style="color:#4a7aaa;font-size:.85rem;">Enable AI Annotations in the sidebar to view the NEUROSIGHT clinical interpretation.</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:12px;">Clinical Timeline</div>',unsafe_allow_html=True)
    tls={"alert":[("#ff6b6b","Apr 2024","⚠ Relapse — visual disturbance"),("#ffaa00","Jan 2024","MRI: 2 new T2 lesions"),
            ("#00c6ff","Oct 2023","OCT: RNFL 91μm (−3.1 from prior)"),("#69ffb0","Mar 2023","Natalizumab dose 12 — stable"),
            ("#6a8caa","Jan 2023","Baseline OCT established")],
        "warn":[("#ffaa00","Mar 2024","OCT: Vascular density decline noted"),("#00c6ff","Jan 2024","MRI: Stable"),
            ("#69ffb0","Jul 2023","Ocrelizumab infusion cycle 6"),("#6a8caa","Jan 2023","Baseline OCT established")],
        "ok":[("#69ffb0","Apr 2024","OCT: All biomarkers within limits"),("#69ffb0","Jan 2024","MRI: Stable — no lesion activity"),
            ("#69ffb0","Oct 2023","Cladribine cycle 2 completed"),("#6a8caa","Jan 2023","Baseline OCT established")],
    }[status]
    tl='<div class="ns-card" style="padding:14px 16px;">'
    for col,date,event in tls:
        tl+=f'<div class="tl-item"><div class="tl-dot" style="background:{col};box-shadow:0 0 5px {col}50;"></div><div><div style="font-family:\'DM Mono\',monospace;font-size:.7rem;color:#4a7aaa;">{date}</div><div style="font-size:.82rem;color:#c0d0e8;">{event}</div></div></div>'
    st.markdown(tl+"</div>",unsafe_allow_html=True)

st.markdown("""<hr class='ns'><div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;'>
  <div style='font-family:DM Mono,monospace;font-size:.68rem;color:#2a4a6a;'>NEUROSIGHT · EP PerMed Hackathon 2025 · Personalised MS Monitoring</div>
  <div style='font-size:.68rem;color:#2a4a6a;'>⚠️ Research prototype — not validated for clinical decision-making</div></div>""",unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════════
# NETWORK INTELLIGENCE SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='ns'>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin:8px 0 20px;">
  <div style="width:42px;height:42px;border-radius:50%;
    background:linear-gradient(135deg,#0055ff,#7b2fff);
    display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;">🌐</div>
  <div>
    <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
      background:linear-gradient(90deg,#00c6ff,#7b2fff);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
      NEUROSIGHT Network Intelligence</div>
    <div style="font-size:.8rem;color:#6a8caa;margin-top:2px;">
      Federated · 47 hospitals · 12,840 anonymised MS patients · 8 countries</div>
  </div>
  <div style="margin-left:auto;display:flex;gap:8px;align-items:center;">
    <div style="width:8px;height:8px;border-radius:50%;background:#00e676;
      box-shadow:0 0 8px #00e676;animation:pulse 2s infinite;"></div>
    <div style="font-family:'DM Mono',monospace;font-size:.75rem;color:#00e676;">LIVE NETWORK</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Generate synthetic network population ─────────────────────────────────────
rng_net = np.random.default_rng(42)
N = 350  # network patients for scatter

# Each patient: 2D UMAP-like coords, DMT, outcome, MS type, RNFL rate
net_dmts    = rng_net.choice(["Natalizumab","Ocrelizumab","Cladribine","Interferon","Dimethyl Fumarate"], N,
                              p=[0.22,0.28,0.18,0.17,0.15])
net_types   = rng_net.choice(["RRMS","PPMS","SPMS"], N, p=[0.65,0.20,0.15])
net_edss    = rng_net.uniform(0.5, 7.0, N)
net_rnfl_r  = rng_net.uniform(-3.5, 0.2, N)   # μm/quarter thinning rate
net_outcome = np.where(net_rnfl_r > -1.2, "Stable", np.where(net_rnfl_r > -2.2, "Monitoring", "Progressing"))

# UMAP-like 2D embedding — clusters by DMT + type
umap_x = np.zeros(N); umap_y = np.zeros(N)
centres = {"Natalizumab":(1.8,2.2),"Ocrelizumab":(-1.5,1.8),"Cladribine":(0.5,-2.1),
           "Interferon":(-2.5,-1.2),"Dimethyl Fumarate":(2.8,-0.8)}
for i in range(N):
    cx,cy = centres[net_dmts[i]]
    umap_x[i] = cx + rng_net.normal(0, 0.85)
    umap_y[i] = cy + rng_net.normal(0, 0.85)
    if net_types[i] == "PPMS":
        umap_x[i] += rng_net.normal(-0.6, 0.3)
        umap_y[i] += rng_net.normal(-0.6, 0.3)

# Current patient position (based on their profile)
pt_x = {"alert":-1.2, "warn":0.2, "ok":0.6}[status]
pt_y = {"alert":1.4,  "warn":-1.8, "ok":-2.3}[status]

# ── Tab layout for three network features ────────────────────────────────────
net_tab1, net_tab2, net_tab3 = st.tabs([
    "  🗺  Patient Stratification Map  ",
    "  📊  Network Statistics  ",
    "  🆘  SOS — Clinical Decision Support  ",
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1: PATIENT STRATIFICATION MAP
# ────────────────────────────────────────────────────────────────────────────
with net_tab1:
    st.markdown('<div class="section-title">Population Embedding · OCT + Clinical Feature Space (UMAP)</div>',
                unsafe_allow_html=True)

    col_scatter, col_scatter_info = st.columns([3, 1])

    with col_scatter:
        dmt_palette = {"Natalizumab":"#0072ff","Ocrelizumab":"#7b2fff",
                       "Cladribine":"#00e676","Interferon":"#ffaa00","Dimethyl Fumarate":"#ff6b6b"}
        outcome_shape = {"Stable":"circle","Monitoring":"diamond","Progressing":"x"}

        fig_sc = go.Figure()

        # Network patients by DMT
        for dmt_n, col_n in dmt_palette.items():
            mask = net_dmts == dmt_n
            fig_sc.add_trace(go.Scatter(
                x=umap_x[mask], y=umap_y[mask],
                mode='markers', name=dmt_n,
                marker=dict(
                    size=7, color=col_n, opacity=0.55,
                    line=dict(width=0),
                    symbol=[outcome_shape[o] for o in net_outcome[mask]],
                ),
                hovertemplate=(
                    f'<b>{dmt_n}</b><br>'
                    'EDSS: %{customdata[0]:.1f}<br>'
                    'RNFL rate: %{customdata[1]:.2f} μm/q<br>'
                    'Outcome: %{customdata[2]}<extra></extra>'
                ),
                customdata=np.column_stack([
                    net_edss[mask],
                    net_rnfl_r[mask],
                    net_outcome[mask],
                ]),
            ))

        # Highlight zone — similar patients to current
        theta_z = np.linspace(0, 2*np.pi, 60)
        zone_r = 1.2
        fig_sc.add_trace(go.Scatter(
            x=pt_x + zone_r*np.cos(theta_z),
            y=pt_y + zone_r*np.sin(theta_z),
            mode='lines', showlegend=False,
            line=dict(color='#ffaa00', width=1.5, dash='dash'),
            hoverinfo='skip',
        ))
        fig_sc.add_annotation(
            x=pt_x + zone_r + 0.05, y=pt_y,
            text="Similar patients", showarrow=False,
            font=dict(size=9, color='#ffaa00', family='DM Mono'),
            xanchor='left',
        )

        # Current patient marker
        fig_sc.add_trace(go.Scatter(
            x=[pt_x], y=[pt_y], mode='markers',
            name=f'▶ {selected_patient.split("·")[1].strip()} (current)',
            marker=dict(size=18, color='#ffffff', symbol='star',
                        line=dict(width=2, color='#ffaa00')),
            hovertemplate='<b>Current Patient</b><br>EDSS: ' + str(pt["edss"]) +
                          '<br>Risk: ' + str(risk) + '%<extra></extra>',
        ))

        fig_sc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(8,18,38,0.8)',
            height=440,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(size=9, color='#8ab0d0'), bgcolor='rgba(6,13,26,0.8)',
                        bordercolor='#1a2d4a', borderwidth=1, x=0.01, y=0.99),
            xaxis=dict(showgrid=True, gridcolor='#0d1e35', zeroline=False,
                       showticklabels=False, title='UMAP dim 1',
                       title_font=dict(size=9, color='#4a7aaa')),
            yaxis=dict(showgrid=True, gridcolor='#0d1e35', zeroline=False,
                       showticklabels=False, title='UMAP dim 2',
                       title_font=dict(size=9, color='#4a7aaa')),
            hoverlabel=dict(bgcolor='#0d1e35', bordercolor='#1e3354', font_color='#e8edf5'),
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown(
            '<div style="font-size:.72rem;color:#4a7aaa;text-align:center;">'
            '◆ = Monitoring &nbsp;|&nbsp; ✕ = Progressing &nbsp;|&nbsp; ● = Stable &nbsp;|&nbsp;'
            ' ⭐ = Current patient &nbsp;|&nbsp; Dashed ring = Similar phenotype cluster</div>',
            unsafe_allow_html=True)

    with col_scatter_info:
        # Count similar patients in zone
        dist = np.sqrt((umap_x - pt_x)**2 + (umap_y - pt_y)**2)
        similar_mask = dist < 1.2
        n_sim = similar_mask.sum()
        sim_dmts = net_dmts[similar_mask]
        sim_outcomes = net_outcome[similar_mask]

        st.markdown('<div class="section-title">Cluster Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ns-card" style="padding:14px 16px;">'
            f'<div class="ns-metric-label">Similar Patients Found</div>'
            f'<div class="ns-metric-val" style="color:#ffaa00;">{n_sim}</div>'
            f'<div style="font-size:.75rem;color:#6a8caa;margin-top:3px;">within phenotype cluster</div>'
            f'</div>', unsafe_allow_html=True)

        # DMT breakdown in cluster
        st.markdown('<div style="font-size:.65rem;letter-spacing:.12em;text-transform:uppercase;'
                    'color:#4a7aaa;margin-bottom:8px;">DMT Distribution in Cluster</div>',
                    unsafe_allow_html=True)
        for dmt_n, col_n in dmt_palette.items():
            n_d = (sim_dmts == dmt_n).sum()
            if n_d == 0: continue
            pct = n_d / n_sim * 100
            st.markdown(
                f'<div style="margin-bottom:7px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                f'<span style="font-size:.77rem;color:#c0d0e8;">{dmt_n}</span>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:.75rem;color:{col_n};">{n_d}</span></div>'
                f'<div style="background:#0d1e35;border-radius:4px;height:5px;">'
                f'<div style="background:{col_n};width:{pct:.0f}%;height:5px;border-radius:4px;"></div>'
                f'</div></div>',
                unsafe_allow_html=True)

        # Outcome breakdown
        st.markdown('<div style="font-size:.65rem;letter-spacing:.12em;text-transform:uppercase;'
                    'color:#4a7aaa;margin:12px 0 8px;">Outcomes in Cluster</div>',
                    unsafe_allow_html=True)
        for outcome, ocol in [("Stable","#69ffb0"),("Monitoring","#ffaa00"),("Progressing","#ff6b6b")]:
            n_o = (sim_outcomes == outcome).sum()
            pct_o = n_o / n_sim * 100 if n_sim > 0 else 0
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:5px 0;border-bottom:1px solid #0d1e35;">'
                f'<span style="font-size:.77rem;color:{ocol};">● {outcome}</span>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:.77rem;color:{ocol};">'
                f'{pct_o:.0f}%</span></div>',
                unsafe_allow_html=True)

        # Outlier flag
        if status == "alert":
            st.markdown(
                '<div style="background:rgba(255,68,68,0.08);border:1px solid #ff444433;'
                'border-radius:10px;padding:10px 12px;margin-top:12px;">'
                '<div style="font-size:.72rem;color:#ff6b6b;font-weight:600;margin-bottom:4px;">'
                '⚠ Outlier Flag</div>'
                '<div style="font-size:.76rem;color:#c0d0e8;line-height:1.5;">'
                'This patient\'s RNFL trajectory places them in the bottom 8th percentile '
                'of their DMT cohort. Review warranted.</div></div>',
                unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2: NETWORK STATISTICS
# ────────────────────────────────────────────────────────────────────────────
with net_tab2:
    st.markdown('<div class="section-title">Network-Wide DMT Outcome Statistics · 12,840 Patients · 47 Centres</div>',
                unsafe_allow_html=True)

    ns_col1, ns_col2, ns_col3 = st.columns(3)

    # DMT efficacy comparison bar chart
    with ns_col1:
        dmt_names_net  = ["Natalizumab","Ocrelizumab","Cladribine","Interferon","Dim. Fumarate"]
        stable_pct     = [72, 68, 65, 48, 52]
        monitoring_pct = [18, 20, 22, 30, 28]
        progressing_pct= [10, 12, 13, 22, 20]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='Stable', x=dmt_names_net, y=stable_pct,
            marker_color='#00e676', marker_line_width=0))
        fig_bar.add_trace(go.Bar(name='Monitoring', x=dmt_names_net, y=monitoring_pct,
            marker_color='#ffaa00', marker_line_width=0))
        fig_bar.add_trace(go.Bar(name='Progressing', x=dmt_names_net, y=progressing_pct,
            marker_color='#ff6b6b', marker_line_width=0))

        fig_bar.update_layout(
            barmode='stack', height=300,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(8,18,38,0.6)',
            margin=dict(l=5,r=5,t=30,b=5),
            title=dict(text='Outcome by DMT (%)', font=dict(size=10,color='#8ab0d0',family='DM Mono'), x=0.5),
            legend=dict(font=dict(size=9,color='#8ab0d0'),bgcolor='rgba(0,0,0,0)',orientation='h',y=1.12),
            xaxis=dict(tickfont=dict(size=8,color='#6a8caa'), gridcolor='#0d1e35', linecolor='#1a2d4a'),
            yaxis=dict(tickfont=dict(size=8,color='#6a8caa'), gridcolor='#1a2d4a',
                       title='%', title_font=dict(size=9,color='#4a7aaa')),
            hoverlabel=dict(bgcolor='#0d1e35',font_color='#e8edf5'),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # RNFL thinning rate distribution
    with ns_col2:
        rng_dist = np.random.default_rng(77)
        rnfl_rates_pop = rng_dist.normal(-1.1, 0.9, 500)
        pt_rate = -2.1 if status=="alert" else -0.9 if status=="warn" else -0.4

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=rnfl_rates_pop, nbinsx=35,
            marker_color='#0072ff', opacity=0.6,
            name='Network population',
        ))
        fig_hist.add_vline(x=pt_rate, line_color='#ff6b6b' if status=='alert' else '#ffaa00' if status=='warn' else '#69ffb0',
            line_width=2.5, line_dash='dash',
            annotation_text=f'This patient ({pt_rate} μm/q)',
            annotation_font=dict(size=9, color='#e8edf5'),
            annotation_position='top left')
        fig_hist.add_vline(x=-1.1, line_color='#6a8caa', line_width=1, line_dash='dot',
            annotation_text='Network median',
            annotation_font=dict(size=8, color='#6a8caa'),
            annotation_position='bottom right')

        fig_hist.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(8,18,38,0.6)',
            margin=dict(l=5,r=5,t=30,b=5),
            title=dict(text='RNFL Thinning Rate Distribution (μm/quarter)', font=dict(size=10,color='#8ab0d0',family='DM Mono'), x=0.5),
            showlegend=False,
            xaxis=dict(tickfont=dict(size=8,color='#6a8caa'), title='μm/quarter',
                       title_font=dict(size=9,color='#4a7aaa'), gridcolor='#1a2d4a', linecolor='#1a2d4a'),
            yaxis=dict(tickfont=dict(size=8,color='#6a8caa'), title='Patients',
                       title_font=dict(size=9,color='#4a7aaa'), gridcolor='#1a2d4a'),
            hoverlabel=dict(bgcolor='#0d1e35',font_color='#e8edf5'),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Time to treatment switch heatmap by DMT x MS type
    with ns_col3:
        ms_types_h  = ["RRMS","SPMS","PPMS"]
        dmt_names_h = ["Natalizumab","Ocrelizumab","Cladribine","Interferon"]
        switch_months = np.array([
            [28, 22, 19, 14],
            [18, 24, 16, 11],
            [12, 20, 14,  9],
        ])
        fig_hm = go.Figure(go.Heatmap(
            z=switch_months, x=dmt_names_h, y=ms_types_h,
            colorscale=[[0,'#cc2200'],[0.4,'#ffaa00'],[0.7,'#aaff00'],[1.0,'#00e676']],
            showscale=True,
            colorbar=dict(thickness=8, len=0.8,
                tickfont=dict(size=8,color='#8ab0d0',family='DM Mono'),
                title=dict(text='months', font=dict(size=8,color='#6a8caa')),
                outlinecolor='#1a2d4a', outlinewidth=1),
            text=switch_months, texttemplate='%{text}m',
            textfont=dict(size=10, color='white', family='DM Mono'),
            hovertemplate='%{y} + %{x}<br>Median time stable: %{z} months<extra></extra>',
        ))
        fig_hm.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(8,18,38,0.6)',
            margin=dict(l=5,r=5,t=30,b=5),
            title=dict(text='Median Months Stable Before Switch', font=dict(size=10,color='#8ab0d0',family='DM Mono'), x=0.5),
            xaxis=dict(tickfont=dict(size=8,color='#6a8caa'), side='bottom'),
            yaxis=dict(tickfont=dict(size=8,color='#6a8caa')),
            hoverlabel=dict(bgcolor='#0d1e35',font_color='#e8edf5'),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # Network KPI strip
    st.markdown('<div class="section-title">Network Summary Metrics</div>', unsafe_allow_html=True)
    nk1,nk2,nk3,nk4,nk5,nk6 = st.columns(6)
    net_kpis = [
        ("Hospitals Connected","47","🏥"),
        ("Patients in Network","12,840","👥"),
        ("Countries","8","🌍"),
        ("Avg RNFL scans/patient","11.4","👁"),
        ("DMTs Tracked","12","💊"),
        ("Models Last Updated","6h ago","🔄"),
    ]
    chip_s = "background:rgba(255,255,255,0.03);border:1px solid #1e3354;border-radius:12px;padding:12px 14px;text-align:center;"
    for col_nk, (label, val, icon) in zip([nk1,nk2,nk3,nk4,nk5,nk6], net_kpis):
        with col_nk:
            st.markdown(
                f'<div style="{chip_s}">'
                f'<div style="font-size:1.1rem;margin-bottom:4px;">{icon}</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:1rem;color:#00c6ff;font-weight:500;">{val}</div>'
                f'<div style="font-size:.65rem;color:#4a7aaa;text-transform:uppercase;letter-spacing:.08em;margin-top:3px;">{label}</div>'
                f'</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 3: SOS CLINICAL DECISION SUPPORT
# ────────────────────────────────────────────────────────────────────────────
with net_tab3:
    # SOS header
    sos_col_hdr, sos_col_btn = st.columns([3,1])
    with sos_col_hdr:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a0505,#120308);
            border:1px solid #ff444444;border-radius:16px;padding:16px 20px;margin-bottom:12px;">
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="font-size:2rem;">🆘</div>
            <div>
              <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#ff8888;">
                SOS — Clinical Decision Support</div>
              <div style="font-size:.82rem;color:#8ab0d0;margin-top:3px;">
                When you need a second opinion from 12,840 patients — search the network
                for the closest matching cases and surface what worked for them.</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with sos_col_btn:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("🆘 Activate SOS Search", use_container_width=True, type="primary"):
            st.session_state.sos_active = True
            st.rerun()
        if st.session_state.sos_active:
            if st.button("✕ Clear Results", use_container_width=True):
                st.session_state.sos_active = False
                st.rerun()

    if not st.session_state.sos_active:
        # Search parameters form
        st.markdown('<div class="section-title">Search Parameters</div>', unsafe_allow_html=True)
        sp1, sp2, sp3 = st.columns(3)
        with sp1:
            st.markdown('<div style="font-size:.7rem;color:#6a8caa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">MS Type</div>', unsafe_allow_html=True)
            sos_type = st.selectbox("", ["RRMS","PPMS","SPMS"], index=0 if pt["type"]=="RRMS" else 1, label_visibility="collapsed", key="sos_type")
            st.markdown('<div style="font-size:.7rem;color:#6a8caa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;margin-top:10px;">Prior DMT</div>', unsafe_allow_html=True)
            sos_prior = st.selectbox("", list(DMT_INFO.keys()), label_visibility="collapsed", key="sos_prior")
        with sp2:
            st.markdown('<div style="font-size:.7rem;color:#6a8caa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">EDSS Range</div>', unsafe_allow_html=True)
            sos_edss = st.slider("", 0.0, 9.0, (max(0.0, pt["edss"]-1.0), min(9.0, pt["edss"]+1.0)), 0.5, label_visibility="collapsed", key="sos_edss")
            st.markdown('<div style="font-size:.7rem;color:#6a8caa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;margin-top:10px;">RNFL Rate (μm/q)</div>', unsafe_allow_html=True)
            sos_rnfl = st.slider("", -4.0, 0.0, (-2.5, -1.0), 0.1, label_visibility="collapsed", key="sos_rnfl")
        with sp3:
            st.markdown('<div style="font-size:.7rem;color:#6a8caa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">Disease Duration</div>', unsafe_allow_html=True)
            sos_dur = st.select_slider("", ["<2 yrs","2–5 yrs","5–10 yrs",">10 yrs"], value="5–10 yrs", label_visibility="collapsed", key="sos_dur")
            st.markdown('<div style="font-size:.7rem;color:#6a8caa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;margin-top:10px;">Outcome Target</div>', unsafe_allow_html=True)
            sos_target = st.selectbox("", ["Stable at 18 months","Reduced relapse rate","Stable RNFL at 12 months"], label_visibility="collapsed", key="sos_tgt")

        st.markdown(
            '<div style="font-size:.76rem;color:#6a8caa;background:rgba(255,255,255,0.02);'
            'border:1px solid #1a2d4a;border-radius:10px;padding:10px 14px;margin-top:6px;">'
            '💡 The SOS engine will search the federated network for patients matching this profile '
            'and return ranked treatment outcomes. All patient data is anonymised and never leaves '
            'its source hospital. Results represent real clinical precedents, not guidelines.</div>',
            unsafe_allow_html=True)

    else:
        # ── SOS RESULTS ──────────────────────────────────────────────────────
        # Simulate matched cases
        rng_sos = np.random.default_rng(pt["seed"] + 200)

        sos_results = {
            "alert": {
                "n_matched": 23,
                "profile": f"RRMS · EDSS {pt['edss']} · RNFL −2.1μm/q · Natalizumab failure · 5y disease duration",
                "cases": [
                    {"rank":1,"dmt":"Ocrelizumab","n":9,"stable_pct":78,"rnfl_change":"-0.6μm/q at 18mo",
                     "median_switch":"6 weeks","hospitals":"UZ Leuven, Amsterdam UMC, CHU Liège",
                     "note":"Fastest stabilisation profile. Recommended for JCV+ patients switching from Natalizumab.",
                     "color":"#7b2fff"},
                    {"rank":2,"dmt":"Cladribine","n":7,"stable_pct":71,"rnfl_change":"-0.8μm/q at 18mo",
                     "median_switch":"8 weeks","hospitals":"UZ Gent, KU Leuven, Erasmus MC",
                     "note":"Suitable if patient prefers oral pulse therapy. Requires lymphocyte monitoring.",
                     "color":"#00e676"},
                    {"rank":3,"dmt":"Natalizumab (dose optimisation)","n":7,"stable_pct":58,"rnfl_change":"-1.2μm/q at 18mo",
                     "median_switch":"Continued","hospitals":"CHUPS Rouen, ZOL Genk",
                     "note":"Partial response in subset. Only viable if JCV index <0.9 and adherence confirmed.",
                     "color":"#0072ff"},
                ],
            },
            "warn": {
                "n_matched": 31,
                "profile": f"RRMS · EDSS {pt['edss']} · RNFL −0.9μm/q · Ocrelizumab · vascular dropout signal · 8y disease duration",
                "cases": [
                    {"rank":1,"dmt":"Continue Ocrelizumab + increased monitoring","n":14,"stable_pct":82,"rnfl_change":"-0.5μm/q at 12mo",
                     "median_switch":"No switch","hospitals":"UZ Leuven, MS Centre Brussels, AKH Wien",
                     "note":"Majority stabilised with quarterly OCT monitoring and infusion schedule adherence. No switch needed.",
                     "color":"#7b2fff"},
                    {"rank":2,"dmt":"Ocrelizumab + add serum NfL monitoring","n":11,"stable_pct":76,"rnfl_change":"-0.6μm/q at 12mo",
                     "median_switch":"No switch","hospitals":"CHU Bordeaux, LUMC Leiden",
                     "note":"NfL cross-referencing improved early detection in this subcohort.",
                     "color":"#00c6ff"},
                    {"rank":3,"dmt":"Switch to Ofatumumab","n":6,"stable_pct":68,"rnfl_change":"-0.7μm/q at 12mo",
                     "median_switch":"4 weeks","hospitals":"Hôpital Pitié-Salpêtrière",
                     "note":"Consider if IV access is problematic — subcutaneous equivalent with similar B-cell depletion.",
                     "color":"#ffaa00"},
                ],
            },
            "ok": {
                "n_matched": 44,
                "profile": f"RRMS · EDSS {pt['edss']} · RNFL −0.4μm/q · Cladribine · stable trajectory · 2y disease duration",
                "cases": [
                    {"rank":1,"dmt":"Continue Cladribine · Year 2 course","n":22,"stable_pct":91,"rnfl_change":"+0.1μm/q at 18mo",
                     "median_switch":"No switch","hospitals":"UZ Leuven, MS Centre Hasselt, CHU Namur",
                     "note":"Excellent response profile. Complete year 2 course. No changes indicated.",
                     "color":"#00e676"},
                    {"rank":2,"dmt":"Cladribine + extend OCT interval to 6 months","n":14,"stable_pct":89,"rnfl_change":"-0.2μm/q at 18mo",
                     "median_switch":"No switch","hospitals":"AMC Amsterdam, ZNA Antwerp",
                     "note":"Stable patients in this cohort safely extended OCT interval without missing early signals.",
                     "color":"#00e676"},
                    {"rank":3,"dmt":"Monitor — no DMT change","n":8,"stable_pct":85,"rnfl_change":"-0.3μm/q at 18mo",
                     "median_switch":"No switch","hospitals":"Cliniques St-Luc Brussels",
                     "note":"Watchful waiting appropriate given excellent current trajectory.",
                     "color":"#69ffb0"},
                ],
            },
        }[status]

        # Results header
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#0a1628,#070d1e);'
            f'border:1px solid #2a6acc55;border-radius:14px;padding:16px 20px;margin-bottom:16px;">'
            f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:.65rem;color:#2a6acc;'
            f'letter-spacing:.12em;text-transform:uppercase;">🔍 SOS SEARCH COMPLETE</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:.9rem;color:#69ffb0;font-weight:600;">'
            f'{sos_results["n_matched"]} matching patients found</div></div>'
            f'<div style="font-size:.8rem;color:#8ab0d0;margin-top:6px;">Profile searched: '
            f'<span style="color:#c0d0e8;">{sos_results["profile"]}</span></div>'
            f'</div>', unsafe_allow_html=True)

        # Case cards
        for case in sos_results["cases"]:
            rank_color = ["#ffaa00","#8ab0d0","#6a8caa"][case["rank"]-1]
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#0a1628,#080f20);'
                f'border:1px solid {case["color"]}44;border-radius:14px;'
                f'padding:16px 20px;margin-bottom:12px;position:relative;overflow:hidden;">'
                f'<div style="position:absolute;top:0;left:0;bottom:0;width:4px;background:{case["color"]};"></div>'
                f'<div style="padding-left:8px;">'

                f'<div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:10px;">'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:.7rem;color:{rank_color};'
                f'background:rgba(255,255,255,0.05);padding:2px 8px;border-radius:20px;'
                f'border:1px solid {rank_color}44;">#{case["rank"]} MATCH</span>'
                f'<span style="font-family:\'Syne\',sans-serif;font-size:1rem;font-weight:700;color:{case["color"]};">'
                f'{case["dmt"]}</span></div>'
                f'<div style="display:flex;gap:12px;flex-wrap:wrap;">'
                f'<div style="text-align:center;background:rgba(255,255,255,0.04);border-radius:8px;padding:6px 12px;">'
                f'<div style="font-size:.6rem;color:#4a7aaa;text-transform:uppercase;letter-spacing:.08em;">Patients</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:.9rem;color:#c0d0e8;">{case["n"]}</div></div>'
                f'<div style="text-align:center;background:rgba(255,255,255,0.04);border-radius:8px;padding:6px 12px;">'
                f'<div style="font-size:.6rem;color:#4a7aaa;text-transform:uppercase;letter-spacing:.08em;">Stable at target</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:.9rem;color:#69ffb0;">{case["stable_pct"]}%</div></div>'
                f'<div style="text-align:center;background:rgba(255,255,255,0.04);border-radius:8px;padding:6px 12px;">'
                f'<div style="font-size:.6rem;color:#4a7aaa;text-transform:uppercase;letter-spacing:.08em;">RNFL outcome</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:.9rem;color:#c0d0e8;">{case["rnfl_change"]}</div></div>'
                f'<div style="text-align:center;background:rgba(255,255,255,0.04);border-radius:8px;padding:6px 12px;">'
                f'<div style="font-size:.6rem;color:#4a7aaa;text-transform:uppercase;letter-spacing:.08em;">Median switch</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:.9rem;color:#c0d0e8;">{case["median_switch"]}</div></div>'
                f'</div></div>'

                f'<div style="font-size:.8rem;color:#c0d0e8;line-height:1.6;margin-bottom:8px;">'
                f'💡 {case["note"]}</div>'

                f'<div style="font-size:.72rem;color:#4a7aaa;">'
                f'🏥 Evidence from: {case["hospitals"]}</div>'
                f'</div></div>',
                unsafe_allow_html=True)

        # Confidence footer
        st.markdown(
            '<div style="font-size:.72rem;color:#4a7aaa;text-align:center;margin-top:8px;">'
            '⚠️ SOS results are federated evidence summaries from anonymised real-world data. '
            'All treatment decisions remain the responsibility of the treating clinician. '
            'Not a substitute for clinical judgement.</div>',
            unsafe_allow_html=True)

st.markdown("""<hr class='ns'><div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;'>
  <div style='font-family:DM Mono,monospace;font-size:.68rem;color:#2a4a6a;'>NEUROSIGHT · EP PerMed Hackathon 2025 · Personalised MS Monitoring · Network Intelligence</div>
  <div style='font-size:.68rem;color:#2a4a6a;'>⚠️ Research prototype — not validated for clinical decision-making</div></div>""",unsafe_allow_html=True)
