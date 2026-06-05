"""
EngageIQ — Streamlit Dashboard
Run with: streamlit run src/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import threading
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Page config — MUST be first Streamlit command ────────────────────────────
st.set_page_config(
    page_title="EngageIQ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports — show error in UI if anything fails ─────────────────────
try:
    from database import (
        init_db, _load_seed, get_record_count, log_feedback,
        get_feedback, get_saved_opportunities, upsert_persona, get_persona,
        reset_opportunities,
    )
    from personas import seed_personas, PERSONAS, persona_vector_text
    from embeddings import (
        embed_all_opportunities, embed_persona,
        build_faiss_index, encode_text,
    )
    from ranking import rank_opportunities, rank_from_rows, ndcg_at_k
    from analytics import (
        load_df, top_domains, source_distribution,
        top_opportunities, trending_by_stars,
        engagement_volume_over_time, summary_stats,
    )
except Exception as _import_err:
    import traceback
    st.error(f"**Import error:** {_import_err}")
    st.code(traceback.format_exc())
    st.stop()

# ── Force light mode + design system ─────────────────────────────────────────
st.markdown("""
<style>
/* ════════════════════════════════════════════
   NUCLEAR LIGHT MODE — override Streamlit dark theme completely
   ════════════════════════════════════════════ */
html, body, #root, #root > div,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="stMainBlockContainer"],
[data-testid="stMain"],
.main, .block-container,
section.main > div { background-color: #f8f9fb !important; }

/* All text dark by default */
html, body, * { color: #1a202c; }

/* ── Layout ── */
.block-container { padding-top: 0.8rem !important; padding-bottom: 0.8rem !important; max-width: 1200px; }

/* ── Sidebar — warm peach from illustration ── */
[data-testid="stSidebar"] { background-color: #fde8d0 !important; border-right: 1px solid #f4c4a0 !important; }
[data-testid="stSidebar"] * { color: #1a202c !important; background-color: transparent; }

/* ── Top header bar — hidden ── */
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Tabs ── */
[data-testid="stTabs"] { background: transparent !important; }
button[data-baseweb="tab"] {
    font-size: 0.875rem !important; font-weight: 600 !important;
    color: #64748b !important; background: transparent !important; padding: 8px 18px !important;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #F4674A !important; border-bottom: 2px solid #F4674A !important; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #ffffff !important; border: 1px solid #e8ecf0 !important;
    border-radius: 10px !important; padding: 14px 18px !important;
}
div[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.76rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #1a202c !important; font-size: 1.55rem !important; font-weight: 700 !important; }

/* ── Expanders — warm peach cards ── */
[data-testid="stExpander"] { background: #fde8d0 !important; border: 1px solid #f4c4a0 !important; border-radius: 8px !important; margin-bottom: 6px !important; }
[data-testid="stExpander"] summary { background: #fde8d0 !important; color: #000000 !important; font-weight: 600 !important; font-size: 0.88rem !important; padding: 8px 14px !important; border-radius: 8px !important; }
[data-testid="stExpander"] summary:hover { background: #f9d8bc !important; }
[data-testid="stExpander"] > div > div { background: #fde8d0 !important; padding-top: 0 !important; }
[data-testid="stExpander"][open] { border-color: #f4c4a0 !important; }
/* Force all text inside cards to black */
[data-testid="stExpander"] *:not(button) { color: #000000 !important; }
/* Tighten metric cards inside expanders — inherit peach bg */
[data-testid="stExpander"] div[data-testid="metric-container"] { background: #f9d8bc !important; border-color: #f4c4a0 !important; padding: 8px 10px !important; }
[data-testid="stExpander"] div[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
[data-testid="stExpander"] div[data-testid="metric-container"] label { font-size: 0.68rem !important; }

/* ── ALL BUTTONS base ── */
.stButton > button {
    background-color: #f8fafc !important; color: #1e293b !important;
    border: 1px solid #e2e8f0 !important; border-radius: 5px !important;
    font-weight: 500 !important; font-size: 0.8rem !important;
    padding: 4px 10px !important; min-height: 32px !important; width: 100% !important;
    transition: all 0.12s ease !important; cursor: pointer !important;
}
.stButton > button:hover { background-color: #f1f5f9 !important; border-color: #cbd5e1 !important; }

/* Primary button (Update Rankings) — illustration coral */
[data-testid="stSidebar"] .stButton > button[kind="primary"],
.stButton > button[kind="primary"] {
    background-color: #F4674A !important; color: #ffffff !important;
    border: none !important; font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #e0543a !important;
    box-shadow: 0 2px 8px rgba(244,103,74,0.35) !important;
}

/* ── Action button color classes injected by JS (defined here, applied via JS) ── */

/* ── Dropdowns / selectbox — white bg, dark text ── */
[data-baseweb="select"] > div {
    background-color: #ffffff !important; border-color: #e2e8f0 !important;
}
[data-baseweb="select"] * { color: #1a202c !important; background-color: #ffffff !important; }
[data-baseweb="popover"], [data-baseweb="menu"],
[role="listbox"], [role="option"] {
    background-color: #ffffff !important; color: #1a202c !important;
}
[role="option"]:hover { background-color: #eff6ff !important; }
[data-baseweb="select"] svg { color: #475569 !important; }

/* Multiselect tags */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] { background-color: #ffffff !important; color: #1a202c !important; border: 1px solid #cbd5e1 !important; }
[data-testid="stMultiSelect"] span[data-baseweb="tag"] span { color: #1a202c !important; }

/* ── Input / textarea ── */
input, textarea { background-color: #ffffff !important; color: #1a202c !important; border: 1px solid #e2e8f0 !important; }

/* ── Dataframe / table — white bg, dark text ── */
[data-testid="stDataFrame"], .stDataFrame,
[data-testid="stDataFrame"] * {
    background-color: #ffffff !important; color: #1a202c !important;
}
[data-testid="stDataFrame"] th {
    background-color: #f1f5f9 !important; color: #1a202c !important;
    font-weight: 700 !important; border-bottom: 2px solid #e2e8f0 !important;
}
[data-testid="stDataFrame"] td { color: #1a202c !important; border-color: #e8ecf0 !important; }
/* Glide data grid canvas text */
.dvn-scroller { background: #ffffff !important; }

/* ── Slider ── */
[data-testid="stSlider"] * { color: #1a202c !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div { background-color: #e2e8f0 !important; }

/* ── Divider ── */
hr { border-color: #e8ecf0 !important; margin: 0.6rem 0 !important; }

/* ── Toast notifications ── */
[data-testid="stToast"] { background-color: #ffffff !important; color: #1a202c !important; border: 1px solid #e2e8f0 !important; }

/* ── Triage action buttons — equal size, centred text ── */
button.iq-engage, button.iq-skip, button.iq-save {
    height: 34px !important; min-height: 34px !important; max-height: 34px !important;
    font-size: 0.74rem !important; font-weight: 600 !important;
    padding: 0 4px !important; width: 100% !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    text-align: center !important; white-space: nowrap !important; overflow: hidden !important;
}
button.iq-engage {
    background-color: #F4674A !important; color: #ffffff !important;
    border: 1px solid #F4674A !important;
}
button.iq-engage:hover { background-color: #e0543a !important; border-color: #e0543a !important; }
button.iq-skip {
    background-color: #e2e8f0 !important; color: #475569 !important;
    border: 1px solid #cbd5e1 !important;
}
button.iq-skip:hover { background-color: #cbd5e1 !important; border-color: #94a3b8 !important; color: #1e293b !important; }
button.iq-save {
    background-color: #dcfce7 !important; color: #166534 !important;
    border: 1px solid #86efac !important;
}
button.iq-save:hover { background-color: #bbf7d0 !important; border-color: #4ade80 !important; color: #14532d !important; }
button.iq-engage p, button.iq-skip p, button.iq-save p {
    margin: 0 !important; line-height: 1 !important; text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# ── JS: tag action buttons with color classes via MutationObserver ────────────
_BUTTON_TAGGER_JS = """
<script>
(function() {
  var doc = window.parent.document;

  // ── 1. Colour triage buttons by label ───────────────────────────────────
  function tagButtons() {
    doc.querySelectorAll('button').forEach(function(btn) {
      var t = btn.textContent.trim();
      if (t === 'Engage') {
        btn.classList.add('iq-engage'); btn.classList.remove('iq-skip','iq-save');
      } else if (t === 'Skip') {
        btn.classList.add('iq-skip'); btn.classList.remove('iq-engage','iq-save');
      } else if (t === 'Save for later') {
        btn.classList.add('iq-save'); btn.classList.remove('iq-engage','iq-skip');
      }
    });
  }

  // ── 2. Persist active tab across Streamlit reruns ───────────────────────
  var TAB_KEY = 'engageiq_active_tab';

  function saveTab(idx) { localStorage.setItem(TAB_KEY, idx); }

  function restoreTab() {
    var saved = localStorage.getItem(TAB_KEY);
    if (saved === null) return;
    var tabs = doc.querySelectorAll('button[data-baseweb="tab"]');
    var idx = parseInt(saved, 10);
    if (tabs[idx] && tabs[idx].getAttribute('aria-selected') !== 'true') {
      tabs[idx].click();
    }
  }

  doc.addEventListener('click', function(e) {
    var tab = e.target.closest('button[data-baseweb="tab"]');
    if (tab) {
      var tabs = doc.querySelectorAll('button[data-baseweb="tab"]');
      saveTab(Array.from(tabs).indexOf(tab));
    }
  }, true);

  // ── 3. Open URL if Engage was clicked (stored in hidden marker) ─────────
  function openPendingUrl() {
    var marker = doc.getElementById('engageiq-open-url');
    if (marker && marker.dataset.url) {
      window.open(marker.dataset.url, '_blank');
      marker.dataset.url = '';
    }
  }

  // Run on load + watch DOM changes
  tagButtons(); restoreTab(); openPendingUrl();
  new MutationObserver(function() {
    tagButtons(); restoreTab(); openPendingUrl();
  }).observe(doc.body, {childList:true, subtree:true});
})();
</script>
"""
st.iframe(_BUTTON_TAGGER_JS, height=1)

# ── Pastel chart palette ──────────────────────────────────────────────────────
PASTEL = ["#a8c5e8", "#b5d5c5", "#f5c5a3", "#c5b5e8", "#f5d5a3",
          "#a3c5f5", "#e8c5b5", "#b5e8d5", "#e8d5a3", "#c5a3e8",
          "#a3e8c5", "#f5a3c5", "#d5e8a3", "#a3d5e8", "#e8a3b5"]

CHART_LAYOUT = dict(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#f8f9fb",
    font=dict(family="Inter, system-ui, sans-serif", size=11, color="#1a202c"),
)
CHART_MARGIN       = dict(l=0,  r=20, t=14, b=8)
CHART_MARGIN_PIE   = dict(l=10, r=10, t=14, b=40)

# ── Init ──────────────────────────────────────────────────────────────────────
try:
    init_db()
    _load_seed()
    seed_personas()
except Exception as _startup_err:
    import traceback
    st.error(f"**Startup error:** {_startup_err}")
    st.code(traceback.format_exc())
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
if "persona"          not in st.session_state: st.session_state.persona = "Sofia"
if "ranked"           not in st.session_state: st.session_state.ranked = []
if "index"            not in st.session_state: st.session_state.index = None
if "rows"             not in st.session_state: st.session_state.rows = []
if "pvec"             not in st.session_state: st.session_state.pvec = None
if "custom_interests" not in st.session_state: st.session_state.custom_interests = ""
if "_open_url"        not in st.session_state: st.session_state._open_url = None
if "_last_filter"     not in st.session_state: st.session_state._last_filter = None
if "_filter_ranked"   not in st.session_state: st.session_state._filter_ranked = []

# ── Background poller ─────────────────────────────────────────────────────────
def _poll_loop():
    while True:
        time.sleep(300)
        try: embed_all_opportunities(batch_size=32)
        except Exception: pass

if "poller_started" not in st.session_state:
    threading.Thread(target=_poll_loop, daemon=True).start()
    st.session_state.poller_started = True

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:6px;'>
          <!-- Speech bubble logo -->
          <svg width="48" height="44" viewBox="0 0 48 44" xmlns="http://www.w3.org/2000/svg">
            <!-- Bubble body -->
            <rect x="1" y="1" width="40" height="32" rx="10" fill="#F4674A"/>
            <!-- Tail -->
            <polygon points="8,30 2,43 20,30" fill="#F4674A"/>
            <!-- /// lines -->
            <line x1="10" y1="22" x2="16" y2="10" stroke="white" stroke-width="3" stroke-linecap="round"/>
            <line x1="18" y1="22" x2="24" y2="10" stroke="white" stroke-width="3" stroke-linecap="round"/>
            <line x1="26" y1="22" x2="32" y2="10" stroke="white" stroke-width="3" stroke-linecap="round"/>
          </svg>
          <div>
            <div style='font-size:1.45rem;font-weight:800;color:#1a202c;line-height:1.1;letter-spacing:-0.02em;'>EngageIQ</div>
            <div style='font-size:0.72rem;color:#64748b;margin-top:1px;'>Smart Engagement Opportunity Scorer</div>
          </div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("<div style='font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:#64748b;margin-bottom:4px;'>Persona</div>", unsafe_allow_html=True)
    persona_options = list(PERSONAS.keys()) + ["Custom"]
    idx = persona_options.index(st.session_state.persona) if st.session_state.persona in persona_options else 0
    selected = st.selectbox("Active persona", persona_options, index=idx, label_visibility="collapsed")

    _label_style = "font-size:0.75rem;font-weight:600;color:#64748b;margin:8px 0 2px;"

    if selected == "Custom":
        st.session_state.persona = "Custom"
        st.markdown(f"<div style='{_label_style}'>Background</div>", unsafe_allow_html=True)
        bg   = st.text_area("Background",  placeholder="e.g. PhD student in NLP at UC Davis",           key="c_bg",   label_visibility="collapsed", height=68)
        st.markdown(f"<div style='{_label_style}'>Interests</div>", unsafe_allow_html=True)
        ints = st.text_area("Interests",   placeholder="e.g. machine learning, Python, transformers",   key="c_int",  label_visibility="collapsed", height=68)
        st.markdown(f"<div style='{_label_style}'>Goal</div>", unsafe_allow_html=True)
        goal = st.text_area("Goal",        placeholder="e.g. Find repos to contribute to for portfolio", key="c_goal", label_visibility="collapsed", height=68)
        st.markdown(f"<div style='{_label_style}'>Time Budget (hours/week)</div>", unsafe_allow_html=True)
        time_budget = st.number_input("Time Budget", min_value=1, max_value=40, value=5, key="c_time", label_visibility="collapsed")
        if st.button("Save Persona", width="stretch"):
            combined = f"{ints.strip()} {goal.strip()}".strip()
            if combined:
                upsert_persona("Custom", combined)
                st.session_state.custom_interests = combined
                st.success("Saved.")
            else:
                st.warning("Please fill in at least Interests or Goal.")
    else:
        st.session_state.persona = selected
        p = PERSONAS[selected]
        st.markdown(f"<div style='{_label_style}'>Background</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8rem;color:#1a202c;margin-bottom:2px;'>{p['background']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{_label_style}'>Interests</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8rem;color:#1a202c;margin-bottom:2px;'>{p['interests']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{_label_style}'>Goal</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8rem;color:#1a202c;margin-bottom:2px;'>{p['goal']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{_label_style}'>Time Budget (hours/week)</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8rem;color:#1a202c;margin-bottom:2px;'>{p['time_budget']} hrs/week</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("<div style='font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:#64748b;margin-bottom:6px;'>Data Pipeline</div>", unsafe_allow_html=True)
    st.metric("Records in DB", f"{get_record_count():,}")

    if st.button("Run Ingestion", width="stretch"):
        with st.spinner("Ingesting from GitHub, HN, Reddit..."):
            from scraper import run_ingestion
            run_ingestion(target=10000)
        st.success("Done."); st.rerun()

    if st.button("Compute Embeddings", width="stretch"):
        with st.spinner("Encoding opportunities..."):
            embed_all_opportunities(batch_size=64)
        st.success("Done."); st.rerun()

    if st.button("Reset to Clean Data", width="stretch"):
        with st.spinner("Wiping DB and reloading clean seed..."):
            reset_opportunities()
        st.success(f"Done. {get_record_count():,} records loaded."); st.rerun()

    st.divider()

    if st.button("Update Rankings", width="stretch", type="primary"):
        with st.spinner("Building FAISS index..."):
            index, rows = build_faiss_index()
            st.session_state.index = index
            st.session_state.rows  = rows
            p_name = st.session_state.persona
            p = get_persona(p_name)
            if p and p.get("embedding") is None:
                embed_persona(p_name); p = get_persona(p_name)
            if p and p.get("embedding"):
                pvec = json.loads(p["embedding"])
            else:
                p_data = PERSONAS.get(p_name, {})
                fallback = (persona_vector_text(p_data) if isinstance(p_data, dict) else p_data) or st.session_state.custom_interests
                pvec = encode_text(fallback)
            st.session_state.pvec  = pvec
            st.session_state._last_filter  = None   # invalidate filter cache
            st.session_state._filter_ranked = []
            st.session_state.ranked = rank_opportunities(pvec, index, rows, persona_name=p_name, top_n=50)
        st.success("Ready."); st.rerun()

# ── Page header ───────────────────────────────────────────────────────────────
persona = st.session_state.persona
st.markdown(
    f"<div style='padding-top:4px;'>"
    f"<div style='display:flex;align-items:baseline;gap:10px;margin-bottom:3px;'>"
    f"<span style='font-size:1.5rem;font-weight:700;color:#1a202c;letter-spacing:-0.02em;'>EngageIQ</span>"
    f"<span style='font-size:1.1rem;color:#cbd5e1;font-weight:300;'>/</span>"
    f"<span style='font-size:1.05rem;color:#64748b;font-weight:500;'>{persona}</span>"
    f"</div>"
    f"<div style='font-size:0.82rem;color:#94a3b8;margin-bottom:10px;letter-spacing:0.01em;'>"
    f"Ranked engagement opportunities &nbsp;·&nbsp; GitHub &nbsp;·&nbsp; Hacker News &nbsp;·&nbsp; Reddit"
    f"</div></div>",
    unsafe_allow_html=True
)
st.divider()


# ── Inject URL marker for JS to pick up and open ─────────────────────────────
_pending_url = st.session_state.get("_open_url", "") or ""
st.session_state._open_url = None
st.markdown(
    f'<div id="engageiq-open-url" data-url="{_pending_url}" style="display:none;"></div>',
    unsafe_allow_html=True
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab0, tab1, tab3, tab4 = st.tabs(["Home", "Opportunities", "Feedback", "Export Brief"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — HOME / HERO
# ─────────────────────────────────────────────────────────────────────────────
with tab0:
    import base64 as _b64
    _HERO_IMG = Path(__file__).parent / "assets" / "hero_illustration.webp"

    # ── Hero row ──────────────────────────────────────────────────────────────
    hero_left, hero_right = st.columns([3, 2], gap="large")
    with hero_left:
        st.markdown(
            "<div style='padding: 32px 0 20px;'>"
            "<div style='font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;"
            "color:#F4674A;margin-bottom:10px;'>Smart Engagement Discovery</div>"
            "<h1 style='font-size:2.15rem;font-weight:800;line-height:1.2;color:#1a202c;"
            "letter-spacing:-0.03em;margin:0 0 18px;'>"
            "Find and act on your best engagement opportunities across GitHub, Hacker&nbsp;News, and Reddit"
            "</h1>"
            "<p style='font-size:1rem;color:#475569;line-height:1.7;max-width:520px;margin:0 0 10px;'>"
            "EngageIQ surfaces the highest-signal repositories, discussions, and issues "
            "ranked for <em>your</em> interests. Spend zero time searching and all your time contributing."
            "</p>"
            "<p style='font-size:0.9rem;color:#64748b;line-height:1.6;max-width:520px;margin:0 0 32px;'>"
            "Pick a persona, load your rankings, and triage with a single click. "
            "The model learns from every Engage, Skip, and Save action you make."
            "</p>"
            "</div>",
            unsafe_allow_html=True
        )

    with hero_right:
        if _HERO_IMG.exists():
            _img_b64 = _b64.b64encode(_HERO_IMG.read_bytes()).decode()
            st.markdown(
                f'<div style="padding-top:16px;">'
                f'<img src="data:image/webp;base64,{_img_b64}" '
                f'style="width:100%;border-radius:16px;object-fit:cover;" />'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<hr style='border:none;border-top:1.5px solid #f0e8de;margin:0 0 28px;'/>",
        unsafe_allow_html=True
    )

    # ── Stats strip ───────────────────────────────────────────────────────────
    try:
        _df_home = load_df()
        if not _df_home.empty:
            _stats = summary_stats(_df_home)
            s1, s2, s3, s4 = st.columns(4)
            for _col, _label, _val in [
                (s1, "Total Records",   f"{_stats['total_records']:,}"),
                (s2, "Unique Domains",  str(_stats['unique_domains'])),
                (s3, "Avg Stars",       str(_stats['avg_stars'])),
                (s4, "Top Domain",      _stats['top_domain']),
            ]:
                _col.markdown(
                    f'<div style="background:#ffffff;border:1px solid #e8ecf0;border-radius:10px;'
                    f'padding:16px 18px;text-align:center;">'
                    f'<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.08em;color:#94a3b8;margin-bottom:4px;">{_label}</div>'
                    f'<div style="font-size:1.5rem;font-weight:800;color:#1a202c;">{_val}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.markdown(
                "<hr style='border:none;border-top:1.5px solid #f0e8de;margin:0 0 20px;'/>",
                unsafe_allow_html=True
            )

            # ── Records by Domain + Source Distribution ────────────────────────
            _ch1, _ch2 = st.columns([3, 2])
            with _ch1:
                st.markdown("<div style='font-size:0.83rem;font-weight:600;color:#1a202c;margin-bottom:4px;'>Records by Domain</div>", unsafe_allow_html=True)
                _td = top_domains(_df_home)
                _fig1 = px.bar(
                    _td.sort_values("count"), x="count", y="domain", orientation="h",
                    color="domain", color_discrete_sequence=PASTEL,
                    text="count", labels={"count": "Records", "domain": ""},
                )
                _fig1.update_traces(textposition="outside", textfont_size=10,
                                    textfont_color="#1a202c", marker_line_width=0)
                _fig1.update_layout(
                    **CHART_LAYOUT, height=400, showlegend=False, margin=CHART_MARGIN,
                    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False,
                               tickfont=dict(color="#1a202c", size=11)),
                    yaxis=dict(showgrid=False, tickfont=dict(color="#1a202c", size=11)),
                )
                st.plotly_chart(_fig1, width="stretch")

            with _ch2:
                st.markdown("<div style='font-size:0.83rem;font-weight:600;color:#1a202c;margin-bottom:4px;'>Source Distribution</div>", unsafe_allow_html=True)
                _sd = source_distribution(_df_home)
                _fig2 = px.pie(
                    _sd, names="source", values="count", hole=0.52,
                    color_discrete_sequence=["#F4674A", "#C5B8F0", "#6BBFEA", "#F0508A"],
                )
                _fig2.update_traces(
                    textposition="inside", textinfo="label+percent",
                    textfont_size=11, textfont_color="#1a202c",
                    marker=dict(line=dict(color="#f8f9fb", width=2)),
                    insidetextorientation="horizontal",
                )
                _fig2.update_layout(
                    **CHART_LAYOUT, height=400, showlegend=True, margin=CHART_MARGIN_PIE,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5,
                                xanchor="center", font=dict(size=11, color="#1a202c")),
                )
                st.plotly_chart(_fig2, width="stretch")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OPPORTUNITIES
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if not st.session_state.rows or st.session_state.pvec is None:
        st.markdown(
            "<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:16px 20px;"
            "color:#1e40af;font-size:0.9rem;'>Click <strong>Update Rankings</strong> in the sidebar to get started.</div>",
            unsafe_allow_html=True
        )
    else:
        # ── Filter row ──
        fc1, fc2 = st.columns([2, 2])
        with fc1:
            source_filter = st.multiselect("Source", ["github", "hackernews", "reddit"],
                                           default=["github", "hackernews", "reddit"])
        with fc2:
            from scraper import DOMAINS as _ALL_DOMAINS
            domain_filter = st.multiselect("Domain", sorted(_ALL_DOMAINS.keys()))

        # ── Filter rows, then rank filtered subset ──
        all_sources = source_filter or ["github", "hackernews", "reddit"]
        # github_issue is treated as github in the UI
        expanded_sources = all_sources + (["github_issue"] if "github" in all_sources else [])
        filter_key = (tuple(sorted(all_sources)), tuple(sorted(domain_filter)))

        if filter_key != st.session_state._last_filter:
            filtered_rows = [
                r for r in st.session_state.rows
                if r["source"] in expanded_sources
                and (not domain_filter or r.get("domain") in domain_filter)
            ]
            p_name = st.session_state.persona
            st.session_state._filter_ranked = rank_from_rows(
                st.session_state.pvec, filtered_rows, persona_name=p_name, top_n=15
            )
            st.session_state._last_filter = filter_key

        filtered = st.session_state._filter_ranked
        showing  = len(filtered)
        total_pool = sum(
            1 for r in st.session_state.rows
            if r["source"] in expanded_sources
            and (not domain_filter or r.get("domain") in domain_filter)
        )
        st.markdown(
            f"<div style='font-size:0.82rem;color:#64748b;margin:6px 0 10px;'>"
            f"Showing top {showing} of {total_pool:,} matching opportunities · ranked by relevance, community health, and recency</div>",
            unsafe_allow_html=True
        )

        src_label = {"github": "GitHub", "github_issue": "GitHub Issue", "hackernews": "Hacker News", "reddit": "Reddit"}
        src_badge_style = {
            "github":       "background:#dbeafe;color:#1e40af;",
            "github_issue": "background:#ede9fe;color:#5b21b6;",
            "hackernews":   "background:#fff3e0;color:#b45309;",
            "reddit":       "background:#ffe4e6;color:#be123c;",
        }
        action_text = {
            "github":       "Star this repo, explore open issues, and consider opening a PR or comment.",
            "github_issue": "Read the issue thread, reproduce the problem, and submit a fix or proposal.",
            "hackernews":   "Join the discussion with a concrete insight or question — top comments get high visibility.",
            "reddit":       "Reply with real value — share a project, tool, or relevant experience.",
        }

        for i, opp in enumerate(filtered, 1):
            with st.expander(f"#{i}  {opp['title'][:85]}", expanded=(i <= 3)):
                col_a, col_b = st.columns([3, 1])

                with col_a:
                    src = opp["source"]
                    badge_s = src_badge_style.get(src, "background:#f1f5f9;color:#475569;")
                    body_preview = opp["body"][:160] if opp.get("body") else ""
                    st.markdown(
                        f'<div style="margin-bottom:4px;">'
                        f'<span style="display:inline-block;padding:1px 8px;border-radius:3px;'
                        f'font-size:0.68rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;{badge_s}">'
                        f'{src_label.get(src, src)}</span>'
                        f'&nbsp;<span style="color:#94a3b8;font-size:0.75rem;">{opp.get("domain","")}</span>'
                        f'</div>'
                        f'<a href="{opp["url"]}" style="font-size:0.9rem;font-weight:600;color:#1e293b;text-decoration:none;" target="_blank">{opp["title"]}</a>'
                        + (f'<p style="color:#94a3b8;font-size:0.78rem;margin:3px 0 6px;line-height:1.4;">{body_preview}</p>' if body_preview else '<div style="margin-bottom:6px;"></div>'),
                        unsafe_allow_html=True
                    )
                    # Why this + Suggested Action — compact two-line blocks
                    explanation = opp.get("explanation", "—")
                    suggestion  = action_text.get(src, "Engage with this opportunity.")
                    st.markdown(
                        f'<div style="background:#f9d8bc;border-left:2px solid #f4a878;padding:5px 10px;'
                        f'border-radius:0 4px 4px 0;font-size:0.79rem;color:#4a2c1a;line-height:1.45;margin-bottom:4px;">'
                        f'<span style="font-weight:600;color:#2d1a0e;">Why: </span>{explanation}</div>'
                        f'<div style="background:#f9d8bc;border-left:2px solid #f4a878;padding:5px 10px;'
                        f'border-radius:0 4px 4px 0;font-size:0.79rem;color:#4a2c1a;line-height:1.45;">'
                        f'<span style="font-weight:600;color:#2d1a0e;">Do: </span>{suggestion}</div>',
                        unsafe_allow_html=True
                    )

                with col_b:
                    # Compact inline stats + triage buttons
                    score    = opp.get("final_score", 0)
                    stars    = opp.get("stars", 0)
                    comments = opp.get("comments", 0)
                    st.markdown(
                        f'<div style="font-size:0.72rem;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px;">Score</div>'
                        f'<div style="font-size:1.3rem;font-weight:700;color:#1e293b;line-height:1.1;margin-bottom:8px;">{score:.3f}</div>'
                        f'<div style="display:flex;gap:12px;margin-bottom:10px;">'
                        f'<div><div style="font-size:0.68rem;color:#94a3b8;font-weight:600;text-transform:uppercase;">Stars</div>'
                        f'<div style="font-size:0.9rem;font-weight:600;color:#334155;">{stars:,}</div></div>'
                        f'<div><div style="font-size:0.68rem;color:#94a3b8;font-weight:600;text-transform:uppercase;">Comments</div>'
                        f'<div style="font-size:0.9rem;font-weight:600;color:#334155;">{comments:,}</div></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    # ── Triage buttons ──
                    b1, b2, b3 = st.columns([2, 2, 3])
                    pname = st.session_state.persona
                    with b1:
                        st.markdown('<div class="btn-engage">', unsafe_allow_html=True)
                        if st.button("Engage", key=f"e_{opp['id']}_{i}", help="Opens link and marks as relevant", width="stretch"):
                            log_feedback(opp["id"], "engage", pname)
                            st.session_state._open_url = opp["url"]
                        st.markdown('</div>', unsafe_allow_html=True)
                    with b2:
                        st.markdown('<div class="btn-skip">', unsafe_allow_html=True)
                        if st.button("Skip", key=f"s_{opp['id']}_{i}", help="Not relevant — deprioritises this type", width="stretch"):
                            log_feedback(opp["id"], "skip", pname)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with b3:
                        st.markdown('<div class="btn-save">', unsafe_allow_html=True)
                        if st.button("Save for later", key=f"b_{opp['id']}_{i}", help="Bookmark to review in the Feedback tab", width="stretch"):
                            log_feedback(opp["id"], "bookmark", pname)
                        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — FEEDBACK
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    persona = st.session_state.persona
    feedback = get_feedback(persona)

    if not feedback:
        st.markdown(
            "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px 20px;"
            "color:#475569;font-size:0.9rem;'>No feedback recorded yet. Use the <strong>Engage / Skip / Save</strong> "
            "buttons on the Opportunities tab to start training the ranking model.</div>",
            unsafe_allow_html=True
        )
    else:
        fb_df = pd.DataFrame(feedback)
        action_counts = fb_df["action"].value_counts().reset_index()
        action_counts.columns = ["action", "count"]

        total_fb = len(fb_df)
        engages  = int((fb_df["action"] == "engage").sum())
        skips    = int((fb_df["action"] == "skip").sum())
        saves    = int((fb_df["action"] == "bookmark").sum())

        st.markdown(
            f'<div style="background:#fde8d0;border:1px solid #f4c4a0;border-radius:10px;'
            f'padding:16px 20px;display:flex;gap:0;margin-bottom:16px;">'
            f'<div style="flex:1;text-align:center;border-right:1px solid #f4c4a0;">'
            f'<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#000000;margin-bottom:4px;">Total Feedback</div>'
            f'<div style="font-size:1.6rem;font-weight:800;color:#000000;">{total_fb}</div>'
            f'</div>'
            f'<div style="flex:1;text-align:center;border-right:1px solid #f4c4a0;">'
            f'<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#000000;margin-bottom:4px;">Engaged</div>'
            f'<div style="font-size:1.6rem;font-weight:800;color:#000000;">{engages}</div>'
            f'</div>'
            f'<div style="flex:1;text-align:center;border-right:1px solid #f4c4a0;">'
            f'<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#000000;margin-bottom:4px;">Skipped</div>'
            f'<div style="font-size:1.6rem;font-weight:800;color:#000000;">{skips}</div>'
            f'</div>'
            f'<div style="flex:1;text-align:center;">'
            f'<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#000000;margin-bottom:4px;">Saved</div>'
            f'<div style="font-size:1.6rem;font-weight:800;color:#000000;">{saves}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='font-size:0.83rem;font-weight:600;color:#1a202c;margin-bottom:4px;'>Action Breakdown</div>", unsafe_allow_html=True)
        fig_fb = px.pie(
            action_counts, names="action", values="count", hole=0.52,
            color="action",
            color_discrete_map={"engage": "#bbf7d0", "skip": "#e2e8f0", "bookmark": "#bfdbfe"},
        )
        fig_fb.update_traces(
            textposition="inside", textinfo="label+percent",
            textfont_size=12, textfont_color="#1a202c",
            marker=dict(line=dict(color="#f8f9fb", width=2)),
            insidetextorientation="horizontal",
        )
        fig_fb.update_layout(
            **CHART_LAYOUT, height=300, showlegend=True, margin=CHART_MARGIN_PIE,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5,
                        xanchor="center", font=dict(size=11, color="#1a202c")),
        )
        st.plotly_chart(fig_fb, width="stretch")


    # ── Simulation (always shown) ──
    st.divider()
    st.markdown("<div style='font-size:0.83rem;font-weight:600;color:#1a202c;margin-bottom:2px;'>Learning Improvement — 50-Round Simulation</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;color:#94a3b8;margin-bottom:8px;'>Demonstrates how the UCB bandit improves recommendation quality as feedback accumulates.</div>", unsafe_allow_html=True)

    if st.button("Run simulation"):
        from ranking import UCBBandit
        import random
        bandit = UCBBandit()
        scores = []
        for rnd in range(1, 51):
            fid    = f"sim_{random.randint(1, 20)}"
            action = random.choices(["engage", "skip"], weights=[0.4, 0.6])[0]
            reward = 1.0 if action == "engage" else 0.0
            bandit.counts[fid]  = bandit.counts.get(fid, 0) + 1
            bandit.rewards[fid] = bandit.rewards.get(fid, 0.0) + reward
            avg = sum(bandit.rewards.values()) / max(1, sum(bandit.counts.values()))
            scores.append({"Round": rnd, "Avg Reward": round(avg, 4)})

        sim_df = pd.DataFrame(scores)
        fig_sim = px.line(sim_df, x="Round", y="Avg Reward", color_discrete_sequence=["#60a5fa"])
        fig_sim.update_traces(line_width=2.5)
        fig_sim.update_layout(
            **CHART_LAYOUT, height=260, margin=CHART_MARGIN,
            xaxis=dict(showgrid=False, tickfont=dict(color="#1a202c", size=11)),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Avg Reward",
                       tickfont=dict(color="#1a202c", size=11),
                       title_font=dict(color="#1a202c")),
        )
        st.plotly_chart(fig_sim, width="stretch")
        st.success("Reward improves as the bandit learns user preferences over time.")

    # ── Saved for Later ───────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='font-size:0.95rem;font-weight:700;color:#1a202c;margin-bottom:4px;'>Saved for Later</div>"
        "<div style='font-size:0.8rem;color:#64748b;margin-bottom:12px;'>Opportunities you bookmarked — click to open.</div>",
        unsafe_allow_html=True
    )
    _saved = get_saved_opportunities(st.session_state.persona)
    if not _saved:
        st.markdown(
            "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px 18px;"
            "color:#64748b;font-size:0.85rem;'>Nothing saved yet. Hit <strong>Save for later</strong> on any opportunity.</div>",
            unsafe_allow_html=True
        )
    else:
        _src_label = {"github": "GitHub", "github_issue": "GitHub Issue", "hackernews": "Hacker News", "reddit": "Reddit"}
        _src_color = {"github": "#dbeafe", "github_issue": "#ede9fe", "hackernews": "#fff3e0", "reddit": "#ffe4e6"}
        _src_text  = {"github": "#1e40af", "github_issue": "#5b21b6", "hackernews": "#b45309", "reddit": "#be123c"}
        for _s in _saved:
            _bg   = _src_color.get(_s["source"], "#f1f5f9")
            _tc   = _src_text.get(_s["source"], "#475569")
            _lbl  = _src_label.get(_s["source"], _s["source"])
            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #e8ecf0;border-radius:8px;'
                f'padding:10px 16px;margin-bottom:6px;display:flex;align-items:center;gap:12px;">'
                f'<span style="background:{_bg};color:{_tc};font-size:0.65rem;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.05em;padding:2px 8px;border-radius:3px;white-space:nowrap;">'
                f'{_lbl}</span>'
                f'<span style="font-size:0.85rem;color:#64748b;white-space:nowrap;">{_s.get("domain","")}</span>'
                f'<a href="{_s["url"]}" target="_blank" style="font-size:0.88rem;font-weight:600;'
                f'color:#1e293b;text-decoration:none;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
                f'{_s["title"]}</a>'
                f'<span style="font-size:0.75rem;color:#94a3b8;white-space:nowrap;">⭐ {_s.get("stars",0):,}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — EXPORT BRIEF
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    persona = st.session_state.persona

    if not st.session_state.ranked:
        st.markdown(
            "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px 20px;"
            "color:#475569;font-size:0.9rem;'>Load rankings first to generate the brief.</div>",
            unsafe_allow_html=True
        )
    else:
        top10 = st.session_state.ranked[:10]
        rows_data = [{
            "Rank":   i,
            "Title":  opp["title"][:60],
            "Source": opp["source"],
            "Domain": opp.get("domain", ""),
            "Stars":  opp.get("stars", 0),
            "Score":  round(opp.get("final_score", 0), 3),
            "URL":    opp["url"],
        } for i, opp in enumerate(top10, 1)]

        brief_df = pd.DataFrame(rows_data)
        st.markdown(f"<div style='font-size:0.9rem;font-weight:600;color:#1a202c;margin-bottom:8px;'>Top 10 Opportunities — {persona}</div>", unsafe_allow_html=True)

        # Render as plain HTML table — immune to dark-mode glitches
        tbl_html = """
        <style>
        .brief-table { width:100%; border-collapse:collapse; font-size:0.85rem; background:#ffffff; }
        .brief-table th { background:#f1f5f9; color:#1a202c; font-weight:700; padding:9px 12px;
                          text-align:left; border-bottom:2px solid #e2e8f0; }
        .brief-table td { padding:8px 12px; color:#1a202c; border-bottom:1px solid #f1f5f9; }
        .brief-table tr:hover td { background:#f8faff; }
        .brief-table a { color:#1e40af; text-decoration:none; }
        .brief-table a:hover { text-decoration:underline; }
        </style>
        <table class="brief-table"><thead><tr>
        <th>#</th><th>Title</th><th>Source</th><th>Domain</th><th>Stars</th><th>Score</th><th>Link</th>
        </tr></thead><tbody>
        """
        for row in rows_data:
            tbl_html += (
                f"<tr><td>{row['Rank']}</td>"
                f"<td>{row['Title']}</td>"
                f"<td>{row['Source']}</td>"
                f"<td>{row['Domain']}</td>"
                f"<td>{row['Stars']:,}</td>"
                f"<td>{row['Score']}</td>"
                f"<td><a href='{row['URL']}' target='_blank'>Open</a></td></tr>"
            )
        tbl_html += "</tbody></table>"
        st.markdown(tbl_html, unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        dl1, _ = st.columns([1, 3])
        with dl1:
            st.download_button("Download CSV", brief_df.to_csv(index=False),
                               file_name=f"engageiq_brief_{persona}.csv",
                               mime="text/csv", width="stretch")
