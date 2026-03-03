"""NeuroSight Knowledge Base — same app as the OCT dashboard, this page runs the KB view."""
import sys
from pathlib import Path

# Ensure parent directory is on path so we can import neurosight_app_final
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# Set tab title and icon when this page is shown (no underscores in tab)
st.set_page_config(page_title="NeuroSight Knowledge Base", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

from neurosight_app_final import run_kb

run_kb()
