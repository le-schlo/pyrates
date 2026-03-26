from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import re

import streamlit as st
import numpy as np
from utils import init_dependencies
from pyrates.utilities.plotting.visualize_spin_population import visualize_density, visualize_density_interactive

def render_spin_distribution(smiles: str, spin_population: list, uncertainty, title: Optional[str] = None):
    """Render a modal-like view of the spin distribution.
    """
    if title:
        st.subheader(title)

    on = st.toggle("Show uncertainty")

    img_unc = visualize_density_interactive(smiles, spin_population, uncertainty)
    img = visualize_density_interactive(smiles, spin_population)

    if img is None:
        st.info("Could not render molecule.")
        return

    if on:
        st.plotly_chart(img_unc, width='content')
    else:
        st.plotly_chart(img, width='content')

