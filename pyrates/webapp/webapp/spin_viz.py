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
    #todo: toggle between uncertainty and not!
    img = visualize_density_interactive(smiles, spin_population, uncertainty)

    if title:
        st.subheader(title)

    if img is None:
        st.info("Could not render molecule.")
        return

    st.plotly_chart(img, width='content')

    st.caption("Orange shading for atoms with spin density > 0.25; top-5 heavy atoms are annotated with numeric values.")

