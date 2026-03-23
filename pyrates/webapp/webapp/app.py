from __future__ import annotations

import streamlit as st
from pyrates.webapp.webapp.design import apply_global_styles
from pyrates.webapp.webapp.utils import init_dependencies


def main():
    st.set_page_config(page_title="Start - PyRates", layout="wide")
    apply_global_styles()

    # Initialize optional dependencies once per session
    deps = init_dependencies()

    st.title("EnTdecker 2.0")
    st.write("Web interface for predicting excited state properties for energy transfer catalysis.")
    st.caption("Use the sidebar to navigate to 'Input Molecules' and 'ML results'.")

    # One-time dependency status
    with st.expander("Environment status", expanded=False):
        st.write(
            "The app supports optional features depending on installed packages."
        )
        st.write(
            "- RDKit for SMILES validation and canonicalization\n"
            "- streamlit-ketcher for drawing molecules"
        )
        st.write(
            f"RDKit: {'available' if deps.get('has_rdkit') else 'not available'} | "
            f"Molecule drawer: {'available' if deps.get('has_ketcher') else 'not available'}"
        )


if __name__ == "__main__":
    main()
