from __future__ import annotations

from typing import List

import streamlit as st

from pyrates.webapp.webapp.design import apply_global_styles, tab_banner
from pyrates.webapp.webapp.utils import (
    gather_smiles,
    render_molecule_drawer,
    convert_drawn_to_smiles,
    init_dependencies,
)
from pyrates.webapp.webapp.routes import predict_stub
from pyrates.webapp.science_utils.smiles_utils import is_valid_smiles, canonicalize_smiles


def _navigate_to_results():
    try:
        st.switch_page("pages/1_ML_results.py")  # type: ignore[attr-defined]
    except Exception:
        st.success("Analysis complete. Open the ML results page from the sidebar.")


def _validate_and_canonicalize(smiles_list: List[str]) -> tuple[list[str], list[str]]:
    """Validate with science_utils and canonicalize; return (valid, invalid)."""
    valid: list[str] = []
    invalid: list[str] = []
    for s in smiles_list:
        if not is_valid_smiles(s):
            invalid.append(s)
            continue
        cs = canonicalize_smiles(s) or s.strip()
        valid.append(cs)

    # Deduplicate while preserving order
    seen = set()
    uniq_valid: list[str] = []
    for s in valid:
        if s not in seen:
            uniq_valid.append(s)
            seen.add(s)
    return uniq_valid, invalid


def main():
    st.set_page_config(page_title="Input Molecules - PyRates", layout="wide")
    apply_global_styles()

    # Initialize optional dependencies ONCE per session (safe to call; it's idempotent)
    init_dependencies()

    # If there is a pending SMILES from the drawer, apply it BEFORE creating the widget
    pending = st.session_state.pop("_pending_single_smiles", None)
    if pending:
        st.session_state["single_smiles"] = pending

    st.title("Input Molecules")
    st.caption("Provide molecules via SMILES or drawing; batch input and file upload supported.")

    # Distinctive tabs with emojis to improve visual distinction
    tab_single, tab_batch = st.tabs(["🧪 Single molecule", "📦 Batch mode"])

    with tab_single:
        tab_banner("Single molecule", variant="single")
        with st.container(border=True):
            st.subheader("Single molecule input")
            cols = st.columns([1, 1])
            with cols[0]:
                st.text_input("SMILES", key="single_smiles", placeholder="e.g., C1=CC=CC=C1")
                st.caption("Type a SMILES or use the drawer on the right, then click 'Get SMILES'.")
            with cols[1]:
                st.subheader("Molecule drawer")
                raw = render_molecule_drawer()
                # Single 'Get SMILES' button that copies the drawn molecule into the text field
                if st.button("Get SMILES", help="Transfer the drawn molecule into the SMILES field"):
                    drawn_smiles = convert_drawn_to_smiles(raw)
                    if drawn_smiles:
                        # Do not modify the widget key directly after instantiation.
                        # Store as pending and rerun so it is applied before the widget is created.
                        st.session_state["_pending_single_smiles"] = drawn_smiles
                        st.rerun()
                    else:
                        st.warning("Could not extract SMILES from the drawing. Please try again.")

    with tab_batch:
        tab_banner("Batch mode", variant="batch")
        with st.container(border=True):
            st.subheader("Batch input")
            st.text_area(
                "SMILES list (one per line)",
                key="batch_text",
                height=160,
                placeholder="CCO\nO=C(O)C(O)C\n# Comments and blank lines are ignored",
            )
            up_col, col_col = st.columns([1, 1])
            with up_col:
                st.file_uploader(
                    "Upload .smi/.txt or .csv with SMILES",
                    type=["smi", "smiles", "txt", "csv"],
                    key="upload",
                )
            with col_col:
                st.text_input("CSV SMILES column name (optional)", key="smiles_column", placeholder="smiles")

    st.divider()
    if st.button("Analyse Molecule(s)"):
        smiles_list = gather_smiles(
            single_smiles=st.session_state.get("single_smiles"),
            drawn_smiles=None,  # handled via Get SMILES button into single_smiles
            batch_text=st.session_state.get("batch_text"),
            upload=st.session_state.get("upload"),
            smiles_column=st.session_state.get("smiles_column") or "smiles",
        )
        if not smiles_list:
            st.warning("Please provide at least one valid SMILES via the inputs above.")
        else:
            valid, invalid = _validate_and_canonicalize(smiles_list)
            if invalid:
                st.warning(f"Ignoring {len(invalid)} invalid SMILES: {', '.join(invalid[:5])}{' ...' if len(invalid) > 5 else ''}")
            if not valid:
                st.error("No valid SMILES found. Please correct your input.")
            else:
                with st.spinner("Running ML models (stub)..."):
                    df = predict_stub(valid)
                st.session_state["smiles_list"] = valid
                st.session_state["results_df"] = df
                _navigate_to_results()


if __name__ == "__main__":
    main()
