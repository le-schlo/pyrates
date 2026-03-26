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
from pyrates.webapp.webapp.routes import predict
from pyrates.webapp.science_utils.smiles_utils import is_valid_smiles, canonicalize_smiles, kekulize_smiles


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


@st.dialog("Draw Molecule", width='medium')
def _open_molecule_drawer_dialog() -> None:
    raw = render_molecule_drawer(kekulize_smiles(st.session_state.get("drawer_input_smiles", "")))
    drawn_smiles = convert_drawn_to_smiles(raw)

    # If drawer output exists but cannot be converted into a valid SMILES, surface a warning.
    if raw and not drawn_smiles:
        st.warning("Could not extract a valid SMILES from the drawing. Please adjust and click 'Apply' again.")
        return

    if drawn_smiles:
        if not is_valid_smiles(drawn_smiles):
            st.warning("The drawn molecule produced an invalid SMILES. Please adjust and click 'Apply' again.")
            return

        current_val = st.session_state.get("single_smiles", "")

        # Canonicalize both to ensure we don't loop on minor format differences
        canon_drawn = canonicalize_smiles(drawn_smiles)
        canon_current = canonicalize_smiles(current_val)

        if canon_drawn and (canon_drawn != canon_current):
            st.session_state["_pending_single_smiles"] = drawn_smiles
            st.rerun()


def _clear_single_smiles() -> None:
    # Queue clear so main() applies it before the text_input is instantiated.
    st.session_state["_pending_single_smiles"] = ""
    st.session_state["drawer_input_smiles"] = ""


def main():
    st.set_page_config(page_title="Input Molecules - PyRates", layout="wide")
    apply_global_styles()

    init_dependencies()

    if "drawer_input_smiles" not in st.session_state:
        st.session_state["drawer_input_smiles"] = st.session_state.get("saved_single_smiles", "")

    # If there is a pending SMILES from the drawer, apply it BEFORE creating the widget
    pending = st.session_state.pop("_pending_single_smiles", None)
    if pending is not None:
        st.session_state["single_smiles"] = pending
    else:
        # 3. Normal run: Sync text input -> drawer
        # If the user typed in the text box (or first load), we update the drawer.
        if "single_smiles" in st.session_state:
             st.session_state["drawer_input_smiles"] = st.session_state["single_smiles"]
        else:
             st.session_state["drawer_input_smiles"] = st.session_state.get("saved_single_smiles", "")

    st.title("Input Molecules")
    #st.caption("Provide molecules via SMILES or drawing; batch input and file upload supported.")

    tab_single, tab_batch = st.tabs(["Single molecule", "Batch mode"])

    with tab_single:
        #tab_banner("Single molecule input", variant="single")
        with st.container(border=True):
            st.subheader("Single molecule input")

            # Align the input and action button on the same baseline.
            try:
                input_col, draw_col = st.columns([2, 2], vertical_alignment="bottom")
            except TypeError:
                # Fallback for older Streamlit versions without vertical_alignment.
                input_col, draw_col = st.columns([2, 2])
            with input_col:
                st.text_input(
                    "SMILES",
                    key="single_smiles",
                    placeholder="e.g., C1=CC=CC=C1",
                    value=st.session_state.get("saved_single_smiles", ""),
                )
            with draw_col:
                if st.button("Draw Molecule", key="open_molecule_drawer_dialog"):
                    _open_molecule_drawer_dialog()

            if st.button("Clear", key="clear_single_smiles", on_click=_clear_single_smiles):
                st.rerun()
            st.caption("Type a SMILES or use the drawer on the right, then click 'Apply'.")


    with tab_batch:
        #tab_banner("Batch mode", variant="batch")
        with st.container(border=True):
            st.subheader("Batch input")
            st.text_area(
                "SMILES list (one per line)",
                key="batch_text",
                height=160,
                placeholder="CCO\nO=C(O)C(O)C\n# Comments and blank lines are ignored",
                value=st.session_state.get("saved_batch_text", "")
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

    #st.divider()
    if st.button("Analyse Molecule(s)"):
        if "single_smiles" in st.session_state:
            st.session_state["saved_single_smiles"] = st.session_state["single_smiles"]
        if "batch_text" in st.session_state:
            st.session_state["saved_batch_text"] = st.session_state["batch_text"]
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
                with st.spinner("Running ML models..."):
                    df = predict(valid)
                st.session_state["smiles_list"] = valid
                st.session_state["results_df"] = df
                _navigate_to_results()


if __name__ == "__main__":
    main()
