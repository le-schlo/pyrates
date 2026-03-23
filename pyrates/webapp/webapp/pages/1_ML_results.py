from __future__ import annotations

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pyrates.webapp.webapp.design import apply_global_styles
from pyrates.webapp.webapp.spin_viz import render_spin_distribution
from pyrates.webapp.science_utils.units import convert_energy, SUPPORTED_UNITS
from pyrates.webapp.science_utils.photocatalysts import list_photocatalyst_names, get_photocatalyst_row
from pyrates.utilities.plotting.plot_modes import plot_mode_distribution, plot_mode_distribution_single

BASE_UNIT = "kcal/mol"  # underlying values returned by the stub (per user)
DEFAULT_DISPLAY_UNIT = "kcal/mol"


def _convert_row_values(row: pd.Series, to_unit: str) -> dict:
    return {
        "triplet_energy_adiabatic": convert_energy(row["triplet_energy_adiabatic"], BASE_UNIT, to_unit),
        "triplet_energy_adiabatic_uncertainty": convert_energy(row["triplet_energy_adiabatic_uncertainty"], BASE_UNIT, to_unit),
        "triplet_energy_adiabatic_mdn_mu": convert_energy(row["triplet_energy_adiabatic_mdn_mu"], BASE_UNIT, to_unit),
        "triplet_energy_adiabatic_mdn_sigma": convert_energy(row["triplet_energy_adiabatic_mdn_sigma"], BASE_UNIT, to_unit),
        "lamda_S_T": convert_energy(row["lamda_S_T"], BASE_UNIT, to_unit),
        "lamda_S_T_uncertainty": convert_energy(row["lamda_S_T_uncertainty"], BASE_UNIT, to_unit),
        "lamda_S_T_mdn_mu": convert_energy(row["lamda_S_T_mdn_mu"], BASE_UNIT, to_unit),
        "lamda_S_T_mdn_sigma": convert_energy(row["lamda_S_T_mdn_sigma"], BASE_UNIT, to_unit),
        "lamda_T_S": convert_energy(row["lamda_T_S"], BASE_UNIT, to_unit),
        "lamda_T_S_uncertainty": convert_energy(row["lamda_T_S_uncertainty"], BASE_UNIT, to_unit),
        "lamda_T_S_mdn_mu": convert_energy(row["lamda_T_S_mdn_mu"], BASE_UNIT, to_unit),
        "lamda_T_S_mdn_sigma": convert_energy(row["lamda_T_S_mdn_sigma"], BASE_UNIT, to_unit),
    }


def _make_display_df(df: pd.DataFrame, to_unit: str) -> pd.DataFrame:
    conv = df.apply(lambda r: _convert_row_values(r, to_unit), axis=1, result_type="expand")
    # Use explicit assign to avoid any optional typing inference on DataFrame copy
    return df.assign(
        triplet_energy_adiabatic=conv["triplet_energy_adiabatic"],
        lamda_S_T=conv["lamda_S_T"],
        lamda_T_S=conv["lamda_T_S"],
    )


def _unit_selector() -> str:
    # Put selector on the right using columns
    left, mid, right = st.columns([1, 1, 1])
    with right:
        unit = st.selectbox(
            "Select units",
            options=SUPPORTED_UNITS,
            index=SUPPORTED_UNITS.index(DEFAULT_DISPLAY_UNIT),
            key="energy_unit",
            help="Applies to triplet energy and reorganization parameters.",
        )
    return unit


def _show_single_result(df: pd.DataFrame, unit: str):
    st.markdown("""
    <style>
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
    row = df.iloc[0]
    vals = _convert_row_values(row, unit)

    st.subheader("Predicted values")
    c1, c2, c3 = st.columns(3)

    metrics = [
        (c1, "Triplet energy (adiabatic)", "triplet_energy_adiabatic", 0),
        (c2, "λ(S→T)", "lamda_S_T", 1),
        (c3, "λ(T→S)", "lamda_T_S", 2),
    ]

    for col, label, key, target_idx in metrics:
        with col:
            st.metric(
                f"{label} [{unit}]",
                value=f"{vals[key]:.4g}",
                delta=f"Uncertainty: ± {vals[key + '_uncertainty']:.2f}",
                delta_color="off",
            )
            with st.popover("Display prediction modes"):
                fig = plot_mode_distribution_single(
                    mu=vals[key+'_mdn_mu'], sigma=vals[key+'_mdn_sigma'], pi=row[key+'_mdn_pi'],
                    predicted_value=vals[key],
                    uncertainty=vals[key+'_uncertainty'],
                    target_name=f"{label}",
                    unit=unit
                )
                st.pyplot(fig)
                plt.close(fig)

    # Always show spin density visualization directly for single molecule
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        render_spin_distribution(row["smiles"],row["spin_map"], row["spin_map_uncertainty"], title="Spin distribution")

    with st.expander("Details", expanded=False):
        disp = _make_display_df(df, unit).drop(columns=[c for c in ["spin_map"] if c in df.columns])
        st.dataframe(disp, width="stretch")


def _show_batch_results(df: pd.DataFrame, unit: str):
    st.subheader("Predicted values")
    disp = _make_display_df(df, unit).drop(columns=[c for c in ["spin_map"] if c in df.columns])
    st.dataframe(disp, width="stretch")
    st.caption(f"{len(df)} molecules analyzed.")

    st.markdown("---")
    st.subheader("Details")
    for display_idx, (row_idx, row) in enumerate(df.iterrows(), start=1):
        with st.expander(f"Molecule {display_idx}: {row['smiles']}"):
            vals = _convert_row_values(row, unit)
            st.write(
                {
                    "smiles": row["smiles"],
                    "triplet_energy_adiabatic": f"{vals['triplet_energy_adiabatic']:.6g} {unit}",
                    "lamda_S_T": f"{vals['lamda_S_T']:.6g} {unit}",
                    "lamda_T_S": f"{vals['lamda_T_S']:.6g} {unit}",
                }
            )

            @st.dialog("Spin distribution", width="large")
            def _spin_modal(smiles: str, spin_map):
                render_spin_distribution(smiles, spin_map)

            if st.button("Visualize spin densities", key=f"spin_plot_{row_idx}"):
                _spin_modal(row["smiles"], row.get("spin_map", {}))


def _download_buttons(df: pd.DataFrame, unit: str):
    disp = _make_display_df(df, unit).drop(columns=[c for c in ["spin_map"] if c in df.columns])
    st.download_button(
        label=f"Download results (CSV, {unit})",
        data=disp.to_csv(index=False).encode("utf-8"),
        file_name="pyrates_ml_results.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="PyRates - ML results", layout="wide")
    apply_global_styles()
    st.title("ML results")

    df = st.session_state.get("results_df")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No results available. Please analyze molecules from the 'Input Molecules' page.")
        return

    # Unit selector (default kcal/mol)
    unit = st.session_state.get("energy_unit") or DEFAULT_DISPLAY_UNIT
    unit = _unit_selector()

    if len(df) == 1:
        _show_single_result(df, unit)
    else:
        _show_batch_results(df, unit)

    _download_buttons(df, unit)

    st.divider()
    # --- Marcus handoff section ---
    st.subheader("Transfer Results to Marcus Theory Analysis")
    names = list_photocatalyst_names()
    if not names:
        st.info("No photocatalysts found in the package data.")
    else:
        pc = st.selectbox(
            "Select photocatalyst",
            options=names,
            index=0,
            key="selected_photocatalyst_name",
            help="Choose a photocatalyst to continue with Marcus Theory analysis.",
        )
        go = st.button("Go", type="primary")
        if go and pc:
            # Store selected row only; do NOT overwrite the selectbox's key in session_state
            row = get_photocatalyst_row(pc) or {"photocatalyst_name": pc}
            st.session_state["selected_photocatalyst"] = row
            try:
                st.switch_page("pages/2_Marcus_Theory_Analysis.py")  # type: ignore[attr-defined]
            except Exception:
                st.success("Selection stored. Open 'Marcus Theory Analysis' from the sidebar.")

    # Remove placeholder coming soon note
    st.empty()


if __name__ == "__main__":
    main()
