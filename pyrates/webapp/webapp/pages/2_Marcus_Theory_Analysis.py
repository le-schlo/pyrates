from __future__ import annotations

import pandas as pd
import streamlit as st

from pyrates.webapp.design import apply_global_styles
from pyrates.science_utils.units import SUPPORTED_UNITS, convert_energy
from pyrates.model.photocatalysts import get_photocatalyst_row, list_photocatalyst_names
from pyrates.science_utils import marcus_theory

DEFAULT_DISPLAY_UNIT = "kcal/mol"
BASE_QUENCHER_UNIT = "kcal/mol"      # ML predictions base unit (per user)
BASE_PC_UNIT = "kcal/mol"            # CSV data base unit


def _selectors() -> str:
    """Render photocatalyst selector (left) and unit selector (right). Return unit."""
    left, mid, right = st.columns([1, 0.2, 1])

    with left:
        names = list_photocatalyst_names()
        if names:
            current = st.session_state.get("selected_photocatalyst_name")
            idx = names.index(current) if isinstance(current, str) and current in names else 0
            sel = st.selectbox(
                "Select photocatalyst",
                options=names,
                index=idx,
                key="selected_photocatalyst_name",
                help="Choose a photocatalyst for the analysis.",
            )
            # Update the cached row for downstream usage (safe; does not touch the widget key)
            row = get_photocatalyst_row(sel)
            if row:
                st.session_state["selected_photocatalyst"] = row
        else:
            st.info("No photocatalysts found in the package data.")

    with right:
        unit = st.selectbox(
            "Select units",
            options=SUPPORTED_UNITS,
            index=SUPPORTED_UNITS.index(DEFAULT_DISPLAY_UNIT),
            key="energy_unit",
            help="Applies to triplet energy and reorganization parameters.",
        )
    return unit


def _load_selected_photocatalyst() -> dict:
    # Try to get from session; if only the name is present, reload the row
    row = st.session_state.get("selected_photocatalyst")
    name = st.session_state.get("selected_photocatalyst_name")
    if isinstance(row, dict) and row.get("photocatalyst_name"):
        return row
    if isinstance(name, str) and name:
        row2 = get_photocatalyst_row(name)
        if row2:
            st.session_state["selected_photocatalyst"] = row2
            return row2
    return {}


def _convert_pc_values(pc_row: dict, to_unit: str) -> dict:
    """Extract and convert photocatalyst parameters to the target unit."""
    return {
        "E_T_adiabatic": convert_energy(float(pc_row["E_T_adiabatic"]), BASE_PC_UNIT, to_unit),
        "lamda_S_T": convert_energy(float(pc_row["lamda_S_T"]), BASE_PC_UNIT, to_unit),
        "lamda_T_S": convert_energy(float(pc_row["lamda_T_S"]), BASE_PC_UNIT, to_unit),
    }


def _convert_quencher_row(row: pd.Series, to_unit: str) -> dict:
    """Convert a quencher (ML) row into the target unit."""
    return {
        "triplet_energy_adiabatic": convert_energy(float(row["triplet_energy_adiabatic"]), BASE_QUENCHER_UNIT, to_unit),
        "lamda_S_T": convert_energy(float(row["lamda_S_T"]), BASE_QUENCHER_UNIT, to_unit),
        "lamda_T_S": convert_energy(float(row["lamda_T_S"]), BASE_QUENCHER_UNIT, to_unit),
    }


def _compute_marcus(pc_vals: dict, q_vals: dict) -> dict:
    """Compute Marcus theory driving force and forward/backward barriers in consistent units."""
    dG, dG_act_fwd, dG_act_bwd = marcus_theory.calculate_driving_force_and_barrier(
        pc_adiabatic_E_T=pc_vals["E_T_adiabatic"],
        pc_lambda_S_T=pc_vals["lamda_S_T"],
        pc_lambda_T_S=pc_vals["lamda_T_S"],
        quencher_E_T=q_vals["triplet_energy_adiabatic"],
        quencher_lambda_S_T=q_vals["lamda_S_T"],
        quencher_lambda_T_S=q_vals["lamda_T_S"],
        symmetric_approx=False,
    )
    return {
        "dG": float(dG),
        "dG_activation": float(dG_act_fwd),
        "dG_activation_backEnT": float(dG_act_bwd),
    }


def _render_plot(pc_vals: dict, q_vals: dict, unit: str, *, smiles: str | None = None):
    """Render the Marcus theory parabolas using the helper in science_utils.marcus_theory.

    Falls die symmetrische Approximation verwendet werden musste (Parabeln schneiden sich
    nicht sauber), wird eine Warnung direkt auf der Seite angezeigt.
    """
    res = marcus_theory.plot_reaction_coordinate(
        pc_adiabatic_E_T=pc_vals["E_T_adiabatic"],
        pc_lambda_S_T=pc_vals["lamda_S_T"],
        pc_lambda_T_S=pc_vals["lamda_T_S"],
        quencher_E_T=q_vals["triplet_energy_adiabatic"],
        quencher_lambda_S_T=q_vals["lamda_S_T"],
        quencher_lambda_T_S=q_vals["lamda_T_S"],
        unit=unit,
        render=True,
        show=False,
    )

    fig = res.get("fig", None)
    if fig is not None:
        st.pyplot(fig)

    if res.get("symmetric_approx"):
        target = f" for quencher {smiles}" if smiles else ""
        reason = res.get("symmetric_reason") or "parabolas do not intersect properly"
        st.warning(
            "Symmetric approximation used in Marcus plot" + target +
            f" (reason: {reason}). Energetic barriers are shown using a symmetric reorganization energy.",
            icon="⚠️",
        )


def _build_display_table(df: pd.DataFrame, pc_row: dict, unit: str) -> pd.DataFrame:
    pc_vals = _convert_pc_values(pc_row, unit)

    rows = []
    for _, r in df.iterrows():
        q_vals = _convert_quencher_row(r, unit)
        res = _compute_marcus(pc_vals, q_vals)
        rows.append(
            {
                "smiles": r["smiles"],
                "triplet_energy_adiabatic": q_vals["triplet_energy_adiabatic"],
                "lamda_S_T": q_vals["lamda_S_T"],
                "lamda_T_S": q_vals["lamda_T_S"],
                "dG": res["dG"],
                "dG_activation": res["dG_activation"],
                "dG_activation_backEnT": res["dG_activation_backEnT"],
            }
        )

    disp = pd.DataFrame(rows)
    # Nice ordering and rounding for display
    disp = disp[
        [
            "smiles",
            "triplet_energy_adiabatic",
            "lamda_S_T",
            "lamda_T_S",
            "dG",
            "dG_activation",
            "dG_activation_backEnT",
        ]
    ].copy()
    for col in [c for c in disp.columns if c != "smiles"]:
        disp[col] = disp[col].astype(float).round(6)
    return disp


def _build_pc_table(pc_row: dict, unit: str) -> pd.DataFrame:
    vals = _convert_pc_values(pc_row, unit)
    return pd.DataFrame([
        {
            "photocatalyst": pc_row.get("photocatalyst_name", "(unknown)"),
            "E_T_adiabatic": round(vals["E_T_adiabatic"], 6),
            "lamda_S_T": round(vals["lamda_S_T"], 6),
            "lamda_T_S": round(vals["lamda_T_S"], 6),
        }
    ])


def _build_summary_csv(disp_quenchers: pd.DataFrame, pc_row: dict, unit: str) -> pd.DataFrame:
    """Create a combined summary for download: PC row first, then all quenchers.

    PC row has 'photocatalyst' name and energy parameters; dG columns left empty.
    """
    pc_vals = _convert_pc_values(pc_row, unit)
    pc_summary = pd.DataFrame([
        {
            "photocatalyst": pc_row.get("photocatalyst_name", ""),
            "smiles": "",
            "E_T_adiabatic": round(pc_vals["E_T_adiabatic"], 6),
            "lamda_S_T": round(pc_vals["lamda_S_T"], 6),
            "lamda_T_S": round(pc_vals["lamda_T_S"], 6),
            "dG": "",
            "dG_activation": "",
            "dG_activation_backEnT": "",
        }
    ])

    # Align columns
    cols = [
        "photocatalyst",
        "smiles",
        "E_T_adiabatic",
        "lamda_S_T",
        "lamda_T_S",
        "dG",
        "dG_activation",
        "dG_activation_backEnT",
    ]

    # Ensure we always operate on a DataFrame
    q_df = disp_quenchers.copy() if isinstance(disp_quenchers, pd.DataFrame) else pd.DataFrame()

    # First, if present, rename ML column to the target name to avoid duplicates later
    if "triplet_energy_adiabatic" in q_df.columns and "E_T_adiabatic" not in q_df.columns:
        q_df = q_df.rename(columns={"triplet_energy_adiabatic": "E_T_adiabatic"})
    elif "triplet_energy_adiabatic" in q_df.columns and "E_T_adiabatic" in q_df.columns:
        # Edge case: both present; prefer ML values and drop the placeholder
        q_df.drop(columns=["E_T_adiabatic"], inplace=True)
        q_df = q_df.rename(columns={"triplet_energy_adiabatic": "E_T_adiabatic"})

    # Ensure missing columns exist AFTER rename to avoid duplicates
    for c in cols:
        if c not in q_df.columns:
            q_df[c] = ""

    # Ensure the photocatalyst column is empty for quencher rows
    q_df["photocatalyst"] = ""

    # Reorder/select expected columns only
    q_df = q_df[cols]

    combined = pd.concat([pc_summary[cols], q_df], axis=0, ignore_index=True)
    return combined


def main():
    st.set_page_config(page_title="Marcus Theory Analysis - PyRates", layout="wide")
    apply_global_styles()
    st.title("Marcus Theory Analysis")

    # Selectors: PC on the left, units on the right
    unit = st.session_state.get("energy_unit") or DEFAULT_DISPLAY_UNIT
    unit = _selectors()

    # Load selected photocatalyst row
    pc_row = _load_selected_photocatalyst()
    if not pc_row:
        st.info("No photocatalyst selected yet. Please go back to 'ML results' and choose one.")
        return

    # Load ML results (quenchers)
    df: pd.DataFrame | None = st.session_state.get("results_df")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No quencher predictions available. Please run the ML analysis first.")
        return

    st.subheader("Photocatalyst")
    st.dataframe(_build_pc_table(pc_row, unit), width="stretch")

    st.subheader("Results (per quencher)")
    disp = _build_display_table(df, pc_row, unit)
    st.dataframe(disp, width="stretch")

    # CSV download
    combined = _build_summary_csv(disp, pc_row, unit)
    st.download_button(
        label="Download summary (CSV)",
        data=combined.to_csv(index=False).encode("utf-8"),
        file_name="marcus_summary.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Marcus theory plots")

    pc_vals = _convert_pc_values(pc_row, unit)

    # Eine Plot-Sektion pro Quencher
    for _, r in df.iterrows():
        q_vals = _convert_quencher_row(r, unit)
        smiles = str(r.get("smiles", ""))
        with st.expander(f"Marcus plot for quencher: {smiles}"):
            _render_plot(pc_vals, q_vals, unit, smiles=smiles)


if __name__ == "__main__":
    main()
