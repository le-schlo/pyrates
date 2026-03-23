import math
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Any


def _compute_marcus_core(
    pc_adiabatic_E_T: float,
    pc_lambda_S_T: float,
    pc_lambda_T_S: float,
    quencher_E_T: float,
    quencher_lambda_S_T: float,
    quencher_lambda_T_S: float,
    symmetric_approx: bool = False,
) -> Tuple[float, float, float, float, float, bool, str]:
    """Shared core logic for computing driving force and activation barriers.

    Returns
    -------
    dG : float
        Thermodynamic driving force.
    dG_activation_forward : float
        Forward activation barrier.
    dG_activation_backward : float
        Backward activation barrier.
    lambda_forward : float
        Forward reorganization energy.
    lambda_reverse : float
        Reverse reorganization energy.
    symmetric_flag : bool
        Whether the symmetric approximation has been applied.
    reason : str
        Short reason why the symmetric approximation was used (if any).
    """

    dG = quencher_E_T - pc_adiabatic_E_T
    lambda_forward = pc_lambda_T_S + quencher_lambda_S_T
    lambda_reverse = pc_lambda_S_T + quencher_lambda_T_S

    reason = ""
    # Expression from the original calculate_driving_force_and_barrier
    root_arg = lambda_forward * lambda_reverse + ((lambda_forward - lambda_reverse) * dG)
    auto_symmetric = root_arg < 0 or abs(lambda_forward - lambda_reverse) < 1e-10

    if auto_symmetric:
        symmetric_approx = True
        reason = "parabolas_do_not_intersect_or_symmetric"
    if not symmetric_approx:
        root_term = math.sqrt(root_arg)
        dG_activation = lambda_forward * (
            ((root_term - lambda_reverse) / (lambda_forward - lambda_reverse)) ** 2
        )
    else:
        lambda_rxn = math.sqrt(max(lambda_forward * lambda_reverse, 0.0))
        if lambda_rxn <= 0:
            # Fallback in case of numerical issues
            dG_activation = 0.0
            reason = reason or "invalid_lambda_product"
        else:
            dG_activation = (lambda_rxn + dG) ** 2 / (4 * lambda_rxn)

    dG_activation_backward = dG_activation + abs(dG)
    return (
        dG,
        dG_activation,
        dG_activation_backward,
        lambda_forward,
        lambda_reverse,
        symmetric_approx,
        reason,
    )


def calculate_driving_force_and_barrier(
    pc_adiabatic_E_T: float,
    pc_lambda_S_T: float,
    pc_lambda_T_S: float,
    quencher_E_T: float,
    quencher_lambda_S_T: float,
    quencher_lambda_T_S: float,
    symmetric_approx: bool = False,
) -> Tuple[float, float, float]:
    """Calculate thermodynamic driving force and barrier heights for EnT.

    In ``symmetric_approx`` mode, uses the symmetric approximation for the
    reorganization energy (geometric mean of forward and reverse
    reorganization energies).

    If the product and reactant parabolas do not intersect properly
    (negative square-root argument or nearly identical reorganization
    energies), the symmetric approximation is applied automatically.
    """

    (
        dG,
        dG_activation,
        dG_activation_backward,
        _lambda_forward,
        _lambda_reverse,
        symmetric_flag,
        reason,
    ) = _compute_marcus_core(
        pc_adiabatic_E_T,
        pc_lambda_S_T,
        pc_lambda_T_S,
        quencher_E_T,
        quencher_lambda_S_T,
        quencher_lambda_T_S,
        symmetric_approx,
    )

    if symmetric_flag and not symmetric_approx:
        # Only warn if the user did not explicitly request symmetric_approx=True
        print(
            "Warning: Product and reactant parabolas do not intersect properly. Using symmetric approximation.",
            f"Reason: {reason}",
        )

    return dG, dG_activation, dG_activation_backward


def plot_reaction_coordinate(
    pc_adiabatic_E_T: float,
    pc_lambda_S_T: float,
    pc_lambda_T_S: float,
    quencher_E_T: float,
    quencher_lambda_S_T: float,
    quencher_lambda_T_S: float,
    unit: str = "kcal/mol",
    *,
    render: bool = False,
    show: bool = False,
) -> Dict[str, Any]:
    """Plot the Marcus-theory parabolas for a triplet–triplet energy transfer reaction.

    Returns a dictionary that contains the matplotlib figure, metadata
    (limits, key energies, TS position) and a flag that indicates
    whether the symmetric approximation was applied.

    This keeps compatibility with existing callers that only use the
    figure (access via ``result["fig"]``).
    """

    # --- 1. Compute core values via shared logic ---
    (
        dG,
        dG_activation,
        dG_activation_backward,
        lambda_f,
        lambda_r,
        symmetric_flag,
        reason,
    ) = _compute_marcus_core(
        pc_adiabatic_E_T,
        pc_lambda_S_T,
        pc_lambda_T_S,
        quencher_E_T,
        quencher_lambda_S_T,
        quencher_lambda_T_S,
        symmetric_approx=False,
    )

    # --- 2. Intersection / reaction coordinate of the transition state ---
    a = lambda_f - lambda_r
    b = 2 * lambda_r
    c = -(lambda_r + dG)
    discriminant = b**2 - 4 * a * c

    x_intersect = 0.5  # sensible default position

    if not symmetric_flag and abs(a) >= 1e-10 and discriminant >= 0:
        # Regular case: quadratic equation is well behaved
        sol1 = (-b + math.sqrt(discriminant)) / (2 * a)
        sol2 = (-b - math.sqrt(discriminant)) / (2 * a)
        x_intersect = sol1 if abs(sol1 - 0.5) < abs(sol2 - 0.5) else sol2
    else:
        # Symmetric approximation or numerically problematic case
        if not symmetric_flag:
            symmetric_flag = True
            reason = reason or "quadratic_intersection_failed"
        # Choose a TS position based on curve shape; for strongly
        # exergonic reactions the TS tends to shift towards reactants/products.
        try:
            lambda_rxn = math.sqrt(max(lambda_f * lambda_r, 0.0))
            if lambda_rxn > 0:
                # Inspired by the standard Marcus expression:
                # x roughly proportional to (lambda_r + dG) / (lambda_f + lambda_r)
                denom = lambda_f + lambda_r
                if abs(denom) > 1e-12:
                    x_intersect = (lambda_r + dG) / denom
                else:
                    x_intersect = 0.5
            else:
                x_intersect = 0.5
        except Exception:
            x_intersect = 0.5

    # --- 3. Plotting with robust scaling ---
    # x-scale: always include 0, 1 and the TS position
    key_x_points = [0.0, 1.0, float(x_intersect)]
    x_min_core = min(key_x_points)
    x_max_core = max(key_x_points)
    # Padding around the key points
    x_span = max(x_max_core - x_min_core, 1.0)
    xmin = x_min_core - 0.5 * x_span
    xmax = x_max_core + 0.5 * x_span

    # Ensure a minimal useful window
    xmin = min(xmin, -1.0)
    xmax = max(xmax, 2.0)

    x_coords = np.linspace(xmin - 0.5 * x_span, xmax + 0.5 * x_span, 1000)
    G_initial = lambda_f * x_coords**2
    G_final = lambda_r * (x_coords - 1) ** 2 + dG

    plt.figure(figsize=(10, 7))
    plt.plot(x_coords, G_initial, "b-", label="Reactants Parabola")
    plt.plot(x_coords, G_final, "r-", label="Products Parabola")

    # y-scale: include all relevant energy values
    key_y_points = [
        0.0,
        float(dG),
        float(dG_activation),
        float(dG_activation_backward),
        float(lambda_f),
        float(lambda_r + dG),
    ]
    ymin_core = min(key_y_points + [min(G_initial.min(), G_final.min())])
    ymax_core = max(key_y_points + [max(G_initial.max(), G_final.max())])

    # Safety padding so annotations are not clipped
    y_span = max(ymax_core - ymin_core, 1.0)
    ymin = ymin_core - 0.15 * y_span
    ymax = ymax_core + 0.3 * y_span

    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)

    # ΔG line (vertical at x = 1)
    plt.plot([1, 1], [0, dG], "-", color="black")
    plt.plot(
        [1, 1],
        [0, dG],
        marker="_",
        markersize=12,
        linestyle="None",
        color="black",
        markeredgewidth=2,
    )
    plt.text(
        1.05,
        dG / 2,
        f"ΔG = {dG:.2f} {unit}",
        va="center",
        ha="left",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # ΔG‡ (forward) at x_intersect
    plt.plot([x_intersect, x_intersect], [0, dG_activation], "-", color="green")
    plt.plot(
        [x_intersect, x_intersect],
        [0, dG_activation],
        marker="_",
        markersize=12,
        linestyle="None",
        color="green",
        markeredgewidth=2,
    )
    plt.text(
        x_intersect,
        dG_activation * 1.05,
        f"ΔG‡ = {dG_activation:.2f} {unit}",
        va="bottom",
        ha="center",
        fontsize=12,
        color="green",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # λ_forward at x = 1
    plt.plot([1, 1], [0, lambda_f], "-", color="blue")
    plt.plot(
        [1, 1],
        [0, lambda_f],
        marker="_",
        markersize=12,
        linestyle="None",
        color="blue",
        markeredgewidth=2,
    )
    plt.text(
        1.05,
        lambda_f / 2,
        f"λ_forward = {lambda_f:.2f} {unit}",
        va="center",
        ha="left",
        fontsize=12,
        color="blue",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # λ_reverse at x = 0 (from the product level)
    plt.plot([0, 0], [dG, lambda_r + dG], "-", color="red")
    plt.plot(
        [0, 0],
        [dG, lambda_r + dG],
        marker="_",
        markersize=12,
        linestyle="None",
        color="red",
        markeredgewidth=2,
    )
    plt.text(
        -0.05,
        dG + (lambda_r / 2),
        f"λ_reverse = {lambda_r:.2f} {unit}",
        va="center",
        ha="right",
        fontsize=12,
        color="red",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title("Marcus Theory Plot for Energy Transfer", fontsize=16)
    ax.set_xlabel("Normalized Reaction Coordinate", fontsize=14)
    ax.set_ylabel("ΔG", fontsize=14)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=12)
    fig.tight_layout()

    if render:
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
    if show:
        plt.show()

    result: Dict[str, Any] = {
        "fig": fig,
        "ax": ax,
        "dG": dG,
        "dG_activation_forward": dG_activation,
        "dG_activation_backward": dG_activation_backward,
        "lambda_forward": lambda_f,
        "lambda_reverse": lambda_r,
        "x_intersect": x_intersect,
        "xlim": (xmin, xmax),
        "ylim": (ymin, ymax),
        "symmetric_approx": symmetric_flag,
        "symmetric_reason": reason,
    }
    return result


if __name__ == "__main__":
    res = plot_reaction_coordinate(
        pc_adiabatic_E_T=62.9,
        pc_lambda_S_T=6.5,
        pc_lambda_T_S=6.5,
        quencher_E_T=60.9,
        quencher_lambda_S_T=13.2,
        quencher_lambda_T_S=13.2,
        unit="kcal/mol",
        render=True,
        show=True,
    )

    dG, dG_act, dG_act_bwd = calculate_driving_force_and_barrier(
        pc_adiabatic_E_T=60.0,
        pc_lambda_S_T=2.0,
        pc_lambda_T_S=3.0,
        quencher_E_T=55.0,
        quencher_lambda_S_T=55.0,
        quencher_lambda_T_S=55.0,
        symmetric_approx=False,
    )

    print(dG, dG_act, dG_act_bwd)
