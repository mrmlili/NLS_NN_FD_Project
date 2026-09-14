"""
========================================================================
PAPER RESULTS FINAL — 20-08-2026
========================================================================

Purpose
-------
Generate the final manuscript tables, supplementary error-metric table,
and final figures directly from the canonical raw experiment files.

This script is a hardened successor to:
    paper_results_final.py

The original script is intentionally left unchanged.

Canonical sources
-----------------
Table 1
    TEST #28-HOLDOUT
    Held-out pointwise test subset
    N = 32, 64, 128, 256

Table 2
    TEST #29
    Five-seed unseen-resolution robustness at N = 512

Table 3
    TEST #37
    Independent-realization generalization
    N = 32, 64, 128, 256

Table 4
    TEST #38S
    Robust single-state CPU benchmark

Table S1
    TEST #28-HOLDOUT error-distribution metrics
    MAE, P95, P99, Linf
    Source: test28_error_metrics.npz

Figures
-------
Figure 1
    In-range physical spatial-correction RMSE
    Source: TEST #28-HOLDOUT

Figure 2
    Five-seed N=512 robustness
    Source: TEST #29

Figure 3
    Accuracy-cost trade-off
    Accuracy source: TEST #28-HOLDOUT
    Cost source: TEST #38S

Integrity principles
--------------------
1. No numerical values are hard-coded.
2. No ratios are recomputed from rounded CSV presentation values.
3. Stored ratios are checked against full-precision source values.
4. Figure 3 aligns accuracy and cost explicitly by resolution N.
5. Table S1 RMSE is checked against canonical TEST #28 RMSE.
6. The canonical source files are read-only; this script does not modify them.

========================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# PROJECT ROOT
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ======================================================================
# OUTPUT DIRECTORY
# ======================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "nn_fd"
    / "paper_results_final_20_8"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# CANONICAL INPUT FILES
# ======================================================================

TABLE1_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "nn_fd"
    / "nn_ff_fd_ra_holdout_test28.npz"
)

TABLE2_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "nn_fd"
    / "nn_ff_fd_ra_robustness_n512_test29.npz"
)

TABLE3_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "nn_fd"
    / "independent_realization_generalization_test37.npz"
)

TABLE4_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "nn_fd"
    / "operator_cpu_benchmark_test38s.npz"
)

TABLES1_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "nn_fd"
    / "paper_results_final"
    / "test28_error_metrics.npz"
)


# ======================================================================
# HELPERS
# ======================================================================

def require_file(path: Path) -> None:
    """Raise a clear error when a canonical source file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired canonical source file not found:\n{path}"
        )


def require_keys(
    npz_obj,
    required_keys,
    source_name,
) -> None:
    """Require canonical fields before using a source file."""
    missing = [
        key
        for key in required_keys
        if key not in npz_obj.files
    ]

    if missing:
        raise KeyError(
            f"\n{source_name} is missing required fields:\n"
            f"{missing}"
        )


def save_csv(
    dataframe: pd.DataFrame,
    filename: str,
) -> Path:
    """Save dataframe to the dedicated 20-08-2026 output directory."""
    path = OUTPUT_DIR / filename

    dataframe.to_csv(
        path,
        index=False,
        float_format="%.12e",
    )

    print(f"Saved: {path}")
    return path


def assert_close(
    actual,
    expected,
    label,
    rtol=1e-10,
    atol=1e-15,
) -> None:
    """
    Check numerical consistency using full-precision values.
    """
    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    expected = np.asarray(
        expected,
        dtype=np.float64,
    )

    if not np.allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
    ):
        raise AssertionError(
            f"\n{label} failed consistency check.\n"
            f"Actual   = {actual}\n"
            f"Expected = {expected}"
        )


def configure_axes(ax) -> None:
    """Common figure formatting."""
    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )


# ======================================================================
# TABLE 1
# TEST #28-HOLDOUT
# ======================================================================

print("=" * 78)
print("TABLE 1 — HELD-OUT IN-RANGE ACCURACY")
print("=" * 78)

require_file(TABLE1_SOURCE)

z1 = np.load(
    TABLE1_SOURCE
)

require_keys(
    z1,
    [
        "N",
        "unseen",
        "parent_rmse",
        "nn_rmse",
        "ra_rmse",
    ],
    "TEST #28-HOLDOUT",
)

N1_all = z1["N"]

unseen1 = z1["unseen"]

mask1 = ~unseen1

N1 = N1_all[mask1]

parent1 = z1["parent_rmse"][mask1]
nn1 = z1["nn_rmse"][mask1]
ra1 = z1["ra_rmse"][mask1]

order1 = np.argsort(N1)

N1 = N1[order1]
parent1 = parent1[order1]
nn1 = nn1[order1]
ra1 = ra1[order1]

assert np.array_equal(
    N1,
    np.array(
        [32, 64, 128, 256],
        dtype=np.int32,
    ),
)

assert np.all(
    np.isfinite(parent1)
)

assert np.all(
    np.isfinite(nn1)
)

assert np.all(
    np.isfinite(ra1)
)

assert np.all(
    parent1 >= 0.0
)

assert np.all(
    nn1 >= 0.0
)

assert np.all(
    ra1 >= 0.0
)

table1 = pd.DataFrame(
    {
        "N": N1.astype(np.int32),
        "Parent_FD_RMSE": parent1,
        "NN_FD_RMSE": nn1,
        "NN_FF_FD_RA_RMSE": ra1,
    }
)

print(
    table1.to_string(
        index=False
    )
)

table1_path = save_csv(
    table1,
    "Table1_main_accuracy.csv",
)


# ======================================================================
# FIGURE 1
# ======================================================================

print()
print("=" * 78)
print("FIGURE 1 — IN-RANGE ACCURACY")
print("=" * 78)

fig, ax = plt.subplots(
    figsize=(7.2, 4.8)
)

ax.plot(
    N1,
    parent1,
    marker="o",
    linewidth=2.0,
    markersize=6,
    label="Parent FD",
)

ax.plot(
    N1,
    nn1,
    marker="s",
    linewidth=2.0,
    markersize=6,
    label="NN-FD",
)

ax.plot(
    N1,
    ra1,
    marker="^",
    linewidth=2.0,
    markersize=6,
    label="NN-FF-FD-RA",
)

ax.set_yscale("log")

ax.set_xlabel(
    r"Resolution $N$"
)

ax.set_ylabel(
    "Physical correction RMSE"
)

ax.set_title(
    "In-range spatial-correction accuracy"
)

ax.set_xticks(
    N1
)

configure_axes(ax)

ax.legend()

fig.tight_layout()

figure1_path = (
    OUTPUT_DIR
    / "figure1_main_accuracy.png"
)

fig.savefig(
    figure1_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    f"Saved: {figure1_path}"
)


# ======================================================================
# TABLE 2
# TEST #29 — FIVE-SEED N=512 ROBUSTNESS
# ======================================================================

print()
print("=" * 78)
print("TABLE 2 — UNSEEN N=512 FIVE-SEED ROBUSTNESS")
print("=" * 78)

require_file(TABLE2_SOURCE)

z2 = np.load(
    TABLE2_SOURCE
)

require_keys(
    z2,
    [
        "seeds",
        "parent_rmse",
        "nn_rmse",
        "ra_rmse",
        "ra_vs_parent",
        "ra_vs_nn",
        "parent_rmse_mean",
        "parent_rmse_std",
        "nn_rmse_mean",
        "nn_rmse_std",
        "ra_rmse_mean",
        "ra_rmse_std",
    ],
    "TEST #29",
)

seeds2 = z2["seeds"].astype(
    np.int64
)

parent2 = z2["parent_rmse"]
nn2 = z2["nn_rmse"]
ra2 = z2["ra_rmse"]

parent_over_ra2 = z2["ra_vs_parent"]
nn_over_ra2 = z2["ra_vs_nn"]

# Check stored ratios against full-precision RMSE values.
assert_close(
    parent_over_ra2,
    parent2 / ra2,
    "TEST #29 Parent/RA ratios",
)

assert_close(
    nn_over_ra2,
    nn2 / ra2,
    "TEST #29 NN/RA ratios",
)

table2 = pd.DataFrame(
    {
        "Seed": seeds2,
        "Parent_FD_RMSE": parent2,
        "NN_FD_RMSE": nn2,
        "NN_FF_FD_RA_RMSE": ra2,
        "Parent_over_NN_FF_FD_RA": parent_over_ra2,
        "NN_FD_over_NN_FF_FD_RA": nn_over_ra2,
    }
)

print(
    table2.to_string(
        index=False
    )
)

table2_path = save_csv(
    table2,
    "Table2_unseen_512_robustness.csv",
)


# ======================================================================
# FIGURE 2
# ======================================================================

print()
print("=" * 78)
print("FIGURE 2 — N=512 GENERALIZATION")
print("=" * 78)

fig, ax = plt.subplots(
    figsize=(8.2, 4.8)
)

x = np.arange(
    len(seeds2)
)

width = 0.24

ax.bar(
    x - width,
    parent2,
    width=width,
    label="Parent FD",
)

ax.bar(
    x,
    nn2,
    width=width,
    label="NN-FD",
)

ax.bar(
    x + width,
    ra2,
    width=width,
    label="NN-FF-FD-RA",
)

ax.set_yscale("log")

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        str(seed)
        for seed in seeds2
    ],
    rotation=30,
    ha="right",
)

ax.set_xlabel(
    "Independent seed"
)

ax.set_ylabel(
    "Physical correction RMSE"
)

ax.set_title(
    r"Unseen-resolution robustness at $N=512$"
)

configure_axes(ax)

ax.legend()

fig.tight_layout()

figure2_path = (
    OUTPUT_DIR
    / "figure2_generalization_512.png"
)

fig.savefig(
    figure2_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    f"Saved: {figure2_path}"
)


# ======================================================================
# TABLE 3
# TEST #37
# ======================================================================

print()
print("=" * 78)
print("TABLE 3 — INDEPENDENT-REALIZATION GENERALIZATION")
print("=" * 78)

require_file(TABLE3_SOURCE)

z3 = np.load(
    TABLE3_SOURCE
)

require_keys(
    z3,
    [
        "N",
        "parent_rmse",
        "nn_rmse",
        "ra_rmse",
        "nn_parent",
        "ra_parent",
        "ra_nn",
    ],
    "TEST #37",
)

N3 = z3["N"].astype(
    np.int32
)

parent3 = z3["parent_rmse"]
nn3 = z3["nn_rmse"]
ra3 = z3["ra_rmse"]

parent_over_nn3 = z3["nn_parent"]
parent_over_ra3 = z3["ra_parent"]
ra_over_nn3 = z3["ra_nn"]

assert_close(
    parent_over_nn3,
    parent3 / nn3,
    "TEST #37 Parent/NN ratios",
)

assert_close(
    parent_over_ra3,
    parent3 / ra3,
    "TEST #37 Parent/RA ratios",
)

nn_over_ra3 = z3["ra_nn"]

assert_close(
    nn_over_ra3,
    nn3 / ra3,
    "TEST #37 NN/RA ratios",
)

order3 = np.argsort(N3)

N3 = N3[order3]
parent3 = parent3[order3]
nn3 = nn3[order3]
ra3 = ra3[order3]
parent_over_nn3 = parent_over_nn3[order3]
parent_over_ra3 = parent_over_ra3[order3]
nn_over_ra3 = nn_over_ra3[order3]

assert np.array_equal(
    N3,
    np.array(
        [32, 64, 128, 256],
        dtype=np.int32,
    ),
)

table3 = pd.DataFrame(
    {
        "N": N3,
        "Parent_FD_RMSE": parent3,
        "NN_FD_RMSE": nn3,
        "NN_FF_FD_RA_RMSE": ra3,
        "Parent_over_NN_FD": parent_over_nn3,
        "Parent_over_NN_FF_FD_RA": parent_over_ra3,
    }
)

print(
    table3.to_string(
        index=False
    )
)

table3_path = save_csv(
    table3,
    "Table3_independent_generalization.csv",
)


# ======================================================================
# TABLE 4
# TEST #38S
# ======================================================================

print()
print("=" * 78)
print("TABLE 4 — OPERATOR CPU COST")
print("=" * 78)

require_file(TABLE4_SOURCE)

z4 = np.load(
    TABLE4_SOURCE
)

require_keys(
    z4,
    [
        "N",
        "parent_mean",
        "nn_mean",
        "ra_mean",
        "nn_ratio",
        "ra_ratio",
    ],
    "TEST #38S",
)

N4 = z4["N"].astype(
    np.int32
)

parent4 = z4["parent_mean"]
nn4 = z4["nn_mean"]
ra4 = z4["ra_mean"]

nn_over_fd4 = z4["nn_ratio"]
ra_over_fd4 = z4["ra_ratio"]

assert_close(
    nn_over_fd4,
    nn4 / parent4,
    "TEST #38S NN/FD ratios",
)

assert_close(
    ra_over_fd4,
    ra4 / parent4,
    "TEST #38S RA/FD ratios",
)

order4 = np.argsort(N4)

N4_sorted = N4[order4]
parent4_sorted = parent4[order4]
nn4_sorted = nn4[order4]
ra4_sorted = ra4[order4]
nn_over_fd4_sorted = nn_over_fd4[order4]
ra_over_fd4_sorted = ra_over_fd4[order4]

assert np.array_equal(
    N4_sorted,
    np.array(
        [32, 64, 128, 256],
        dtype=np.int32,
    ),
)

table4 = pd.DataFrame(
    {
        "N": N4_sorted,
        "Parent_FD_s": parent4_sorted,
        "NN_FD_s": nn4_sorted,
        "NN_FF_FD_RA_s": ra4_sorted,
        "NN_FD_over_FD": nn_over_fd4_sorted,
        "NN_FF_FD_RA_over_FD": ra_over_fd4_sorted,
    }
)

print(
    table4.to_string(
        index=False
    )
)

table4_path = save_csv(
    table4,
    "Table4_operator_cpu_cost.csv",
)


# ======================================================================
# TABLE S1
# TEST #28-HOLDOUT ERROR-DISTRIBUTION METRICS
# ======================================================================

print()
print("=" * 78)
print("TABLE S1 — ERROR DISTRIBUTION METRICS")
print("=" * 78)

require_file(TABLES1_SOURCE)

zs1 = np.load(
    TABLES1_SOURCE
)

require_keys(
    zs1,
    [
        "N",
        "unseen",
        "samples",
        "method",
        "rmse",
        "mae",
        "p95",
        "p99",
        "linf",
    ],
    "TEST #28 error-metrics export",
)

tableS1 = pd.DataFrame(
    {
        "N": zs1["N"].astype(
            np.int32
        ),
        "Unseen": zs1["unseen"].astype(
            bool
        ),
        "Samples": zs1["samples"].astype(
            np.int32
        ),
        "Method": zs1["method"].astype(
            str
        ),
        "RMSE": zs1["rmse"],
        "MAE": zs1["mae"],
        "P95": zs1["p95"],
        "P99": zs1["p99"],
        "Linf": zs1["linf"],
    }
)

# --------------------------------------------------------------
# Check S1 RMSE against canonical TEST #28.
# --------------------------------------------------------------

for method, canonical_rmse, method_label in [
    (
        "Parent FD",
        z1["parent_rmse"],
        "Parent FD",
    ),
    (
        "NN-FD",
        z1["nn_rmse"],
        "NN-FD",
    ),
    (
        "NN-FF-FD-RA",
        z1["ra_rmse"],
        "NN-FF-FD-RA",
    ),
]:

    method_rows = tableS1[
        tableS1["Method"] == method
    ].sort_values(
        ["Unseen", "N"]
    )

    # Only compare entries corresponding to canonical
    # TEST #28 N=32,64,128,256,512.
    expected_N = np.array(
        [32, 64, 128, 256, 512],
        dtype=np.int32,
    )

    actual_N = method_rows[
        "N"
    ].to_numpy(
        dtype=np.int32
    )

    assert np.array_equal(
        actual_N,
        expected_N,
    ), (
        f"Table S1 N mismatch for {method_label}: "
        f"{actual_N}"
    )

    assert_close(
        method_rows["RMSE"].to_numpy(),
        np.array(
            [
                canonical_rmse[
                    np.where(
                        z1["N"] == n
                    )[0][0]
                ]
                for n in expected_N
            ]
        ),
        f"Table S1 / TEST #28 RMSE consistency: {method_label}",
    )

# --------------------------------------------------------------
# Basic S1 checks.
# --------------------------------------------------------------

assert len(tableS1) == 15

assert set(
    tableS1["Method"]
) == {
    "Parent FD",
    "NN-FD",
    "NN-FF-FD-RA",
}

assert set(
    tableS1["N"].unique()
) == {
    32,
    64,
    128,
    256,
    512,
}

for col in [
    "RMSE",
    "MAE",
    "P95",
    "P99",
    "Linf",
]:
    assert np.all(
        np.isfinite(
            tableS1[col].to_numpy(
                dtype=np.float64
            )
        )
    )

    assert np.all(
        tableS1[col].to_numpy(
            dtype=np.float64
        ) >= 0.0
    )

tableS1_path = save_csv(
    tableS1,
    "TableS1_error_distribution_metrics.csv",
)


# ======================================================================
# FIGURE 3
# ACCURACY-COST TRADE-OFF
# ======================================================================

print()
print("=" * 78)
print("FIGURE 3 — ACCURACY-COST TRADE-OFF")
print("=" * 78)

# ----------------------------------------------------------------------
# Explicit N-based alignment.
#
# Accuracy:
#     TEST #28-HOLDOUT
#
# Cost:
#     TEST #38S
#
# Each scatter point therefore corresponds to the same N in both
# canonical sources.
# ----------------------------------------------------------------------

assert np.array_equal(
    N1,
    N4_sorted,
), (
    "Figure 3 source mismatch: "
    "TEST #28-HOLDOUT and TEST #38S "
    "do not contain the same resolutions."
)

fig, ax = plt.subplots(
    figsize=(7.4, 5.2)
)

ax.scatter(
    parent4_sorted,
    parent1,
    marker="o",
    s=55,
    label="Parent FD",
)

ax.scatter(
    nn4_sorted,
    nn1,
    marker="s",
    s=55,
    label="NN-FD",
)

ax.scatter(
    ra4_sorted,
    ra1,
    marker="^",
    s=60,
    label="NN-FF-FD-RA",
)

# Label only the RA points, matching the previously approved
# figure design.
for xval, yval, nval in zip(
    ra4_sorted,
    ra1,
    N1,
):
    ax.annotate(
        f"$N={int(nval)}$",
        (xval, yval),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)

ax.set_xlabel(
    "Operator evaluation time (s)"
)

ax.set_ylabel(
    "Physical spatial-correction RMSE"
)

ax.set_title(
    "Accuracy--cost trade-off"
)

configure_axes(ax)

ax.legend()

fig.tight_layout()

figure3_path = (
    OUTPUT_DIR
    / "figure3_accuracy_cost_tradeoff.png"
)

fig.savefig(
    figure3_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    f"Saved: {figure3_path}"
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print()
print("=" * 78)
print("PAPER RESULTS FINAL — 20-08-2026 SUMMARY")
print("=" * 78)

parent512_mean = float(
    z2["parent_rmse_mean"]
)

parent512_std = float(
    z2["parent_rmse_std"]
)

nn512_mean = float(
    z2["nn_rmse_mean"]
)

nn512_std = float(
    z2["nn_rmse_std"]
)

ra512_mean = float(
    z2["ra_rmse_mean"]
)

ra512_std = float(
    z2["ra_rmse_std"]
)

print()
print(
    "Table 1: TEST #28-HOLDOUT "
    "held-out in-range evaluation"
)

print(
    "Table 2: TEST #29 "
    "five-seed unseen N=512 robustness"
)

print(
    "Table 3: TEST #37 "
    "independent-realization generalization"
)

print(
    "Table 4: TEST #38S "
    "robust single-state CPU benchmark"
)

print(
    "Table S1: TEST #28-HOLDOUT "
    "error-distribution metrics"
)

print()
print(
    "N=512 five-seed exact mean +/- std:"
)

print(
    f"Parent FD   = "
    f"{parent512_mean:.12e} +/- "
    f"{parent512_std:.12e}"
)

print(
    f"NN-FD       = "
    f"{nn512_mean:.12e} +/- "
    f"{nn512_std:.12e}"
)

print(
    f"NN-FF-FD-RA = "
    f"{ra512_mean:.12e} +/- "
    f"{ra512_std:.12e}"
)


# ======================================================================
# SAVE SUMMARY TXT
# ======================================================================

summary_lines = [
    "PAPER RESULTS FINAL — 20-08-2026",
    "",
    "Canonical sources:",
    f"Table 1: {TABLE1_SOURCE}",
    f"Table 2: {TABLE2_SOURCE}",
    f"Table 3: {TABLE3_SOURCE}",
    f"Table 4: {TABLE4_SOURCE}",
    f"Table S1: {TABLES1_SOURCE}",
    "",
    "Outputs:",
    f"Table 1: {table1_path}",
    f"Table 2: {table2_path}",
    f"Table 3: {table3_path}",
    f"Table 4: {table4_path}",
    f"Table S1: {tableS1_path}",
    f"Figure 1: {figure1_path}",
    f"Figure 2: {figure2_path}",
    f"Figure 3: {figure3_path}",
    "",
    "N=512 five-seed exact mean +/- std:",
    f"Parent FD   = "
    f"{parent512_mean:.12e} +/- "
    f"{parent512_std:.12e}",
    f"NN-FD       = "
    f"{nn512_mean:.12e} +/- "
    f"{nn512_std:.12e}",
    f"NN-FF-FD-RA = "
    f"{ra512_mean:.12e} +/- "
    f"{ra512_std:.12e}",
]

summary_path = (
    OUTPUT_DIR
    / "paper_results_summary.txt"
)

summary_path.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

print()
print(
    f"Saved: {summary_path}"
)

print()
print("=" * 78)
print("PAPER RESULTS FINAL — 20-08-2026 COMPLETED")
print("=" * 78)