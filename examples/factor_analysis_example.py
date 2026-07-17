"""
Example: PCA (scree plot + loadings) and Factor Analysis (no rotation,
orthogonal varimax rotation, oblique oblimin rotation) on an extended
10-asset portfolio designed to exhibit clear factor structure.

Requires network access (yfinance), scikit-learn, and `factor_analyzer`.

Run from the repo root:
    python examples/factor_analysis_example.py
"""

from riskfunctions.models.factor_analysis import (
    factor_analysis_loadings,
    fit_factor_analysis,
    fit_pca,
    load_extended_returns,
    pca_loadings,
    plot_correlation_heatmap,
    plot_loadings_heatmap,
    plot_pca_scree,
)

N_FACTORS_FULL = 10
N_COMPONENTS_REDUCED = 5  # chosen by inspecting the scree graph


def main():
    returns = load_extended_returns()
    plot_correlation_heatmap(returns)

    # PCA: scree graph on the full component count, then loadings for the
    # reduced number of components suggested by the scree graph.
    pca_full = fit_pca(returns, n_components=N_FACTORS_FULL)
    plot_pca_scree(pca_full)

    pca_reduced = fit_pca(returns, n_components=N_COMPONENTS_REDUCED)
    plot_loadings_heatmap(pca_loadings(pca_reduced, returns), "Correlation of each Principle Component with each Asset")

    # Factor Analysis: no rotation, orthogonal (varimax), oblique (oblimin).
    for rotation, title in [
        (None, "Correlation of each Factor with each Asset"),
        ("varimax", "Correlation of each Factor with Orthogonal Rotation with each Asset"),
        ("oblimin", "Correlation of each Factor with Oblique Rotation with each Asset"),
    ]:
        fa = fit_factor_analysis(returns, n_factors=N_FACTORS_FULL, rotation=rotation)
        plot_loadings_heatmap(factor_analysis_loadings(fa, returns), title)


if __name__ == "__main__":
    main()
