"""Generate tail-diagnostic plots and save them to examples/output/."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extremeloss.plotting import (  # noqa: E402
    plot_exceedance_curve,
    plot_hill_curve,
    plot_mean_excess,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(321)
    losses = rng.pareto(a=3.0, size=20_000) * 30.0
    thresholds = np.linspace(10.0, 80.0, 15)

    ax = plot_exceedance_curve(losses, thresholds)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT_DIR / "exceedance_curve.png", dpi=150)
    plt.close(ax.figure)

    ax = plot_mean_excess(losses, thresholds)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT_DIR / "mean_excess.png", dpi=150)
    plt.close(ax.figure)

    positive_losses = np.clip(losses, 1e-8, None)
    ax = plot_hill_curve(positive_losses)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT_DIR / "hill_curve.png", dpi=150)
    plt.close(ax.figure)

    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
