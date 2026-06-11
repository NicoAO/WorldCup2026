"""Expected goals -> exact-score probability matrix (Dixon-Coles Poisson)."""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from . import config


def score_matrix(mu_h: float, mu_a: float, rho: float | None = None) -> np.ndarray:
    """P[h, a] for h, a in 0..MAX_GOALS with Dixon-Coles low-score correction."""
    if rho is None:
        rho = config.DEFAULTS["rho"]
    n = config.MAX_GOALS + 1
    ph = poisson.pmf(np.arange(n), mu_h)
    pa = poisson.pmf(np.arange(n), mu_a)
    m = np.outer(ph, pa)
    # Dixon-Coles tau on {0,1}x{0,1}
    m[0, 0] *= 1 - mu_h * mu_a * rho
    m[0, 1] *= 1 + mu_h * rho
    m[1, 0] *= 1 + mu_a * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def outcome_probs(m: np.ndarray) -> tuple[float, float, float]:
    home = np.tril(m, -1).sum()   # h > a
    draw = np.trace(m)
    away = np.triu(m, 1).sum()
    return float(home), float(draw), float(away)


def market_summary(m: np.ndarray) -> dict:
    h, d, a = outcome_probs(m)
    n = m.shape[0]
    tot = np.add.outer(np.arange(n), np.arange(n))
    return {
        "p_home": h, "p_draw": d, "p_away": a,
        "over_2_5": float(m[tot > 2.5].sum()),
        "under_2_5": float(m[tot < 2.5].sum()),
        "btts": float(m[1:, 1:].sum()),
    }


def top_scorelines(m: np.ndarray, k: int = 8) -> list[tuple[str, float]]:
    flat = [((h, a), m[h, a]) for h in range(m.shape[0]) for a in range(m.shape[1])]
    flat.sort(key=lambda x: -x[1])
    return [(f"{h}-{a}", float(p)) for (h, a), p in flat[:k]]


def blend_with_market(m: np.ndarray, market_1x2: tuple[float, float, float],
                      weight: float) -> np.ndarray:
    """Rescale the score matrix so its 1X2 matches a model/market blend.

    Scales the win/draw/loss regions of the matrix proportionally (one IPF step
    is exact here because the regions partition the matrix).
    """
    h, d, a = outcome_probs(m)
    th = (1 - weight) * h + weight * market_1x2[0]
    td = (1 - weight) * d + weight * market_1x2[1]
    ta = (1 - weight) * a + weight * market_1x2[2]
    out = m.copy()
    ih, ia = np.tril_indices_from(m, -1)
    out[ih, ia] *= th / max(h, 1e-9)
    out[np.diag_indices_from(out)] *= td / max(d, 1e-9)
    iu, ja = np.triu_indices_from(m, 1)
    out[iu, ja] *= ta / max(a, 1e-9)
    return out / out.sum()
