"""
Classify_Clusters.py (refactor)

Objetivo:
- Combinar métricas de similitud (p.ej. Jaccard y Wang) de forma robusta.
- Estimar umbrales para separar "baja" vs "alta" similitud.
- Incluir validación estadística para detectar si realmente existe bimodalidad
  (GMM 2 componentes) y evitar falsos cortes.

Principales mejoras:
1) API consistente (methods y options).
2) Validación de entradas (NaN/Inf/rangos/size).
3) Umbral GMM con validación estadística:
   - Comparación BIC: 2 componentes vs 1 componente
   - Índice de separación: |m1-m2| / sqrt(v1+v2)
   - Checks de degeneración (varianzas demasiado pequeñas)
   - Chequeo de solapamiento (aprox. por separación)
4) Umbral por histograma con métricas de confianza.
5) Umbral por percentil robusto.
6) Logging (sin prints).

Nota:
- Se asume que los scores (Jaccard/Wang/combined) viven en [0,1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union, Dict, Any

import logging
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Tipos
# ──────────────────────────────────────────────────────────────

CombineMethod = Literal["linear", "geometric", "product", "harmonic"]
ThresholdMethod = Literal["gmm", "histogram_valley", "percentile"]

# ──────────────────────────────────────────────────────────────
# Dataclasses (Resultados + Opciones)
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CombinedScoreOptions:
    method: CombineMethod = "geometric"
    alpha: float = 0.5  # solo aplica a "linear"


@dataclass(frozen=True)
class ValleyThreshold:
    threshold: float
    peak_left: float
    peak_right: float
    # métricas auxiliares (útiles para validar “calidad” del valle)
    separation: Optional[float] = None      # diferencia entre picos (peak_right - peak_left)
    valley_depth: Optional[float] = None    # profundidad relativa (heurística) para histograma


@dataclass(frozen=True)
class GMMValidation:
    accepted: bool
    reason: str

    # Diagnóstico estadístico
    bic_1: float
    bic_2: float
    bic_improvement: float

    means: Tuple[float, float]
    vars: Tuple[float, float]
    weights: Tuple[float, float]

    separation_index: float  # |m1-m2| / sqrt(v1+v2)
    min_weight: float
    min_var: float


@dataclass(frozen=True)
class GMMThresholdResult:
    threshold: float
    validation: GMMValidation


@dataclass(frozen=True)
class GMMThresholdOptions:
    random_state: int = 0
    min_samples: int = 50

    # Validación estadística
    require_bic_improvement: bool = True
    min_bic_improvement: float = 10.0       # >0: 2-comp mejor que 1-comp. 10 suele ser evidencia fuerte
    min_separation_index: float = 1.0       # <1 => componentes muy solapados
    min_component_weight: float = 0.05      # evita “componentes fantasma”
    min_variance: float = 1e-6              # evita degeneración numérica

    # Rango esperado
    clip_to_unit_interval: bool = True


@dataclass(frozen=True)
class HistogramValleyOptions:
    bins: int = 80
    min_samples: int = 50
    # kernel simple para suavizado; se puede ajustar si tu distribución es ruidosa
    kernel: Tuple[float, ...] = (1, 2, 3, 2, 1)
    value_range: Tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class PercentileOptions:
    percentile: float = 10.0
    min_samples: int = 1


# ──────────────────────────────────────────────────────────────
# Helpers de validación
# ──────────────────────────────────────────────────────────────

def _as_float_array(x: Union[pd.Series, np.ndarray, list, tuple]) -> np.ndarray:
    if isinstance(x, pd.Series):
        arr = x.dropna().astype(float).to_numpy()
    else:
        arr = np.asarray(x, dtype=float)
        arr = arr[np.isfinite(arr)]
    return arr


def _validate_scores_unit_interval(arr: np.ndarray, name: str, *, allow_outside: bool = False) -> None:
    if arr.size == 0:
        raise ValueError(f"{name} is empty after removing NaN/Inf.")
    if not allow_outside:
        if np.nanmin(arr) < 0.0 - 1e-12 or np.nanmax(arr) > 1.0 + 1e-12:
            raise ValueError(f"{name} must be within [0,1]. Got min={arr.min()}, max={arr.max()}.")


def _validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite.")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0,1].")


# ──────────────────────────────────────────────────────────────
# 1) Combinación de scores
# ──────────────────────────────────────────────────────────────

def combined_score(
    jaccard: Union[float, np.ndarray],
    wang: Union[float, np.ndarray],
    *,
    options: CombinedScoreOptions = CombinedScoreOptions(),
) -> Union[float, np.ndarray]:
    """
    Combina dos scores (idealmente en [0,1]) con distintos métodos.

    methods:
      - linear: alpha*J + (1-alpha)*W
      - geometric: sqrt(J*W)
      - product: J*W
      - harmonic: 2 / (1/J + 1/W)  (penaliza fuerte si uno es bajo)

    Nota:
    - Para harmonic, si J o W son 0, el resultado es 0 (evita div/0).
    """
    J = np.asarray(jaccard, dtype=float)
    W = np.asarray(wang, dtype=float)

    # Validación básica
    if np.any(~np.isfinite(J)) or np.any(~np.isfinite(W)):
        raise ValueError("jaccard/wang contain NaN or Inf.")
    if np.any(J < 0) or np.any(J > 1) or np.any(W < 0) or np.any(W > 1):
        raise ValueError("jaccard/wang must be within [0,1].")

    method = options.method

    if method == "linear":
        _validate_alpha(options.alpha)
        return options.alpha * J + (1.0 - options.alpha) * W

    if method == "geometric":
        return np.sqrt(J * W)

    if method == "product":
        return J * W

    if method == "harmonic":
        # 2 / (1/J + 1/W) = 2JW / (J+W)
        denom = (J + W)
        out = np.zeros_like(denom, dtype=float)
        mask = denom > 0
        out[mask] = (2.0 * J[mask] * W[mask]) / denom[mask]
        return out

    raise ValueError(f"Unsupported combine method: {method}")


# ──────────────────────────────────────────────────────────────
# 2) Umbral por valle (histograma suavizado)
# ──────────────────────────────────────────────────────────────

def valley_threshold_from_series(
    series: Union[pd.Series, np.ndarray, list, tuple],
    *,
    options: HistogramValleyOptions = HistogramValleyOptions(),
) -> Optional[ValleyThreshold]:
    """
    Detecta 2 picos principales sobre un histograma suavizado y toma el mínimo entre ellos.

    Retorna None si:
    - pocos datos
    - no hay al menos 2 máximos locales
    """
    values = _as_float_array(series)
    if values.size < options.min_samples:
        return None

    # Esperamos [0,1] típicamente
    lo, hi = options.value_range
    hist, edges = np.histogram(values, bins=int(options.bins), range=(float(lo), float(hi)))

    kernel = np.array(options.kernel, dtype=float)
    if kernel.size < 3:
        raise ValueError("kernel must have length >= 3.")
    kernel /= kernel.sum()

    smooth = np.convolve(hist.astype(float), kernel, mode="same")

    # máximos locales
    locmax = np.where((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] > smooth[2:]))[0] + 1
    if locmax.size < 2:
        return None

    # tomar 2 picos más altos
    peaks = locmax[np.argsort(smooth[locmax])[-2:]]
    peaks.sort()

    pL, pR = int(peaks[0]), int(peaks[1])
    if pR <= pL:
        return None

    valley_idx_rel = int(np.argmin(smooth[pL:pR + 1]))
    valley_idx = pL + valley_idx_rel

    thr = 0.5 * (edges[valley_idx] + edges[valley_idx + 1])
    peak_left = 0.5 * (edges[pL] + edges[pL + 1])
    peak_right = 0.5 * (edges[pR] + edges[pR + 1])

    # heurística de “profundidad del valle”
    # (peak heights - valley height) / peak height promedio
    peak_h = 0.5 * (smooth[pL] + smooth[pR])
    valley_h = smooth[valley_idx]
    valley_depth = None
    if peak_h > 0:
        valley_depth = float(max(0.0, (peak_h - valley_h) / peak_h))

    return ValleyThreshold(
        threshold=float(thr),
        peak_left=float(peak_left),
        peak_right=float(peak_right),
        separation=float(peak_right - peak_left),
        valley_depth=valley_depth,
    )


# ──────────────────────────────────────────────────────────────
# 3) Umbral por percentil
# ──────────────────────────────────────────────────────────────

def percentile_threshold(
    series: Union[pd.Series, np.ndarray, list, tuple],
    *,
    options: PercentileOptions = PercentileOptions(),
) -> float:
    values = _as_float_array(series)
    if values.size < options.min_samples:
        raise ValueError(f"Not enough samples for percentile threshold: {values.size} < {options.min_samples}")

    p = float(options.percentile)
    if not (0.0 <= p <= 100.0):
        raise ValueError("percentile must be in [0,100].")

    return float(np.percentile(values, p))


# ──────────────────────────────────────────────────────────────
# 4) Umbral GMM con validación estadística
# ──────────────────────────────────────────────────────────────

def _gmm_intersection_threshold(
    m1: float, v1: float, w1: float,
    m2: float, v2: float, w2: float,
) -> Optional[float]:
    """
    Intersección de dos gaussianas ponderadas:
      w1*N(x|m1,v1) = w2*N(x|m2,v2)

    Retorna raíz dentro [min(m1,m2), max(m1,m2)] si existe; si no, retorna raíz más cercana al centro.
    """
    # ecuación cuadrática en x:
    # a x^2 + b x + c = 0
    a = (1.0 / (2.0 * v2)) - (1.0 / (2.0 * v1))
    b = (m1 / v1) - (m2 / v2)
    c = (m2**2) / (2.0 * v2) - (m1**2) / (2.0 * v1) + np.log((w2 * np.sqrt(v1)) / (w1 * np.sqrt(v2)))

    # caso lineal
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return None
        return float(-c / b)

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None

    sqrt_disc = float(np.sqrt(disc))
    r1 = float((-b + sqrt_disc) / (2.0 * a))
    r2 = float((-b - sqrt_disc) / (2.0 * a))

    lo, hi = (min(m1, m2), max(m1, m2))
    candidates = [r for r in (r1, r2) if lo <= r <= hi]
    if candidates:
        # el “valle” razonable
        return float(sorted(candidates, key=lambda r: abs(r - (m1 + m2) / 2.0))[0])

    # fallback: raíz más cercana al centro entre medias
    center = 0.5 * (m1 + m2)
    return float(sorted([r1, r2], key=lambda r: abs(r - center))[0])


def gmm_valley_threshold(
    series: Union[pd.Series, np.ndarray, list, tuple],
    *,
    options: GMMThresholdOptions = GMMThresholdOptions(),
) -> Optional[GMMThresholdResult]:
    """
    Ajusta GMM con 1 y 2 componentes y decide si 2 componentes es estadísticamente defendible.

    Retorna None si:
    - no hay suficientes datos
    - falla la validación estadística
    """
    x = _as_float_array(series)
    if x.size < options.min_samples:
        return None

    X = x.reshape(-1, 1)

    # Modelos
    gmm1 = GaussianMixture(n_components=1, covariance_type="full", random_state=int(options.random_state))
    gmm2 = GaussianMixture(n_components=2, covariance_type="full", random_state=int(options.random_state))

    gmm1.fit(X)
    gmm2.fit(X)

    bic_1 = float(gmm1.bic(X))
    bic_2 = float(gmm2.bic(X))
    bic_improvement = float(bic_1 - bic_2)  # positivo => 2 componentes mejora

    means = gmm2.means_.ravel().astype(float)
    vars_ = gmm2.covariances_.ravel().astype(float)   # varianzas (1D)
    weights = gmm2.weights_.ravel().astype(float)

    # ordenar por media
    idx = np.argsort(means)
    m1, m2 = float(means[idx[0]]), float(means[idx[1]])
    v1, v2 = float(vars_[idx[0]]), float(vars_[idx[1]])
    w1, w2 = float(weights[idx[0]]), float(weights[idx[1]])

    # chequeos de degeneración
    min_var = float(min(v1, v2))
    min_weight = float(min(w1, w2))

    sep_index = float(abs(m2 - m1) / np.sqrt(max(v1 + v2, 1e-300)))

    # Validación estadística
    accepted = True
    reasons = []

    if options.require_bic_improvement and bic_improvement < options.min_bic_improvement:
        accepted = False
        reasons.append(f"BIC improvement too small (bic1-bic2={bic_improvement:.3g} < {options.min_bic_improvement}).")

    if min_weight < options.min_component_weight:
        accepted = False
        reasons.append(f"Component weight too small (min_weight={min_weight:.3g} < {options.min_component_weight}).")

    if min_var < options.min_variance:
        accepted = False
        reasons.append(f"Variance too small/degenerate (min_var={min_var:.3g} < {options.min_variance}).")

    if sep_index < options.min_separation_index:
        accepted = False
        reasons.append(f"Components overlap too much (separation_index={sep_index:.3g} < {options.min_separation_index}).")

    reason = "accepted" if accepted else " | ".join(reasons)

    validation = GMMValidation(
        accepted=accepted,
        reason=reason,
        bic_1=bic_1,
        bic_2=bic_2,
        bic_improvement=bic_improvement,
        means=(m1, m2),
        vars=(v1, v2),
        weights=(w1, w2),
        separation_index=sep_index,
        min_weight=min_weight,
        min_var=min_var,
    )

    if not accepted:
        logger.info("[gmm_threshold] Rejected: %s", reason)
        return None

    thr = _gmm_intersection_threshold(m1, v1, w1, m2, v2, w2)
    if thr is None:
        logger.info("[gmm_threshold] Rejected: could not compute intersection threshold.")
        return None

    if options.clip_to_unit_interval:
        thr = float(max(0.0, min(1.0, thr)))

    return GMMThresholdResult(threshold=float(thr), validation=validation)


# ──────────────────────────────────────────────────────────────
# 5) API unificada (recomendado para tu pipeline)
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThresholdResult:
    method: ThresholdMethod
    threshold: float
    details: Dict[str, Any]


def compute_threshold(
    series: Union[pd.Series, np.ndarray, list, tuple],
    *,
    method: ThresholdMethod = "gmm",
    gmm: GMMThresholdOptions = GMMThresholdOptions(),
    hist: HistogramValleyOptions = HistogramValleyOptions(),
    perc: PercentileOptions = PercentileOptions(),
) -> Optional[ThresholdResult]:
    """
    Compute a threshold with a unified interface.

    Returns None if method can't find a valid threshold (e.g. gmm validation fails, no bimodality in histogram).
    """
    if method == "gmm":
        res = gmm_valley_threshold(series, options=gmm)
        if res is None:
            return None
        return ThresholdResult(
            method="gmm",
            threshold=float(res.threshold),
            details={
                "validation": res.validation,
            },
        )

    if method == "histogram_valley":
        vt = valley_threshold_from_series(series, options=hist)
        if vt is None:
            return None
        return ThresholdResult(
            method="histogram_valley",
            threshold=float(vt.threshold),
            details={
                "peak_left": vt.peak_left,
                "peak_right": vt.peak_right,
                "separation": vt.separation,
                "valley_depth": vt.valley_depth,
                "bins": int(hist.bins),
            },
        )

    if method == "percentile":
        thr = percentile_threshold(series, options=perc)
        return ThresholdResult(
            method="percentile",
            threshold=float(thr),
            details={
                "percentile": float(perc.percentile),
            },
        )

    raise ValueError(f"Unsupported threshold method: {method}")