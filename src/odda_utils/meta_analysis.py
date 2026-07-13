# Cross-study meta-analysis of effect sizes (fixed-effect + DerSimonian-Laird
# random-effects). Given per-study effect sizes and their uncertainty (variances,
# standard errors, or two-sided p-values), returns pooled estimates, standard
# errors, 95% CIs, z/p, and heterogeneity statistics (Q, Q_p, df, I^2, tau^2).
# The core ``se_from_p``/``meta_analyze`` functions mirror the validated reference
# at $HOME/data/odda_supplemental/analysis_code/meta_analysis.py. On top of
# that core this module adds JSON-serializable dataclass results and a batch driver
# so many entities (e.g. proteins/genes) can be meta-analyzed in a single call.
# Depends only on numpy + scipy.stats. Exposed via the odda_utils `meta_analysis`
# MCP tool.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union

import numpy as np
from scipy.stats import norm, chi2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core statistics (mirrors the validated reference implementation)
# ---------------------------------------------------------------------------


def se_from_p(effect, p, eps=1e-300):
    """Back out the standard error of an effect estimate from a two-sided p-value:
    SE = |effect| / z, where z is the standard-normal deviate for p (two-sided)."""
    p = float(min(max(p, eps), 1 - 1e-12))
    z = norm.isf(p / 2.0)
    return abs(effect) / z if z > 0 else np.nan


def meta_analyze(yi, vi):
    """Fixed-effect and random-effects (DerSimonian-Laird) meta-analysis.

    Parameters
    ----------
    yi : array-like
        Per-study effect sizes (e.g. log2 fold changes).
    vi : array-like
        Per-study variances of the effect sizes (SE**2).

    Returns
    -------
    dict or None
        Pooled fixed/random estimates with SEs and 95% CIs, z, p, and
        heterogeneity (Q, its p-value, df, I2 [%], tau2). None if no valid studies.
    """
    yi = np.asarray(yi, float); vi = np.asarray(vi, float)
    m = np.isfinite(yi) & np.isfinite(vi) & (vi > 0)
    yi, vi = yi[m], vi[m]
    k = int(len(yi))
    if k == 0:
        return None
    wi = 1.0 / vi
    theta_f = float(np.sum(wi * yi) / np.sum(wi)); var_f = float(1.0 / np.sum(wi))
    df = k - 1
    Q = float(np.sum(wi * (yi - theta_f) ** 2))
    Qp = float(chi2.sf(Q, df)) if df > 0 else float("nan")
    C = float(np.sum(wi) - np.sum(wi ** 2) / np.sum(wi))
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0
    wr = 1.0 / (vi + tau2)
    theta_r = float(np.sum(wr * yi) / np.sum(wr)); var_r = float(1.0 / np.sum(wr))
    se_r = float(np.sqrt(var_r)); z = theta_r / se_r; p = float(2 * norm.sf(abs(z)))
    return {
        "k": k,
        "fixed": {"estimate": theta_f, "se": float(np.sqrt(var_f)),
                  "ci_low": theta_f - 1.96 * np.sqrt(var_f), "ci_high": theta_f + 1.96 * np.sqrt(var_f)},
        "random": {"estimate": theta_r, "se": se_r,
                   "ci_low": theta_r - 1.96 * se_r, "ci_high": theta_r + 1.96 * se_r, "z": z, "p": p},
        "heterogeneity": {"Q": Q, "Q_p": Qp, "df": df, "I2": I2, "tau2": tau2},
    }


# ---------------------------------------------------------------------------
# Result containers (JSON-serializable primitives only)
# ---------------------------------------------------------------------------


@dataclass
class PooledEstimate:
    """A pooled effect estimate with its standard error and 95% confidence interval.

    Parameters
    ----------
    estimate : float
        Pooled effect size.
    se : float
        Standard error of the pooled estimate.
    ci_low, ci_high : float
        Lower and upper bounds of the 95% confidence interval.
    z : float, optional
        Wald z-statistic (``estimate / se``). Populated for the random-effects
        estimate; ``None`` for the fixed-effect estimate (matching the reference).
    p : float, optional
        Two-sided p-value for the pooled estimate being non-zero. Populated for
        the random-effects estimate; ``None`` for the fixed-effect estimate.
    """

    estimate: float
    se: float
    ci_low: float
    ci_high: float
    z: Optional[float] = None
    p: Optional[float] = None


@dataclass
class Heterogeneity:
    """Between-study heterogeneity statistics.

    Parameters
    ----------
    Q : float
        Cochran's Q statistic.
    Q_p : float
        P-value of Q against a chi-squared distribution with ``df`` degrees of
        freedom (NaN when ``df == 0``).
    df : int
        Degrees of freedom (``k - 1``).
    I2 : float
        I-squared statistic as a percentage (0-100).
    tau2 : float
        DerSimonian-Laird estimate of the between-study variance.
    """

    Q: float
    Q_p: float
    df: int
    I2: float
    tau2: float


@dataclass
class MetaAnalysisResult:
    """Meta-analysis result for a single entity (e.g. one protein or gene).

    Parameters
    ----------
    name : str, optional
        Entity label (e.g. protein/gene identifier).
    k : int
        Number of valid studies actually pooled (after dropping studies with
        non-finite effects or non-positive variances).
    fixed : PooledEstimate, optional
        Fixed-effect (inverse-variance) pooled estimate. ``None`` when no valid
        studies were available.
    random : PooledEstimate, optional
        Random-effects (DerSimonian-Laird) pooled estimate. ``None`` when no
        valid studies were available.
    heterogeneity : Heterogeneity, optional
        Between-study heterogeneity statistics. ``None`` when no valid studies
        were available.
    error : str, optional
        Human-readable message when the entity could not be analyzed (e.g. no
        valid studies, or an exception during batch processing).
    """

    name: Optional[str] = None
    k: int = 0
    fixed: Optional[PooledEstimate] = None
    random: Optional[PooledEstimate] = None
    heterogeneity: Optional[Heterogeneity] = None
    error: Optional[str] = None


@dataclass
class MetaAnalysisBatchResult:
    """Meta-analysis results for one or more entities.

    Parameters
    ----------
    results : dict of str to MetaAnalysisResult
        Per-entity results keyed by entity name. Single-entity calls yield a
        one-entry mapping.
    n_entities : int
        Number of entities processed.
    n_succeeded : int
        Number of entities pooled without error (``error is None``).
    n_failed : int
        Number of entities that produced an ``error`` (e.g. no valid studies or
        an exception during processing).
    """

    results: dict[str, MetaAnalysisResult] = field(default_factory=dict)
    n_entities: int = 0
    n_succeeded: int = 0
    n_failed: int = 0


# ---------------------------------------------------------------------------
# Input resolution helpers
# ---------------------------------------------------------------------------

# Accepted per-study dictionary keys (first present key of each group wins).
_EFFECT_KEYS = ("yi", "effect", "effect_size", "es", "log2fc", "logfc")
_VARIANCE_KEYS = ("vi", "variance", "var")
_SE_KEYS = ("se", "standard_error", "std_error", "se_")
_P_KEYS = ("p", "pvalue", "p_value", "pval")


def _resolve_variance(
    yi: float,
    vi: Optional[float] = None,
    se: Optional[float] = None,
    p: Optional[float] = None,
) -> float:
    """Resolve a per-study variance from a variance, a standard error, or a p-value.

    Exactly one uncertainty source is expected; they are checked in the order
    variance, standard error, p-value and the first non-``None`` value is used.

    Parameters
    ----------
    yi : float
        The study's effect size (needed to back out the SE from a p-value).
    vi : float, optional
        Variance of the effect size (``SE ** 2``).
    se : float, optional
        Standard error of the effect size.
    p : float, optional
        Two-sided p-value for the effect size.

    Returns
    -------
    float
        The variance to use for pooling, or ``NaN`` when no uncertainty source is
        provided (such studies are dropped by :func:`meta_analyze`).
    """
    if vi is not None:
        return float(vi)
    if se is not None:
        return float(se) ** 2
    if p is not None:
        return float(se_from_p(yi, p)) ** 2
    return float("nan")


def _first_present(study: Mapping, keys: Sequence[str]) -> Optional[float]:
    """Return the first non-``None`` value among ``keys`` in ``study``, else ``None``."""
    for key in keys:
        if key in study and study[key] is not None:
            return study[key]
    return None


def _parse_study(study: Union[Mapping, Sequence]) -> tuple[float, float]:
    """Parse a single per-study record into an ``(effect, variance)`` pair.

    Parameters
    ----------
    study : mapping or sequence
        Either a mapping with an effect key (one of ``yi``/``effect``/
        ``effect_size``/``es``/``log2fc``/``logfc``) and an uncertainty key
        (a variance ``vi``/``variance``/``var``, a standard error
        ``se``/``standard_error``/``std_error``, or a p-value
        ``p``/``pvalue``/``p_value``/``pval``), or a two-element ``(effect,
        variance)`` sequence.

    Returns
    -------
    tuple of (float, float)
        The effect size and its resolved variance.

    Raises
    ------
    ValueError
        If the record is malformed or is missing an effect size.
    """
    if isinstance(study, Mapping):
        yi = _first_present(study, _EFFECT_KEYS)
        if yi is None:
            raise ValueError(
                "study dict is missing an effect size; expected one of %s"
                % (_EFFECT_KEYS,)
            )
        yi = float(yi)
        vi = _first_present(study, _VARIANCE_KEYS)
        se = _first_present(study, _SE_KEYS)
        p = _first_present(study, _P_KEYS)
        return yi, _resolve_variance(yi, vi=vi, se=se, p=p)
    if isinstance(study, Sequence) and not isinstance(study, (str, bytes)):
        if len(study) != 2:
            raise ValueError(
                "sequence study must be a 2-element (effect, variance) pair, "
                "got length %d" % len(study)
            )
        yi, vi = study
        return float(yi), float(vi)
    raise ValueError(
        "study must be a mapping or a 2-element (effect, variance) sequence, "
        "got %r" % type(study).__name__
    )


def _studies_to_arrays(
    studies: Sequence[Union[Mapping, Sequence]],
) -> tuple[list[float], list[float]]:
    """Convert a list of per-study records into parallel effect/variance lists."""
    yi_list: list[float] = []
    vi_list: list[float] = []
    for study in studies:
        yi, vi = _parse_study(study)
        yi_list.append(yi)
        vi_list.append(vi)
    return yi_list, vi_list


def _arrays_to_yi_vi(
    effects: Sequence[float],
    variances: Optional[Sequence[float]],
    standard_errors: Optional[Sequence[float]],
    pvalues: Optional[Sequence[float]],
) -> tuple[list[float], list[float]]:
    """Convert parallel effect + uncertainty arrays into effect/variance lists.

    Exactly one of ``variances``, ``standard_errors``, or ``pvalues`` must be
    supplied and must be the same length as ``effects``.
    """
    if effects is None or len(effects) == 0:
        raise ValueError("`effects` must be a non-empty list of per-study effect sizes")
    provided = [x for x in (variances, standard_errors, pvalues) if x is not None]
    if len(provided) == 0:
        raise ValueError(
            "provide the study uncertainties as one of `variances`, "
            "`standard_errors`, or `pvalues`"
        )
    if len(provided) > 1:
        raise ValueError(
            "provide exactly one of `variances`, `standard_errors`, or `pvalues`"
        )
    n = len(effects)
    uncertainty = provided[0]
    if len(uncertainty) != n:
        raise ValueError(
            "uncertainty list length (%d) must match effects length (%d)"
            % (len(uncertainty), n)
        )
    yi_list: list[float] = []
    vi_list: list[float] = []
    for i in range(n):
        yi = float(effects[i])
        if variances is not None:
            vi = _resolve_variance(yi, vi=variances[i])
        elif standard_errors is not None:
            vi = _resolve_variance(yi, se=standard_errors[i])
        else:
            vi = _resolve_variance(yi, p=pvalues[i])
        yi_list.append(yi)
        vi_list.append(vi)
    return yi_list, vi_list


def _result_from_meta_dict(
    meta: Optional[dict], name: Optional[str]
) -> MetaAnalysisResult:
    """Wrap the dict returned by :func:`meta_analyze` in a :class:`MetaAnalysisResult`."""
    if meta is None:
        return MetaAnalysisResult(
            name=name,
            k=0,
            error=(
                "no valid studies: need at least one study with a finite effect "
                "and a positive variance"
            ),
        )
    fixed = meta["fixed"]
    random = meta["random"]
    het = meta["heterogeneity"]
    return MetaAnalysisResult(
        name=name,
        k=int(meta["k"]),
        fixed=PooledEstimate(
            estimate=float(fixed["estimate"]),
            se=float(fixed["se"]),
            ci_low=float(fixed["ci_low"]),
            ci_high=float(fixed["ci_high"]),
        ),
        random=PooledEstimate(
            estimate=float(random["estimate"]),
            se=float(random["se"]),
            ci_low=float(random["ci_low"]),
            ci_high=float(random["ci_high"]),
            z=float(random["z"]),
            p=float(random["p"]),
        ),
        heterogeneity=Heterogeneity(
            Q=float(het["Q"]),
            Q_p=float(het["Q_p"]),
            df=int(het["df"]),
            I2=float(het["I2"]),
            tau2=float(het["tau2"]),
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_meta_analysis(
    effects: Sequence[float],
    variances: Optional[Sequence[float]] = None,
    standard_errors: Optional[Sequence[float]] = None,
    pvalues: Optional[Sequence[float]] = None,
    name: str = "effect",
) -> MetaAnalysisResult:
    """Meta-analyze a single entity from parallel effect + uncertainty arrays.

    Parameters
    ----------
    effects : sequence of float
        Per-study effect sizes (e.g. log2 fold changes).
    variances : sequence of float, optional
        Per-study variances (``SE ** 2``). Mutually exclusive with
        ``standard_errors`` and ``pvalues``.
    standard_errors : sequence of float, optional
        Per-study standard errors. Squared internally to obtain variances.
    pvalues : sequence of float, optional
        Per-study two-sided p-values. Standard errors are backed out via
        :func:`se_from_p` and then squared to obtain variances.
    name : str, optional
        Label for the entity, stored on the result. Defaults to ``"effect"``.

    Returns
    -------
    MetaAnalysisResult
        Pooled fixed- and random-effects estimates plus heterogeneity. When no
        study has a finite effect and a positive variance, ``k`` is 0 and
        ``error`` explains why.

    Raises
    ------
    ValueError
        If ``effects`` is empty, or the uncertainty arguments are missing,
        ambiguous, or mismatched in length.

    Examples
    --------
    >>> res = run_meta_analysis([1.0, 1.2, 0.8, 1.1], variances=[0.05, 0.06, 0.07, 0.05])
    >>> res.k
    4
    >>> round(res.random.estimate, 3)
    1.035
    >>> res.random.p < 0.001
    True

    Standard errors or p-values can be supplied instead of variances:

    >>> res = run_meta_analysis([0.5, 0.7], standard_errors=[0.2, 0.25])
    >>> res.k
    2
    >>> res = run_meta_analysis([0.5, 0.7], pvalues=[0.01, 0.02])
    >>> res.k
    2
    """
    yi_list, vi_list = _arrays_to_yi_vi(effects, variances, standard_errors, pvalues)
    meta = meta_analyze(yi_list, vi_list)
    return _result_from_meta_dict(meta, name)


def run_meta_analysis_batch(
    entities: Mapping[str, Sequence[Union[Mapping, Sequence]]],
) -> MetaAnalysisBatchResult:
    """Meta-analyze many entities at once (e.g. one entry per protein/gene).

    Errors on individual entities are caught, logged, and recorded on that
    entity's result so that the remaining entities are still processed.

    Parameters
    ----------
    entities : mapping of str to sequence of per-study records
        Maps an entity name to its list of per-study records. Each record is
        either a mapping with an effect key and an uncertainty key (variance,
        standard error, or p-value; see :func:`_parse_study`) or a two-element
        ``(effect, variance)`` sequence.

    Returns
    -------
    MetaAnalysisBatchResult
        Per-entity :class:`MetaAnalysisResult` objects keyed by name, plus
        success/failure counts.

    Examples
    --------
    >>> batch = run_meta_analysis_batch({
    ...     "P12345": [{"yi": 1.0, "vi": 0.05}, {"yi": 1.2, "se": 0.24}],
    ...     "Q9Y6K9": [{"effect": -0.4, "p": 0.03}, {"effect": -0.6, "p": 0.01}],
    ... })
    >>> batch.n_entities
    2
    >>> batch.results["P12345"].k
    2
    """
    results: dict[str, MetaAnalysisResult] = {}
    n_succeeded = 0
    n_failed = 0
    for name, studies in entities.items():
        try:
            yi_list, vi_list = _studies_to_arrays(studies)
            meta = meta_analyze(yi_list, vi_list)
            result = _result_from_meta_dict(meta, name)
        except Exception as exc:  # noqa: BLE001 - one bad entity must not abort the batch
            logger.warning("Meta-analysis failed for entity %r: %s", name, exc)
            result = MetaAnalysisResult(name=name, k=0, error=str(exc))
        results[name] = result
        if result.error is None:
            n_succeeded += 1
        else:
            n_failed += 1
    return MetaAnalysisBatchResult(
        results=results,
        n_entities=len(results),
        n_succeeded=n_succeeded,
        n_failed=n_failed,
    )


if __name__ == "__main__":  # tiny self-test
    r = run_meta_analysis([1.0, 1.2, 0.8, 1.1], variances=[0.05, 0.06, 0.07, 0.05])
    print(
        "single random estimate=%.3f CI[%.3f,%.3f] I2=%.1f%% p=%.3g k=%d"
        % (
            r.random.estimate,
            r.random.ci_low,
            r.random.ci_high,
            r.heterogeneity.I2,
            r.random.p,
            r.k,
        )
    )

    # Compare against the reference core function directly (should be identical).
    ref = meta_analyze([1.0, 1.2, 0.8, 1.1], [0.05, 0.06, 0.07, 0.05])
    assert abs(ref["random"]["estimate"] - r.random.estimate) < 1e-12
    assert abs(ref["heterogeneity"]["I2"] - r.heterogeneity.I2) < 1e-12

    # Standard-error and p-value inputs.
    r_se = run_meta_analysis([0.5, 0.7], standard_errors=[0.2, 0.25])
    r_p = run_meta_analysis([0.5, 0.7], pvalues=[0.01, 0.02])
    print("se-input k=%d, p-input k=%d" % (r_se.k, r_p.k))

    # Batch over several entities, including one deliberately empty entity.
    batch = run_meta_analysis_batch(
        {
            "P12345": [{"yi": 1.0, "vi": 0.05}, {"yi": 1.2, "se": 0.24}],
            "Q9Y6K9": [{"effect": -0.4, "p": 0.03}, {"effect": -0.6, "p": 0.01}],
            "EMPTY": [],
        }
    )
    print(
        "batch entities=%d succeeded=%d failed=%d"
        % (batch.n_entities, batch.n_succeeded, batch.n_failed)
    )
    for entity_name, entity_result in batch.results.items():
        summary = (
            "k=%d random=%.3f p=%.3g"
            % (
                entity_result.k,
                entity_result.random.estimate,
                entity_result.random.p,
            )
            if entity_result.random is not None
            else "error=%s" % entity_result.error
        )
        print("  %-8s %s" % (entity_name, summary))
