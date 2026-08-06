"""Threshold-dependent validation metrics for the landslide model."""

import numpy as np
import pandas as pd


def find_control_date(
    x, y, event_date, inventory_df, *, max_years=5, window_days=5, rain_start_year=1991
):
    """A same-location control date ~1 year from the event with no recorded
    landslide nearby in `inventory_df`, so the control day isn't secretly a
    real event (label leakage).
    """
    control_date = event_date - pd.DateOffset(years=1)
    step = -1
    if control_date.year < rain_start_year:
        control_date = event_date + pd.DateOffset(years=1)
        step = 1

    same_location = (inventory_df["x"] == x) & (inventory_df["y"] == y)
    for _ in range(max_years):
        nearby = inventory_df[
            same_location
            & (inventory_df["date"] >= control_date - pd.Timedelta(days=window_days))
            & (inventory_df["date"] <= control_date + pd.Timedelta(days=window_days))
        ]
        if nearby.empty:
            return control_date
        control_date += pd.DateOffset(years=step)
    return None


def confusion_counts(event_flags, control_flags):
    """Return (tp, fn, fp, tn) from boolean failure predictions.

    ``event_flags`` are predictions on days with a recorded landslide,
    ``control_flags`` on matched control days.
    """
    event_flags = np.asarray(event_flags, dtype=bool)
    control_flags = np.asarray(control_flags, dtype=bool)
    tp = int(np.sum(event_flags))
    fn = int(event_flags.size - tp)
    fp = int(np.sum(control_flags))
    tn = int(control_flags.size - fp)
    return tp, fn, fp, tn


def tpr_fpr(event_flags, control_flags):
    """Return (sensitivity, false-positive rate)."""
    tp, fn, fp, tn = confusion_counts(event_flags, control_flags)
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    return tpr, fpr


def youden_j(event_flags, control_flags):
    """Youden's J = TPR - FPR (also called the True Skill Statistic).

    Ranges from -1 to 1; 0 means no skill beyond chance. Unlike AUC this
    depends on where the decision threshold actually sits.
    """
    tpr, fpr = tpr_fpr(event_flags, control_flags)
    return tpr - fpr


def mcc(event_flags, control_flags):
    """Matthews correlation coefficient.

    More robust than Youden's J when events and controls are imbalanced,
    because it uses all four confusion-matrix cells rather than two rates.
    Returns 0.0 when a whole row or column of the matrix is empty.
    """
    tp, fn, fp, tn = confusion_counts(event_flags, control_flags)
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float(numerator / denominator) if denominator > 0 else 0.0


def fpr_at_sensitivity(event_scores, control_scores, target_sensitivity=0.80):
    """False-positive rate when the threshold is set to hit a target sensitivity.

    Lower scores are more landslide-like (pass FoS, or negated saturation).
    This is the operationally meaningful number for a warning system: "if we
    want to catch 80% of landslides, how often do we cry wolf?"

    Returns (fpr, threshold_used, achieved_sensitivity).
    """
    event_scores = np.asarray(event_scores, dtype=float)
    control_scores = np.asarray(control_scores, dtype=float)
    event_scores = event_scores[np.isfinite(event_scores)]
    control_scores = control_scores[np.isfinite(control_scores)]
    if event_scores.size == 0 or control_scores.size == 0:
        return np.nan, np.nan, np.nan

    # Threshold at the sensitivity-th percentile of event scores: predicting
    # "failure" below it captures the requested fraction of real events.
    threshold = float(np.quantile(event_scores, target_sensitivity))
    achieved = float(np.mean(event_scores <= threshold))
    fpr = float(np.mean(control_scores <= threshold))
    return fpr, threshold, achieved


def reliability_curve(scores, labels, n_bins=10):
    """Observed event frequency per predictor bin (calibration check).

    ``scores`` is the continuous predictor (e.g. simulated saturation) and
    ``labels`` is 1 for landslide days and 0 for control days. A well-behaved
    predictor produces a monotonically rising curve: wetter bins should
    contain a higher fraction of real landslide days.

    Returns (bin_centres, observed_fraction, bin_counts).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    if scores.size == 0:
        return np.array([]), np.array([]), np.array([])

    edges = np.linspace(scores.min(), scores.max(), n_bins + 1)
    centres, fractions, counts = [], [], []
    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        # Include the right edge in the final bin so no point is dropped.
        in_bin = (scores >= lower) & (
            (scores < upper) if i < n_bins - 1 else (scores <= upper)
        )
        counts.append(int(in_bin.sum()))
        centres.append(float((lower + upper) / 2.0))
        fractions.append(float(labels[in_bin].mean()) if in_bin.any() else np.nan)
    return np.array(centres), np.array(fractions), np.array(counts)


def pairwise_auc(event_scores, control_scores):
    """Probability a random event scores lower than a random control.

    Retained for backward comparison only. See the module docstring for why
    this must not be used to calibrate physical parameters.
    """
    event_scores = np.asarray(event_scores, dtype=float)
    control_scores = np.asarray(control_scores, dtype=float)
    event_scores = event_scores[np.isfinite(event_scores)]
    control_scores = control_scores[np.isfinite(control_scores)]
    if event_scores.size == 0 or control_scores.size == 0:
        return np.nan
    differences = event_scores[:, None] - control_scores[None, :]
    return float(np.mean(differences < 0.0) + 0.5 * np.mean(differences == 0.0))


def critical_pore_pressure(gamma, gamma_w, beta_rad, phi_rad, c=0.0, h_perp=None):
    """Pore-pressure ratio m_pp at which FoS = 1.

    From FoS = A - B * m_pp with
    A = c / (gamma * h_perp * sin(beta)) + tan(phi) / tan(beta)
    B = (gamma_w / gamma) * (tan(phi) / tan(beta))

    Returns a value clipped to [0, 1]. A result of 0 means the slope fails
    while dry; a result of 1 means it is stable even fully saturated.
    """
    ratio = np.tan(phi_rad) / np.tan(beta_rad)
    a = ratio
    if c > 0.0:
        if h_perp is None:
            raise ValueError("h_perp is required when cohesion is non-zero.")
        a = a + c / (gamma * h_perp * np.sin(beta_rad))
    b = (gamma_w / gamma) * ratio
    return float(np.clip((a - 1.0) / b, 0.0, 1.0))


def critical_saturation(onset, m_pp_critical):
    """Bucket saturation at failure: S_crit = onset + m_crit * (1 - onset).

    This is the single quantity that determines whether a day is classified
    as a failure. Because many (onset, beta) pairs map to the same S_crit,
    those parameters are confounded and only S_crit is identifiable from an
    event inventory.
    """
    return float(onset + m_pp_critical * (1.0 - onset))


def onset_for_critical_saturation(s_crit, m_pp_critical):
    """Invert ``critical_saturation``: the onset giving a target S_crit.

    Useful for reporting which (onset, beta) combinations are consistent with
    a calibrated S_crit. Returns NaN when m_pp_critical is 1 (no solution).
    """
    if np.isclose(m_pp_critical, 1.0):
        return np.nan
    return float((s_crit - m_pp_critical) / (1.0 - m_pp_critical))
