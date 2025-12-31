"""
Day 0 vs Day 1 comparison metrics.

This module provides functionality for:
- Comparing parser outputs between different days
- Calculating drift metrics
- Detecting significant changes in output patterns

DRIFT DETECTION PHILOSOPHY:
Parser behavior should be relatively stable between deployments. Significant
changes in extraction patterns can indicate:
1. Model degradation or overfitting
2. Prompt drift or configuration changes
3. Data distribution shifts
4. Bugs introduced in new versions

Each threshold is calibrated based on acceptable variance in health data
processing, where consistency is crucial for user trust and safety.

DESIGN PRINCIPLE: NO GROUND TRUTH REQUIRED
Drift detection compares Day 0 (baseline) to Day 1 (new version) outputs.
We don't need labeled data - we only need consistent behavior between versions.
This is a relative comparison, not an absolute accuracy measurement.

NOTE: Uses only Python standard library (statistics module).
No external ML libraries or model training required.
"""

import statistics
from datetime import datetime
from typing import List, Dict, Any

from .utils import calculate_percentage


def calculate_extraction_volume(outputs: List[Dict]) -> Dict[str, Any]:
    """
    Calculate statistics on number of items extracted per journal.

    WHY THIS MATTERS:
    Extraction volume reflects parser sensitivity. A sudden increase may
    indicate over-parsing (finding things that aren't there), while a
    decrease may indicate under-parsing (missing important health signals).
    For health journals, consistent extraction behavior is critical.

    Args:
        outputs: List of parsed output records, each containing 'items'.

    Returns:
        Dictionary with extraction volume statistics.
    """
    item_counts = []
    for output in outputs:
        items = output.get("items", [])
        item_counts.append(len(items))

    if not item_counts:
        return {
            "average": 0.0,
            "min": 0,
            "max": 0,
            "median": 0.0,
            "standard_deviation": 0.0,
            "total_journals": 0,
            "total_items": 0
        }

    avg = statistics.mean(item_counts)
    med = statistics.median(item_counts)
    std = statistics.stdev(item_counts) if len(item_counts) > 1 else 0.0

    return {
        "average": round(avg, 2),
        "min": min(item_counts),
        "max": max(item_counts),
        "median": round(med, 2),
        "standard_deviation": round(std, 2),
        "total_journals": len(item_counts),
        "total_items": sum(item_counts)
    }


def calculate_uncertainty_rate(outputs: List[Dict]) -> float:
    """
    Calculate percentage of items with uncertain polarity.

    WHY THIS MATTERS:
    Uncertainty indicates the parser's confidence. A high uncertainty rate
    suggests the parser is struggling with ambiguous text. For health data,
    uncertain extractions should be minimized - users expect definitive
    parsing of what they wrote. Rising uncertainty may indicate prompt
    degradation or model confusion.

    Args:
        outputs: List of parsed output records, each containing 'items'.

    Returns:
        Percentage of items with unknown/uncertain polarity.
    """
    total_items = 0
    uncertain_count = 0

    for output in outputs:
        items = output.get("items", [])
        for item in items:
            total_items += 1
            polarity = item.get("polarity", "").lower()
            if polarity in ("unknown", "uncertain"):
                uncertain_count += 1

    return calculate_percentage(uncertain_count, total_items)


def calculate_intensity_distribution(outputs: List[Dict]) -> Dict[str, float]:
    """
    Calculate distribution of items by intensity level.

    WHY THIS MATTERS:
    Intensity distribution reflects how the parser interprets severity.
    A shift toward "high" intensity could cause unnecessary user alarm,
    while a shift toward "low" could cause users to underestimate symptoms.
    Consistent intensity calibration is essential for appropriate health
    guidance.

    Args:
        outputs: List of parsed output records, each containing 'items'.

    Returns:
        Dictionary with percentages for each intensity level.
    """
    intensity_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    total_items = 0

    for output in outputs:
        items = output.get("items", [])
        for item in items:
            total_items += 1
            intensity = item.get("intensity", "unknown")
            if intensity is None:
                intensity = "unknown"
            intensity = intensity.lower()
            if intensity in intensity_counts:
                intensity_counts[intensity] += 1
            else:
                intensity_counts["unknown"] += 1

    return {
        level: calculate_percentage(count, total_items)
        for level, count in intensity_counts.items()
    }


def calculate_domain_mix(outputs: List[Dict]) -> Dict[str, float]:
    """
    Calculate distribution of items by domain category.

    WHY THIS MATTERS:
    Domain mix reflects what the parser focuses on. A shift toward more
    "emotion" extraction at the expense of "symptom" could mean missing
    physical health signals. For balanced health monitoring, domain
    distribution should remain stable unless journal content changes.

    Args:
        outputs: List of parsed output records, each containing 'items'.

    Returns:
        Dictionary with percentages for each domain.
    """
    domain_counts = {"symptom": 0, "food": 0, "emotion": 0, "mind": 0, "other": 0}
    total_items = 0

    for output in outputs:
        items = output.get("items", [])
        for item in items:
            total_items += 1
            domain = item.get("domain", "other")
            if domain is None:
                domain = "other"
            domain = domain.lower()
            if domain in domain_counts:
                domain_counts[domain] += 1
            else:
                domain_counts["other"] += 1

    return {
        domain: calculate_percentage(count, total_items)
        for domain, count in domain_counts.items()
    }


def _calculate_percent_change(baseline: float, current: float) -> float:
    """Calculate absolute percentage point change."""
    return round(current - baseline, 2)


def _calculate_relative_change(baseline: float, current: float) -> float:
    """Calculate relative percentage change from baseline."""
    if baseline == 0:
        return 0.0 if current == 0 else 100.0
    return round(((current - baseline) / baseline) * 100, 2)


def compare_drift(day0_outputs: List[Dict], day1_outputs: List[Dict]) -> Dict[str, Any]:
    """
    Compare parser outputs between Day 0 (baseline) and Day 1 (new version).

    DRIFT THRESHOLD JUSTIFICATION:
    
    - Extraction Volume (>20% change): Significant change in how many items
      are extracted suggests parser sensitivity shift. For health data,
      missing items or over-extraction both have safety implications.
    
    - Uncertainty Rate (>15% change): Rising uncertainty indicates parser
      confidence degradation. Users expect consistent, confident parsing.
    
    - High Intensity (>25% change): Intensity affects user perception of
      severity. Over-reporting high intensity causes anxiety; under-reporting
      may cause users to dismiss important symptoms.
    
    - Domain Mix (>30% change): Large domain shifts indicate the parser is
      categorizing differently. This affects downstream analytics and
      recommendations specific to each health domain.

    Args:
        day0_outputs: Baseline parser outputs (Day 0).
        day1_outputs: New version parser outputs (Day 1).

    Returns:
        Comprehensive drift report with flags, explanations, and recommendations.
    """
    # Calculate all metrics for both days
    day0_volume = calculate_extraction_volume(day0_outputs)
    day1_volume = calculate_extraction_volume(day1_outputs)

    day0_uncertainty = calculate_uncertainty_rate(day0_outputs)
    day1_uncertainty = calculate_uncertainty_rate(day1_outputs)

    day0_intensity = calculate_intensity_distribution(day0_outputs)
    day1_intensity = calculate_intensity_distribution(day1_outputs)

    day0_domain = calculate_domain_mix(day0_outputs)
    day1_domain = calculate_domain_mix(day1_outputs)

    # Track drift flags
    drift_flags = []
    severity_scores = []

    # Check extraction volume drift (>20% relative change)
    volume_change = _calculate_relative_change(day0_volume["average"], day1_volume["average"])
    if abs(volume_change) > 20:
        severity = "severe" if abs(volume_change) > 50 else ("moderate" if abs(volume_change) > 35 else "minor")
        severity_scores.append(severity)
        drift_flags.append({
            "metric": "extraction_volume",
            "baseline_value": day0_volume["average"],
            "baseline_range": f"Expected {day0_volume['average'] * 0.8:.1f}-{day0_volume['average'] * 1.2:.1f} items/journal based on Day 0",
            "observed_value": day1_volume["average"],
            "change_percent": volume_change,
            "threshold": "20%",
            "reason": "Extraction volume shift indicates parser sensitivity change - over-parsing creates noise, under-parsing misses health signals",
            "drift_severity": severity
        })

    # Check uncertainty rate drift (>15 percentage points change)
    uncertainty_change = _calculate_percent_change(day0_uncertainty, day1_uncertainty)
    if abs(uncertainty_change) > 15:
        severity = "severe" if abs(uncertainty_change) > 30 else ("moderate" if abs(uncertainty_change) > 20 else "minor")
        severity_scores.append(severity)
        drift_flags.append({
            "metric": "uncertainty_rate",
            "baseline_value": day0_uncertainty,
            "baseline_range": f"Expected {max(0, day0_uncertainty - 15):.1f}-{day0_uncertainty + 15:.1f}% based on Day 0",
            "observed_value": day1_uncertainty,
            "change_percent": uncertainty_change,
            "threshold": "15 percentage points",
            "reason": "Rising uncertainty suggests parser confidence degradation - users expect definitive health data parsing",
            "drift_severity": severity
        })

    # Check high intensity drift (>25 percentage points change)
    high_intensity_change = _calculate_percent_change(
        day0_intensity.get("high", 0),
        day1_intensity.get("high", 0)
    )
    if abs(high_intensity_change) > 25:
        severity = "severe" if abs(high_intensity_change) > 40 else ("moderate" if abs(high_intensity_change) > 30 else "minor")
        severity_scores.append(severity)
        drift_flags.append({
            "metric": "high_intensity_rate",
            "baseline_value": day0_intensity.get("high", 0),
            "baseline_range": f"Expected {max(0, day0_intensity.get('high', 0) - 25):.1f}-{day0_intensity.get('high', 0) + 25:.1f}% based on Day 0",
            "observed_value": day1_intensity.get("high", 0),
            "change_percent": high_intensity_change,
            "threshold": "25 percentage points",
            "reason": "High intensity over-reporting causes user anxiety; under-reporting may cause dismissal of serious symptoms",
            "drift_severity": severity
        })

    # Check domain mix drift (>30 percentage points change for any domain)
    for domain in ["symptom", "food", "emotion", "mind"]:
        domain_change = _calculate_percent_change(
            day0_domain.get(domain, 0),
            day1_domain.get(domain, 0)
        )
        if abs(domain_change) > 30:
            severity = "severe" if abs(domain_change) > 50 else ("moderate" if abs(domain_change) > 40 else "minor")
            severity_scores.append(severity)
            
            reason_map = {
                "symptom": "Symptom under-detection risks missing physical health signals critical for user wellbeing",
                "food": "Food tracking drift affects nutritional insights and dietary pattern analysis",
                "emotion": "Emotional over-parsing increases risk of inappropriate mental health nudges",
                "mind": "Mind/cognitive drift affects stress and mental state monitoring accuracy"
            }
            
            drift_flags.append({
                "metric": f"domain_mix_{domain}",
                "baseline_value": day0_domain.get(domain, 0),
                "baseline_range": f"Expected {max(0, day0_domain.get(domain, 0) - 30):.1f}-{day0_domain.get(domain, 0) + 30:.1f}% based on Day 0",
                "observed_value": day1_domain.get(domain, 0),
                "change_percent": domain_change,
                "threshold": "30 percentage points",
                "reason": reason_map.get(domain, "Domain distribution shift affects downstream analytics"),
                "drift_severity": severity
            })

    # Determine overall drift status
    if not severity_scores:
        overall_status = "none"
        recommended_action = "deploy"
    elif "severe" in severity_scores:
        overall_status = "severe"
        recommended_action = "rollback"
    elif "moderate" in severity_scores:
        overall_status = "moderate"
        recommended_action = "human_review"
    else:
        overall_status = "minor"
        recommended_action = "human_review"

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "comparison": {
            "day0_journals": day0_volume["total_journals"],
            "day1_journals": day1_volume["total_journals"],
            "day0_items": day0_volume["total_items"],
            "day1_items": day1_volume["total_items"]
        },
        "metrics": {
            "extraction_volume": {
                "day0": day0_volume,
                "day1": day1_volume,
                "change_percent": volume_change
            },
            "uncertainty_rate": {
                "day0": day0_uncertainty,
                "day1": day1_uncertainty,
                "change": uncertainty_change
            },
            "intensity_distribution": {
                "day0": day0_intensity,
                "day1": day1_intensity
            },
            "domain_mix": {
                "day0": day0_domain,
                "day1": day1_domain
            }
        },
        "drift_flags": drift_flags,
        "drift_count": len(drift_flags),
        "overall_drift_status": overall_status,
        "recommended_action": recommended_action,
        "thresholds_used": {
            "extraction_volume": "20% relative change",
            "uncertainty_rate": "15 percentage points",
            "high_intensity": "25 percentage points",
            "domain_mix": "30 percentage points per domain"
        }
    }
