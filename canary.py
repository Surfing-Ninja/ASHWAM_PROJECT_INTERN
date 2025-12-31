"""
Small labeled test set evaluation.

This module provides functionality for:
- Evaluating parser outputs against gold-labeled test data
- Computing accuracy and other evaluation metrics

CRITICAL DESIGN DECISION: NO PRECISION/RECALL/F1

Traditional IR metrics like precision, recall, and F1 are NOT appropriate here.
This is because:

1. NO CLOSED LABEL SET: We don't have a canonical list of all items that
   "should" be extracted from each journal. Health journals are open-ended
   - there's no ground truth for "all possible extractions."

2. NO RECALL PENALTY: We cannot penalize the parser for NOT extracting
   something, because we don't know everything that could be extracted.
   A journal about "headache and fatigue" might mention 10 different
   things - we can't enumerate them all.

3. WHAT WE CAN VERIFY:
   - Evidence Validity: Is every extracted quote actually in the journal?
   - Polarity Correctness: For items that match gold labels, is polarity right?
   - Contradictions: Does the parser contradict itself?

This approach ensures we only evaluate what we can definitively verify,
avoiding false negatives that could unfairly penalize valid extractions.

DESIGN PRINCIPLE: MINIMAL GROUND TRUTH
The canary set is intentionally small (typically 5 journals) because:
1. Complete labeling is expensive and subjective
2. We only need sanity checks, not comprehensive coverage
3. Other checks (invariants, drift) don't need any labels

NOTE: Uses only Python standard library.
No external ML libraries or model training required.
"""

from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

from .utils import find_text_in_journal, calculate_percentage


def calculate_evidence_validity_rate(predicted_items: List[Dict], journal_text: str) -> float:
    """
    Calculate percentage of predicted items with valid evidence spans.

    WHY 100% REQUIRED:
    Every piece of evidence must exist verbatim in the journal. There is
    no acceptable rate of hallucination for health data - if the parser
    claims a user wrote something they didn't, that's a critical failure.

    Args:
        predicted_items: List of items extracted by the parser.
        journal_text: The original journal text to verify against.

    Returns:
        Percentage of items with evidence found in journal (0-100).
    """
    if not predicted_items:
        return 100.0  # No items = no invalid evidence

    valid_count = 0
    for item in predicted_items:
        evidence_span = item.get("evidence_span", "")
        if find_text_in_journal(evidence_span, journal_text):
            valid_count += 1

    return calculate_percentage(valid_count, len(predicted_items))


def calculate_polarity_correctness(predicted_items: List[Dict], gold_items: List[Dict]) -> float:
    """
    Calculate polarity agreement for items that match between predicted and gold.

    WHY ONLY MATCHED ITEMS:
    We only evaluate polarity for items that exist in BOTH predicted and gold.
    This avoids penalizing the parser for:
    - Extracting valid items not in gold (not a mistake)
    - Missing items in gold (can't verify what's "required")

    Args:
        predicted_items: List of items extracted by the parser.
        gold_items: List of gold-labeled items for comparison.

    Returns:
        Percentage of matched items with correct polarity (0-100).
    """
    if not predicted_items or not gold_items:
        return 100.0  # No comparison possible = no errors detected

    # Build lookup of gold items by evidence_span (lowercase for matching)
    gold_by_evidence = {}
    for gold_item in gold_items:
        evidence = gold_item.get("evidence_span", "").strip().lower()
        if evidence:
            gold_by_evidence[evidence] = gold_item

    matched_count = 0
    correct_count = 0

    for pred_item in predicted_items:
        pred_evidence = pred_item.get("evidence_span", "").strip().lower()
        
        if pred_evidence in gold_by_evidence:
            matched_count += 1
            gold_item = gold_by_evidence[pred_evidence]
            
            pred_polarity = pred_item.get("polarity", "").lower()
            gold_polarity = gold_item.get("polarity", "").lower()
            
            if pred_polarity == gold_polarity:
                correct_count += 1

    if matched_count == 0:
        return 100.0  # No matched items = can't evaluate polarity

    return calculate_percentage(correct_count, matched_count)


def check_contradictions_in_output(predicted_items: List[Dict]) -> int:
    """
    Count contradictions where same evidence has conflicting polarities.

    WHY ZERO TOLERANCE:
    It is logically impossible for the same evidence to indicate both
    presence and absence. Any contradiction is a critical logic error
    that invalidates the parser's reliability.

    Args:
        predicted_items: List of items extracted by the parser.

    Returns:
        Number of contradictory evidence spans (should be 0).
    """
    # Group items by evidence_span
    evidence_groups = defaultdict(list)
    for item in predicted_items:
        evidence = item.get("evidence_span", "").strip().lower()
        if evidence:
            polarity = item.get("polarity", "").lower()
            evidence_groups[evidence].append(polarity)

    contradiction_count = 0
    for evidence, polarities in evidence_groups.items():
        polarity_set = set(polarities)
        
        # Check for contradictory polarities
        has_present = "present" in polarity_set or "positive" in polarity_set
        has_absent = "absent" in polarity_set or "negative" in polarity_set
        
        if has_present and has_absent:
            contradiction_count += 1

    return contradiction_count


def evaluate_canary(
    canary_outputs: List[Dict],
    gold_labels: List[Dict],
    canary_journals: List[Dict]
) -> Dict[str, Any]:
    """
    Evaluate parser outputs against canary test set.

    EVALUATION PHILOSOPHY:
    The canary test is a small (typically 5) set of pre-labeled journals
    used as a sanity check before deployment. We evaluate:

    1. Evidence Validity (100% required): Every quote must be real
    2. Polarity Correctness (90% required): Matched items must agree
    3. Contradictions (0 required): No self-contradictions allowed

    Decision Logic:
    - Evidence < 100% → ROLLBACK (hallucinations are unacceptable)
    - Contradictions > 0 → ROLLBACK (logic errors are critical)
    - Polarity < 90% → HUMAN_REVIEW (may be acceptable with explanation)
    - All pass → PASS (safe to deploy)

    Args:
        canary_outputs: Parser outputs for canary journals.
        gold_labels: Gold-labeled items for each canary journal.
        canary_journals: Original canary journal texts.

    Returns:
        Comprehensive evaluation report with decision and reasoning.
    """
    per_journal_results = []
    
    total_evidence_valid = 0
    total_evidence_count = 0
    total_polarity_correct = 0
    total_polarity_matched = 0
    total_contradictions = 0

    num_journals = min(len(canary_outputs), len(gold_labels), len(canary_journals))

    for idx in range(num_journals):
        # Get data for this journal
        output = canary_outputs[idx]
        gold = gold_labels[idx]
        journal = canary_journals[idx]

        predicted_items = output.get("items", [])
        gold_items = gold.get("items", [])
        journal_text = journal.get("text", "")

        # Calculate metrics for this journal
        evidence_validity = calculate_evidence_validity_rate(predicted_items, journal_text)
        polarity_correctness = calculate_polarity_correctness(predicted_items, gold_items)
        contradictions = check_contradictions_in_output(predicted_items)

        # Track for aggregation
        valid_evidence_count = sum(
            1 for item in predicted_items
            if find_text_in_journal(item.get("evidence_span", ""), journal_text)
        )
        total_evidence_valid += valid_evidence_count
        total_evidence_count += len(predicted_items)

        # Track polarity matches
        gold_by_evidence = {
            g.get("evidence_span", "").strip().lower(): g
            for g in gold_items if g.get("evidence_span")
        }
        for pred_item in predicted_items:
            pred_evidence = pred_item.get("evidence_span", "").strip().lower()
            if pred_evidence in gold_by_evidence:
                total_polarity_matched += 1
                if pred_item.get("polarity", "").lower() == gold_by_evidence[pred_evidence].get("polarity", "").lower():
                    total_polarity_correct += 1

        total_contradictions += contradictions

        per_journal_results.append({
            "journal_index": idx,
            "predicted_item_count": len(predicted_items),
            "gold_item_count": len(gold_items),
            "evidence_validity_rate": evidence_validity,
            "polarity_correctness": polarity_correctness,
            "contradiction_count": contradictions,
            "status": "PASS" if evidence_validity == 100.0 and contradictions == 0 and polarity_correctness >= 90.0 else "FAIL"
        })

    # Calculate aggregate metrics
    aggregate_evidence_validity = calculate_percentage(total_evidence_valid, total_evidence_count)
    aggregate_polarity_correctness = calculate_percentage(total_polarity_correct, total_polarity_matched)

    # Determine alert level and reasoning
    reasons = []
    alert_level = "PASS"

    if aggregate_evidence_validity < 100.0:
        alert_level = "ROLLBACK"
        invalid_count = total_evidence_count - total_evidence_valid
        reasons.append(
            f"Evidence validity is {aggregate_evidence_validity}% ({invalid_count} hallucinated spans). "
            "Hallucinated evidence is unacceptable for health data - cannot deploy."
        )

    if total_contradictions > 0:
        alert_level = "ROLLBACK"
        reasons.append(
            f"Found {total_contradictions} contradiction(s) where same evidence has conflicting polarity. "
            "This indicates a critical logic error in the parser."
        )

    if aggregate_polarity_correctness < 90.0 and alert_level != "ROLLBACK":
        alert_level = "HUMAN_REVIEW"
        reasons.append(
            f"Polarity correctness is {aggregate_polarity_correctness}% (below 90% threshold). "
            "Parser may be misinterpreting presence/absence of health signals."
        )

    if alert_level == "PASS":
        reasons.append(
            "All canary checks passed: 100% evidence validity, no contradictions, "
            f"and {aggregate_polarity_correctness}% polarity correctness (≥90%)."
        )

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "canary_journals_evaluated": num_journals,
        "metrics": {
            "evidence_validity": aggregate_evidence_validity,
            "polarity_correctness": aggregate_polarity_correctness,
            "contradiction_count": total_contradictions,
            "total_predicted_items": total_evidence_count,
            "total_matched_items": total_polarity_matched
        },
        "thresholds": {
            "evidence_validity": "100% required",
            "polarity_correctness": "90% required",
            "contradictions": "0 required"
        },
        "alert_level": alert_level,
        "reasoning": " ".join(reasons),
        "recommended_action": alert_level,
        "details": per_journal_results
    }


def run_canary_test(
    canary_outputs: List[Dict],
    gold_labels: List[Dict],
    canary_journals: List[Dict]
) -> Dict[str, Any]:
    """
    Run the complete canary test suite and return a comprehensive report.

    The canary test is the final safety check before deployment. It uses
    a small set of pre-labeled journals to verify the parser behaves
    correctly on known cases.

    Args:
        canary_outputs: Parser outputs for canary journals.
        gold_labels: Gold-labeled items for each canary journal.
        canary_journals: Original canary journal texts.

    Returns:
        Comprehensive canary test report with pass/fail decision.
    """
    result = evaluate_canary(canary_outputs, gold_labels, canary_journals)
    
    # Add test metadata
    result["test_name"] = "canary_evaluation"
    result["test_description"] = (
        "Evaluates parser outputs against a small labeled test set. "
        "Checks evidence validity (must be 100%), polarity correctness (must be ≥90%), "
        "and contradictions (must be 0). Does NOT use recall/F1 as there is no closed label set."
    )
    
    return result
