"""
Hard rule checks (schema, evidence validity, contradictions).

This module implements invariant checks including:
- Schema validation
- Evidence validity verification
- Contradiction detection

These checks are CRITICAL for production safety. Health data requires
absolute correctness - invalid schema or hallucinated evidence can lead
to incorrect health recommendations and potential harm to users.

DESIGN PRINCIPLE: NO GROUND TRUTH REQUIRED
These checks work WITHOUT complete ground truth labels. We verify:
1. Structural correctness (schema)
2. Evidence grounding (quotes exist in source)
3. Logical consistency (no contradictions)

This allows production monitoring without expensive labeling.

NOTE: Uses only Python standard library. No ML models or training required.
"""

from typing import List, Dict, Any
from collections import defaultdict
from datetime import datetime

from .utils import find_text_in_journal, calculate_percentage


# Valid enum values for schema validation
VALID_DOMAINS = {"symptom", "food", "emotion", "mind"}
VALID_POLARITIES = {"present", "absent", "unknown"}
VALID_INTENSITIES = {"low", "medium", "high", "unknown"}
REQUIRED_FIELDS = {"domain", "item_name", "polarity", "evidence_span"}


def check_schema_validity(parsed_items: List[Dict]) -> Dict[str, Any]:
    """
    Validate that all parsed items conform to the required schema.

    WHAT: Checks that every parsed item has required fields with valid values.
    
    WHY THIS CHECK EXISTS:
    Downstream systems (analytics, recommendations, storage) depend on
    consistent data structure. Missing fields cause crashes, invalid enum
    values cause undefined behavior in decision logic. For health data,
    this can mean incorrect dietary advice or missed symptom patterns.

    ON FAILURE:
    - Status: FAIL
    - Action: STOP processing and alert engineers
    - Recommendation: rollback (do not deploy this parser version)

    Required fields: domain, item_name, polarity, evidence_span
    Valid domains: symptom, food, emotion, mind
    Valid polarities: present, absent, unknown
    Valid intensities (if present): low, medium, high, unknown

    Args:
        parsed_items: List of parsed item dictionaries to validate.

    Returns:
        Dictionary containing validation results with pass/fail status,
        counts, and actionable recommendations.
    """
    total_count = len(parsed_items)
    valid_count = 0
    invalid_items = []

    for idx, item in enumerate(parsed_items):
        errors = []

        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in item or item[field] is None or item[field] == "":
                errors.append(f"missing_or_empty_field: {field}")

        # Validate domain enum
        domain = item.get("domain")
        if domain and domain not in VALID_DOMAINS:
            errors.append(f"invalid_domain: {domain}")

        # Validate polarity enum
        polarity = item.get("polarity")
        if polarity and polarity not in VALID_POLARITIES:
            errors.append(f"invalid_polarity: {polarity}")

        # Validate intensity enum (if present)
        intensity = item.get("intensity")
        if intensity is not None and intensity not in VALID_INTENSITIES:
            errors.append(f"invalid_intensity: {intensity}")

        if errors:
            if len(invalid_items) < 10:  # Limit to first 10
                invalid_items.append({
                    "index": idx,
                    "item": item,
                    "errors": errors
                })
        else:
            valid_count += 1

    validity_rate = calculate_percentage(valid_count, total_count)
    # Empty dataset is considered valid (no invalid items)
    passed = (total_count == 0) or (validity_rate == 100.0)

    return {
        "metric": "schema_validity",
        "validity_rate": 100.0 if total_count == 0 else validity_rate,
        "valid_count": valid_count,
        "total_count": total_count,
        "status": "PASS" if passed else "FAIL",
        "risk": "Missing or invalid data breaks downstream systems",
        "action": None if passed else "STOP processing and alert engineers",
        "invalid_items": invalid_items,
        "recommended_action": "deploy" if passed else "rollback"
    }


def check_evidence_validity(parsed_items: List[Dict], journal_text: str) -> Dict[str, Any]:
    """
    Verify that evidence_span values exist verbatim in the source journal.

    WHY THIS CHECK EXISTS:
    LLM parsers can hallucinate evidence - generating plausible-sounding
    text that never appeared in the original journal. For health data,
    hallucinated evidence is DANGEROUS: it could attribute symptoms,
    foods, or emotions to a user that they never reported. This check
    ensures every piece of evidence is grounded in the actual source text.

    The 95% threshold allows for minor formatting differences while
    catching systematic hallucination problems.

    Args:
        parsed_items: List of parsed item dictionaries with evidence_span.
        journal_text: The original journal text to verify against.

    Returns:
        Dictionary containing validation results with pass/fail status,
        counts, and actionable recommendations.
    """
    total_count = len(parsed_items)
    valid_count = 0
    invalid_items = []

    for idx, item in enumerate(parsed_items):
        evidence_span = item.get("evidence_span", "")

        # Check if evidence exists verbatim in journal (case-insensitive)
        if find_text_in_journal(evidence_span, journal_text):
            valid_count += 1
        else:
            if len(invalid_items) < 10:  # Limit to first 10
                invalid_items.append({
                    "index": idx,
                    "item": item,
                    "evidence_span": evidence_span,
                    "error": "evidence_not_found_in_journal"
                })

    validity_rate = calculate_percentage(valid_count, total_count)
    passed = validity_rate >= 95.0

    return {
        "metric": "evidence_validity",
        "validity_rate": validity_rate,
        "valid_count": valid_count,
        "total_count": total_count,
        "status": "PASS" if passed else "FAIL",
        "risk": "Hallucinated evidence is unsafe for health data",
        "action": None if passed else "Quarantine batch and human review",
        "invalid_items": invalid_items,
        "recommended_action": "deploy" if passed else "rollback"
    }


def check_hallucination_rate(parsed_items: List[Dict], journal_text: str) -> Dict[str, Any]:
    """
    Detect hallucinated evidence spans that don't exist in the source journal.

    WHY THIS CHECK EXISTS:
    LLMs can fabricate plausible-sounding evidence that was never written
    by the user. This is CRITICAL for health data - we cannot attribute
    symptoms, foods, or emotions to users based on fabricated text.
    Hallucinated evidence destroys user trust and can lead to harmful
    health recommendations based on false information.

    IMPORTANT: We ONLY check evidence_span, NOT item_name. Item names are
    abstractions/labels derived from evidence, not direct quotes.

    Args:
        parsed_items: List of parsed item dictionaries with evidence_span.
        journal_text: The original journal text to verify against.

    Returns:
        Dictionary containing hallucination detection results.
    """
    total_count = len(parsed_items)
    hallucinated_count = 0
    hallucinated_items = []

    for idx, item in enumerate(parsed_items):
        evidence_span = item.get("evidence_span", "")

        # Hallucination: evidence is empty/missing OR not found in journal
        is_hallucinated = False
        if not evidence_span or evidence_span.strip() == "":
            is_hallucinated = True
        elif not find_text_in_journal(evidence_span, journal_text):
            is_hallucinated = True

        if is_hallucinated:
            hallucinated_count += 1
            if len(hallucinated_items) < 10:  # Limit to first 10
                hallucinated_items.append({
                    "index": idx,
                    "item": item,
                    "evidence_span": evidence_span,
                    "error": "hallucinated_or_missing_evidence"
                })

    hallucination_rate = calculate_percentage(hallucinated_count, total_count)
    passed = hallucination_rate <= 5.0

    return {
        "metric": "hallucination_rate",
        "rate": hallucination_rate,
        "hallucinated_count": hallucinated_count,
        "total_count": total_count,
        "status": "PASS" if passed else "FAIL",
        "risk": "Fabricated evidence compromises user trust and safety",
        "action": None if passed else "Reject batch, investigate parser",
        "hallucinated_items": hallucinated_items,
        "recommended_action": "deploy" if passed else "rollback"
    }


def check_contradictions(parsed_items: List[Dict]) -> Dict[str, Any]:
    """
    Detect contradictions where same evidence has conflicting polarities.

    WHY THIS CHECK EXISTS:
    It is logically IMPOSSIBLE for the same piece of evidence to indicate
    both presence and absence of something. If the user wrote "I had a
    headache", that cannot simultaneously mean they had AND didn't have
    a headache. Contradictions indicate a fundamental parser logic error.

    For health data, contradictions are ZERO TOLERANCE - any contradiction
    means the parser cannot be trusted and must be investigated.

    Args:
        parsed_items: List of parsed item dictionaries.

    Returns:
        Dictionary containing contradiction detection results.
    """
    # Group items by evidence_span
    evidence_groups = defaultdict(list)
    for idx, item in enumerate(parsed_items):
        evidence_span = item.get("evidence_span", "").strip().lower()
        if evidence_span:
            evidence_groups[evidence_span].append({
                "index": idx,
                "item": item,
                "polarity": item.get("polarity", "")
            })

    contradictions = []
    contradiction_count = 0

    # Check each group for conflicting polarities
    for evidence_span, items in evidence_groups.items():
        polarities = set()
        for item_info in items:
            polarity = item_info["polarity"].lower() if item_info["polarity"] else ""
            polarities.add(polarity)

        # Check for contradiction: both present/positive AND absent/negative
        has_present = "present" in polarities or "positive" in polarities
        has_absent = "absent" in polarities or "negative" in polarities

        if has_present and has_absent:
            contradiction_count += 1
            contradictions.append({
                "evidence_span": evidence_span,
                "conflicting_items": items,
                "polarities_found": list(polarities)
            })

    total_evidence_groups = len(evidence_groups)
    contradiction_rate = calculate_percentage(contradiction_count, total_evidence_groups)
    passed = contradiction_count == 0  # Zero tolerance

    return {
        "metric": "contradiction_rate",
        "rate": contradiction_rate,
        "contradiction_count": contradiction_count,
        "total_evidence_groups": total_evidence_groups,
        "status": "PASS" if passed else "FAIL",
        "risk": "Same evidence with conflicting polarity is logically impossible and unsafe",
        "action": None if passed else "STOP - critical logic error",
        "contradictions": contradictions,
        "recommended_action": "deploy" if passed else "rollback"
    }


def run_invariant_checks(parsed_outputs: List[Dict], journals: List[Dict]) -> Dict[str, Any]:
    """
    Run all invariant checks and produce a comprehensive report.

    This function orchestrates all validation checks:
    1. Schema validity (100% required)
    2. Evidence validity (95% required)
    3. Hallucination rate (<=5% required)
    4. Contradiction detection (0% required)

    Args:
        parsed_outputs: List of parsed output records, each containing 'items'.
        journals: List of journal records, each containing 'text'.

    Returns:
        Comprehensive report with all check results and overall status.
    """
    # Collect all parsed items and match with journal text
    all_items = []
    all_journal_text = ""

    for idx, parsed_output in enumerate(parsed_outputs):
        items = parsed_output.get("items", [])
        all_items.extend(items)

        # Match journal by index (assuming same order)
        if idx < len(journals):
            journal_text = journals[idx].get("text", "")
            all_journal_text += " " + journal_text

    # Run all checks
    schema_result = check_schema_validity(all_items)
    evidence_result = check_evidence_validity(all_items, all_journal_text)
    hallucination_result = check_hallucination_rate(all_items, all_journal_text)
    contradiction_result = check_contradictions(all_items)

    # Determine overall status
    all_checks = [
        schema_result,
        evidence_result,
        hallucination_result,
        contradiction_result
    ]

    all_passed = all(check["status"] == "PASS" for check in all_checks)
    overall_status = "PASS" if all_passed else "FAIL"

    # Determine final recommended action
    if all_passed:
        final_action = "deploy"
    else:
        final_action = "rollback"

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "overall_status": overall_status,
        "final_recommended_action": final_action,
        "total_items_checked": len(all_items),
        "total_journals_checked": len(journals),
        "checks": {
            "schema_validity": schema_result,
            "evidence_validity": evidence_result,
            "hallucination_rate": hallucination_result,
            "contradiction_rate": contradiction_result
        },
        "summary": {
            "passed_checks": sum(1 for c in all_checks if c["status"] == "PASS"),
            "failed_checks": sum(1 for c in all_checks if c["status"] == "FAIL"),
            "total_checks": len(all_checks)
        }
    }
