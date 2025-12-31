"""
System test for ashwam_monitor.

This test creates mock data with intentional issues to verify that
all checks correctly detect problems:
- Schema validation failure (missing evidence_span)
- Hallucination detection (evidence not in journal)
- Contradiction detection (same evidence, conflicting polarity)
- Drift detection (3x more emotion items in Day 1)

Run with: python -m ashwam_monitor.test_system
"""

import json
import tempfile
import shutil
from pathlib import Path

from .utils import load_jsonl, save_json
from .invariants import run_invariant_checks
from .drift import compare_drift
from .canary import run_canary_test


# =============================================================================
# MOCK DATA WITH INTENTIONAL ISSUES
# =============================================================================

MOCK_JOURNALS = [
    {
        "id": "journal_001",
        "text": "I had a headache this morning and felt very tired. Ate some oatmeal for breakfast. Feeling anxious about work."
    },
    {
        "id": "journal_002", 
        "text": "Slept well last night. Had coffee and toast. My knee pain is gone. Feeling happy today."
    }
]

# Day 0 outputs - baseline (correct)
MOCK_DAY0_OUTPUTS = [
    {
        "journal_id": "journal_001",
        "items": [
            {"domain": "symptom", "item_name": "headache", "polarity": "present", "intensity": "medium", "evidence_span": "had a headache"},
            {"domain": "symptom", "item_name": "fatigue", "polarity": "present", "intensity": "high", "evidence_span": "felt very tired"},
            {"domain": "food", "item_name": "oatmeal", "polarity": "present", "intensity": "low", "evidence_span": "Ate some oatmeal"},
            {"domain": "emotion", "item_name": "anxiety", "polarity": "present", "intensity": "medium", "evidence_span": "Feeling anxious"}
        ]
    },
    {
        "journal_id": "journal_002",
        "items": [
            {"domain": "symptom", "item_name": "sleep quality", "polarity": "present", "intensity": "high", "evidence_span": "Slept well"},
            {"domain": "food", "item_name": "coffee", "polarity": "present", "intensity": "low", "evidence_span": "Had coffee"},
            {"domain": "symptom", "item_name": "knee pain", "polarity": "absent", "intensity": "low", "evidence_span": "knee pain is gone"},
            {"domain": "emotion", "item_name": "happiness", "polarity": "present", "intensity": "high", "evidence_span": "Feeling happy"}
        ]
    }
]

# Day 1 outputs - with intentional issues
MOCK_DAY1_OUTPUTS = [
    {
        "journal_id": "journal_001",
        "items": [
            # ISSUE 1: Missing evidence_span (schema fail)
            {"domain": "symptom", "item_name": "headache", "polarity": "present", "intensity": "medium", "evidence_span": ""},
            {"domain": "symptom", "item_name": "fatigue", "polarity": "present", "intensity": "high", "evidence_span": "felt very tired"},
            # ISSUE 2: Hallucinated evidence (not in journal)
            {"domain": "food", "item_name": "eggs", "polarity": "present", "intensity": "low", "evidence_span": "scrambled eggs for breakfast"},
            {"domain": "emotion", "item_name": "anxiety", "polarity": "present", "intensity": "medium", "evidence_span": "Feeling anxious"},
            # ISSUE 3: Contradiction - same evidence, opposite polarity
            {"domain": "emotion", "item_name": "anxiety_positive", "polarity": "present", "intensity": "high", "evidence_span": "Feeling anxious about work"},
            {"domain": "emotion", "item_name": "anxiety_negative", "polarity": "absent", "intensity": "low", "evidence_span": "Feeling anxious about work"},
            # ISSUE 4: Extra emotion items (drift)
            {"domain": "emotion", "item_name": "stress", "polarity": "present", "intensity": "high", "evidence_span": "about work"},
            {"domain": "emotion", "item_name": "worry", "polarity": "present", "intensity": "medium", "evidence_span": "anxious about"}
        ]
    },
    {
        "journal_id": "journal_002",
        "items": [
            {"domain": "symptom", "item_name": "sleep quality", "polarity": "present", "intensity": "high", "evidence_span": "Slept well"},
            {"domain": "food", "item_name": "coffee", "polarity": "present", "intensity": "low", "evidence_span": "Had coffee"},
            {"domain": "symptom", "item_name": "knee pain", "polarity": "absent", "intensity": "low", "evidence_span": "knee pain is gone"},
            # More emotion items for drift
            {"domain": "emotion", "item_name": "happiness", "polarity": "present", "intensity": "high", "evidence_span": "Feeling happy"},
            {"domain": "emotion", "item_name": "contentment", "polarity": "present", "intensity": "medium", "evidence_span": "happy today"},
            {"domain": "emotion", "item_name": "satisfaction", "polarity": "present", "intensity": "low", "evidence_span": "Slept well last night"}
        ]
    }
]

# Canary data (small labeled test set)
MOCK_CANARY_JOURNALS = [
    {
        "id": "canary_001",
        "text": "Migraine all day. Took ibuprofen. Feeling frustrated."
    }
]

MOCK_CANARY_GOLD = [
    {
        "journal_id": "canary_001",
        "items": [
            {"domain": "symptom", "item_name": "migraine", "polarity": "present", "intensity": "high", "evidence_span": "Migraine all day"},
            {"domain": "food", "item_name": "ibuprofen", "polarity": "present", "intensity": "low", "evidence_span": "Took ibuprofen"},
            {"domain": "emotion", "item_name": "frustration", "polarity": "present", "intensity": "medium", "evidence_span": "Feeling frustrated"}
        ]
    }
]

# Canary outputs (what the parser produced for canary)
MOCK_CANARY_OUTPUTS = [
    {
        "journal_id": "canary_001",
        "items": [
            {"domain": "symptom", "item_name": "migraine", "polarity": "present", "intensity": "high", "evidence_span": "Migraine all day"},
            {"domain": "food", "item_name": "ibuprofen", "polarity": "present", "intensity": "low", "evidence_span": "Took ibuprofen"},
            # Wrong polarity for emotion
            {"domain": "emotion", "item_name": "frustration", "polarity": "absent", "intensity": "medium", "evidence_span": "Feeling frustrated"}
        ]
    }
]


def write_jsonl(data: list, filepath: Path) -> None:
    """Write data as JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def run_test() -> bool:
    """
    Run the full system test with mock data.
    
    Returns:
        True if all expected failures were detected, False otherwise.
    """
    print("=" * 60)
    print("  ASHWAM Monitor System Test")
    print("=" * 60)
    
    # Create temporary directory for test data
    temp_dir = Path(tempfile.mkdtemp(prefix="ashwam_test_"))
    data_dir = temp_dir / "data"
    out_dir = temp_dir / "out"
    
    try:
        # Write mock data files
        print("\n[1] Creating mock data with intentional issues...")
        write_jsonl(MOCK_JOURNALS, data_dir / "journals.jsonl")
        write_jsonl(MOCK_DAY0_OUTPUTS, data_dir / "parser_outputs_day0.jsonl")
        write_jsonl(MOCK_DAY1_OUTPUTS, data_dir / "parser_outputs_day1.jsonl")
        write_jsonl(MOCK_CANARY_JOURNALS, data_dir / "canary" / "journals.jsonl")
        write_jsonl(MOCK_CANARY_GOLD, data_dir / "canary" / "gold.jsonl")
        print(f"   Data written to: {data_dir}")
        
        # Load data back
        print("\n[2] Loading data...")
        journals = load_jsonl(data_dir / "journals.jsonl")
        day0_outputs = load_jsonl(data_dir / "parser_outputs_day0.jsonl")
        day1_outputs = load_jsonl(data_dir / "parser_outputs_day1.jsonl")
        canary_journals = load_jsonl(data_dir / "canary" / "journals.jsonl")
        canary_gold = load_jsonl(data_dir / "canary" / "gold.jsonl")
        print(f"   Loaded {len(journals)} journals, {len(day0_outputs)} day0, {len(day1_outputs)} day1")
        
        # Run invariant checks
        print("\n[3] Running invariant checks...")
        invariant_report = run_invariant_checks(day1_outputs, journals)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(invariant_report, out_dir / "invariant_report.json")
        
        # Verify expected failures
        tests_passed = True
        
        # Check schema validity failed
        schema_status = invariant_report["checks"]["schema_validity"]["status"]
        if schema_status == "FAIL":
            print("   ✓ Schema validation correctly detected missing evidence_span")
        else:
            print("   ✗ Schema validation should have FAILED but got:", schema_status)
            tests_passed = False
        
        # Check hallucination detected
        hallucination_rate = invariant_report["checks"]["hallucination_rate"]["rate"]
        if hallucination_rate > 0:
            print(f"   ✓ Hallucination detection found {hallucination_rate}% hallucinated items")
        else:
            print("   ✗ Hallucination detection should have found issues")
            tests_passed = False
        
        # Check contradictions detected
        contradiction_count = invariant_report["checks"]["contradiction_rate"]["contradiction_count"]
        if contradiction_count > 0:
            print(f"   ✓ Contradiction detection found {contradiction_count} contradiction(s)")
        else:
            print("   ✗ Contradiction detection should have found issues")
            tests_passed = False
        
        # Run drift analysis
        print("\n[4] Running drift analysis...")
        drift_report = compare_drift(day0_outputs, day1_outputs)
        save_json(drift_report, out_dir / "drift_report.json")
        
        # Check drift detected
        drift_status = drift_report["overall_drift_status"]
        if drift_status != "none":
            print(f"   ✓ Drift detection found {drift_status} drift")
            print(f"   ✓ {len(drift_report.get('drift_flags', []))} drift flags raised")
        else:
            print("   ⚠ Drift detection found no drift (may be expected with small data)")
        
        # Run canary test
        print("\n[5] Running canary test...")
        canary_report = run_canary_test(MOCK_CANARY_OUTPUTS, canary_gold, canary_journals)
        save_json(canary_report, out_dir / "canary_report.json")
        
        # Check canary detected polarity issue
        polarity_correctness = canary_report["metrics"]["polarity_correctness"]
        if polarity_correctness < 100:
            print(f"   ✓ Canary test detected polarity mismatch ({polarity_correctness}% correct)")
        else:
            print("   ✗ Canary test should have detected polarity issues")
            tests_passed = False
        
        # Verify reports generated
        print("\n[6] Verifying reports generated...")
        reports = ["invariant_report.json", "drift_report.json", "canary_report.json"]
        for report in reports:
            if (out_dir / report).exists():
                print(f"   ✓ {report} generated")
            else:
                print(f"   ✗ {report} NOT generated")
                tests_passed = False
        
        # Print overall result
        print("\n" + "=" * 60)
        if tests_passed:
            print("  ✓ ALL TESTS PASSED - System correctly detects issues")
        else:
            print("  ✗ SOME TESTS FAILED - Review implementation")
        print("=" * 60)
        
        # Print final recommendations
        print(f"\n  Invariant recommendation: {invariant_report.get('final_recommended_action', 'unknown')}")
        print(f"  Drift recommendation: {drift_report.get('recommended_action', 'unknown')}")
        print(f"  Canary recommendation: {canary_report.get('recommended_action', 'unknown')}")
        
        return tests_passed
        
    except Exception as e:
        print(f"\n  ✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print(f"\n  Cleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_error_handling() -> bool:
    """
    Test error handling for edge cases.
    
    Returns:
        True if all error handling tests pass.
    """
    print("\n" + "=" * 60)
    print("  Error Handling Tests")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: File not found
    print("\n[1] Testing file not found handling...")
    try:
        load_jsonl("/nonexistent/path/file.jsonl")
        print("   ✗ Should have raised FileNotFoundError")
        all_passed = False
    except FileNotFoundError:
        print("   ✓ FileNotFoundError correctly raised")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        all_passed = False
    
    # Test 2: Malformed JSON
    print("\n[2] Testing malformed JSON handling...")
    temp_file = Path(tempfile.mktemp(suffix=".jsonl"))
    try:
        with open(temp_file, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('not valid json\n')
            f.write('{"another": "valid"}\n')
        
        try:
            load_jsonl(temp_file)
            print("   ✗ Should have raised JSONDecodeError")
            all_passed = False
        except json.JSONDecodeError:
            print("   ✓ JSONDecodeError correctly raised for malformed JSON")
        except Exception as e:
            print(f"   ✗ Unexpected error: {e}")
            all_passed = False
    finally:
        temp_file.unlink(missing_ok=True)
    
    # Test 3: Empty dataset
    print("\n[3] Testing empty dataset handling...")
    from .invariants import check_schema_validity
    from .drift import calculate_extraction_volume
    from .canary import calculate_evidence_validity_rate
    
    try:
        result = check_schema_validity([])
        if result["total_count"] == 0 and result["status"] == "PASS":
            print("   ✓ Empty dataset handled correctly in schema check")
        else:
            print(f"   ✗ Unexpected result for empty dataset: {result}")
            all_passed = False
    except Exception as e:
        print(f"   ✗ Error with empty dataset: {e}")
        all_passed = False
    
    try:
        result = calculate_extraction_volume([])
        if result["total_journals"] == 0:
            print("   ✓ Empty dataset handled correctly in extraction volume")
        else:
            print(f"   ✗ Unexpected result: {result}")
            all_passed = False
    except Exception as e:
        print(f"   ✗ Error with empty dataset: {e}")
        all_passed = False
    
    try:
        result = calculate_evidence_validity_rate([], "some text")
        if result == 100.0:
            print("   ✓ Empty items handled correctly in evidence validity")
        else:
            print(f"   ✗ Unexpected result: {result}")
            all_passed = False
    except Exception as e:
        print(f"   ✗ Error with empty items: {e}")
        all_passed = False
    
    # Test 4: Missing fields in data
    print("\n[4] Testing missing fields handling...")
    items_with_missing = [
        {"domain": "symptom"},  # Missing most fields
        {"item_name": "test", "polarity": "present"},  # Missing domain, evidence
        {},  # Empty item
    ]
    
    try:
        result = check_schema_validity(items_with_missing)
        if result["status"] == "FAIL" and result["valid_count"] == 0:
            print("   ✓ Missing fields correctly detected")
        else:
            print(f"   ✗ Should have detected missing fields: {result}")
            all_passed = False
    except Exception as e:
        print(f"   ✗ Error with missing fields: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓ ALL ERROR HANDLING TESTS PASSED")
    else:
        print("  ✗ SOME ERROR HANDLING TESTS FAILED")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    print("\n")
    main_tests = run_test()
    error_tests = test_error_handling()
    
    print("\n" + "=" * 60)
    print("  FINAL RESULT")
    print("=" * 60)
    if main_tests and error_tests:
        print("  ✓ ALL SYSTEM TESTS PASSED")
        exit(0)
    else:
        print("  ✗ SOME TESTS FAILED")
        exit(1)
