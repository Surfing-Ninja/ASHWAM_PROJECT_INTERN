"""
CLI Entry Point and Orchestration for ASHWAM Parser Monitor.

This module provides the command-line interface for running monitoring checks,
including invariants validation, drift detection, and canary evaluation.

Usage:
    python -m ashwam_monitor run --data ./data --out ./out

The orchestrator runs all checks in sequence and produces:
1. invariant_report.json - Schema, evidence, hallucination, contradiction checks
2. drift_report.json - Day 0 vs Day 1 comparison metrics
3. canary_report.json - Gold label evaluation results

Final recommendation is based on the most severe status across all checks:
- deploy: All checks pass, safe to deploy
- human_review: Minor issues detected, needs manual review
- rollback: Critical issues detected, do not deploy

Author: Mohit
Version: 1.0.0
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from .utils import load_jsonl, save_json
from .invariants import run_invariant_checks
from .drift import compare_drift
from .canary import run_canary_test


# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_status(label: str, status: str, details: str = "") -> None:
    """Print a status line with appropriate coloring."""
    if status.upper() in ("PASS", "DEPLOY", "NONE"):
        icon = f"{Colors.GREEN}✓{Colors.RESET}"
        status_colored = f"{Colors.GREEN}{status}{Colors.RESET}"
    elif status.upper() in ("FAIL", "ROLLBACK", "SEVERE"):
        icon = f"{Colors.RED}✗{Colors.RESET}"
        status_colored = f"{Colors.RED}{status}{Colors.RESET}"
    else:  # WARNING, HUMAN_REVIEW, MODERATE, MINOR
        icon = f"{Colors.YELLOW}⚠{Colors.RESET}"
        status_colored = f"{Colors.YELLOW}{status}{Colors.RESET}"
    
    detail_str = f" - {details}" if details else ""
    print(f"  {icon} {label}: {status_colored}{detail_str}")


def print_metric(label: str, value: str) -> None:
    """Print a metric value."""
    print(f"    • {label}: {value}")


def run_monitoring(data_path: Path, out_path: Path) -> int:
    """
    Run all monitoring checks and generate reports.
    
    Args:
        data_path: Path to data directory containing input files.
        out_path: Path to output directory for reports.
    
    Returns:
        Exit code (0 for deploy, 1 for human_review, 2 for rollback).
    """
    print_header("ASHWAM Parser Monitor")
    print(f"  Timestamp: {datetime.utcnow().isoformat()}Z")
    print(f"  Data path: {data_path}")
    print(f"  Output path: {out_path}")
    
    # Validate data path exists
    if not data_path.exists():
        print(f"\n{Colors.RED}✗ Error: Data directory does not exist: {data_path}{Colors.RESET}")
        print(f"  Please provide a valid path to the data directory.")
        return 2
    
    # Ensure output directory exists
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"\n{Colors.RED}✗ Error: Cannot create output directory: {out_path}{Colors.RESET}")
        print(f"  Please check permissions.")
        return 2
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print_header("Loading Data")
    
    try:
        journals = load_jsonl(data_path / "journals.jsonl")
        print_metric("Journals loaded", str(len(journals)))
        
        day0_outputs = load_jsonl(data_path / "parser_outputs_day0.jsonl")
        print_metric("Day 0 outputs loaded", str(len(day0_outputs)))
        
        day1_outputs = load_jsonl(data_path / "parser_outputs_day1.jsonl")
        print_metric("Day 1 outputs loaded", str(len(day1_outputs)))
        
        canary_journals = load_jsonl(data_path / "canary" / "journals.jsonl")
        print_metric("Canary journals loaded", str(len(canary_journals)))
        
        canary_gold = load_jsonl(data_path / "canary" / "gold.jsonl")
        print_metric("Canary gold labels loaded", str(len(canary_gold)))
        
    except FileNotFoundError as e:
        print(f"\n{Colors.RED}✗ Error: Could not find data file: {e.filename}{Colors.RESET}")
        print(f"  Please ensure all required files exist in {data_path}")
        print(f"  Required files:")
        print(f"    - journals.jsonl")
        print(f"    - parser_outputs_day0.jsonl")
        print(f"    - parser_outputs_day1.jsonl")
        print(f"    - canary/journals.jsonl")
        print(f"    - canary/gold.jsonl")
        return 2
    except json.JSONDecodeError as e:
        print(f"\n{Colors.RED}✗ Error: Malformed JSON in data file{Colors.RESET}")
        print(f"  Line {e.lineno}, Column {e.colno}: {e.msg}")
        print(f"  Please check the JSONL files for syntax errors.")
        return 2
    except PermissionError as e:
        print(f"\n{Colors.RED}✗ Error: Permission denied reading file: {e.filename}{Colors.RESET}")
        return 2
    except Exception as e:
        print(f"\n{Colors.RED}✗ Error loading data: {type(e).__name__}: {e}{Colors.RESET}")
        return 2
    
    # Validate data is not empty
    if not journals:
        print(f"\n{Colors.YELLOW}⚠ Warning: No journals loaded{Colors.RESET}")
    if not day0_outputs:
        print(f"\n{Colors.YELLOW}⚠ Warning: No Day 0 outputs loaded{Colors.RESET}")
    if not day1_outputs:
        print(f"\n{Colors.RED}✗ Error: No Day 1 outputs to evaluate{Colors.RESET}")
        return 2
    
    # Track overall status
    critical_issues = []
    warnings = []
    
    # =========================================================================
    # INVARIANT CHECKS
    # =========================================================================
    print_header("Invariant Checks")
    
    try:
        invariant_report = run_invariant_checks(day1_outputs, journals)
        save_json(invariant_report, out_path / "invariant_report.json")
    except Exception as e:
        print(f"\n{Colors.RED}✗ Error running invariant checks: {e}{Colors.RESET}")
        critical_issues.append(f"Invariant checks failed with error: {e}")
        invariant_report = {"checks": {}, "overall_status": "ERROR", "final_recommended_action": "rollback"}
    
    # Print invariant results
    checks = invariant_report.get("checks", {})
    
    schema_check = checks.get("schema_validity", {})
    print_status(
        "Schema Validity",
        schema_check.get("status", "UNKNOWN"),
        f"{schema_check.get('validity_rate', 0)}% valid ({schema_check.get('valid_count', 0)}/{schema_check.get('total_count', 0)})"
    )
    if schema_check.get("status") == "FAIL":
        invalid_count = schema_check.get('total_count', 0) - schema_check.get('valid_count', 0)
        critical_issues.append(
            f"Schema validation failed ({invalid_count} invalid items) - "
            "stop processing, check invalid_items in invariant_report.json"
        )
    
    evidence_check = checks.get("evidence_validity", {})
    print_status(
        "Evidence Validity",
        evidence_check.get("status", "UNKNOWN"),
        f"{evidence_check.get('validity_rate', 0)}% valid"
    )
    if evidence_check.get("status") == "FAIL":
        critical_issues.append(
            f"Evidence validity at {evidence_check.get('validity_rate', 0)}% (below 95%) - "
            "quarantine batch and review hallucinated evidence spans"
        )
    
    hallucination_check = checks.get("hallucination_rate", {})
    print_status(
        "Hallucination Rate",
        hallucination_check.get("status", "UNKNOWN"),
        f"{hallucination_check.get('rate', 0)}% hallucinated"
    )
    if hallucination_check.get("status") == "FAIL":
        critical_issues.append(
            f"Hallucination rate at {hallucination_check.get('rate', 0)}% (above 5%) - "
            "reject batch, investigate parser prompt or model changes"
        )
    
    contradiction_check = checks.get("contradiction_rate", {})
    print_status(
        "Contradictions",
        contradiction_check.get("status", "UNKNOWN"),
        f"{contradiction_check.get('contradiction_count', 0)} found"
    )
    if contradiction_check.get("status") == "FAIL":
        critical_issues.append(
            f"Found {contradiction_check.get('contradiction_count', 0)} contradiction(s) - "
            "critical logic error, stop deployment and fix parser"
        )
    
    print(f"\n  {Colors.BOLD}Invariants Overall:{Colors.RESET} {invariant_report.get('overall_status', 'UNKNOWN')}")
    print(f"  Recommendation: {invariant_report.get('final_recommended_action', 'unknown')}")
    
    # =========================================================================
    # DRIFT ANALYSIS
    # =========================================================================
    print_header("Drift Analysis (Day 0 → Day 1)")
    
    try:
        drift_report = compare_drift(day0_outputs, day1_outputs)
        save_json(drift_report, out_path / "drift_report.json")
    except Exception as e:
        print(f"\n{Colors.RED}✗ Error running drift analysis: {e}{Colors.RESET}")
        warnings.append(f"Drift analysis failed with error: {e}")
        drift_report = {"metrics": {}, "drift_flags": [], "overall_drift_status": "error", "recommended_action": "human_review"}
    
    # Print drift summary
    metrics = drift_report.get("metrics", {})
    
    volume_metrics = metrics.get("extraction_volume", {})
    print_metric(
        "Extraction Volume",
        f"Day 0: {volume_metrics.get('day0', {}).get('average', 0):.1f} → "
        f"Day 1: {volume_metrics.get('day1', {}).get('average', 0):.1f} items/journal "
        f"({volume_metrics.get('change_percent', 0):+.1f}%)"
    )
    
    uncertainty = metrics.get("uncertainty_rate", {})
    print_metric(
        "Uncertainty Rate",
        f"Day 0: {uncertainty.get('day0', 0):.1f}% → Day 1: {uncertainty.get('day1', 0):.1f}%"
    )
    
    # Print drift flags
    drift_flags = drift_report.get("drift_flags", [])
    if drift_flags:
        print(f"\n  {Colors.YELLOW}Drift Flags ({len(drift_flags)}):{Colors.RESET}")
        for flag in drift_flags[:5]:  # Show first 5
            print_status(
                flag.get("metric", "unknown"),
                flag.get("drift_severity", "unknown").upper(),
                flag.get("reason", "")[:60] + "..."
            )
    
    drift_status = drift_report.get("overall_drift_status", "none")
    print(f"\n  {Colors.BOLD}Drift Status:{Colors.RESET} {drift_status}")
    print(f"  Recommendation: {drift_report.get('recommended_action', 'unknown')}")
    
    if drift_status == "severe":
        drift_count = len(drift_report.get("drift_flags", []))
        critical_issues.append(
            f"Severe drift detected ({drift_count} flags) - "
            "significant behavior change, review drift_report.json before proceeding"
        )
    elif drift_status in ("moderate", "minor"):
        drift_count = len(drift_report.get("drift_flags", []))
        warnings.append(
            f"{drift_status.capitalize()} drift detected ({drift_count} flags) - "
            "review drift_flags in drift_report.json"
        )
    
    # =========================================================================
    # CANARY TEST
    # =========================================================================
    print_header("Canary Test Evaluation")
    
    try:
        # Use day1_outputs for canary (assuming first N match canary journals)
        canary_outputs = day1_outputs[:len(canary_journals)]
        canary_report = run_canary_test(canary_outputs, canary_gold, canary_journals)
        save_json(canary_report, out_path / "canary_report.json")
    except Exception as e:
        print(f"\n{Colors.RED}✗ Error running canary test: {e}{Colors.RESET}")
        warnings.append(f"Canary test failed with error: {e}")
        canary_report = {"metrics": {}, "alert_level": "HUMAN_REVIEW", "reasoning": f"Error: {e}", "recommended_action": "human_review"}
    
    # Print canary results
    canary_metrics = canary_report.get("metrics", {})
    print_metric("Evidence Validity", f"{canary_metrics.get('evidence_validity', 0):.1f}%")
    print_metric("Polarity Correctness", f"{canary_metrics.get('polarity_correctness', 0):.1f}%")
    print_metric("Contradictions", str(canary_metrics.get("contradiction_count", 0)))
    
    canary_alert = canary_report.get("alert_level", "UNKNOWN")
    print(f"\n  {Colors.BOLD}Canary Alert Level:{Colors.RESET} ", end="")
    print_status("", canary_alert, "")
    print(f"  Reasoning: {canary_report.get('reasoning', 'N/A')[:100]}...")
    
    if canary_alert == "ROLLBACK":
        evidence_val = canary_metrics.get('evidence_validity', 0)
        contradictions = canary_metrics.get('contradiction_count', 0)
        critical_issues.append(
            f"Canary test failed (evidence: {evidence_val}%, contradictions: {contradictions}) - "
            "parser not safe, review canary_report.json details"
        )
    elif canary_alert == "HUMAN_REVIEW":
        polarity = canary_metrics.get('polarity_correctness', 0)
        warnings.append(
            f"Canary polarity correctness at {polarity}% (below 90%) - "
            "review mismatched items in canary_report.json"
        )
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_header("Final Summary")
    
    # Determine final recommendation
    if critical_issues:
        final_recommendation = "ROLLBACK"
        final_color = Colors.RED
        exit_code = 2
    elif warnings:
        final_recommendation = "HUMAN_REVIEW"
        final_color = Colors.YELLOW
        exit_code = 1
    else:
        final_recommendation = "DEPLOY"
        final_color = Colors.GREEN
        exit_code = 0
    
    # Print summary
    print(f"  {Colors.BOLD}Critical Issues:{Colors.RESET} {len(critical_issues)}")
    for issue in critical_issues:
        print(f"    {Colors.RED}• {issue}{Colors.RESET}")
    
    print(f"\n  {Colors.BOLD}Warnings:{Colors.RESET} {len(warnings)}")
    for warning in warnings:
        print(f"    {Colors.YELLOW}• {warning}{Colors.RESET}")
    
    print(f"\n  {Colors.BOLD}Reports Generated:{Colors.RESET}")
    print(f"    • {out_path / 'invariant_report.json'}")
    print(f"    • {out_path / 'drift_report.json'}")
    print(f"    • {out_path / 'canary_report.json'}")
    
    # Final recommendation box
    print(f"\n  {'─' * 50}")
    print(f"  {Colors.BOLD}FINAL RECOMMENDATION:{Colors.RESET} {final_color}{final_recommendation}{Colors.RESET}")
    print(f"  {'─' * 50}")
    
    if final_recommendation == "DEPLOY":
        print(f"\n  {Colors.GREEN}✓ All checks passed. Parser is safe to deploy.{Colors.RESET}")
        print(f"\n  {Colors.BOLD}Next Steps:{Colors.RESET}")
        print(f"    1. Proceed with production deployment")
        print(f"    2. Monitor production metrics for first 24 hours")
        print(f"    3. Keep rollback ready if issues emerge")
    elif final_recommendation == "HUMAN_REVIEW":
        print(f"\n  {Colors.YELLOW}⚠ Some concerns detected. Manual review required.{Colors.RESET}")
        print(f"\n  {Colors.BOLD}Next Steps:{Colors.RESET}")
        print(f"    1. Review drift_report.json for specific concerns")
        print(f"    2. Compare sample outputs between Day 0 and Day 1")
        print(f"    3. If changes are intentional, proceed with caution")
        print(f"    4. If changes are unexpected, investigate parser/prompt changes")
    else:
        print(f"\n  {Colors.RED}✗ Critical issues detected. DO NOT deploy.{Colors.RESET}")
        print(f"\n  {Colors.BOLD}Next Steps:{Colors.RESET}")
        print(f"    1. Review invariant_report.json for failure details")
        print(f"    2. Check invalid_items and hallucinated_items lists")
        print(f"    3. Investigate parser logic or prompt changes")
        print(f"    4. Fix issues and re-run monitoring before deployment")
        print(f"    5. If in production, initiate rollback immediately")
    
    print()
    return exit_code


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="ashwam_monitor",
        description="Monitor and validate ASHWAM parser outputs for production safety.",
        epilog="Run 'python -m ashwam_monitor run --help' for subcommand options."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Run all monitoring checks"
    )
    run_parser.add_argument(
        "--data",
        type=Path,
        default=Path("./data"),
        help="Path to data directory (default: ./data)"
    )
    run_parser.add_argument(
        "--out",
        type=Path,
        default=Path("./out"),
        help="Path to output directory (default: ./out)"
    )
    
    return parser


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "run":
        return run_monitoring(args.data, args.out)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
