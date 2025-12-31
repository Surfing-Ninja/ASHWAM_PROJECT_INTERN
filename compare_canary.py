"""
Canary Data Comparison Script.

Compares parser outputs (Day 0 and Day 1) against canary gold labels
to generate detailed comparison reports.

Usage:
    python -m ashwam_monitor.compare_canary --data <path> --out <path>

Author: Mohit
Version: 1.0.0
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of dictionaries."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(filepath: str, data: Dict[str, Any]) -> None:
    """Save data to a JSON file with pretty formatting."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def find_text_in_journal(journal_text: str, evidence_span: str) -> bool:
    """Check if evidence span exists in journal text (case-insensitive)."""
    if not evidence_span or not journal_text:
        return False
    return evidence_span.lower() in journal_text.lower()


def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize item for comparison by extracting key fields."""
    return {
        "domain": item.get("domain", "").lower(),
        "polarity": item.get("polarity", "").lower(),
        "evidence_span": item.get("evidence_span", "").lower().strip(),
    }


def compare_items(parser_items: List[Dict], gold_items: List[Dict], journal_text: str) -> Dict[str, Any]:
    """
    Compare parser items against gold items for a single journal.
    
    Returns detailed comparison results.
    """
    results = {
        "parser_item_count": len(parser_items),
        "gold_item_count": len(gold_items),
        "matched_items": [],
        "missing_from_parser": [],  # In gold but not in parser
        "extra_in_parser": [],  # In parser but not in gold
        "polarity_mismatches": [],
        "hallucinated_evidence": [],
        "evidence_validity": {"valid": 0, "invalid": 0},
    }
    
    # Normalize all items
    parser_normalized = [normalize_item(item) for item in parser_items]
    gold_normalized = [normalize_item(item) for item in gold_items]
    
    # Track which gold items have been matched
    gold_matched = [False] * len(gold_items)
    parser_matched = [False] * len(parser_items)
    
    # Check each parser item against gold
    for p_idx, (p_item, p_norm) in enumerate(zip(parser_items, parser_normalized)):
        # Check evidence validity
        evidence_valid = find_text_in_journal(journal_text, p_item.get("evidence_span", ""))
        if evidence_valid:
            results["evidence_validity"]["valid"] += 1
        else:
            results["evidence_validity"]["invalid"] += 1
            results["hallucinated_evidence"].append({
                "evidence_span": p_item.get("evidence_span", ""),
                "domain": p_item.get("domain", ""),
            })
        
        # Try to match with gold items
        for g_idx, (g_item, g_norm) in enumerate(zip(gold_items, gold_normalized)):
            if gold_matched[g_idx]:
                continue
            
            # Match by domain and evidence overlap
            if p_norm["domain"] == g_norm["domain"]:
                # Check if evidence spans overlap
                p_evidence = p_norm["evidence_span"]
                g_evidence = g_norm["evidence_span"]
                
                if p_evidence and g_evidence and (p_evidence in g_evidence or g_evidence in p_evidence):
                    gold_matched[g_idx] = True
                    parser_matched[p_idx] = True
                    
                    # Check polarity
                    if p_norm["polarity"] == g_norm["polarity"]:
                        results["matched_items"].append({
                            "domain": p_item.get("domain"),
                            "parser_evidence": p_item.get("evidence_span"),
                            "gold_evidence": g_item.get("evidence_span"),
                            "polarity": p_item.get("polarity"),
                        })
                    else:
                        results["polarity_mismatches"].append({
                            "domain": p_item.get("domain"),
                            "evidence": p_item.get("evidence_span"),
                            "parser_polarity": p_item.get("polarity"),
                            "gold_polarity": g_item.get("polarity"),
                        })
                    break
    
    # Find missing from parser (in gold but not matched)
    for g_idx, (g_item, matched) in enumerate(zip(gold_items, gold_matched)):
        if not matched:
            results["missing_from_parser"].append({
                "domain": g_item.get("domain"),
                "evidence_span": g_item.get("evidence_span"),
                "polarity": g_item.get("polarity"),
            })
    
    # Find extra in parser (in parser but not matched)
    for p_idx, (p_item, matched) in enumerate(zip(parser_items, parser_matched)):
        if not matched:
            results["extra_in_parser"].append({
                "domain": p_item.get("domain"),
                "evidence_span": p_item.get("evidence_span"),
                "polarity": p_item.get("polarity"),
            })
    
    return results


def generate_comparison_report(
    parser_outputs: List[Dict],
    canary_journals: List[Dict],
    gold_labels: List[Dict],
    version_name: str
) -> Dict[str, Any]:
    """
    Generate comprehensive comparison report.
    """
    # Create lookup maps
    journal_map = {j["journal_id"]: j["text"] for j in canary_journals}
    parser_map = {p["journal_id"]: p.get("items", []) for p in parser_outputs}
    gold_map = {g["journal_id"]: g.get("items", []) for g in gold_labels}
    
    # Only compare journals that exist in canary set
    canary_journal_ids = set(journal_map.keys())
    
    report = {
        "version": version_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_canary_journals": len(canary_journal_ids),
            "journals_with_parser_output": 0,
            "total_gold_items": 0,
            "total_parser_items": 0,
            "total_matched": 0,
            "total_missing": 0,
            "total_extra": 0,
            "total_polarity_mismatches": 0,
            "total_hallucinated": 0,
            "evidence_validity_rate": 0.0,
            "recall_rate": 0.0,  # Items in gold that parser found
            "precision_proxy": 0.0,  # Items in parser that match gold
        },
        "per_journal": {},
        "domain_breakdown": defaultdict(lambda: {"matched": 0, "missing": 0, "extra": 0}),
    }
    
    total_valid_evidence = 0
    total_evidence = 0
    
    for journal_id in canary_journal_ids:
        journal_text = journal_map.get(journal_id, "")
        parser_items = parser_map.get(journal_id, [])
        gold_items = gold_map.get(journal_id, [])
        
        if parser_items:
            report["summary"]["journals_with_parser_output"] += 1
        
        report["summary"]["total_gold_items"] += len(gold_items)
        report["summary"]["total_parser_items"] += len(parser_items)
        
        # Compare
        comparison = compare_items(parser_items, gold_items, journal_text)
        report["per_journal"][journal_id] = comparison
        
        # Update totals
        report["summary"]["total_matched"] += len(comparison["matched_items"])
        report["summary"]["total_missing"] += len(comparison["missing_from_parser"])
        report["summary"]["total_extra"] += len(comparison["extra_in_parser"])
        report["summary"]["total_polarity_mismatches"] += len(comparison["polarity_mismatches"])
        report["summary"]["total_hallucinated"] += len(comparison["hallucinated_evidence"])
        
        total_valid_evidence += comparison["evidence_validity"]["valid"]
        total_evidence += comparison["evidence_validity"]["valid"] + comparison["evidence_validity"]["invalid"]
        
        # Domain breakdown
        for item in comparison["matched_items"]:
            report["domain_breakdown"][item["domain"]]["matched"] += 1
        for item in comparison["missing_from_parser"]:
            report["domain_breakdown"][item["domain"]]["missing"] += 1
        for item in comparison["extra_in_parser"]:
            report["domain_breakdown"][item["domain"]]["extra"] += 1
    
    # Calculate rates
    if total_evidence > 0:
        report["summary"]["evidence_validity_rate"] = round(total_valid_evidence / total_evidence * 100, 2)
    
    if report["summary"]["total_gold_items"] > 0:
        report["summary"]["recall_rate"] = round(
            report["summary"]["total_matched"] / report["summary"]["total_gold_items"] * 100, 2
        )
    
    if report["summary"]["total_parser_items"] > 0:
        matched_or_polarity = report["summary"]["total_matched"] + report["summary"]["total_polarity_mismatches"]
        report["summary"]["precision_proxy"] = round(
            matched_or_polarity / report["summary"]["total_parser_items"] * 100, 2
        )
    
    # Convert defaultdict to regular dict
    report["domain_breakdown"] = dict(report["domain_breakdown"])
    
    return report


def print_report(report: Dict[str, Any], color: bool = True) -> None:
    """Print formatted report to console."""
    GREEN = "\033[92m" if color else ""
    YELLOW = "\033[93m" if color else ""
    RED = "\033[91m" if color else ""
    BOLD = "\033[1m" if color else ""
    RESET = "\033[0m" if color else ""
    
    summary = report["summary"]
    
    print(f"\n{'='*60}")
    print(f"{BOLD}  Canary Comparison Report: {report['version']}{RESET}")
    print(f"{'='*60}")
    
    print(f"\n{BOLD}  Overview:{RESET}")
    print(f"    • Canary journals: {summary['total_canary_journals']}")
    print(f"    • Journals with output: {summary['journals_with_parser_output']}")
    print(f"    • Gold items: {summary['total_gold_items']}")
    print(f"    • Parser items: {summary['total_parser_items']}")
    
    print(f"\n{BOLD}  Matching Results:{RESET}")
    matched_color = GREEN if summary['total_matched'] > 0 else YELLOW
    print(f"    {matched_color}✓ Matched: {summary['total_matched']}{RESET}")
    
    missing_color = RED if summary['total_missing'] > 0 else GREEN
    print(f"    {missing_color}✗ Missing (in gold, not in parser): {summary['total_missing']}{RESET}")
    
    extra_color = YELLOW if summary['total_extra'] > 0 else GREEN
    print(f"    {extra_color}⚠ Extra (in parser, not in gold): {summary['total_extra']}{RESET}")
    
    polarity_color = RED if summary['total_polarity_mismatches'] > 0 else GREEN
    print(f"    {polarity_color}✗ Polarity mismatches: {summary['total_polarity_mismatches']}{RESET}")
    
    halluc_color = RED if summary['total_hallucinated'] > 0 else GREEN
    print(f"    {halluc_color}✗ Hallucinated evidence: {summary['total_hallucinated']}{RESET}")
    
    print(f"\n{BOLD}  Rates:{RESET}")
    ev_rate = summary['evidence_validity_rate']
    ev_color = GREEN if ev_rate >= 95 else (YELLOW if ev_rate >= 80 else RED)
    print(f"    • Evidence Validity: {ev_color}{ev_rate}%{RESET}")
    
    recall = summary['recall_rate']
    recall_color = GREEN if recall >= 70 else (YELLOW if recall >= 50 else RED)
    print(f"    • Recall (gold items found): {recall_color}{recall}%{RESET}")
    
    precision = summary['precision_proxy']
    prec_color = GREEN if precision >= 70 else (YELLOW if precision >= 50 else RED)
    print(f"    • Precision proxy (parser items matching): {prec_color}{precision}%{RESET}")
    
    print(f"\n{BOLD}  Domain Breakdown:{RESET}")
    for domain, stats in report["domain_breakdown"].items():
        print(f"    {domain}: matched={stats['matched']}, missing={stats['missing']}, extra={stats['extra']}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare parser outputs against canary gold labels"
    )
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to data directory containing parser outputs and canary folder"
    )
    parser.add_argument(
        "--out", "-o",
        required=True,
        help="Output directory for comparison reports"
    )
    
    args = parser.parse_args()
    data_path = Path(args.data)
    out_path = Path(args.out)
    
    # Load data
    print(f"\n{'='*60}")
    print("  Loading Data for Comparison")
    print(f"{'='*60}\n")
    
    try:
        canary_journals = load_jsonl(data_path / "canary" / "journals.jsonl")
        gold_labels = load_jsonl(data_path / "canary" / "gold.jsonl")
        day0_outputs = load_jsonl(data_path / "parser_outputs_day0.jsonl")
        day1_outputs = load_jsonl(data_path / "parser_outputs_day1.jsonl")
        
        print(f"    • Canary journals: {len(canary_journals)}")
        print(f"    • Gold labels: {len(gold_labels)}")
        print(f"    • Day 0 outputs: {len(day0_outputs)}")
        print(f"    • Day 1 outputs: {len(day1_outputs)}")
    except FileNotFoundError as e:
        print(f"Error: Could not find file: {e}")
        return 1
    
    # Generate reports
    print(f"\n{'='*60}")
    print("  Generating Comparison Reports")
    print(f"{'='*60}")
    
    day0_report = generate_comparison_report(day0_outputs, canary_journals, gold_labels, "Day 0")
    day1_report = generate_comparison_report(day1_outputs, canary_journals, gold_labels, "Day 1")
    
    # Print reports
    print_report(day0_report)
    print_report(day1_report)
    
    # Generate diff report
    diff_report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "comparison": "Day 0 vs Day 1",
        "day0_summary": day0_report["summary"],
        "day1_summary": day1_report["summary"],
        "changes": {
            "matched_change": day1_report["summary"]["total_matched"] - day0_report["summary"]["total_matched"],
            "missing_change": day1_report["summary"]["total_missing"] - day0_report["summary"]["total_missing"],
            "extra_change": day1_report["summary"]["total_extra"] - day0_report["summary"]["total_extra"],
            "recall_change": round(day1_report["summary"]["recall_rate"] - day0_report["summary"]["recall_rate"], 2),
            "precision_change": round(day1_report["summary"]["precision_proxy"] - day0_report["summary"]["precision_proxy"], 2),
            "evidence_validity_change": round(day1_report["summary"]["evidence_validity_rate"] - day0_report["summary"]["evidence_validity_rate"], 2),
        }
    }
    
    # Print diff summary
    print(f"\n{'='*60}")
    print("  Day 0 → Day 1 Changes")
    print(f"{'='*60}\n")
    
    changes = diff_report["changes"]
    
    def format_change(val, higher_is_better=True):
        if val > 0:
            color = "\033[92m" if higher_is_better else "\033[91m"
            return f"{color}+{val}\033[0m"
        elif val < 0:
            color = "\033[91m" if higher_is_better else "\033[92m"
            return f"{color}{val}\033[0m"
        else:
            return f"{val}"
    
    print(f"    • Matched items: {format_change(changes['matched_change'])}")
    print(f"    • Missing items: {format_change(changes['missing_change'], higher_is_better=False)}")
    print(f"    • Extra items: {format_change(changes['extra_change'], higher_is_better=False)}")
    print(f"    • Recall: {format_change(changes['recall_change'])}pp")
    print(f"    • Precision proxy: {format_change(changes['precision_change'])}pp")
    print(f"    • Evidence validity: {format_change(changes['evidence_validity_change'])}pp")
    
    # Save reports
    out_path.mkdir(parents=True, exist_ok=True)
    save_json(out_path / "canary_comparison_day0.json", day0_report)
    save_json(out_path / "canary_comparison_day1.json", day1_report)
    save_json(out_path / "canary_comparison_diff.json", diff_report)
    
    print(f"\n{'='*60}")
    print("  Reports Saved")
    print(f"{'='*60}\n")
    print(f"    • {out_path}/canary_comparison_day0.json")
    print(f"    • {out_path}/canary_comparison_day1.json")
    print(f"    • {out_path}/canary_comparison_diff.json")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
