# Ashwam Production Monitor

## Overview

Production monitoring system for journal parser that detects drift and unsafe behavior **without ground truth labels**.

This system is designed for health journaling applications where:
- Users write free-form health journals (symptoms, food, emotions, mental state)
- An LLM parser extracts structured items from journals
- We need to validate parser behavior before production deployment
- We cannot rely on complete ground truth labels (open-ended extraction)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd ashwam_monitor

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# No external dependencies needed - uses Python standard library only
python -m ashwam_monitor --help
```

### Requirements
- Python 3.8+
- No external packages required (uses stdlib: json, pathlib, argparse, statistics, datetime, collections)

## Usage

### Run Full Monitoring Suite

```bash
python -m ashwam_monitor run --data ./data --out ./out
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | `./data` | Path to input data directory |
| `--out` | `./out` | Path to output reports directory |

### Run System Tests

```bash
python -m ashwam_monitor.test_system
```

## Data Structure

### Expected Input Format

```
data/
├── journals.jsonl              # Raw journal texts
├── parser_outputs_day0.jsonl   # Baseline parser outputs (previous version)
├── parser_outputs_day1.jsonl   # Current parser outputs (new version)
└── canary/
    ├── journals.jsonl          # Small labeled test set journals
    └── gold.jsonl              # Gold labels for canary journals
```

### File Formats

#### journals.jsonl
Each line is a JSON object with the original journal text:
```json
{"id": "journal_001", "text": "I had a headache this morning and felt very tired..."}
```

#### parser_outputs_dayX.jsonl
Each line contains extracted items for one journal:
```json
{
  "journal_id": "journal_001",
  "items": [
    {
      "domain": "symptom",
      "item_name": "headache",
      "polarity": "present",
      "intensity": "medium",
      "evidence_span": "had a headache"
    }
  ]
}
```

#### Item Schema

| Field | Required | Valid Values | Description |
|-------|----------|--------------|-------------|
| `domain` | Yes | `symptom`, `food`, `emotion`, `mind` | Category of extraction |
| `item_name` | Yes | Any string | Name/label of the item |
| `polarity` | Yes | `present`, `absent`, `unknown` | Whether item is present or absent |
| `intensity` | No | `low`, `medium`, `high`, `unknown` | Severity/intensity level |
| `evidence_span` | Yes | Any string | Verbatim quote from journal |

## Output Reports

The monitor generates three JSON reports:

### 1. invariant_report.json

Hard rule checks that must pass for safe deployment.

```json
{
  "timestamp": "2025-12-31T12:00:00Z",
  "overall_status": "PASS|FAIL",
  "final_recommended_action": "deploy|rollback",
  "checks": {
    "schema_validity": {...},
    "evidence_validity": {...},
    "hallucination_rate": {...},
    "contradiction_rate": {...}
  }
}
```

| Check | Threshold | Failure Meaning |
|-------|-----------|-----------------|
| Schema Validity | 100% | Missing required fields or invalid enum values |
| Evidence Validity | ≥95% | Evidence not found verbatim in journal |
| Hallucination Rate | ≤5% | Fabricated evidence spans |
| Contradiction Rate | 0% | Same evidence with conflicting polarity |

### 2. drift_report.json

Day 0 vs Day 1 comparison metrics.

```json
{
  "timestamp": "2025-12-31T12:00:00Z",
  "overall_drift_status": "none|minor|moderate|severe",
  "recommended_action": "deploy|human_review|rollback",
  "metrics": {
    "extraction_volume": {...},
    "uncertainty_rate": {...},
    "intensity_distribution": {...},
    "domain_mix": {...}
  },
  "drift_flags": [...]
}
```

| Metric | Threshold | Concern |
|--------|-----------|---------|
| Extraction Volume | >20% change | Parser sensitivity shift |
| Uncertainty Rate | >15pp change | Confidence degradation |
| High Intensity | >25pp change | Severity calibration drift |
| Domain Mix | >30pp change | Categorization shift |

### 3. canary_report.json

Evaluation against small labeled test set.

```json
{
  "timestamp": "2025-12-31T12:00:00Z",
  "alert_level": "PASS|HUMAN_REVIEW|ROLLBACK",
  "metrics": {
    "evidence_validity": 100.0,
    "polarity_correctness": 95.0,
    "contradiction_count": 0
  }
}
```

## Monitoring Philosophy

### Safety-First
Health data requires absolute correctness. We prioritize **false positives** (flagging safe parsers) over **false negatives** (missing unsafe parsers). A rollback recommendation is conservative but protects users.

### Evidence-Grounded
Every extracted item must have evidence that exists **verbatim** in the source journal. Hallucinated evidence is unacceptable - we cannot attribute symptoms, foods, or emotions to users based on fabricated text.

### No Ground Truth
Traditional ML metrics (precision, recall, F1) require complete ground truth labels. Health journals are open-ended - we cannot enumerate all possible extractions. Instead, we use:
- **Invariant checks**: Rules that must always hold (schema, no hallucinations, no contradictions)
- **Drift detection**: Relative comparison to baseline behavior
- **Canary tests**: Small labeled set for sanity checks

### Human-in-Loop
The system recommends one of three actions:
- `deploy`: All checks passed, safe for production
- `human_review`: Some concerns, needs manual verification
- `rollback`: Critical issues, do not deploy

## Decision Thresholds

### Why These Specific Thresholds?

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| Schema Validity | 100% | Any invalid data breaks downstream systems (analytics, storage, recommendations) |
| Evidence Validity | ≥95% | Small tolerance for formatting differences; systematic issues are caught |
| Hallucination Rate | ≤5% | Fabricated health data is dangerous; strict limit protects users |
| Contradictions | 0% | Logically impossible for same evidence to mean opposite things |
| Extraction Volume | ±20% | Significant sensitivity change affects user experience |
| Uncertainty Rate | ±15pp | Rising uncertainty indicates parser confusion |
| High Intensity | ±25pp | Over/under-reporting severity affects user perception |
| Domain Mix | ±30pp | Large shifts indicate categorization problems |
| Polarity Correctness | ≥90% | For matched items, high agreement expected |

### Health Data Safety Rationale

1. **Hallucinated symptoms** could cause unnecessary anxiety or missed real symptoms
2. **Wrong polarity** (present vs absent) inverts the meaning of health signals
3. **Over-reporting high intensity** may cause users to overreact to minor issues
4. **Under-reporting symptoms** may cause users to dismiss serious conditions
5. **Domain misclassification** affects which health insights are surfaced

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All checks passed - safe to deploy |
| 1 | Some concerns - human review recommended |
| 2 | Critical issues - rollback recommended |

## Example Output

```
============================================================
  ASHWAM Parser Monitor
============================================================

  Timestamp: 2025-12-31T12:00:00Z
  Data path: ./data
  Output path: ./out

============================================================
  Invariant Checks
============================================================

  ✓ Schema Validity: PASS - 100.0% valid (150/150)
  ✓ Evidence Validity: PASS - 98.5% valid
  ✓ Hallucination Rate: PASS - 1.5% hallucinated
  ✓ Contradictions: PASS - 0 found

  Invariants Overall: PASS
  Recommendation: deploy

============================================================
  Final Summary
============================================================

  Critical Issues: 0
  Warnings: 0

  ──────────────────────────────────────────────────────────
  FINAL RECOMMENDATION: DEPLOY
  ──────────────────────────────────────────────────────────

  All checks passed. Parser is safe to deploy.
```

## Contributing

1. Run tests before submitting changes: `python -m ashwam_monitor.test_system`
2. Ensure all checks pass on test data
3. Document any new thresholds with safety rationale

## License

[Your License Here]
