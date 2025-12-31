# Design Decisions

This document explains the key design decisions behind the Ashwam Production Monitor, particularly the unconventional choices that differ from standard ML evaluation practices.

## Why We Don't Use Precision/Recall in Canary Evaluation

### The Standard Approach (and Why It Doesn't Work Here)

Traditional information extraction evaluation uses:
- **Precision**: Of items extracted, how many are correct?
- **Recall**: Of items that should be extracted, how many did we get?
- **F1 Score**: Harmonic mean of precision and recall

This requires a **closed label set** - a complete enumeration of all items that should be extracted from each document.

### The Health Journal Problem

Health journals are **open-ended**. Consider this journal:

> "Woke up with a headache. Had coffee and toast for breakfast. Feeling anxious about my presentation. The weather is nice today."

What "should" be extracted?

| Possible Extraction | Valid? |
|---------------------|--------|
| headache (symptom, present) | ✓ Yes |
| coffee (food, present) | ✓ Yes |
| toast (food, present) | ✓ Yes |
| anxiety (emotion, present) | ✓ Yes |
| presentation stress (mind, present) | ✓ Maybe? |
| good weather (emotion, positive) | ? Unclear |
| morning (time context) | ? Not in schema |

There is no "correct" answer for what the complete extraction should be. Different annotators would produce different gold sets. This makes recall **undefined**.

### Our Solution: Verify What Was Extracted

Instead of measuring what's missing, we verify what's present:

1. **Evidence Validity (100% required)**
   - Every `evidence_span` must exist verbatim in the journal
   - This catches hallucinations without needing complete labels
   - If the parser claims "scrambled eggs" but the journal says "toast", that's a failure

2. **Polarity Correctness (90% required)**
   - For items that **match** between predicted and gold, check polarity agreement
   - Only evaluates overlapping items, not missing ones
   - A parser that extracts fewer items isn't penalized (no recall)

3. **Contradiction Detection (0 tolerance)**
   - Same evidence cannot have conflicting polarities
   - This is a logical invariant, not a label comparison

### Why This Works for Health Data

- **False negatives are acceptable**: Missing an extraction is not dangerous
- **False positives are dangerous**: Fabricating health data is harmful
- **Consistency matters**: Users expect stable behavior

---

## Why Contradictions Are Zero-Tolerance

### The Logical Impossibility

A contradiction occurs when the same piece of evidence is used to claim opposite things:

```json
{"evidence_span": "I had a headache", "polarity": "present"}
{"evidence_span": "I had a headache", "polarity": "absent"}
```

This is **logically impossible**. The text "I had a headache" cannot simultaneously mean the user had AND didn't have a headache.

### Why Any Contradiction Is a Critical Failure

1. **Indicates Fundamental Parser Bug**
   - A well-functioning parser should never produce contradictions
   - If it happens once, the logic is broken and may affect other extractions

2. **Destroys User Trust**
   - If users see contradictory information, they lose confidence in the system
   - Health apps require high trust to be useful

3. **Unsafe for Health Recommendations**
   - Downstream systems may use extracted data for recommendations
   - Contradictory data could lead to conflicting health advice

4. **No Acceptable Rate**
   - Unlike other errors where 95% correct is good, contradictions are binary
   - Even 0.1% contradiction rate means the parser is unreliable

### Implementation Note

We detect contradictions by:
1. Grouping items by `evidence_span` (case-insensitive)
2. Checking if any group contains both `present`/`positive` AND `absent`/`negative`
3. Flagging any non-empty contradiction set

---

## How Drift Thresholds Were Chosen

### Philosophy: Detect Anomalies, Not Normal Variance

Parser outputs naturally vary somewhat between runs. Our thresholds are calibrated to:
- **Ignore normal variance** (small day-to-day differences)
- **Flag anomalies** (significant behavior changes)
- **Prioritize safety** (err on the side of caution for health data)

### Extraction Volume: >20% Relative Change

**Why 20%?**
- Parser sensitivity affects user experience significantly
- Over-extraction (>20% more): Creates noise, overwhelms users
- Under-extraction (<20% less): Misses health signals, reduces value

**Example:**
- Day 0: 5 items/journal average
- Day 1: 6.5 items/journal (+30%) → FLAG
- Day 1: 5.5 items/journal (+10%) → OK

**Severity tiers:**
- 20-35%: Minor drift
- 35-50%: Moderate drift
- >50%: Severe drift

### Uncertainty Rate: >15 Percentage Points Change

**Why 15pp?**
- Uncertainty (polarity="unknown") indicates parser confidence
- Rising uncertainty suggests prompt degradation or model confusion
- 15pp is a significant shift in behavior

**Example:**
- Day 0: 5% uncertain items
- Day 1: 25% uncertain (+20pp) → FLAG
- Day 1: 12% uncertain (+7pp) → OK

**Health Impact:**
- High uncertainty means less actionable health insights
- Users expect definitive parsing of what they wrote

### High Intensity: >25 Percentage Points Change

**Why 25pp?**
- Intensity calibration affects user perception of severity
- Over-reporting "high" intensity causes unnecessary anxiety
- Under-reporting may cause users to dismiss serious symptoms

**Example:**
- Day 0: 10% high intensity items
- Day 1: 40% high intensity (+30pp) → FLAG
- Day 1: 20% high intensity (+10pp) → OK

**Health Impact:**
- A user seeing "HIGH INTENSITY" for minor symptoms may panic
- A user seeing "LOW INTENSITY" for serious symptoms may not seek care

### Domain Mix: >30 Percentage Points per Domain

**Why 30pp?**
- Domain distribution reflects what the parser focuses on
- Large shifts indicate categorization problems
- Different domains have different downstream uses

**Example:**
- Day 0: 40% symptom, 30% food, 20% emotion, 10% mind
- Day 1: 20% symptom (-20pp), 30% food, 40% emotion (+20pp), 10% mind → OK
- Day 1: 10% symptom (-30pp), 30% food, 50% emotion (+30pp), 10% mind → FLAG

**Domain-Specific Risks:**
| Domain | Under-detection Risk | Over-detection Risk |
|--------|---------------------|---------------------|
| Symptom | Missing physical health signals | False symptom alarms |
| Food | Incomplete nutrition tracking | Noise in diet logs |
| Emotion | Missing mental health signals | Over-pathologizing |
| Mind | Missing cognitive patterns | Privacy concerns |

---

## Threshold Summary Table

| Metric | Threshold | Type | Rationale |
|--------|-----------|------|-----------|
| Schema Validity | 100% | Absolute | Data integrity for downstream systems |
| Evidence Validity | ≥95% | Percentage | Allow minor formatting variance |
| Hallucination Rate | ≤5% | Percentage | Strict limit on fabrication |
| Contradictions | 0 | Absolute | Logical impossibility |
| Extraction Volume | ±20% | Relative | Parser sensitivity |
| Uncertainty Rate | ±15pp | Absolute | Confidence stability |
| High Intensity | ±25pp | Absolute | Severity calibration |
| Domain Mix | ±30pp each | Absolute | Categorization stability |
| Polarity Correctness | ≥90% | Percentage | Label agreement |

---

## Future Considerations

### Adaptive Thresholds
- Current thresholds are fixed based on health data safety
- Future: Could adapt based on historical variance per dataset

### Confidence-Weighted Metrics
- Not all extractions are equally important
- Future: Weight by domain or user-defined priority

### Time-Series Drift
- Current: Compare only Day 0 vs Day 1
- Future: Track drift over multiple days, detect trends

### User-Specific Baselines
- Current: Global thresholds for all users
- Future: Personalized thresholds based on user journal patterns
