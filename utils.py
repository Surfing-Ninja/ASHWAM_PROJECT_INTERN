"""
Helper functions for loading JSONL files and text matching.

This module provides utility functions for:
- Loading and parsing JSONL files
- Text matching and comparison utilities

NOTE: This module uses only Python standard library.
No external ML libraries or model training required.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file and return a list of dictionaries.

    WHAT: Reads a JSON Lines file where each line is a separate JSON object.
    WHY: Standard format for streaming health journal data and parser outputs.
    ON FAILURE: Raises FileNotFoundError (missing file) or JSONDecodeError (malformed JSON).

    Args:
        filepath: Path to the .jsonl file to load.

    Returns:
        List of dictionaries, one per line in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If any line contains invalid JSON.
    """
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                records.append(json.loads(line))
    return records


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save a dictionary as a pretty-printed JSON file.

    WHAT: Writes a dictionary to a JSON file with readable formatting.
    WHY: Produces human-readable reports for operator review.
    ON FAILURE: Creates parent directories if needed; raises PermissionError if write fails.

    Args:
        data: Dictionary to save.
        filepath: Path where the JSON file will be saved.

    Raises:
        PermissionError: If the file cannot be written.
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def find_text_in_journal(text: str, journal: str) -> bool:
    """
    Check if text appears in journal (case-insensitive).

    WHAT: Verifies that an evidence span exists verbatim in the source journal.
    WHY: Critical for detecting hallucinations - parser claims must be grounded in actual text.
    ON FAILURE: Returns False, which triggers hallucination detection upstream.

    Args:
        text: The text to search for (evidence_span from parser output).
        journal: The journal content to search in (original user text).

    Returns:
        True if text is found in journal, False otherwise.
    """
    if not text or not journal:
        return False
    return text.lower() in journal.lower()


def calculate_percentage(count: int, total: int) -> float:
    """
    Calculate percentage of count relative to total.

    WHAT: Safe division that handles zero denominators.
    WHY: Many metrics are expressed as percentages; zero-division crashes are unacceptable.
    ON FAILURE: Returns 0.0 for zero total (safe default).

    Args:
        count: The numerator value.
        total: The denominator value.

    Returns:
        Percentage rounded to 2 decimal places, or 0.0 if total is 0.
    """
    if total == 0:
        return 0.0
    return round((count / total) * 100, 2)
