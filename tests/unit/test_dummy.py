# tests/unit/test_dummy.py
"""Unit tests for DocGuard AI Studio tools, fuzzy range matcher, and diagnostics."""

from app.addons.review_board import (
    _extract_doc_id,
    ai_systems_reviewer,
    copy_editor_reviewer,
    fact_checker_reviewer,
    review_board_agent,
    security_reviewer,
)
from app.fast_api_app import REVIEW_PRESETS, _normalize_doc_id
from app.tools import (
    _extract_text_and_index_map,
    _find_doc_range,
    compute_document_diagnostics,
)


def test_extract_text_and_fuzzy_range_matching() -> None:
    """Verifies exact, smart-quote normalized, and whitespace-tolerant range resolution."""
    sample_doc = {
        "title": "Cloud Run RFC",
        "body": {
            "content": [
                {
                    "paragraph": {
                        "elements": [
                            {
                                "startIndex": 1,
                                "textRun": {
                                    "content": "Cloud Run supports\u00a0“up to 1000 instances” by default.\n"
                                },
                            }
                        ]
                    }
                }
            ]
        },
    }
    full_text, indices = _extract_text_and_index_map(sample_doc)
    assert "Cloud Run" in full_text
    assert len(full_text) == len(indices)

    # Exact match
    rng_exact = _find_doc_range(full_text, indices, "Cloud Run")
    assert rng_exact == (1, 10)

    # Smart-quote and non-breaking-space normalized match
    rng_smart = _find_doc_range(full_text, indices, 'supports "up to 1000 instances"')
    assert rng_smart is not None
    assert rng_smart[0] == 11


def test_compute_document_diagnostics_and_secret_scan() -> None:
    """Verifies readability calculation and deterministic secret/IAM regex detection."""
    text = (
        "We configured roles/owner for the service account and hardcoded "
        "GOCSPX-1234567890abcdefghij12345 in the config file. "
        "We also enabled allUsers on the internal endpoint."
    )
    diag = compute_document_diagnostics(text)
    assert diag["word_count"] > 15
    assert 0.0 <= diag["flesch_reading_ease"] <= 100.0
    alerts = diag["deterministic_alerts"]
    assert len(alerts) >= 3
    severities = {a["severity"] for a in alerts}
    assert "HIGH" in severities
    assert "MEDIUM" in severities


def test_url_and_doc_id_normalization() -> None:
    """Verifies extracting clean Google Doc IDs from full URLs and raw strings."""
    url = "https://docs.google.com/document/d/1WvV8RJGqJ-h2VbmB41xvJuRM1ibU-MBkMsBulkSaAN/edit?tab=t.0"
    assert _normalize_doc_id(url) == "1WvV8RJGqJ-h2VbmB41xvJuRM1ibU-MBkMsBulkSaAN"
    assert _extract_doc_id(url) == "1WvV8RJGqJ-h2VbmB41xvJuRM1ibU-MBkMsBulkSaAN"


def test_review_board_has_four_specialists_and_presets() -> None:
    """Verifies the 4-persona ADK 2.0 Graph Workflow and domain presets."""
    assert security_reviewer.name == "security_reviewer"
    assert fact_checker_reviewer.name == "fact_checker_reviewer"
    assert ai_systems_reviewer.name == "ai_systems_reviewer"
    assert copy_editor_reviewer.name == "copy_editor_reviewer"
    assert review_board_agent.name == "multi_persona_review_board"
    assert "gcp_rfc" in REVIEW_PRESETS
    assert "ai_agent_spec" in REVIEW_PRESETS
