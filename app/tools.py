# app/tools.py
"""Google Docs API v1 structural parser, fuzzy range matcher, and batch mutation tools."""
from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Tuple

import google.auth
from google.adk.tools import ToolContext
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive.readonly",
]

# Deterministic pre-scan regex patterns for secrets, insecure IAM, and risky cloud configs
_SECRET_AND_IAM_PATTERNS: List[Tuple[str, str, re.Pattern[str]]] = [
    (
        "HIGH",
        "Potential exposed Google API key (`AIza...`) in document text",
        re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
    ),
    (
        "HIGH",
        "Potential exposed Google OAuth client secret (`GOCSPX-...`) in document text",
        re.compile(r"\bGOCSPX-[0-9A-Za-z\-_]{20,}\b"),
    ),
    (
        "HIGH",
        "Potential exposed GitHub Personal Access Token (`ghp_` / `gho_`) in document text",
        re.compile(r"\b(?:ghp|gho|ghs|ghu)_[0-9A-Za-z]{30,}\b"),
    ),
    (
        "HIGH",
        "Private cryptographic key block detected in document text",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "MEDIUM",
        "Overly permissive primitive IAM role (`roles/owner` or `roles/editor`) referenced",
        re.compile(r"\broles/(?:owner|editor)\b"),
    ),
    (
        "MEDIUM",
        "Public IAM binding (`allUsers` or `allAuthenticatedUsers`) referenced",
        re.compile(r"\b(?:allUsers|allAuthenticatedUsers)\b"),
    ),
]


def _get_docs_service(tool_context: ToolContext):
    """Builds the Google Docs API client using the logged-in user's OAuth token (or ADC fallback)."""
    user_token = tool_context.state.get("user_access_token")
    if user_token:
        creds = Credentials(token=user_token)
        return build("docs", "v1", credentials=creds)
    creds, _ = google.auth.default(scopes=SCOPES)
    return build("docs", "v1", credentials=creds)


def _extract_text_and_index_map(doc: Dict[str, Any]) -> Tuple[str, List[int]]:
    """Extracts plain text and maps each character to its exact 1-based Docs API index."""
    tabs = doc.get("tabs", [])
    if tabs:
        body = tabs[0].get("documentTab", {}).get("body", {})
    else:
        body = doc.get("body", {})

    chars: List[str] = []
    doc_indices: List[int] = []

    def _walk_elements(content_list: List[Dict[str, Any]]):
        for element in content_list:
            if "paragraph" in element:
                for run in element["paragraph"].get("elements", []):
                    text_run = run.get("textRun")
                    if text_run and "content" in text_run:
                        text = text_run["content"]
                        start_idx = run.get("startIndex", 1)
                        for offset, ch in enumerate(text):
                            chars.append(ch)
                            doc_indices.append(start_idx + offset)
            elif "table" in element:
                for row in element["table"].get("tableRows", []):
                    for cell in row.get("tableCells", []):
                        _walk_elements(cell.get("content", []))

    _walk_elements(body.get("content", []))
    return "".join(chars), doc_indices


def compute_document_diagnostics(full_text: str) -> Dict[str, Any]:
    """Computes readability metrics, structural counts, and deterministic security pre-scan alerts."""
    clean_text = (full_text or "").strip()
    if not clean_text:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "flesch_reading_ease": 0.0,
            "estimated_reading_minutes": 0.0,
            "deterministic_alerts": [],
        }

    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", clean_text)
    word_count = len(words)
    sentences = [s for s in re.split(r"[.!?]+", clean_text) if s.strip()]
    sentence_count = max(1, len(sentences))
    paragraphs = [p for p in clean_text.split("\n") if p.strip()]
    paragraph_count = len(paragraphs)

    # Approximate syllable count for Flesch Reading Ease
    def _count_syllables(word: str) -> int:
        w = re.sub(r"[^a-z]", "", word.lower())
        if not w:
            return 1
        if len(w) <= 3:
            return 1
        w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
        w = re.sub(r"^y", "", w)
        vowel_groups = re.findall(r"[aeiouy]{1,2}", w)
        return max(1, len(vowel_groups))

    syllable_count = sum(_count_syllables(w) for w in words) if word_count else 0
    if word_count > 0:
        asl = word_count / sentence_count
        asw = syllable_count / word_count
        flesch = round(max(0.0, min(100.0, 206.835 - 1.015 * asl - 84.6 * asw)), 1)
    else:
        flesch = 0.0

    alerts: List[Dict[str, str]] = []
    for severity, message, pattern in _SECRET_AND_IAM_PATTERNS:
        for match in pattern.finditer(clean_text):
            matched_str = match.group(0)
            alerts.append(
                {
                    "severity": severity,
                    "description": message,
                    "matched_text": matched_str,
                }
            )

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "flesch_reading_ease": flesch,
        "estimated_reading_minutes": round(max(0.5, word_count / 220.0), 1),
        "deterministic_alerts": alerts,
    }


def _normalize_smart_quotes(text: str) -> str:
    """Normalizes curly quotes and dashes to ASCII equivalents for resilient matching."""
    return (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )


def _find_doc_range(
    full_text: str, doc_indices: List[int], target_text: str
) -> Tuple[int, int] | None:
    """Finds `target_text` and returns its exact (startIndex, endIndex) in the Doc.

    Uses a 3-tier matching strategy:
      1. Exact verbatim substring match.
      2. Smart-quote / non-breaking-space normalized match (length-preserving).
      3. Whitespace-tolerant regex / fuzzy paragraph sliding-window fallback.
    """
    if not target_text or not full_text or not doc_indices:
        return None

    # Tier 1: Exact substring match
    pos = full_text.find(target_text)
    if pos != -1:
        start_index = doc_indices[pos]
        end_index = doc_indices[pos + len(target_text) - 1] + 1
        return start_index, end_index

    # Tier 2: Smart-quote & NBSP normalized match (1:1 character length preserved)
    norm_full = _normalize_smart_quotes(full_text)
    norm_target = _normalize_smart_quotes(target_text).strip()
    if norm_target:
        pos = norm_full.find(norm_target)
        if pos != -1:
            start_index = doc_indices[pos]
            end_index = doc_indices[pos + len(norm_target) - 1] + 1
            return start_index, end_index

        # Case-insensitive fallback on normalized text
        pos_ci = norm_full.lower().find(norm_target.lower())
        if pos_ci != -1:
            start_index = doc_indices[pos_ci]
            end_index = doc_indices[pos_ci + len(norm_target) - 1] + 1
            return start_index, end_index

    # Tier 3: Whitespace-flexible regex match within a single paragraph
    tokens = [re.escape(tok) for tok in norm_target.split() if tok]
    if len(tokens) >= 2:
        ws_pattern = re.compile(r"[ \t]+".join(tokens))
        match = ws_pattern.search(norm_full)
        if match:
            m_start, m_end = match.span()
            if "\n" not in full_text[m_start:m_end]:
                return doc_indices[m_start], doc_indices[m_end - 1] + 1

    # Tier 4: Fuzzy SequenceMatcher fallback for longer single-paragraph phrases
    if len(norm_target) >= 18 and "\n" not in norm_target:
        matcher = difflib.SequenceMatcher(None, norm_full, norm_target, autojunk=False)
        block = matcher.find_longest_match(0, len(norm_full), 0, len(norm_target))
        if block.size / len(norm_target) >= 0.88:
            m_start = block.a
            m_end = block.a + block.size
            if "\n" not in full_text[m_start:m_end]:
                return doc_indices[m_start], doc_indices[m_end - 1] + 1

    return None


def read_google_doc(document_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Reads a Google Document and returns its title, full text, and structural diagnostics.

    Args:
        document_id: The Google Document ID.

    Returns:
        dict with 'status', 'title', 'full_text', and 'diagnostics'.
    """
    try:
        service = _get_docs_service(tool_context)
        doc = (
            service.documents()
            .get(
                documentId=document_id,
                suggestionsViewMode="SUGGESTIONS_INLINE",
                commentsViewMode="COMMENTS_VIEW_MODE_INCLUDED",
                includeTabsContent=True,
            )
            .execute()
        )
        full_text, _ = _extract_text_and_index_map(doc)
        diagnostics = compute_document_diagnostics(full_text)
        tool_context.state["doc_diagnostics"] = diagnostics
        return {
            "status": "success",
            "title": doc.get("title", "Untitled"),
            "full_text": full_text,
            "diagnostics": diagnostics,
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def insert_comments(
    document_id: str, comments: List[Dict[str, str]], tool_context: ToolContext
) -> Dict[str, Any]:
    """Attaches editorial comments to specific phrases in a Google Doc.

    Args:
        document_id: The Google Document ID.
        comments: A list of dicts, each with:
            - 'target_text': Exact phrase or sentence in the document to highlight.
            - 'comment_text': The feedback or question to attach as a comment.

    Returns:
        dict with status and number of comments inserted.
    """
    try:
        service = _get_docs_service(tool_context)
        doc = (
            service.documents()
            .get(
                documentId=document_id,
                suggestionsViewMode="SUGGESTIONS_INLINE",
                includeTabsContent=True,
            )
            .execute()
        )
        full_text, doc_indices = _extract_text_and_index_map(doc)

        requests = []
        skipped = []
        for item in comments:
            target = item.get("target_text", "")
            comment_body = item.get("comment_text", "")
            rng = _find_doc_range(full_text, doc_indices, target)
            if rng:
                requests.append(
                    {
                        "insertComment": {
                            "content": comment_body,
                            "range": {"startIndex": rng[0], "endIndex": rng[1]},
                        }
                    }
                )
            else:
                skipped.append(target)

        if not requests:
            return {
                "status": "warning",
                "message": "No matching text ranges found for comments.",
                "skipped_targets": skipped,
            }

        response = (
            service.documents()
            .batchUpdate(documentId=document_id, body={"requests": requests})
            .execute()
        )

        prev_count = int(tool_context.state.get("inserted_comments_count", 0))
        tool_context.state["inserted_comments_count"] = prev_count + len(requests)

        return {
            "status": "success",
            "inserted_comments_count": len(requests),
            "skipped_targets": skipped,
            "comment_update_state": response.get(
                "commentUpdateState", "ALL_UPDATES_APPLIED"
            ),
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def propose_suggested_edits(
    document_id: str, edits: List[Dict[str, str]], tool_context: ToolContext
) -> Dict[str, Any]:
    """Submits spelling, grammar, security, or fact-check fixes as Suggested Edits in Google Docs.

    Args:
        document_id: The Google Document ID.
        edits: A list of dicts, each with:
            - 'target_text': Exact word or short phrase in the document to replace.
            - 'replacement_text': The corrected word or phrase.

    Returns:
        dict with status and number of suggestions applied.
    """
    try:
        service = _get_docs_service(tool_context)
        doc = (
            service.documents()
            .get(
                documentId=document_id,
                suggestionsViewMode="SUGGESTIONS_INLINE",
                includeTabsContent=True,
            )
            .execute()
        )
        full_text, doc_indices = _extract_text_and_index_map(doc)

        resolved_edits = []
        skipped = []
        for edit in edits:
            target = edit.get("target_text", "")
            replacement = edit.get("replacement_text", "")
            if not target or target == replacement:
                continue
            rng = _find_doc_range(full_text, doc_indices, target)
            if rng:
                # Never allow a deleteContentRange to cross a trailing newline boundary
                start_i, end_i = rng
                while end_i > start_i and full_text[doc_indices.index(end_i - 1)] == "\n":
                    end_i -= 1
                if end_i > start_i:
                    resolved_edits.append(
                        {
                            "startIndex": start_i,
                            "endIndex": end_i,
                            "replacement": replacement,
                            "target": target,
                        }
                    )
            else:
                skipped.append(target)

        if not resolved_edits:
            return {
                "status": "warning",
                "message": "No matching target text found for edits.",
                "skipped_targets": skipped,
            }

        # Sort descending by startIndex and filter out overlapping ranges
        resolved_edits.sort(key=lambda x: x["startIndex"], reverse=True)
        non_overlapping = []
        last_start = float("inf")
        for item in resolved_edits:
            if item["endIndex"] <= last_start:
                non_overlapping.append(item)
                last_start = item["startIndex"]
            else:
                skipped.append(item["target"])

        requests = []
        for item in non_overlapping:
            requests.append(
                {
                    "deleteContentRange": {
                        "range": {
                            "startIndex": item["startIndex"],
                            "endIndex": item["endIndex"],
                        }
                    }
                }
            )
            requests.append(
                {
                    "insertText": {
                        "location": {"index": item["startIndex"]},
                        "text": item["replacement"],
                    }
                }
            )

        response = (
            service.documents()
            .batchUpdate(
                documentId=document_id,
                body={
                    "requests": requests,
                    "writeControl": {"writeMode": "SUGGEST"},
                },
            )
            .execute()
        )

        prev_edits = int(tool_context.state.get("applied_suggestions_count", 0))
        tool_context.state["applied_suggestions_count"] = prev_edits + len(
            non_overlapping
        )

        return {
            "status": "success",
            "applied_suggestions_count": len(non_overlapping),
            "skipped_targets": skipped,
            "comment_update_state": response.get(
                "commentUpdateState", "ALL_UPDATES_APPLIED"
            ),
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}
