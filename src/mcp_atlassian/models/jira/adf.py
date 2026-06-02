"""
Atlassian Document Format (ADF) utilities.

This module provides utilities for converting between ADF and other formats.
Supports both ADF → plain text (for reading) and Markdown → ADF (for writing).

The Markdown → ADF conversion is backed by ``markdown-it-py`` (a CommonMark
compliant tokenizer) rather than hand-rolled regexes. This makes the converter
robust against the classic pitfalls of naive parsers, notably:

* ``**bold**`` / ``*italic*`` / ``_italic_`` / ``~~strike~~`` / ``` `code` ```
  inline marks (including nesting such as ``***both***``).
* Intra-word underscores (``tsm_modems``, ``platform_bnm_prod.yml``, ``a_b_c``)
  stay literal — CommonMark forbids emphasis inside a word.
* Inline code spans are emitted verbatim (no inner mark parsing/escaping).
* Consecutive non-blank lines are grouped into a single paragraph; blank lines
  separate paragraphs.
* Fenced code blocks, ATX headings, bullet/ordered lists, blockquotes and GFM
  tables map to the corresponding ADF nodes.
"""

from datetime import datetime, timezone
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token

# A single shared, CommonMark-compliant parser instance with GFM tables and
# strikethrough enabled. CommonMark's emphasis rules already forbid intra-word
# ``_`` emphasis, which is exactly the behaviour we want for identifiers.
_MD = MarkdownIt("commonmark").enable("table").enable("strikethrough")

# Maps markdown-it inline open/close tag -> ADF mark spec (without attrs).
_MARK_FOR_TAG: dict[str, dict[str, Any]] = {
    "strong": {"type": "strong"},
    "em": {"type": "em"},
    "s": {"type": "strike"},
}


def _apply_marks(text: str, marks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an ADF text node, attaching the active marks (if any)."""
    node: dict[str, Any] = {"type": "text", "text": text}
    if marks:
        # Copy so callers mutating their stack don't corrupt emitted nodes.
        node["marks"] = [dict(m) for m in marks]
    return node


def _inline_tokens_to_adf(tokens: list[Token]) -> list[dict[str, Any]]:
    """Convert the children of a markdown-it ``inline`` token to ADF nodes.

    Walks the flat open/close token stream maintaining a stack of active marks.
    """
    nodes: list[dict[str, Any]] = []
    marks_stack: list[dict[str, Any]] = []

    for tok in tokens:
        ttype = tok.type

        if ttype == "text":
            if tok.content:
                nodes.append(_apply_marks(tok.content, marks_stack))
        elif ttype == "code_inline":
            # Inline code is verbatim: a code mark, no inner parsing.
            node: dict[str, Any] = {
                "type": "text",
                "text": tok.content,
                "marks": [dict(m) for m in marks_stack] + [{"type": "code"}],
            }
            nodes.append(node)
        elif ttype in ("softbreak", "hardbreak"):
            nodes.append({"type": "hardBreak"})
        elif ttype in ("strong_open", "em_open", "s_open"):
            mark = _MARK_FOR_TAG.get(tok.tag)
            if mark is not None:
                marks_stack.append(mark)
        elif ttype in ("strong_close", "em_close", "s_close"):
            if marks_stack:
                marks_stack.pop()
        elif ttype == "link_open":
            href = tok.attrs.get("href", "") if tok.attrs else ""
            marks_stack.append({"type": "link", "attrs": {"href": href}})
        elif ttype == "link_close":
            if marks_stack:
                marks_stack.pop()
        elif ttype == "image":
            # ADF has no inline image mark here; fall back to alt text.
            alt = tok.content or (tok.attrs.get("alt", "") if tok.attrs else "")
            if alt:
                nodes.append(_apply_marks(alt, marks_stack))
        # Any other inline token type is ignored gracefully.

    return nodes


def _parse_inline_formatting(text: str) -> list[dict[str, Any]]:
    """Parse inline Markdown formatting into ADF inline nodes.

    Handles: bold (``**``), italic (``*`` / ``_``), inline code (`` ` ``),
    links (``[text](url)``) and strikethrough (``~~``), including reasonable
    nesting. Intra-word underscores are left literal.

    Args:
        text: Raw text potentially containing inline Markdown formatting.

    Returns:
        List of ADF inline nodes (text nodes with optional marks).
    """
    if not text:
        return []

    # ``parseInline`` yields a single top-level "inline" token whose children
    # are the actual inline nodes.
    tokens = _MD.parseInline(text)
    nodes: list[dict[str, Any]] = []
    for tok in tokens:
        if tok.type == "inline" and tok.children:
            nodes.extend(_inline_tokens_to_adf(tok.children))

    if not nodes:
        nodes.append({"type": "text", "text": text})

    return nodes


def _make_paragraph(text: str) -> dict[str, Any]:
    """Create an ADF paragraph node from text with inline formatting."""
    content = _parse_inline_formatting(text)
    if not content:
        content = [{"type": "text", "text": ""}]
    return {"type": "paragraph", "content": content}


def _make_list_item(text: str) -> dict[str, Any]:
    """Create an ADF listItem node wrapping a paragraph."""
    return {"type": "listItem", "content": [_make_paragraph(text)]}


def _inline_content_from_token(token: Token | None) -> list[dict[str, Any]]:
    """Convert an ``inline`` container token into ADF inline nodes."""
    if token is None or not token.children:
        return []
    return _inline_tokens_to_adf(token.children)


def _block_tokens_to_adf(
    tokens: list[Token], start: int, stop_close: str | None
) -> tuple[list[dict[str, Any]], int]:
    """Convert a run of block tokens to ADF nodes.

    Args:
        tokens: Full flat token list.
        start: Index to start consuming from.
        stop_close: Token type that terminates this run (e.g. the matching
            ``bullet_list_close``), or ``None`` to consume to the end.

    Returns:
        ``(nodes, next_index)`` where ``next_index`` is the index *after* the
        closing token (or after the consumed run).
    """
    nodes: list[dict[str, Any]] = []
    i = start

    while i < len(tokens):
        tok = tokens[i]
        if stop_close is not None and tok.type == stop_close:
            return nodes, i + 1

        ttype = tok.type

        if ttype == "paragraph_open":
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            content = _inline_content_from_token(inline)
            if not content:
                content = [{"type": "text", "text": ""}]
            nodes.append({"type": "paragraph", "content": content})
            # Skip paragraph_open, inline, paragraph_close.
            i += 3
            continue

        if ttype == "heading_open":
            level = int(tok.tag[1]) if len(tok.tag) > 1 else 1
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            content = _inline_content_from_token(inline)
            nodes.append(
                {
                    "type": "heading",
                    "attrs": {"level": level},
                    "content": content,
                }
            )
            i += 3
            continue

        if ttype == "fence" or ttype == "code_block":
            lang = ""
            if ttype == "fence" and tok.info:
                lang = tok.info.strip().split()[0] if tok.info.strip() else ""
            code_text = tok.content
            # markdown-it appends a trailing newline to fence/code content.
            if code_text.endswith("\n"):
                code_text = code_text[:-1]
            cb: dict[str, Any] = {
                "type": "codeBlock",
                "attrs": {"language": lang} if lang else {},
                "content": [{"type": "text", "text": code_text}] if code_text else [],
            }
            nodes.append(cb)
            i += 1
            continue

        if ttype == "bullet_list_open":
            children, i = _block_tokens_to_adf(tokens, i + 1, "bullet_list_close")
            nodes.append({"type": "bulletList", "content": children})
            continue

        if ttype == "ordered_list_open":
            attrs: dict[str, Any] = {}
            start_attr = tok.attrs.get("start") if tok.attrs else None
            if start_attr is not None:
                try:
                    attrs["order"] = int(start_attr)
                except (TypeError, ValueError):
                    pass
            children, i = _block_tokens_to_adf(tokens, i + 1, "ordered_list_close")
            node: dict[str, Any] = {"type": "orderedList", "content": children}
            if attrs:
                node["attrs"] = attrs
            nodes.append(node)
            continue

        if ttype == "list_item_open":
            children, i = _block_tokens_to_adf(tokens, i + 1, "list_item_close")
            if not children:
                children = [{"type": "paragraph", "content": []}]
            nodes.append({"type": "listItem", "content": children})
            continue

        if ttype == "blockquote_open":
            children, i = _block_tokens_to_adf(tokens, i + 1, "blockquote_close")
            if not children:
                children = [{"type": "paragraph", "content": []}]
            nodes.append({"type": "blockquote", "content": children})
            continue

        if ttype == "hr":
            nodes.append({"type": "rule"})
            i += 1
            continue

        if ttype == "table_open":
            table_node, i = _table_to_adf(tokens, i)
            nodes.append(table_node)
            continue

        # Unhandled block token: advance to avoid an infinite loop.
        i += 1

    return nodes, i


def _table_to_adf(tokens: list[Token], start: int) -> tuple[dict[str, Any], int]:
    """Convert a markdown-it table token run into an ADF table node."""
    rows: list[dict[str, Any]] = []
    current_cells: list[dict[str, Any]] | None = None
    cell_type = "tableCell"
    i = start + 1  # skip table_open

    while i < len(tokens):
        tok = tokens[i]
        ttype = tok.type
        if ttype == "table_close":
            i += 1
            break
        if ttype == "tr_open":
            current_cells = []
        elif ttype == "tr_close":
            if current_cells is not None:
                rows.append({"type": "tableRow", "content": current_cells})
            current_cells = None
        elif ttype in ("th_open", "td_open"):
            cell_type = "tableHeader" if ttype == "th_open" else "tableCell"
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            content = _inline_content_from_token(inline)
            if not content:
                content = [{"type": "text", "text": ""}]
            if current_cells is not None:
                current_cells.append(
                    {
                        "type": cell_type,
                        "content": [{"type": "paragraph", "content": content}],
                    }
                )
        i += 1

    table_node = {
        "type": "table",
        "attrs": {"isNumberColumnEnabled": False, "layout": "default"},
        "content": rows,
    }
    return table_node, i


def markdown_to_adf(markdown_text: str) -> dict[str, Any]:
    """Convert Markdown text to ADF (Atlassian Document Format) document.

    Tokenizes the input with ``markdown-it-py`` (CommonMark + GFM tables and
    strikethrough) and maps the token stream to ADF nodes.

    Args:
        markdown_text: Markdown-formatted text to convert.

    Returns:
        ADF document dict with version, type, and content keys.
    """
    doc: dict[str, Any] = {"version": 1, "type": "doc", "content": []}

    if not markdown_text:
        doc["content"].append({"type": "paragraph", "content": []})
        return doc

    tokens = _MD.parse(markdown_text)
    content, _ = _block_tokens_to_adf(tokens, 0, None)
    doc["content"] = content

    # ADF requires at least one content node.
    if not doc["content"]:
        doc["content"].append({"type": "paragraph", "content": []})

    return doc


def adf_to_text(adf_content: dict | list | str | None) -> str | None:
    """
    Convert Atlassian Document Format (ADF) content to plain text.

    ADF is Jira Cloud's rich text format returned for fields like description.
    This function recursively extracts text content from the ADF structure.

    Args:
        adf_content: ADF document (dict), content list, string, or None

    Returns:
        Plain text string or None if no content
    """
    if adf_content is None:
        return None

    if isinstance(adf_content, str):
        return adf_content

    if isinstance(adf_content, list):
        texts = []
        for item in adf_content:
            text = adf_to_text(item)
            if text:
                texts.append(text)
        return "\n".join(texts) if texts else None

    if isinstance(adf_content, dict):
        # Check if this is a text node
        if adf_content.get("type") == "text":
            return adf_content.get("text", "")

        # Check if this is a hardBreak node
        if adf_content.get("type") == "hardBreak":
            return "\n"

        # Check if this is a mention node
        if adf_content.get("type") == "mention":
            attrs = adf_content.get("attrs", {})
            return attrs.get("text") or f"@{attrs.get('id', 'unknown')}"

        # Check if this is an emoji node
        if adf_content.get("type") == "emoji":
            attrs = adf_content.get("attrs", {})
            return attrs.get("text") or attrs.get("shortName", "")

        # Check if this is a date node
        if adf_content.get("type") == "date":
            attrs = adf_content.get("attrs", {})
            timestamp = attrs.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromtimestamp(int(timestamp) / 1000, tz=timezone.utc)
                    return dt.strftime("%Y-%m-%d")
                except (ValueError, OSError, TypeError, OverflowError):
                    return str(timestamp)
            return ""

        # Check if this is a status node
        if adf_content.get("type") == "status":
            attrs = adf_content.get("attrs", {})
            return f"[{attrs.get('text', '')}]"

        # Check if this is an inlineCard node
        if adf_content.get("type") == "inlineCard":
            attrs = adf_content.get("attrs", {})
            url = attrs.get("url")
            if url:
                return url
            data = attrs.get("data", {})
            return data.get("url") or data.get("name", "")

        # Check if this is a codeBlock node
        if adf_content.get("type") == "codeBlock":
            content = adf_content.get("content", [])
            code_text = adf_to_text(content) or ""
            return f"```\n{code_text}\n```"

        # Recursively process content
        content = adf_content.get("content")
        if content:
            return adf_to_text(content)

        return None

    return None


__all__ = [
    "markdown_to_adf",
    "adf_to_text",
    "_parse_inline_formatting",
]
