import re
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, TypedDict, Optional, List, Dict, Callable

def sanitize_filename(filename: str, max_len: int = 96) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    sanitized = sanitized.strip(' .')
    return sanitized[:max_len] if len(sanitized) > max_len else sanitized

EMOJI = {
    "open": "✅",
    "closed": "❌",
    "merged": "🔀",
    "unknown": "⚪️",
    "conversation": "💬",
    "discussion": "🗣️",
    "approved": "✅",
    "changes_requested": "🔄",
    "review_commented": "💬",
    "person": "👤",
}

class Event(TypedDict, total=False):
    date_str: str
    author: str
    body: str
    header_text: str
    replies: list[dict]

def _get_generic_state_str(item: Dict[str, Any]) -> str:
    state = item.get("state", "unknown").lower()
    return f"{EMOJI.get(state, EMOJI['unknown'])} **{state.title()}**"

def _get_pr_state_str(item: Dict[str, Any]) -> str:
    if item.get("merged_at"):
        return f"{EMOJI['merged']} **Merged**"
    return _get_generic_state_str(item)

ITEM_TYPE_DETAILS: Dict[str, Dict[str, Any]] = {
    "issue": {
        "api_key": "issues",
        "display_name": "Issue",
        "timeline_title": f"{EMOJI['conversation']} Conversation",
        "get_state_str": _get_generic_state_str,
    },
    "pull_request": {
        "api_key": "pull_requests",
        "display_name": "Pull Request",
        "timeline_title": f"{EMOJI['conversation']} Conversation",
        "get_state_str": _get_pr_state_str,
    },
    "discussion": {
        "api_key": "discussions",
        "display_name": "Discussion",
        "timeline_title": f"{EMOJI['discussion']} Discussion",
        "get_state_str": lambda item: None,
    },
}

PATH_SEGMENT_MAP = {"pull_requests": "pull", "issues": "issues", "discussions": "discussions"}

def _parse_iso_date(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        logging.warning(f"Could not parse date: {dt_str}")
        return None

def _linkify_references(text: Optional[str], repo_url: str) -> str:
    if not text:
        return ""
    return re.sub(r"#(\d+)", rf"[\g<0>]({repo_url}/issues/\1)", text)

def _format_date_str(dt_str: Optional[str], fmt: str = "%Y-%m-%d") -> str:
    dt = _parse_iso_date(dt_str)
    return dt.strftime(fmt) if dt else "N/A"

def _write_file(filepath: Path, content: str):
    try:
        filepath.write_text(content, encoding="utf-8")
    except (IOError, OSError) as e:
        logging.error(f"Could not write to file {filepath}. Reason: {e}")
        raise

def _format_event_header(author: str, date_str: str, header_text: str) -> str:
    dt = _parse_iso_date(date_str)
    if dt:
        formatted_date = f"on **{dt.strftime('%Y-%m-%d')}** at **{dt.strftime('%H:%M:%S')}**"
    else:
        formatted_date = "on **N/A** at **N/A**"
    return f"{EMOJI['person']} **{author}** {header_text} {formatted_date}"

def _format_replies(replies: List[Dict], indent_level: int) -> str:
    reply_blocks = []
    prefix = "> " * indent_level
    replies.sort(key=lambda x: x.get("created_at", ""))

    for reply in replies:
        body = (reply.get("body") or "").strip() or "_No content provided._"
        header = _format_event_header(reply.get("author", "N/A"), reply.get("created_at", ""), "replied")
        full_reply_content = f"{header}\n\n{body}"
        quoted_block = "\n".join(f"{prefix}{line}" for line in full_reply_content.split("\n"))
        reply_blocks.append(quoted_block)

    return "\n\n".join(reply_blocks)

def _format_timeline(events: List[Event], section_title: str) -> str:
    if not events:
        return ""

    events.sort(key=lambda x: x.get("date_str", ""))
    event_blocks = []
    for event in events:
        body = (event.get("body") or "").strip() or "_No content provided._"
        header = _format_event_header(event["author"], event["date_str"], event["header_text"])

        block = f"{header}\n\n{body}"
        if event.get("replies"):
            block += "\n\n" + _format_replies(event["replies"], 1)
        event_blocks.append(block)

    if not event_blocks:
        return ""
    return f"#### {section_title}\n\n" + "\n\n---\n\n".join(event_blocks)

def _create_event(date_str: str, author: str, body: str, header_text: str, **kwargs: Any) -> Event:
    event: Event = {"date_str": date_str, "author": author, "body": body, "header_text": header_text}
    event.update(kwargs)
    return event

def _format_review_header(review_event: Dict) -> str:
    state = review_event.get("state", "").upper()
    state_emoji_map = {"APPROVED": EMOJI["approved"], "CHANGES_REQUESTED": EMOJI["changes_requested"]}
    emoji = state_emoji_map.get(state, EMOJI["review_commented"])
    return f"submitted a review: {emoji} `{state}`"

def _collect_events(item: Dict) -> List[Event]:
    events: List[Event] = []

    event_sources = [
        {"key": "comments", "date": "created_at", "body": "body", "author": "author", "header_fn": lambda e: "commented"},
        {"key": "review_comments", "date": "created_at", "body": "body", "author": "author", "header_fn": lambda e: f"commented during a code review on `{e.get('path')}`"},
        {"key": "reviews", "date": "submitted_at", "body": "body", "author": "author", "header_fn": _format_review_header},
    ]

    for config in event_sources:
        for source_item in item.get(config["key"], []):
            if not source_item:
                continue
            events.append(_create_event(
                date_str=source_item.get(config["date"], ""),
                author=source_item.get(config["author"], "N/A"),
                body=source_item.get(config["body"]),
                header_text=config["header_fn"](source_item),
                replies=source_item.get("replies", [])
            ))
    return events

def format_item_to_markdown(item: Dict, item_type: str, repo_url: str) -> str:
    details = ITEM_TYPE_DETAILS[item_type]
    item_number = item.get("number")
    path_segment = PATH_SEGMENT_MAP.get(details["api_key"], details["api_key"])
    item_url = f"{repo_url}/{path_segment}/{item_number}"

    title = item.get("title", "Untitled")
    header = f"### [{details['display_name']} #{item_number}]({item_url}) - {_linkify_references(title, repo_url)}"

    meta_fields = [
        ("Author", f"`{item.get('author', 'N/A')}`"),
        ("State", details["get_state_str"](item)),
        ("Created", _format_date_str(item.get("created_at"))),
        ("Updated", _format_date_str(item.get("updated_at")) if item.get("updated_at") != item.get("created_at") else None),
        ("Merged", _format_date_str(item["merged_at"]) if item_type == "pull_request" and item.get("merged_at") else None),
        ("Labels", ", ".join(f"`{l}`" for l in item.get('labels', [])) or None),
        ("Assignees", ", ".join(f"`{a}`" for a in item.get('assignees', [])) or None),
    ]

    valid_meta_fields = [(label, value) for label, value in meta_fields if value]

    if not valid_meta_fields:
        table = ""
    else:
        header_label, header_value = valid_meta_fields[0]
        header_row = f"| **{header_label}** | {header_value} |"
        separator_row = "| :--- | :--- |"
        body_rows = [f"| **{label}** | {value} |" for label, value in valid_meta_fields[1:]]
        table = "\n".join([header_row, separator_row] + body_rows)

    body = (item.get("body") or "").strip() or "_No description provided._"
    body_section = f"#### Description\n\n{body}"

    events = _collect_events(item)
    timeline_section = _format_timeline(events, details["timeline_title"])

    content_parts = [part for part in [body_section, timeline_section] if part]
    content = "\n\n---\n\n".join(content_parts)

    final_parts = [part for part in [header, table, "---", content] if part]
    return "\n\n".join(final_parts)

def convert_to_markdown(data: Dict, output_path: Path):
    repo_name = data.get("repository", "unknown/repository")
    repo_url = f"https://github.com/{repo_name}"
    output_path.mkdir(parents=True, exist_ok=True)

    total_files = 0
    for type_name, details in ITEM_TYPE_DETAILS.items():
        api_key = details["api_key"]
        items = data.get(api_key, [])
        if not items:
            logging.info(f"No {api_key} found to process.")
            continue

        logging.info(f"Processing {len(items)} {api_key}...")
        item_dir = output_path / api_key
        item_dir.mkdir(exist_ok=True)
        for item in items:
            try:
                item_number = int(item['number'])
                item_title = item.get('title', 'untitled').strip()
                safe_basename = sanitize_filename(f"{item_number} - {item_title}", max_len=96)
                safe_filename = f"{safe_basename}.md"

                markdown_content = format_item_to_markdown(item, type_name, repo_url)
                _write_file(item_dir / safe_filename, markdown_content)
                total_files += 1
            except (ValueError, TypeError, KeyError) as e:
                logging.warning(f"Skipping item with invalid data. ID: {item.get('id', 'Unknown')}, Error: {e}")
                continue
            except (IOError, OSError) as e:
                logging.error(f"Halting due to file system error: {e}")
                sys.exit(1)

    logging.info(f"Successfully generated {total_files} Markdown files.")

def main():
    parser = argparse.ArgumentParser(description="Convert a GitHub backup into a directory of readable Markdown files.")
    parser.add_argument("json_data", help="path to the input unified JSON file.")
    parser.add_argument("-o", "--output", help="output directory name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    input_file = Path(args.json_data)
    if not input_file.is_file():
        logging.error(f"Input file '{input_file}' not found or is not a file.")
        sys.exit(1)

    output_path = Path(args.output) if args.output else Path(input_file.stem)

    try:
        with input_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Could not read or parse '{input_file}'. Reason: {e}")
        sys.exit(1)

    convert_to_markdown(data, output_path)
    logging.info(f"Files are in the '{output_path}' directory.")

if __name__ == "__main__":
    main()
