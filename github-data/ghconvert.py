import re
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import Any, TypedDict, Optional, List, Dict, Callable

MAX_FILENAME_LENGTH = 80

EMOJI = {
    "open": "✅",
    "closed": "❌",
    "draft": "📝",
    "merged": "🔀",
    "unknown": "⚪️",
    "issue": "📌",
    "pull_request": "🔀",
    "conversation": "💬",
    "discussion": "🗣️",
    "approved": "✅",
    "NEEDS_CHANGES": "🔄",
    "review_commented": "💬",
    "person": "👤",
}

class Event(TypedDict, total=False):
    date_str: str
    author: str
    body: str
    header_text: str
    replies: list[dict]

class IndexEntry(TypedDict):
    type: str
    number: int
    title: str
    author: str
    status: str
    created_at: str
    updated_at: str
    relative_path: str

def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    sanitized = sanitized.strip(' .')
    return sanitized[:MAX_FILENAME_LENGTH]

def _get_state_str(item: Dict[str, Any]) -> str:
    state = item.get("state", "unknown").lower()
    return f"{EMOJI.get(state, EMOJI['unknown'])} **{state.title()}**"

def _get_pr_state_str(item: Dict[str, Any]) -> str:
    if item.get("merged_at"):
        return f"{EMOJI['merged']} **Merged**"
    if item.get("is_draft"):
        return f"{EMOJI['draft']} **Draft**"
    return _get_state_str(item)

ITEM_DETAILS: Dict[str, Dict[str, Any]] = {
    "issue": {
        "api_key": "issues",
        "display_name": "Issue",
        "timeline_title": f"{EMOJI['conversation']} Conversation",
        "get_state_str": _get_state_str,
        "emoji": EMOJI["issue"],
    },
    "pull_request": {
        "api_key": "pull_requests",
        "display_name": "Pull Request",
        "timeline_title": f"{EMOJI['conversation']} Conversation",
        "get_state_str": _get_pr_state_str,
        "emoji": EMOJI["pull_request"],
    },
    "discussion": {
        "api_key": "discussions",
        "display_name": "Discussion",
        "timeline_title": f"{EMOJI['discussion']} Discussion",
        "get_state_str": _get_state_str,
        "emoji": EMOJI["discussion"],
    },
}

PATH_SEGMENT_MAP = {
    "pull_requests": "pull",
    "issues": "issues",
    "discussions": "discussions",
}

def _parse_iso_date(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        logging.warning(f"Could not parse date: {dt_str}")
        return None

def _link_refs(text: Optional[str], repo_url: str) -> str:
    if not text:
        return ""
    return re.sub(r"#(\d+)", rf"[\g<0>]({repo_url}/issues/\1)", text)

def _fmt_date(dt_str: Optional[str], fmt: str = "%Y-%m-%d") -> str:
    dt = _parse_iso_date(dt_str)
    return dt.strftime(fmt) if dt else "N/A"

def _write_file(filepath: Path, content: str):
    try:
        filepath.write_text(content, encoding="utf-8")
    except (IOError, OSError) as e:
        logging.error(f"Could not write to file {filepath}. Reason: {e}")
        raise

def _fmt_event_header(author: str, date_str: str, header_text: str) -> str:
    dt = _parse_iso_date(date_str)
    if dt:
        date_part = f"on **{dt.strftime('%Y-%m-%d')}**"
        time_part = f"at **{dt.strftime('%H:%M:%S')}**"
        formatted_date = f"{date_part} {time_part}"
    else:
        formatted_date = "on **N/A** at **N/A**"
    return f"{EMOJI['person']} **{author}** {header_text} {formatted_date}"

def _fmt_replies(replies: List[Dict], indent_level: int) -> str:
    reply_blocks = []
    prefix = "> " * indent_level
    replies.sort(key=lambda x: x.get("created_at", ""))

    for reply in replies:
        body = (reply.get("body") or "").strip() or "_No content provided._"
        author = reply.get("author", "N/A")
        date_str = reply.get("created_at", "")
        header = _fmt_event_header(author, date_str, "replied")
        full_reply_content = f"{header}\n\n{body}"

        quoted_block = "\n".join(
            f"{prefix}{line}" for line in full_reply_content.split("\n")
        )
        block_parts = [quoted_block]

        if reply.get("replies"):
            nested_replies_str = _fmt_replies(reply["replies"], indent_level + 1)
            if nested_replies_str:
                block_parts.append(nested_replies_str)

        reply_blocks.append("\n\n".join(block_parts))

    return "\n\n".join(reply_blocks)

def _fmt_timeline(events: List[Event], section_title: str) -> str:
    if not events:
        return ""

    events.sort(key=lambda x: x.get("date_str", ""))
    event_blocks = []
    for event in events:
        body = (event.get("body") or "").strip() or "_No content provided._"
        header = _fmt_event_header(
            event["author"], event["date_str"], event["header_text"]
        )

        block = f"{header}\n\n{body}"
        if event.get("replies"):
            block += "\n\n" + _fmt_replies(event["replies"], 1)
        event_blocks.append(block)

    if not event_blocks:
        return ""
    return f"#### {section_title}\n\n" + "\n\n---\n\n".join(event_blocks)

def _create_event(
    date_str: str, author: str, body: str, header_text: str, **kwargs: Any
) -> Event:
    event: Event = {
        "date_str": date_str, "author": author,
        "body": body, "header_text": header_text
    }
    event.update(kwargs)
    return event

def _fmt_review_header(review_event: Dict) -> str:
    state = review_event.get("state", "").upper()
    state_emoji_map = {
        "APPROVED": EMOJI["approved"],
        "CHANGES_REQUESTED": EMOJI["NEEDS_CHANGES"],
    }
    emoji = state_emoji_map.get(state, EMOJI["review_commented"])
    return f"submitted a review: {emoji} `{state}`"

def _get_events(item: Dict) -> List[Event]:
    events: List[Event] = []

    event_sources = [
        {
            "key": "comments", "date": "created_at", "body": "body",
            "author": "author", "header_fn": lambda e: "commented"
        },
        {
            "key": "review_comments", "date": "created_at", "body": "body",
            "author": "author",
            "header_fn": lambda e: (
                f"commented during a code review on `{e.get('path')}`"
            ),
        },
        {
            "key": "reviews", "date": "submitted_at", "body": "body",
            "author": "author", "header_fn": _fmt_review_header
        },
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

def item_to_md(item: Dict, item_type: str, repo_url: str) -> str:
    details = ITEM_DETAILS[item_type]
    item_number = item.get("number")
    path_segment = PATH_SEGMENT_MAP.get(details["api_key"], details["api_key"])
    item_url = f"{repo_url}/{path_segment}/{item_number}"

    title = item.get("title", "Untitled")
    linked_title = _link_refs(title, repo_url)
    header = (
        f"### [{details['display_name']} #{item_number}]({item_url}) - {linked_title}"
    )

    created_at = item.get("created_at")
    updated_at = item.get("updated_at")
    updated_str = (
        _fmt_date(updated_at)
        if updated_at and updated_at != created_at else None
    )

    meta_fields = [
        ("Author", f"`{item.get('author', 'N/A')}`"),
        ("State", details["get_state_str"](item)),
        (
            "Source Branch",
            f"`{item.get('head_ref')}`"
            if item_type == 'pull_request' and item.get('head_ref') else None
        ),
        (
            "Target Branch",
            f"`{item.get('base_ref')}`"
            if item_type == 'pull_request' and item.get('base_ref') else None
        ),
        ("Created", _fmt_date(created_at)),
        ("Updated", updated_str),
        (
            "Merged",
            _fmt_date(item["merged_at"])
            if item_type == "pull_request" and item.get("merged_at") else None
        ),
        ("Labels", ", ".join(f"`{l}`" for l in item.get('labels', [])) or None),
        ("Assignees", ", ".join(f"`{a}`" for a in item.get('assignees', [])) or None),
    ]

    meta_fields = [(label, value) for label, value in meta_fields if value]

    if not meta_fields:
        table = ""
    else:
        header_label, header_value = meta_fields[0]
        header_row = f"| **{header_label}** | {header_value} |"
        separator_row = "| :--- | :--- |"
        body_rows = [f"| **{label}** | {value} |" for label, value in meta_fields[1:]]
        table = "\n".join([header_row, separator_row] + body_rows)

    body = (item.get("body") or "").strip() or "_No description provided._"
    body_section = f"#### Description\n\n{body}"

    events = _get_events(item)
    timeline_section = _fmt_timeline(events, details["timeline_title"])

    content_parts = [part for part in [body_section, timeline_section] if part]
    content = "\n\n---\n\n".join(content_parts)

    final_parts = [part for part in [header, table, "---", content] if part]
    return "\n\n".join(final_parts)

def _create_index_md(
    repo_data: Dict[str, Any], index_items: List[IndexEntry], output_path: Path
):
    logging.info("Generating index.md summary file...")
    repo_name = repo_data.get("repository", "unknown/repository")
    scraped_at = _fmt_date(
        repo_data.get("scraped_at"), "%Y-%m-%d %H:%M:%S %Z"
    )

    header = [
        f"# Index for `{repo_name}`",
        "This index provides a summary of all issues, pull requests, and "
        "discussions archived from the repository.",
        (f"- **Archive Date:** {scraped_at}\n"
         f"- **Total Items:** {len(index_items)}"),
    ]

    items_by_type: Dict[str, List[IndexEntry]] = defaultdict(list)
    for item in index_items:
        items_by_type[item['type']].append(item)

    sections = []
    for type_name, details in ITEM_DETAILS.items():
        items = items_by_type.get(type_name, [])
        if not items:
            sections.append(f"{details['emoji']} **{details['display_name']}s**: 0 total")
            continue

        counts = Counter(item['status'].replace('*', '') for item in items)
        status_summary = ", ".join(f"{v} {k.strip()}" for k, v in counts.most_common())

        summary = (
            f"<summary>{details['emoji']} <strong>{details['display_name']}s"
            f"</strong> ({len(items)} total: {status_summary})</summary>"
        )

        table_header = ("| Status | # | Title | Author | Created | Updated |\n"
                        "| :--- | :--- | :--- | :--- | :--- | :--- |")
        items.sort(key=lambda x: x['number'], reverse=True)

        table_rows = []
        for item in items:
            title = item['title'].replace('|', '│')
            row = (f"| {item['status']} | [{item['number']}]({item['relative_path']}) | "
                   f"{title} | {item['author']} | {item['created_at']} | "
                   f"{item['updated_at']} |")
            table_rows.append(row)

        table = "\n".join([table_header] + table_rows)
        details_tag = "<details open>" if not sections else "<details>"
        sections.append(f"{details_tag}\n{summary}\n\n{table}\n\n</details>")

    index_content = "\n\n".join(header + sections)
    _write_file(output_path / "index.md", index_content)

def json_to_md(data: Dict, output_path: Path):
    repo_name = data.get("repository", "unknown/repository")
    repo_url = f"https://github.com/{repo_name}"
    output_path.mkdir(parents=True, exist_ok=True)

    index_items: List[IndexEntry] = []
    total_files = 0
    for type_name, details in ITEM_DETAILS.items():
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
                safe_name = sanitize_filename(f"{item_number} - {item_title}")
                safe_filename = f"{safe_name}.md"
                md_content = item_to_md(item, type_name, repo_url)
                _write_file(item_dir / safe_filename, md_content)
                total_files += 1

                index_items.append({
                    "type": type_name,
                    "number": item_number,
                    "title": item_title,
                    "author": f"`{item.get('author', 'N/A')}`",
                    "status": details["get_state_str"](item),
                    "created_at": _fmt_date(item.get("created_at")),
                    "updated_at": _fmt_date(item.get("updated_at")),
                    "relative_path": f"./{api_key}/{safe_filename}"
                })
            except (ValueError, TypeError, KeyError) as e:
                id = item.get('id', 'Unknown')
                logging.warning(f"Skipping item with invalid data. ID: {id}, Error: {e}")
                continue
            except (IOError, OSError) as e:
                logging.error(f"Halting due to file system error: {e}")
                sys.exit(1)

    if index_items:
        try:
            _create_index_md(data, index_items, output_path)
        except (IOError, OSError) as e:
            logging.error(f"Could not generate index file: {e}")
            sys.exit(1)

    logging.info(f"Successfully generated {total_files} Markdown files.")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a GitHub backup into readable Markdown files."
    )
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

    json_to_md(data, output_path)
    logging.info(f"Files are in the '{output_path}' directory.")

if __name__ == "__main__":
    main()
