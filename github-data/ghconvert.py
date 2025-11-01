import re
import sys
import json
import logging
import argparse
from pathlib import Path
from urllib.parse import quote
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import Any, TypedDict, Optional, List, Dict, Tuple

MAX_CELL_LEN = 255
MAX_FILENAME_LENGTH = 80
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")

EMOJI = {
    "open": "✅", "closed": "❌", "draft": "📝", "merged": "🔀",
    "unknown": "⚪️", "issue": "📌", "pull_request": "🔀",
    "discussion": "🗣️", "answered": "✅", "changes_requested": "🔄",
    "commented": "💬", "person": "👤", "description": "📄", "conversation": "💬",
}

DATE_FORMATS = {
    "date_only": "%Y-%m-%d",
    "time_only": "%H:%M:%S",
    "datetime_utc": "%Y-%m-%d %H:%M:%S UTC",
}

class Event(TypedDict, total=False):
    type: str
    date: Optional[datetime]
    author: str
    body: Optional[str]
    replies: list["Event"]
    state: str
    path: str

class IndexEntry(TypedDict):
    type: str
    number: int
    title: str
    author: str
    status: str
    raw_status: str
    created_at: str
    updated_at: str
    path: str

def clean_filename(filename: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9-._ ]", "", filename.strip())
    sanitized = re.sub(r"\s+", " ", sanitized).strip("-._ ")
    return sanitized[:MAX_FILENAME_LENGTH]

def sanitize_cell(text: str) -> str:
    sanitized = CONTROL_CHARS.sub("", text)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if len(sanitized) > MAX_CELL_LEN:
        sanitized = sanitized[: MAX_CELL_LEN - 1].strip() + "…"
    return sanitized.replace("|", r"\|")

def get_status(item: Dict[str, Any], item_type: str) -> Tuple[str, str]:
    if item_type == "pull_requests":
        if item.get("merged_at"):
            raw_status = "Merged"
        elif item.get("is_draft"):
            raw_status = "Draft"
        else:
            raw_status = item.get("state", "unknown").title()
    else:
        raw_status = item.get("state", "unknown").title()

    emoji_map = {
        "Merged": EMOJI["merged"],
        "Draft": EMOJI["draft"],
        "Answered": EMOJI["answered"],
        "Open": EMOJI["open"],
        "Closed": EMOJI["closed"],
    }
    emoji = emoji_map.get(raw_status, EMOJI["unknown"])
    return raw_status, f"{emoji} **{raw_status}**"

ITEM_DETAILS = {
    "issues": {
        "api_key": "issues", "display_name": "Issue",
        "section_title": "Conversation", "url_path": "issues",
        "emoji": EMOJI["issue"]
    },
    "pull_requests": {
        "api_key": "pull_requests", "display_name": "Pull Request",
        "section_title": "Conversation", "url_path": "pull",
        "emoji": EMOJI["pull_request"]
    },
    "discussions": {
        "api_key": "discussions", "display_name": "Discussion",
        "section_title": "Discussion", "url_path": "discussions",
        "emoji": EMOJI["discussion"]
    },
}

def parse_date(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError):
        logging.warning(f"Could not parse date: {dt_str}")
        return None

def format_date(dt: Optional[datetime], fmt: str = "date_only") -> str:
    return dt.strftime(DATE_FORMATS[fmt]) if dt else "N/A"

def linkify(text: Optional[str], repo_url: str) -> str:
    if not text:
        return ""
    parts = re.split(r"(```[\s\S]*?```|`[^`]*?`)", text)
    return "".join(
        re.sub(r"#(\d+)\b", rf"[\g<0>]({repo_url}/issues/\1)", part)
        if i % 2 == 0 else part
        for i, part in enumerate(parts)
    )

def sort_events(events: List[Event]) -> None:
    events.sort(key=lambda x: x.get("date") or datetime.min.replace(tzinfo=timezone.utc))

def format_event_header(event: Event, is_reply: bool = False) -> str:
    author = event.get("author", "N/A")
    dt = event.get("date")
    event_type = event.get("type", "comment")

    date_part = f"on **{format_date(dt)}**" if dt else ""
    time_part = f"at **{format_date(dt, 'time_only')}**" if dt else ""

    if event_type == "review":
        state = event.get("state", "").upper()

        action_text_map = {
            "APPROVED": "approved this pull request",
            "CHANGES_REQUESTED": "requested changes on this pull request",
            "COMMENTED": "reviewed this pull request",
        }
        emoji_map = {
            "APPROVED": EMOJI["answered"],
            "CHANGES_REQUESTED": EMOJI["changes_requested"],
            "COMMENTED": EMOJI["commented"],
        }

        action = action_text_map.get(state, "reviewed this pull request")
        header_text = f"{action} {emoji_map.get(state, EMOJI['commented'])}"
    elif event_type == "review_thread":
        path = event.get('path', 'unknown file')
        header_text = f"started a conversation on `{path}`"
    else:
        header_text = "replied" if is_reply else "commented"

    header_parts = [f"{EMOJI['person']} **{author}**", header_text, date_part, time_part]
    return " ".join(p for p in header_parts if p)

def format_replies(replies: List[Event], indent_level: int, repo_url: str) -> str:
    reply_blocks = []
    prefix = "> " * indent_level
    sort_events(replies)
    for reply in replies:
        header = format_event_header(reply, is_reply=True)
        body = linkify(reply.get("body", ""), repo_url).strip()
        content = f"{header}\n\n{body}" if body else header
        quoted_block = "\n".join(f"{prefix}{line}" for line in content.split("\n"))
        block_parts = [quoted_block]
        if reply.get("replies"):
            nested_str = format_replies(reply["replies"], indent_level + 1, repo_url)
            if nested_str:
                block_parts.append(nested_str)
        reply_blocks.append("\n\n".join(block_parts))
    return "\n\n".join(reply_blocks)

def comments_to_events(comments: List[Dict[str, Any]]) -> List[Event]:
    if not comments:
        return []
    return [
        {
            "type": "comment",
            "date": parse_date(c.get("created_at")),
            "author": c.get("author", "N/A"),
            "body": c.get("body"),
            "replies": comments_to_events(c.get("replies", [])),
        }
        for c in comments
    ]

def extract_events(item: Dict[str, Any], item_type: str) -> List[Event]:
    events: List[Event] = comments_to_events(item.get("comments", []))
    if item_type == "pull_requests":
        for r in item.get("reviews", []):
            if r.get("body") or r.get("state") != "COMMENTED":
                events.append({
                    "type": "review",
                    "date": parse_date(r.get("submitted_at")),
                    "author": r.get("author", "N/A"),
                    "body": r.get("body"),
                    "state": r.get("state"),
                })
        for thread in item.get("review_threads", []):
            if not thread.get("comments"):
                continue
            for root_comment in thread.get("comments", []):
                events.append({
                    "type": "review_thread",
                    "path": thread.get("path", "unknown file"),
                    "date": parse_date(root_comment.get("created_at")),
                    "author": root_comment.get("author", "N/A"),
                    "body": root_comment.get("body"),
                    "replies": comments_to_events(root_comment.get("replies", [])),
                })
    return events

def build_header(
    item: Dict[str, Any], details: Dict[str, Any], repo_url: str
) -> str:
    num = item.get("number", 0)
    title = linkify(item.get('title', 'Untitled'), repo_url)
    url = f"{repo_url}/{details['url_path']}/{num}"
    return f"## {details['emoji']} [{details['display_name']} #{num}]({url}) - {title}"

def build_meta_table(
    item: Dict[str, Any], item_type: str, details: Dict[str, Any]
) -> str:
    is_pr = item_type == "pull_requests"
    _, status_md = get_status(item, item_type)
    created_at = parse_date(item.get("created_at"))
    updated_at = parse_date(item.get("updated_at"))
    merged_at = parse_date(item.get("merged_at"))

    meta_fields = [
        ("Author", f"`{item.get('author', 'N/A')}`"),
        ("State", status_md),
        ("Source Branch", f"`{item.get('head_ref')}`" if is_pr and item.get('head_ref') else None),
        ("Target Branch", f"`{item.get('base_ref')}`" if is_pr and item.get('base_ref') else None),
        ("Created", format_date(created_at)),
        ("Updated", format_date(updated_at) if updated_at and updated_at != created_at else None),
        ("Merged", format_date(merged_at) if is_pr and merged_at else None),
        ("Labels", ", ".join(f"`{l}`" for l in item.get("labels", [])) or None),
        ("Assignees", ", ".join(f"`{a}`" for a in item.get("assignees", [])) or None),
    ]

    active_meta = [f"| **{k}** | {v} |" for k, v in meta_fields if v]
    if not active_meta:
        return ""

    return "\n".join([active_meta[0], "| :--- | :--- |"] + active_meta[1:])

def build_body(item: Dict[str, Any], repo_url: str) -> str:
    body = linkify(item.get("body"), repo_url).strip()
    return f"## {EMOJI['description']} Description\n\n{body or '_No description provided._'}"

def build_timeline(
    events: List[Event], repo_url: str, details: Dict[str, Any]
) -> str:
    if not events:
        return ""

    sort_events(events)
    event_blocks = []
    for event in events:
        header = format_event_header(event)
        body = linkify(event.get("body", ""), repo_url).strip()
        block = f"{header}\n\n{body}" if body else header
        if event.get("replies"):
            block += "\n\n" + format_replies(event["replies"], 1, repo_url)
        event_blocks.append(block)

    if not event_blocks:
        return ""

    joined_blocks = '\n\n---\n\n'.join(event_blocks)
    return f"## {EMOJI['conversation']} {details['section_title']}\n\n{joined_blocks}"

def item_to_md(item: Dict[str, Any], item_type: str, repo_url: str) -> str:
    details = ITEM_DETAILS[item_type]
    header = build_header(item, details, repo_url)
    meta_table = build_meta_table(item, item_type, details)
    body_section = build_body(item, repo_url)
    events = extract_events(item, item_type)
    timeline_section = build_timeline(events, repo_url, details)

    content_parts = [p for p in [body_section, timeline_section] if p]
    full_content = "\n\n---\n\n".join(content_parts)

    all_parts = [p for p in [header, meta_table, "---", full_content] if p]
    return "\n\n".join(all_parts)

def make_index_section(
    type_name: str, items: List[IndexEntry], details: Dict[str, Any]
) -> str:
    if not items:
        return f"## {details['emoji']} {details['display_name']}s\n\n0 total."

    counts = Counter(item["raw_status"] for item in items)
    status_counts = ", ".join(f"{v} {k}" for k, v in counts.most_common())
    summary = (
        f"<summary>{details['emoji']} <strong>{details['display_name']}s</strong> "
        f"({len(items)} total: {status_counts})</summary>"
    )
    table_header = (
        "| Status | # | Title | Author | Created | Updated |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    )
    items.sort(key=lambda x: x["number"], reverse=True)
    table_rows = [
        (
            f"| {i['status']} | [{i['number']}]({i['path']}) | "
            f"{sanitize_cell(i['title'])} | {i['author']} | {i['created_at']} | "
            f"{i['updated_at']} |"
        )
        for i in items
    ]
    return f"<details open>\n{summary}\n\n{table_header}\n" + "\n".join(table_rows) + "\n\n</details>"

def generate_index(
    archive_data: Dict[str, Any], idx_entries: List[IndexEntry], output_path: Path
):
    logging.info("Generating index.md summary file...")
    repo = archive_data.get("repository", "unknown/repository")
    scraped_at_str = archive_data.get('scraped_at')
    scraped_at = format_date(parse_date(scraped_at_str), 'datetime_utc')

    header = [
        f"# Index for `{repo}`",
        "This index provides a summary of all issues, pull requests, and discussions.",
        f"- **Archive Date:** {scraped_at}\n- **Total Items:** {len(idx_entries)}",
    ]

    items_by_type = defaultdict(list)
    for item in idx_entries:
        items_by_type[item["type"]].append(item)

    sections = [
        make_index_section(name, items_by_type.get(name, []), details)
        for name, details in ITEM_DETAILS.items()
    ]
    try:
        (output_path / "index.md").write_text("\n\n".join(header + sections), encoding="utf-8")
    except (IOError, OSError) as e:
        logging.error(f"Could not write to index file. Reason: {e}")
        sys.exit(1)

def make_index_entry(
    item: Dict[str, Any], type_name: str, relative_path: str
) -> IndexEntry:
    raw_status, md_status = get_status(item, type_name)
    created_at = parse_date(item.get("created_at"))
    updated_at = parse_date(item.get("updated_at"))
    encoded_path = "/".join(quote(part) for part in Path(relative_path).parts)
    return {
        "type": type_name,
        "number": item.get("number", 0),
        "title": item.get("title", "Untitled"),
        "author": f"`{item.get('author', 'N/A')}`",
        "status": md_status,
        "raw_status": raw_status,
        "created_at": format_date(created_at),
        "updated_at": format_date(updated_at),
        "path": f"./{encoded_path}",
    }

def process_item_group(
    items: List[Dict[str, Any]], type_name: str, details: Dict[str, Any],
    output_path: Path, repo_url: str
) -> Tuple[List[IndexEntry], int]:
    if not items:
        logging.info(f"No {details['api_key']} found to process.")
        return [], 0

    logging.info(f"Processing {len(items)} {details['api_key']}...")
    item_dir = output_path / details["api_key"]
    item_dir.mkdir(exist_ok=True)
    idx_entries, files_created = [], 0

    for item in items:
        try:
            item_num = int(item.get("number", 0))
            if not item_num:
                logging.warning("Skipping item with missing number.")
                continue

            title = item.get('title', 'Untitled')
            filename = f"{item_num} - {clean_filename(title)}.md"
            md_content = item_to_md(item, type_name, repo_url)
            (item_dir / filename).write_text(md_content, encoding="utf-8")
            files_created += 1

            relative_path = Path(details["api_key"]) / filename
            idx_entries.append(make_index_entry(item, type_name, str(relative_path)))
        except (ValueError, TypeError, KeyError) as e:
            item_num_str = item.get('number', 'N/A')
            logging.warning(f"Skipping item {item_num_str} due to invalid data: {e}")
        except (IOError, OSError) as e:
            logging.error(f"Halting due to file system error: {e}")
            sys.exit(1)
    return idx_entries, files_created

def json_to_md(data: Dict[str, Any], output_path: Path):
    repo_url = f"https://github.com/{data.get('repository', 'unknown/repository')}"
    output_path.mkdir(parents=True, exist_ok=True)
    all_idx_entries, total_files = [], 0

    for type_name, details in ITEM_DETAILS.items():
        items = data.get(details["api_key"], [])
        idx_entries, files_created = process_item_group(
            items, type_name, details, output_path, repo_url
        )
        all_idx_entries.extend(idx_entries)
        total_files += files_created

    if all_idx_entries:
        generate_index(data, all_idx_entries, output_path)
    logging.info(f"Successfully generated {total_files} Markdown files.")

def validate_output_path(path: Path):
    if not (path.is_dir() and any(path.iterdir())):
        return

    if path.resolve() == Path.cwd().resolve():
        conflicting_paths = ["issues", "pull_requests", "discussions", "index.md"]
        conflicting = [p for p in [path / d for d in conflicting_paths] if p.exists()]
        if conflicting:
            logging.error(
                f"Error: Output directory '.' contains conflicting paths: "
                f"{conflicting}. Please remove them or choose a different directory."
            )
            sys.exit(1)
    else:
        logging.error(
            f"Error: Output directory '{path}' exists and is not empty. "
            "Please choose a different directory or clear its contents."
        )
        sys.exit(1)

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
    validate_output_path(output_path)

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
