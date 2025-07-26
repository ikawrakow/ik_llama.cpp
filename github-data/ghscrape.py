import os
import re
import sys
import time
import json
import logging
import requests
import argparse
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, TypedDict, Set

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GRAPHQL_URL = "https://api.github.com/graphql"

PER_PAGE = 100
PER_PAGE_NESTED = 100
PER_PAGE_REVIEW_COMMENTS = 45
INITIAL_REPLIES = 45
BATCH_SIZE = 50
MAX_RETRIES = 5
INITIAL_BACKOFF = 2
REQUEST_TIMEOUT = 60
INTER_REQUEST_DELAY = 0.05
RATE_LIMIT_THRESHOLD = 100

PAGE_INFO_FRAGMENT = "fragment PageInfoFragment on PageInfo { endCursor, hasNextPage }"
AUTHOR_FRAGMENT = "fragment AuthorFragment on Actor { login }"
COMMENT_FIELDS = """
    fragment CommentFields on Comment {
        id, author { ...AuthorFragment }, body, createdAt,
        ... on DiscussionComment {
            replyTo { id }
        }
        ... on PullRequestReviewComment {
            replyTo { id }
        }
    }
"""

BASE_ITEM_FIELDS = """
    fragment BaseItemFields on Node {
        ... on Issue {
            id, number, title, state, body, createdAt, updatedAt
            author { ...AuthorFragment }
            assignees(first: 10) { nodes { login } }
            labels(first: 20) { nodes { name } }
        }
        ... on PullRequest {
            id, number, title, state, body, createdAt, updatedAt, mergedAt,
            isDraft, headRefName, baseRefName
            author { ...AuthorFragment }
            assignees(first: 10) { nodes { login } }
            labels(first: 20) { nodes { name } }
        }
    }
"""

DISCUSSION_FIELDS = """
    fragment DiscussionItemFields on Discussion {
        id, number, title, body, createdAt, updatedAt, closed, isAnswered
        author { ...AuthorFragment }
        labels(first: 20) { nodes { name } }
    }
"""

ITEM_CONFIG = {
    "issues": {
        "graphql_name": "issues",
        "path": ("repository", "issues"),
        "fields": f"""
            ...BaseItemFields,
            comments(first: {PER_PAGE_NESTED}) {{
                pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}}
            }}
        """,
        "required_fragments": {"PageInfoFragment", "CommentFields", "BaseItemFields"},
    },
    "pull_requests": {
        "graphql_name": "pullRequests",
        "path": ("repository", "pullRequests"),
        "fields": f"""
            ...BaseItemFields
            comments(first: {PER_PAGE_NESTED}) {{
                pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}}
            }}
            reviews(first: {PER_PAGE_NESTED}) {{
                pageInfo {{...PageInfoFragment}},
                nodes {{ id, author {{...AuthorFragment}}, body, state, submittedAt }}
            }}
            reviewThreads(first: {PER_PAGE_NESTED}) {{
                pageInfo {{...PageInfoFragment}},
                nodes {{
                    id, path,
                    comments(first: {PER_PAGE_REVIEW_COMMENTS}) {{
                        pageInfo {{...PageInfoFragment}},
                        nodes {{ ...CommentFields }}
                    }}
                }}
            }}
        """,
        "required_fragments": {
            "PageInfoFragment", "CommentFields", "BaseItemFields", "AuthorFragment"
        },
    },
    "discussions": {
        "graphql_name": "discussions",
        "path": ("repository", "discussions"),
        "fields": f"""
            ...DiscussionItemFields
            comments(first: {PER_PAGE_NESTED}) {{
                pageInfo {{...PageInfoFragment}},
                nodes {{
                    ...CommentFields,
                    replies(first: {INITIAL_REPLIES}) {{
                        pageInfo {{...PageInfoFragment}},
                        nodes {{ ...CommentFields }}
                    }}
                }}
            }}
        """,
        "required_fragments": {
            "PageInfoFragment",
            "CommentFields",
            "DiscussionItemFields",
            "AuthorFragment",
        },
    },
}

class GitHubError(Exception): pass
class FatalError(GitHubError): pass
class RetriableError(GitHubError): pass

class PaginationTask(TypedDict):
    node_id: str
    cursor: Optional[str]
    task_type: str
    context: Dict[str, Any]

class SecretFilter(logging.Filter):
    def __init__(self, secret_to_hide: str):
        super().__init__()
        self._secret = secret_to_hide

    def filter(self, record):
        if self._secret:
            original_message = record.getMessage()
            record.msg = original_message.replace(self._secret, "[REDACTED_TOKEN]")
            record.args = ()
        return True

class GQLFragment(TypedDict):
    query: str
    deps: Set[str]

ALL_FRAGMENTS: Dict[str, GQLFragment] = {
    "PageInfoFragment": {"query": PAGE_INFO_FRAGMENT, "deps": set()},
    "AuthorFragment": {"query": AUTHOR_FRAGMENT, "deps": set()},
    "CommentFields": {"query": COMMENT_FIELDS, "deps": {"AuthorFragment"}},
    "BaseItemFields": {"query": BASE_ITEM_FIELDS, "deps": {"AuthorFragment"}},
    "DiscussionItemFields": {
        "query": DISCUSSION_FIELDS, "deps": {"AuthorFragment"}
    },
}

def _get_all_fragments(required: Set[str]) -> Set[str]:
    resolved = set()
    queue = deque(list(required))
    while queue:
        fragment_name = queue.popleft()
        if fragment_name not in resolved:
            resolved.add(fragment_name)
            fragment_def = ALL_FRAGMENTS.get(fragment_name)
            if fragment_def:
                queue.extend(fragment_def["deps"])
    return resolved

def _check_rate_limit(response: requests.Response, session_stats: Dict):
    remaining_str = response.headers.get("X-RateLimit-Remaining")
    reset_str = response.headers.get("X-RateLimit-Reset")
    if not remaining_str or not reset_str:
        return

    remaining = int(remaining_str)
    reset_time = int(reset_str)
    time_until_reset = reset_time - int(time.time())
    minutes, seconds = divmod(max(0, time_until_reset), 60)
    resets_in_str = f"resets in {minutes}m {seconds}s"
    req_num = session_stats.get('requests_made', 0)
    logging.info(
        f"API Rate Limit (Req #{req_num}): {remaining} points remaining, "
        f"{resets_in_str}."
    )
    if remaining < RATE_LIMIT_THRESHOLD:
        sleep_duration = max(0, time_until_reset) + 5
        logging.warning(
            f"Approaching GraphQL rate limit. "
            f"Sleeping for {sleep_duration:.2f}s..."
        )
        time.sleep(sleep_duration)

def _send_request(
    session: requests.Session, json_payload: Dict, session_stats: Dict
) -> Dict:
    session_stats["requests_made"] = session_stats.get("requests_made", 0) + 1
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(
                GRAPHQL_URL,
                json=json_payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code in (401, 404):
                msg = f"Fatal HTTP Error: {response.status_code} - {response.text}"
                raise FatalError(msg)
            if response.status_code == 403 and "Retry-After" in response.headers:
                retry_after = min(int(response.headers["Retry-After"]), 60)
                time.sleep(retry_after + 2)
                continue
            response.raise_for_status()
            response_json = response.json()
            if "errors" in response_json:
                err_summary = "; ".join(
                    f"({e.get('type')}) {e.get('message')}"
                    for e in response_json["errors"]
                )
                error_types = {e.get("type") for e in response_json["errors"]}
                if "NOT_FOUND" in error_types and response_json.get("data"):
                    logging.warning(
                        f"GraphQL returned non-fatal NOT_FOUND errors: {err_summary}"
                    )
                else:
                    raise RetriableError(f"GraphQL API errors: {err_summary}")
            _check_rate_limit(response, session_stats)
            return response_json
        except requests.exceptions.RequestException as e:
            last_exception = RetriableError(str(e))
        logging.warning(f"Retriable error: {last_exception}. Retrying...")
        if attempt < MAX_RETRIES - 1:
            time.sleep(INITIAL_BACKOFF * (2**attempt))
    raise FatalError(f"Max retries reached. Last error: {last_exception}")

def _normalize_author(actor: Optional[Dict]) -> str:
    return (actor or {}).get("login", "ghost")

def _normalize_comment(node: Optional[Dict]) -> Optional[Dict]:
    if not node: return None
    return {
        "id": node.get("id"),
        "author": _normalize_author(node.get("author")),
        "body": node.get("body"),
        "created_at": node.get("createdAt"),
        "reply_to_id": (node.get("replyTo") or {}).get("id"),
        "replies": [],
    }

def _normalize_review(node: Optional[Dict]) -> Optional[Dict]:
    if not node: return None
    return {
        "id": node.get("id"),
        "author": _normalize_author(node.get("author")),
        "body": node.get("body"),
        "state": node.get("state"),
        "submitted_at": node.get("submittedAt"),
    }

def _build_comment_tree(flat_comments: List[Dict]) -> List[Dict]:
    comment_map = {c["id"]: c for c in flat_comments}
    root_comments = []
    for comment in flat_comments:
        parent_id = comment.get("reply_to_id")
        if parent_id and parent_id in comment_map:
            parent_comment = comment_map[parent_id]
            parent_comment.setdefault("replies", []).append(comment)
        else:
            root_comments.append(comment)
    for c in comment_map.values():
        c.get("replies", []).sort(key=lambda x: x.get("created_at", ""))
    root_comments.sort(key=lambda x: x.get("created_at", ""))
    return root_comments

_PAGING_QUERIES = {
    "issue_comments": lambda p, c: f'... on Issue {{ comments(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}} }} }}',
    "pr_comments": lambda p, c: f'... on PullRequest {{ comments(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}} }} }}',
    "pr_reviews": lambda p, c: f'... on PullRequest {{ reviews(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{ id, author {{...AuthorFragment}}, body, state, submittedAt }} }} }}',
    "pr_review_threads": lambda p, c: f'... on PullRequest {{ reviewThreads(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{ id, path, comments(first: {PER_PAGE_REVIEW_COMMENTS}) {{ pageInfo {{...PageInfoFragment}}, nodes {{ ...CommentFields }} }} }} }} }}',
    "pr_thread_comments": lambda p, c: f'... on PullRequestReviewThread {{ comments(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{ ...CommentFields }} }} }}',
    "discussion_comments": lambda p, c: f'... on Discussion {{ comments(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{ ...CommentFields, replies(first: {p}) {{ pageInfo {{...PageInfoFragment}}, nodes {{ ...CommentFields }} }} }} }} }}',
    "discussion_replies": lambda p, c: f'... on DiscussionComment {{ replies(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{ ...CommentFields }} }} }}',
}

PAGING_CONFIG = {
    "issue_comments": {"conn": "comments", "norm": _normalize_comment, "frags": {"PageInfoFragment", "CommentFields"}, "q": _PAGING_QUERIES["issue_comments"]},
    "pr_comments": {"conn": "comments", "norm": _normalize_comment, "frags": {"PageInfoFragment", "CommentFields"}, "q": _PAGING_QUERIES["pr_comments"]},
    "pr_reviews": {"conn": "reviews", "norm": _normalize_review, "frags": {"PageInfoFragment", "AuthorFragment"}, "q": _PAGING_QUERIES["pr_reviews"]},
    "pr_review_threads": {"conn": "reviewThreads", "frags": {"PageInfoFragment", "CommentFields"}, "q": _PAGING_QUERIES["pr_review_threads"]},
    "pr_thread_comments": {"conn": "comments", "norm": _normalize_comment, "frags": {"PageInfoFragment", "CommentFields"}, "q": _PAGING_QUERIES["pr_thread_comments"]},
    "discussion_comments": {"conn": "comments", "frags": {"PageInfoFragment", "CommentFields"}, "q": _PAGING_QUERIES["discussion_comments"]},
    "discussion_replies": {"conn": "replies", "norm": _normalize_comment, "frags": {"PageInfoFragment", "CommentFields"}, "q": _PAGING_QUERIES["discussion_replies"]},
}

def _fetch_nested_data(
    session: requests.Session, batch: List[PaginationTask], session_stats: Dict
) -> Dict:
    query_parts, req_fragments = [], set()
    for i, task in enumerate(batch):
        task_config = PAGING_CONFIG[task["task_type"]]
        query_parts.append(
            f'item{i}: node(id: "{task["node_id"]}") {{ '
            f'{task_config["q"](PER_PAGE_NESTED, task["cursor"] or "")} }}'
        )
        req_fragments.update(task_config["frags"])
    if not query_parts:
        return {}
    all_frags = _get_all_fragments(req_fragments)
    fragments_str = " ".join(ALL_FRAGMENTS[name]["query"] for name in all_frags)
    query = f"query GetBatchedNestedData {{ {' '.join(query_parts)} }} {fragments_str}"
    return _send_request(session, {"query": query}, session_stats).get("data", {})

def _process_paging_queue(
    session: requests.Session, queue: deque[PaginationTask], item_map: Dict, session_stats: Dict
):
    if not queue:
        return
    while queue:
        batch = [queue.popleft() for _ in range(min(BATCH_SIZE, len(queue)))]
        if not batch:
            continue
        time.sleep(INTER_REQUEST_DELAY)
        batched_data = _fetch_nested_data(session, batch, session_stats)

        for i, task in enumerate(batch):
            item_data = batched_data.get(f"item{i}")
            if not item_data:
                continue

            config = PAGING_CONFIG[task["task_type"]]
            connection = item_data.get(config["conn"])
            if not connection or not connection.get("nodes"):
                continue

            target_item = item_map.get(task["context"]["target_item_id"])
            if not target_item:
                logging.warning(
                    f"Could not find target item {task['context']['target_item_id']}"
                    f" for task {task['task_type']}. Skipping."
                )
                continue

            target_list = None
            if task["task_type"] in ("issue_comments", "pr_comments"):
                target_list = target_item.setdefault("comments", [])
            elif task["task_type"] == "pr_reviews":
                target_list = target_item.setdefault("reviews", [])
            elif task["task_type"] == "discussion_replies":
                parent_comment = item_map.get(task["node_id"])
                if not parent_comment:
                    logging.warning(f"Parent comment {task['node_id']} not found.")
                    continue
                target_list = parent_comment.setdefault("replies", [])
            elif task["task_type"] == "pr_thread_comments":
                thread_map = target_item.get("review_threads_map", {})
                thread = thread_map.get(task["context"]["target_thread_id"])
                if not thread:
                    logging.warning(
                        f"Target thread {task['context']['target_thread_id']} not found."
                    )
                    continue
                target_list = thread.setdefault("comments", [])

            if target_list is not None:
                nodes = [config["norm"](n) for n in connection["nodes"]]
                target_list.extend(filter(None, nodes))

            if task["task_type"] == "discussion_comments":
                for comment_node in connection["nodes"]:
                    _process_disc_comment(
                        comment_node, item_map, queue, target_item["id"]
                    )

            if task["task_type"] == "pr_review_threads":
                for thread_node in connection["nodes"]:
                    _process_pr_review_thread(thread_node, target_item, queue)

            if connection.get("pageInfo", {}).get("hasNextPage"):
                task["cursor"] = connection["pageInfo"]["endCursor"]
                queue.append(task)

        logging.info(
            f"Processed batch of {len(batch)} pages. "
            f"{len(queue)} pages remaining in queue."
        )

def _queue_next_page(
    queue: deque[PaginationTask], node: Dict, conn_key: str, task_type: str, ctx: Dict
):
    connection = node.get(conn_key)
    if connection and connection.get("pageInfo", {}).get("hasNextPage"):
        queue.append({
            "node_id": node["id"],
            "cursor": connection["pageInfo"]["endCursor"],
            "task_type": task_type,
            "context": ctx,
        })

def _normalize_item_common(node: Dict) -> Dict:
    return {
        "id": node.get("id"), "number": node.get("number"), "title": node.get("title"),
        "author": _normalize_author(node.get("author")), "body": node.get("body"),
        "created_at": node.get("createdAt"), "updated_at": node.get("updatedAt"),
        "labels": [l["name"] for l in node.get("labels", {}).get("nodes", []) if l],
    }

def _process_disc_comment(
    comment_node: Dict, item_map: Dict, queue: deque, parent_id: str
):
    if not comment_node or not comment_node.get("id"): return
    norm_comment = _normalize_comment(comment_node)
    if not norm_comment: return
    parent = item_map.get(parent_id)
    if not parent: return
    parent.setdefault("comments", []).append(norm_comment)
    item_map[norm_comment["id"]] = norm_comment
    if replies_conn := comment_node.get("replies"):
        norm_comment["replies"].extend(
            filter(None, [_normalize_comment(r) for r in replies_conn.get("nodes", [])])
        )
        _queue_next_page(
            queue, comment_node, "replies", "discussion_replies",
            {"target_item_id": norm_comment["id"]}
        )

def _process_pr_review_thread(thread_node: Dict, item: Dict, queue: deque):
    if not thread_node: return
    thread_obj = {
        "id": thread_node["id"],
        "path": thread_node["path"],
        "comments": list(filter(None, [
            _normalize_comment(c) for c in thread_node.get("comments", {}).get("nodes", [])
        ]))
    }
    item.setdefault('review_threads_map', {})[thread_node['id']] = thread_obj
    _queue_next_page(
        queue, thread_node.get("comments", {}), "comments", "pr_thread_comments",
        {"target_item_id": item["id"], "target_thread_id": thread_node["id"]}
    )

def _queue_item_tasks(
    node: Dict, item_type: str, item_map: Dict, queue: deque
) -> Dict:
    item = _normalize_item_common(node)
    item.update({"comments": []})
    item_map[item["id"]] = item

    if item_type in ("issues", "pull_requests"):
        item.update({
            "state": node.get("state"),
            "assignees": [a["login"] for a in node.get("assignees", {}).get("nodes", []) if a],
        })
        item["comments"].extend(
            filter(None, [_normalize_comment(c) for c in node.get("comments", {}).get("nodes", [])])
        )
        task_type = "issue_comments" if item_type == "issues" else "pr_comments"
        _queue_next_page(
            queue, node, "comments", task_type, {"target_item_id": item["id"]}
        )

    if item_type == "pull_requests":
        item.update({
            "merged_at": node.get("mergedAt"), "is_draft": node.get("isDraft"),
            "head_ref": node.get("headRefName"), "base_ref": node.get("baseRefName"),
            "reviews": list(filter(None, [
                _normalize_review(r) for r in node.get("reviews", {}).get("nodes", [])
            ])),
            "review_threads": [],
        })
        _queue_next_page(
            queue, node, "reviews", "pr_reviews", {"target_item_id": item["id"]}
        )
        for thread_node in node.get("reviewThreads", {}).get("nodes", []):
            _process_pr_review_thread(thread_node, item, queue)
        _queue_next_page(
            queue, node, "reviewThreads", "pr_review_threads", {"target_item_id": item["id"]}
        )

    if item_type == "discussions":
        if node.get("isAnswered"):
            item["state"] = "ANSWERED"
        elif node.get("closed"):
            item["state"] = "CLOSED"
        else:
            item["state"] = "OPEN"
        item.update({"assignees": []})
        for comment_node in node.get("comments", {}).get("nodes", []):
            _process_disc_comment(comment_node, item_map, queue, item["id"])
        _queue_next_page(
            queue, node, "comments", "discussion_comments", {"target_item_id": item["id"]}
        )

    return item

def _paginator(
    session: requests.Session, query: str, variables: Dict, path: List[str], session_stats: Dict
):
    variables.pop("after", None)
    has_next_page = True
    while has_next_page:
        time.sleep(INTER_REQUEST_DELAY)
        payload = {"query": query, "variables": variables}
        response_json = _send_request(session, payload, session_stats)
        connection = response_json.get("data", {})
        for key in path:
            connection = connection.get(key) if connection else None
        if not connection or not connection.get("nodes"):
            break
        yield connection["nodes"]
        page_info = connection.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        if has_next_page:
            variables["after"] = page_info.get("endCursor")

def fetch_items(
    session: requests.Session, owner: str, repo: str, item_type: str, stats: Dict
):
    logging.info(f"Fetching all {item_type.replace('_', ' ')}...")
    item_map, page_queue, processed_items = {}, deque(), []
    config = ITEM_CONFIG[item_type]

    all_frags = _get_all_fragments(config["required_fragments"])
    fragments_str = "\n".join(ALL_FRAGMENTS[name]["query"] for name in all_frags)
    query = f"""
    query GetAllItems($owner: String!, $repo: String!, $perPage: Int!, $after: String) {{
        repository(owner: $owner, name: $repo) {{
            {config["graphql_name"]}(first: $perPage, after: $after, orderBy: {{field: CREATED_AT, direction: ASC}}) {{
                pageInfo {{ ...PageInfoFragment }}
                nodes {{ {config["fields"]} }}
            }}
        }}
    }}
    {fragments_str}
    """
    variables = {"owner": owner, "repo": repo, "perPage": PER_PAGE}
    count = 0
    for page in _paginator(session, query, variables, config["path"], stats):
        for node in page:
            if node and node.get("id"):
                item = _queue_item_tasks(node, item_type, item_map, page_queue)
                processed_items.append(item)
        count += len(page)
        logging.info(f"Processed {count} {item_type}...")

    if not processed_items:
        logging.info(f"No {item_type} found for this repository.")
        return []

    logging.info(
        f"Fetching all nested data for {len(processed_items)} items "
        f"({len(page_queue)} pages)..."
    )
    _process_paging_queue(session, page_queue, item_map, stats)

    if item_type == 'pull_requests':
        logging.info("Structuring review comment threads...")
        for item in processed_items:
            if thread_map := item.pop("review_threads_map", None):
                for thread in thread_map.values():
                    thread['comments'] = _build_comment_tree(thread.get('comments', []))
                item['review_threads'] = list(thread_map.values())

    if item_type == 'discussions':
        for item in processed_items:
             item['comments'] = _build_comment_tree(item.get('comments', []))

    logging.info(
        f"Finished {item_type}: Found and processed {len(processed_items)} items."
    )
    return processed_items

def main():
    parser = argparse.ArgumentParser(
        description="Fetch issues, PRs, and discussions from a GitHub repository."
    )
    parser.add_argument("repository", help="the repository in 'owner/name' format.")
    parser.add_argument(
        "-o", "--output", help="output JSON file name (default: owner__repo.json)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose/debug logging."
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(levelname)s: %(message)s", stream=sys.stdout
    )

    if not GITHUB_TOKEN:
        logging.error("Error: GITHUB_TOKEN environment variable not set.")
        sys.exit(1)
    logging.getLogger().addFilter(SecretFilter(GITHUB_TOKEN))

    try:
        owner, repo_name = args.repository.split("/")
        valid_owner = re.fullmatch(r"^[a-zA-Z0-9.\-_]{1,100}$", owner)
        valid_repo = re.fullmatch(r"^[a-zA-Z0-9.\-_]{1,100}$", repo_name)
        if not (valid_owner and valid_repo):
            raise ValueError("Invalid characters or length in owner/repo name.")
    except (ValueError, IndexError):
        logging.error(
            "Error: Repository must be in 'owner/name' format with valid chars."
        )
        sys.exit(1)

    output_file = args.output or f"{owner}__{repo_name}.json"
    start_time = time.time()
    session_stats = {"requests_made": 0}

    repo_data = {
        "repository": f"{owner}/{repo_name}",
        "scraped_at": datetime.now(timezone.utc).isoformat()
    }
    with requests.Session() as session:
        for item_type in ITEM_CONFIG.keys():
            try:
                data = fetch_items(session, owner, repo_name, item_type, session_stats)
                repo_data[f"{item_type}"] = data
            except (GitHubError, requests.exceptions.RequestException) as e:
                logging.error(
                    f"Could not fetch {item_type} due to an error: {e}. Skipping."
                )
                repo_data[f"{item_type}"] = []

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(repo_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Data successfully saved to {output_file}")
    except (IOError, OSError) as e:
        logging.error(f"Error writing to file {output_file}: {e}")
        sys.exit(1)

    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
