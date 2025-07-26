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
PER_PAGE_REVIEW_COMMENTS = 40
INITIAL_REPLIES = 40
BATCH_SIZE = 1000
MAX_RETRIES = 5
INITIAL_BACKOFF = 2
REQUEST_TIMEOUT = 60
INTER_REQUEST_DELAY = 0.05
RATE_LIMIT_THRESHOLD = 100

PAGE_INFO_FRAGMENT = "fragment PageInfoFragment on PageInfo { endCursor, hasNextPage }"
AUTHOR_FRAGMENT = "fragment AuthorFragment on Actor { login }"
COMMENT_FIELDS = """
    fragment CommentFields on Comment {
        id, author { ...AuthorFragment }, body, createdAt
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
        id, number, title, body, createdAt, updatedAt, closed, locked
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
                        nodes {{
                            id, author {{ ...AuthorFragment }}, body, createdAt,
                            replyTo {{ id }}
                        }}
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
    req_count = session_stats.get('requests_made', 0)

    time_until_reset = reset_time - int(time.time())
    minutes, seconds = divmod(max(0, time_until_reset), 60)
    resets_in_str = f"resets in {minutes}m {seconds}s"

    logging.info(
        f"API Rate Limit (Req #{req_count}): {remaining} points remaining, "
        f"{resets_in_str}."
    )

    if remaining < RATE_LIMIT_THRESHOLD:
        sleep_duration = max(0, reset_time - int(time.time())) + 5
        mins_to_sleep, secs_to_sleep = divmod(sleep_duration, 60)
        if mins_to_sleep > 0:
            sleep_msg = f"{int(mins_to_sleep)}m {int(secs_to_sleep)}s"
        else:
            sleep_msg = f"{sleep_duration:.2f} seconds"
        logging.warning(f"Approaching GraphQL rate limit. Sleeping for {sleep_msg}...")
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
    query_preview = json_payload.get('query', '')[:500]
    logging.debug(f"Executing GraphQL Query (truncated):\n{query_preview}...")

    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(
                GRAPHQL_URL,
                json=json_payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code in (401, 404):
                msg = f"Fatal HTTP Error: {response.status_code} - {response.text}"
                raise FatalError(msg)
            if response.status_code == 403 and "Retry-After" in response.headers:
                retry_after = min(int(response.headers["Retry-After"]), 60)
                sleep_duration = retry_after + 2
                logging.warning(
                    "Rate limit secondary hit. "
                    f"Retrying after {sleep_duration:.2f} seconds..."
                )
                time.sleep(sleep_duration)
                continue
            response.raise_for_status()
            response_json = response.json()

            if "errors" in response_json:
                error_list = []
                for i, error in enumerate(response_json.get("errors", [])):
                    if i >= 3:
                        error_list.append("...")
                        break
                    err_type = error.get("type", "UnknownType")
                    err_msg = error.get("message", "NoMessage")
                    error_list.append(f"({err_type}) {err_msg}")
                err_summary = "; ".join(error_list)

                error_types = {e.get("type") for e in response_json.get("errors", [])}
                data = response_json.get("data")
                if "NOT_FOUND" in error_types and data and any(data.values()):
                    logging.warning(
                        "GraphQL returned non-fatal NOT_FOUND errors with partial "
                        f"data. This may be expected. Errors: {err_summary}"
                    )
                else:
                    raise RetriableError(f"GraphQL API returned errors: {err_summary}")

            _check_rate_limit(response, session_stats)
            return response_json
        except requests.exceptions.RequestException as e:
            last_exception = RetriableError(str(e))
        logging.warning(f"A retriable error occurred: {last_exception}. Retrying...")
        if attempt < MAX_RETRIES - 1:
            time.sleep(INITIAL_BACKOFF * (2**attempt))

    logging.error("Max retries reached. Failing.")
    raise last_exception

def _norm_author(author_node: Optional[Dict]) -> str:
    return (author_node or {}).get("login", "ghost")

def _norm_comment(
    node: Optional[Dict], context: Optional[Dict] = None
) -> Optional[Dict]:
    if not node: return None
    return {
        "id": node.get("id"),
        "author": _norm_author(node.get("author")),
        "body": node.get("body"),
        "created_at": node.get("createdAt"),
    }

def _norm_review(
    node: Optional[Dict], context: Optional[Dict] = None
) -> Optional[Dict]:
    if not node: return None
    return {
        "id": node.get("id"),
        "author": _norm_author(node.get("author")),
        "body": node.get("body"),
        "state": node.get("state"),
        "submitted_at": node.get("submittedAt"),
    }

def _norm_review_comment(
    node: Optional[Dict], context: Dict
) -> Optional[Dict]:
    if not node: return None
    return {
        "id": node.get("id"),
        "author": _norm_author(node.get("author")),
        "body": node.get("body"),
        "created_at": node.get("createdAt"),
        "path": context.get("path", "unknown"),
        "reply_to_id": (node.get("replyTo") or {}).get("id"),
    }

def _norm_disc_comment(
    node: Optional[Dict], context: Dict
) -> Optional[Dict]:
    if not (norm_comment := _norm_comment(node)):
        return None
    norm_comment["replies"] = []
    return norm_comment

def _comment_page_query(item_type: str, p: int, c: str) -> str:
    return f"""
    ... on {item_type} {{
        comments(first: {p}, after: "{c}") {{
            pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}}
        }}
    }}
    """

PAGING_CONFIG = {
    "issue_comments": {
        "conn": "comments", "target": "comments", "norm": _norm_comment,
        "frags": {"PageInfoFragment", "CommentFields"},
        "q": lambda p, c: _comment_page_query("Issue", p, c),
    },
    "pr_comments": {
        "conn": "comments", "target": "comments", "norm": _norm_comment,
        "frags": {"PageInfoFragment", "CommentFields"},
        "q": lambda p, c: _comment_page_query("PullRequest", p, c),
    },
    "pr_reviews": {
        "conn": "reviews", "target": "reviews", "norm": _norm_review,
        "frags": {"PageInfoFragment", "AuthorFragment"},
        "q": lambda p, c: f'''... on PullRequest {{
            reviews(first: {p}, after: "{c}") {{
                pageInfo {{...PageInfoFragment}},
                nodes {{ id, author {{...AuthorFragment}}, body, state, submittedAt }}
            }}
        }}''',
    },
    "pr_review_threads": {
        "conn": "reviewThreads", "target": "review_comments",
        "norm": _norm_review_comment,
        "frags": {"PageInfoFragment", "AuthorFragment"},
        "q": lambda p, c: f'''... on PullRequest {{
            reviewThreads(first: {p}, after: "{c}") {{
                pageInfo {{...PageInfoFragment}},
                nodes {{
                    id, path,
                    comments(first: {p}) {{
                        pageInfo {{...PageInfoFragment}},
                        nodes {{
                            id, author {{ ...AuthorFragment }}, body, createdAt,
                            replyTo {{ id }}
                        }}
                    }}
                }}
            }}
        }}''',
    },
    "PR_THREAD_COMMENTS": {
        "conn": "comments", "target": "review_comments",
        "norm": _norm_review_comment,
        "frags": {"PageInfoFragment", "AuthorFragment"},
        "q": lambda p, c: f'''... on PullRequestReviewThread {{
            comments(first: {p}, after: "{c}") {{
                pageInfo {{...PageInfoFragment}},
                nodes {{
                    id, author {{ ...AuthorFragment }}, body, createdAt,
                    replyTo {{ id }}
                }}
            }}
        }}''',
    },
    "discussion_comments": {
        "conn": "comments", "target": "comments",
        "norm": _norm_disc_comment,
        "frags": {"PageInfoFragment", "CommentFields"},
        "q": lambda p, c: f'''... on Discussion {{
            comments(first: {p}, after: "{c}") {{
                pageInfo {{...PageInfoFragment}},
                nodes {{
                    id, author {{...AuthorFragment}}, body, createdAt,
                    replies(first: {p}) {{
                        pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}}
                    }}
                }}
            }}
        }}''',
    },
    "DISCUSSION_REPLIES": {
        "conn": "replies", "target": "replies", "norm": _norm_comment,
        "frags": {"PageInfoFragment", "CommentFields"},
        "q": lambda p, c: f'''... on DiscussionComment {{
            replies(first: {p}, after: "{c}") {{
                pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}}
            }}
        }}''',
    },
}

def _fetch_nested_data(
    session: requests.Session, batch: List[PaginationTask], session_stats: Dict
) -> Dict:
    query_parts = []
    req_fragments: Set[str] = set()
    for i, task in enumerate(batch):
        task_config = PAGING_CONFIG[task["task_type"]]
        query_parts.append(
            f'item{i}: node(id: "{task["node_id"]}") {{ '
            f'{task_config["q"](PER_PAGE_NESTED, task["cursor"] or "")} }}'
        )
        req_fragments.update(task_config["frags"])

    if not query_parts:
        return {}

    all_required = _get_all_fragments(req_fragments)
    fragments_str = ' '.join(ALL_FRAGMENTS[name]["query"] for name in all_required)
    query = f"query GetBatchedNestedData {{ {' '.join(query_parts)} }} {fragments_str}"

    return _send_request(session, {"query": query}, session_stats).get("data", {})

def _process_paging_queue(
    session: requests.Session,
    queue: deque[PaginationTask],
    item_map: Dict[str, Dict[str, Any]],
    session_stats: Dict,
):
    total_pages = len(queue)
    processed_pages = 0
    pages_since_log = 0
    log_increment = max(1, total_pages // 20)

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
            if not connection:
                continue

            nodes = connection.get("nodes", [])

            if task["task_type"] == "discussion_comments":
                parent_id = task["context"]["target_item_id"]
                for comment_node in nodes:
                    _process_disc_comment(comment_node, item_map, queue, parent_id)
            else:
                target_item = item_map[task["context"]["target_item_id"]]
                target_list = target_item.setdefault(config["target"], [])

                for node in nodes:
                    if not node: continue
                    is_thread = config["conn"] == "reviewThreads"
                    path = node.get("path") if is_thread else task["context"].get("path")
                    context = {"path": path}
                    sub_nodes = (
                        node.get("comments", {}).get("nodes", []) if is_thread else [node]
                    )
                    for sub_node in sub_nodes:
                        if norm_node := config["norm"](sub_node, context):
                            target_list.append(norm_node)

                    comments_conn = node.get("comments", {})
                    if comments_conn.get("pageInfo", {}).get("hasNextPage"):
                        queue.append({
                            "node_id": node["id"],
                            "cursor": comments_conn["pageInfo"]["endCursor"],
                            "task_type": "PR_THREAD_COMMENTS",
                            "context": {
                                "target_item_id": target_item["id"], "path": node["path"]
                            },
                        })

            if connection.get("pageInfo", {}).get("hasNextPage"):
                task["cursor"] = connection["pageInfo"]["endCursor"]
                queue.append(task.copy())

        processed_pages += len(batch)
        pages_since_log += len(batch)
        if total_pages > BATCH_SIZE and pages_since_log >= log_increment:
            logging.info(
                f"Processed {processed_pages}/{total_pages} nested data pages..."
            )
            pages_since_log = 0

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

def _norm_common_fields(node: Dict) -> Dict:
    return {
        "id": node.get("id"),
        "number": node.get("number"),
        "title": node.get("title"),
        "author": _norm_author(node.get("author")),
        "body": node.get("body"),
        "created_at": node.get("createdAt"),
        "updated_at": node.get("updatedAt"),
        "labels": [l["name"] for l in node.get("labels", {}).get("nodes", []) if l],
    }

def _norm_base_item(node: Dict) -> Dict:
    item = _norm_common_fields(node)
    item.update({
        "state": node.get("state"),
        "assignees": [a["login"] for a in node.get("assignees", {}).get("nodes", []) if a],
        "comments": [],
    })
    return item

def _norm_discussion(node: Dict) -> Dict:
    item = _norm_common_fields(node)
    state = "OPEN"
    if node.get("locked"):
        state = "LOCKED"
    elif node.get("closed"):
        state = "CLOSED"
    item.update({"state": state, "assignees": [], "comments": []})
    return item

def _process_disc_comment(
    comment_node: Dict,
    item_map: Dict[str, Dict[str, Any]],
    queue: deque[PaginationTask],
    parent_item_id: str,
):
    if not comment_node or not comment_node.get("id"):
        return

    norm_comment = _norm_disc_comment(comment_node, {})
    if not norm_comment:
        return

    parent_item = item_map[parent_item_id]
    parent_item.setdefault("comments", []).append(norm_comment)
    item_map[norm_comment["id"]] = norm_comment

    replies_conn = comment_node.get("replies", {})
    if replies_conn and replies_conn.get("nodes"):
        norm_comment["replies"].extend(
            [_norm_comment(r) for r in replies_conn["nodes"] if r]
        )

    _queue_next_page(
        queue, comment_node, "replies", "DISCUSSION_REPLIES",
        {"target_item_id": norm_comment["id"]}
    )

def _queue_item_tasks(
    node: Dict,
    item_type: str,
    item_map: Dict[str, Dict[str, Any]],
    queue: deque[PaginationTask],
):
    if item_type == "discussions":
        item = _norm_discussion(node)
    else:
        item = _norm_base_item(node)
    item_map[item["id"]] = item

    if item_type in ("issues", "pull_requests"):
        nodes = node.get("comments", {}).get("nodes", [])
        item["comments"].extend([_norm_comment(c) for c in nodes if c])
        task_type = "issue_comments" if item_type == "issues" else "pr_comments"
        _queue_next_page(
            queue, node, "comments", task_type, {"target_item_id": item["id"]}
        )

    if item_type == "pull_requests":
        item.update({
            "merged_at": node.get("mergedAt"),
            "is_draft": node.get("isDraft"),
            "head_ref": node.get("headRefName"),
            "base_ref": node.get("baseRefName"),
            "reviews": [],
            "review_comments": [],
        })
        review_nodes = node.get("reviews", {}).get("nodes", [])
        item["reviews"].extend([_norm_review(r) for r in review_nodes if r])
        _queue_next_page(
            queue, node, "reviews", "pr_reviews", {"target_item_id": item["id"]}
        )

        for thread in node.get("reviewThreads", {}).get("nodes", []):
            if not thread: continue
            cmt_nodes = thread.get("comments", {}).get("nodes", [])
            item["review_comments"].extend([
                _norm_review_comment(c, {"path": thread.get("path")})
                for c in cmt_nodes if c
            ])
            _queue_next_page(
                queue, thread, "comments", "PR_THREAD_COMMENTS",
                {"target_item_id": item["id"], "path": thread.get("path")}
            )
        _queue_next_page(
            queue, node, "reviewThreads", "pr_review_threads",
            {"target_item_id": item["id"]}
        )

    if item_type == "discussions":
        for comment_node in node.get("comments", {}).get("nodes", []):
            _process_disc_comment(comment_node, item_map, queue, item["id"])
        _queue_next_page(
            queue, node, "comments", "discussion_comments",
            {"target_item_id": item["id"]}
        )

    return item

def _paginator(
    session: requests.Session,
    query: str,
    variables: Dict,
    path: List[str],
    session_stats: Dict,
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
        if not connection:
            break

        nodes = connection.get("nodes", [])
        if not nodes:
            break
        yield nodes

        page_info = connection.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        if has_next_page:
            variables["after"] = page_info.get("endCursor")

def fetch_items(
    session: requests.Session, owner: str, repo: str, item_type: str, stats: Dict
):
    logging.info(f"Fetching all {item_type.replace('_', ' ')}...")
    item_map: Dict[str, Dict[str, Any]] = {}
    page_queue: deque[PaginationTask] = deque()
    processed_items = []
    config = ITEM_CONFIG[item_type]

    def get_query(
        graphql_name: str, fields_fragment: str, req_fragments: Set[str]
    ) -> str:
        all_required = _get_all_fragments(req_fragments)
        fragments_str = "\n".join(ALL_FRAGMENTS[name]["query"] for name in all_required)
        return f"""
        query Get{graphql_name.capitalize()}(
            $owner: String!, $repo: String!, $perPage: Int!, $after: String
        ) {{
            repository(owner: $owner, name: $repo) {{
                {graphql_name}(
                    first: $perPage,
                    after: $after,
                    orderBy: {{field: CREATED_AT, direction: ASC}}
                ) {{
                    pageInfo {{ ...PageInfoFragment }}
                    nodes {{ {fields_fragment} }}
                }}
            }}
        }}
        {fragments_str}
        """

    query = get_query(
        config["graphql_name"], config["fields"], config["required_fragments"]
    )
    variables = {"owner": owner, "repo": repo, "perPage": PER_PAGE}
    paginator = _paginator(session, query, variables, config["path"], stats)

    count = 0
    for page_of_nodes in paginator:
        for node in page_of_nodes:
            if node and node.get("id"):
                item = _queue_item_tasks(node, item_type, item_map, page_queue)
                processed_items.append(item)
        count += len(page_of_nodes)
        logging.info(f"Processed {count} {item_type}...")

    if not processed_items:
        logging.info(f"No {item_type} found for this repository.")
        return []

    logging.info(
        f"Fetching all nested data for {len(processed_items)} items "
        f"({len(page_queue)} pages)..."
    )
    _process_paging_queue(session, page_queue, item_map, stats)

    logging.info(f"Finished {item_type}: Found and processed {len(processed_items)} items.")
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
        is_valid_owner = re.fullmatch(r"^[a-zA-Z0-9.\-_]{1,100}$", owner)
        is_valid_repo = re.fullmatch(r"^[a-zA-Z0-9.\-_]{1,100}$", repo_name)
        if not (is_valid_owner and is_valid_repo):
            raise ValueError("Invalid characters or length in owner/repo name.")
        if owner.startswith('-') or owner.endswith('-') or \
           repo_name.startswith('-') or repo_name.endswith('-'):
            raise ValueError("Owner/repo name cannot begin or end with a hyphen.")
    except (ValueError, IndexError):
        logging.error(
            "Error: Repository must be in 'owner/name' format with valid chars."
        )
        sys.exit(1)

    output_file = args.output or f"{owner}__{repo_name}.json"
    start_time = time.time()
    session_stats = {"requests_made": 0}

    scraped_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    repo_data = {"repository": f"{owner}/{repo_name}", "scraped_at": scraped_at}
    with requests.Session() as session:
        for item_type in ITEM_CONFIG.keys():
            try:
                data = fetch_items(
                    session, owner, repo_name, item_type, session_stats
                )
                repo_data[item_type] = data
            except FatalError as e:
                logging.critical(f"A fatal error occurred: {e}. Aborting.")
                sys.exit(1)
            except GitHubError as e:
                logging.error(
                    f"Could not fetch {item_type} due to an error: {e}. "
                    "Skipping this section."
                )
                repo_data[item_type] = []

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
