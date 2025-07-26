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

MAX_RETRIES = 5
INITIAL_BACKOFF = 2
BATCH_SIZE = 1000
RATE_LIMIT_THRESHOLD = 100
PER_PAGE = 100
PER_PAGE_NESTED = 100
INTER_REQUEST_DELAY = 0.05

PAGE_INFO_FRAGMENT = "fragment PageInfoFragment on PageInfo { endCursor, hasNextPage }"
AUTHOR_FRAGMENT = "fragment AuthorFragment on Actor { login }"
COMMENT_FIELDS = "fragment CommentFields on Comment { id, author { ...AuthorFragment }, body, createdAt }"

BASE_ITEM_FIELDS = """
    fragment BaseItemFields on Node {
        ... on Issue {
            id, number, title, state, body, createdAt, updatedAt
            author { ...AuthorFragment }
            assignees(first: 10) { nodes { login } }
            labels(first: 20) { nodes { name } }
        }
        ... on PullRequest {
            id, number, title, state, body, createdAt, updatedAt
            author { ...AuthorFragment }
            assignees(first: 10) { nodes { login } }
            labels(first: 20) { nodes { name } }
        }
    }
"""

DISCUSSION_ITEM_FIELDS = """
    fragment DiscussionItemFields on Discussion {
        id, number, title, body, createdAt, updatedAt
        author { ...AuthorFragment }
        labels(first: 20) { nodes { name } }
    }
"""

ITEM_TYPE_CONFIG = {
    "issues": {
        "graphql_name": "issues", "path": ("repository", "issues"),
        "fields": f"...BaseItemFields, comments(first: {PER_PAGE_NESTED}) {{ pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}} }}",
        "required_fragments": {"PageInfoFragment", "CommentFields", "BaseItemFields"},
    },
    "pull_requests": {
        "graphql_name": "pullRequests", "path": ("repository", "pullRequests"),
        "fields": f"""...BaseItemFields, mergedAt, isDraft, headRefName, baseRefName
            comments(first: {PER_PAGE_NESTED}) {{ pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}} }}
            reviews(first: {PER_PAGE_NESTED}) {{ pageInfo {{...PageInfoFragment}}, nodes {{ id, author {{...AuthorFragment}}, body, state, submittedAt }} }}
            reviewThreads(first: {PER_PAGE_NESTED}) {{ pageInfo {{...PageInfoFragment}}, nodes {{ id, path, comments(first: 1) {{ pageInfo {{...PageInfoFragment}} }} }} }}
        """,
        "required_fragments": {"PageInfoFragment", "CommentFields", "BaseItemFields", "AuthorFragment"},
    },
    "discussions": {
        "graphql_name": "discussions", "path": ("repository", "discussions"),
        "fields": f"""...DiscussionItemFields
            comments(first: {PER_PAGE_NESTED}) {{ pageInfo {{...PageInfoFragment}}, nodes {{ ...CommentFields, replies(first: 1) {{ pageInfo {{...PageInfoFragment}} }} }} }}
        """,
        "required_fragments": {"PageInfoFragment", "CommentFields", "DiscussionItemFields", "AuthorFragment"},
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
    "DiscussionItemFields": {"query": DISCUSSION_ITEM_FIELDS, "deps": {"AuthorFragment"}},
}

def _resolve_fragments(required: Set[str]) -> Set[str]:
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

def _handle_rate_limit(response: requests.Response, session_stats: Dict):
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

    logging.info(f"API Rate Limit (Req #{req_count}): {remaining} points remaining, {resets_in_str}.")

    if remaining < RATE_LIMIT_THRESHOLD:
        sleep_duration = max(0, reset_time - int(time.time())) + 5
        logging.warning(f"Approaching GraphQL rate limit. Sleeping for {sleep_duration} seconds...")
        time.sleep(sleep_duration)

def _perform_request(session: requests.Session, json_payload: Dict, session_stats: Dict) -> Dict:
    session_stats["requests_made"] = session_stats.get("requests_made", 0) + 1
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    logging.debug("Executing GraphQL Query (truncated):\n%s...", json_payload.get('query', '')[:500])

    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(GRAPHQL_URL, json=json_payload, headers=headers, timeout=60)
            if response.status_code in (401, 404):
                raise FatalError(f"Fatal HTTP Error: {response.status_code} - {response.text}")
            if response.status_code == 403 and "Retry-After" in response.headers:
                retry_after = min(int(response.headers["Retry-After"]), 60)
                sleep_duration = retry_after + 2
                logging.warning(f"Rate limit secondary hit. Retrying after {sleep_duration:.2f} seconds...")
                time.sleep(sleep_duration)
                continue
            response.raise_for_status()
            response_json = response.json()

            if "errors" in response_json:
                error_message = json.dumps(response_json["errors"])
                error_types = {e.get("type") for e in response_json.get("errors", [])}
                if "NOT_FOUND" in error_types and response_json.get("data") and any(val is not None for val in response_json["data"].values()):
                    logging.warning(f"GraphQL returned non-fatal NOT_FOUND errors with partial data. This may be expected (e.g., deleted items). Continuing. Errors: {error_message}")
                else:
                    raise RetriableError(f"GraphQL API returned errors: {error_message}")

            _handle_rate_limit(response, session_stats)
            return response_json
        except requests.exceptions.RequestException as e:
            last_exception = RetriableError(str(e))
        logging.warning(f"A retriable error occurred: {last_exception}. Retrying...")
        if attempt < MAX_RETRIES - 1:
            time.sleep(INITIAL_BACKOFF * (2**attempt))

    logging.error("Max retries reached. Failing.")
    raise last_exception

def _normalize_author(author_node: Optional[Dict]) -> str:
    return (author_node or {}).get("login", "ghost")

def _normalize_comment(node: Optional[Dict], context: Optional[Dict] = None) -> Optional[Dict]:
    if not node: return None
    return {"id": node.get("id"), "author": _normalize_author(node.get("author")), "body": node.get("body"), "created_at": node.get("createdAt")}

def _normalize_review(node: Optional[Dict], context: Optional[Dict] = None) -> Optional[Dict]:
    if not node: return None
    return {"id": node.get("id"), "author": _normalize_author(node.get("author")), "body": node.get("body"), "state": node.get("state"), "submitted_at": node.get("submittedAt")}

def _normalize_review_comment(node: Optional[Dict], context: Dict) -> Optional[Dict]:
    if not (norm_comment := _normalize_comment(node)): return None
    norm_comment["path"] = context.get("path", "unknown")
    return norm_comment

def _normalize_discussion_comment(node: Optional[Dict], context: Dict) -> Optional[Dict]:
    if not (norm_comment := _normalize_comment(node)): return None
    norm_comment["replies"] = []
    return norm_comment

PAGINATION_CONFIG = {
    "issue_comments": {"conn": "comments", "target": "comments", "norm": _normalize_comment, "frags": {"PageInfoFragment", "CommentFields"}, "q": lambda p, c: f'... on Issue {{ comments(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}} }} }}'},
    "pr_comments": {"conn": "comments", "target": "comments", "norm": _normalize_comment, "frags": {"PageInfoFragment", "CommentFields"}, "q": lambda p, c: f'... on PullRequest {{ comments(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}} }} }}'},
    "pr_reviews": {"conn": "reviews", "target": "reviews", "norm": _normalize_review, "frags": {"PageInfoFragment", "AuthorFragment"}, "q": lambda p, c: f'... on PullRequest {{ reviews(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{ id, author {{...AuthorFragment}}, body, state, submittedAt }} }} }}'},
    "pr_review_threads": {"conn": "reviewThreads", "target": "review_comments", "norm": _normalize_review_comment, "frags": {"PageInfoFragment", "CommentFields"}, "q": lambda p, c: f'... on PullRequest {{ reviewThreads(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{ id, path, comments(first: {p}) {{ pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}} }} }} }} }}'},
    "pr_review_thread_comments": {"conn": "comments", "target": "review_comments", "norm": _normalize_review_comment, "frags": {"PageInfoFragment", "CommentFields"}, "q": lambda p, c: f'... on PullRequestReviewThread {{ comments(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}} }} }}'},
    "discussion_comments": {"conn": "comments", "target": "comments", "norm": _normalize_discussion_comment, "frags": {"PageInfoFragment", "CommentFields"}, "q": lambda p, c: f'... on Discussion {{ comments(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{ id, author {{...AuthorFragment}}, body, createdAt, replies(first: {p}) {{ pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}} }} }} }} }}'},
    "discussion_comment_replies": {"conn": "replies", "target": "replies", "norm": _normalize_comment, "frags": {"PageInfoFragment", "CommentFields"}, "q": lambda p, c: f'... on DiscussionComment {{ replies(first: {p}, after: "{c}") {{ pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}} }} }}'},
}

def _batch_fetch_nested_data(session: requests.Session, batch: List[PaginationTask], session_stats: Dict) -> Dict:
    query_parts = []
    required_fragments: Set[str] = set()
    for i, task in enumerate(batch):
        task_config = PAGINATION_CONFIG[task["task_type"]]
        query_parts.append(f'item{i}: node(id: "{task["node_id"]}") {{ {task_config["q"](PER_PAGE_NESTED, task["cursor"] or "")} }}')
        required_fragments.update(task_config["frags"])

    if not query_parts: return {}

    all_required_fragments = _resolve_fragments(required_fragments)
    fragments_str = ' '.join(ALL_FRAGMENTS[name]["query"] for name in all_required_fragments)
    query = f"query GetBatchedNestedData {{ {' '.join(query_parts)} }} {fragments_str}"

    return _perform_request(session, {"query": query}, session_stats).get("data", {})

def _process_pagination_queue(session: requests.Session, queue: deque[PaginationTask], item_map: Dict[str, Dict[str, Any]], session_stats: Dict):
    total_pages = len(queue)
    processed_pages = 0
    pages_since_last_log = 0
    log_increment = max(1, total_pages // 20)

    while queue:
        batch = [queue.popleft() for _ in range(min(BATCH_SIZE, len(queue)))]
        if not batch: continue

        time.sleep(INTER_REQUEST_DELAY)

        batched_data = _batch_fetch_nested_data(session, batch, session_stats)

        for i, task in enumerate(batch):
            item_data = batched_data.get(f"item{i}")
            if not item_data: continue

            config = PAGINATION_CONFIG[task["task_type"]]
            connection = item_data.get(config["conn"])
            if not connection: continue

            nodes_to_process = connection.get("nodes", [])

            if task["task_type"] == "discussion_comments":
                parent_item_id = task["context"]["target_item_id"]
                for comment_node in nodes_to_process:
                    _process_discussion_comment(comment_node, item_map, queue, parent_item_id)
            else:
                target_item = item_map[task["context"]["target_item_id"]]
                target_list = target_item.setdefault(config["target"], [])

                for node in nodes_to_process:
                    if not node: continue
                    context = {"path": node.get("path") if config["conn"] == "reviewThreads" else task["context"].get("path")}

                    comments_to_process = node.get("comments", {}).get("nodes", []) if config["conn"] == "reviewThreads" else [node]
                    for sub_node in comments_to_process:
                        if norm_node := config["norm"](sub_node, context):
                            target_list.append(norm_node)

                    if node.get("comments", {}).get("pageInfo", {}).get("hasNextPage"):
                        queue.append({"node_id": node["id"], "cursor": node["comments"]["pageInfo"]["endCursor"], "task_type": "pr_review_thread_comments", "context": {"target_item_id": target_item["id"], "path": node["path"]}})
                    elif node.get("replies", {}).get("pageInfo", {}).get("hasNextPage"):
                        if norm_node and norm_node.get("id"):
                            queue.append({"node_id": norm_node["id"], "cursor": node["replies"]["pageInfo"]["endCursor"], "task_type": "discussion_comment_replies", "context": {"target_item_id": norm_node["id"]}})

            if connection.get("pageInfo", {}).get("hasNextPage"):
                task["cursor"] = connection["pageInfo"]["endCursor"]
                queue.append(task.copy())

        processed_pages += len(batch)
        pages_since_last_log += len(batch)
        if total_pages > BATCH_SIZE and pages_since_last_log >= log_increment:
            logging.info(f"Processed {processed_pages}/{total_pages} nested data pages...")
            pages_since_last_log = 0

def _queue_paginated(queue: deque[PaginationTask], node: Dict, connection_key: str, task_type: str, context: Dict):
    connection = node.get(connection_key)
    if connection and connection.get("pageInfo", {}).get("hasNextPage"):
        queue.append({"node_id": node["id"], "cursor": connection["pageInfo"]["endCursor"], "task_type": task_type, "context": context})

def _normalize_base(node: Dict) -> Dict:
    return {
        "id": node.get("id"),
        "number": node.get("number"),
        "title": node.get("title"),
        "author": _normalize_author(node.get("author")),
        "state": node.get("state"),
        "body": node.get("body"),
        "created_at": node.get("createdAt"),
        "updated_at": node.get("updatedAt"),
        "labels": [l["name"] for l in node.get("labels", {}).get("nodes", []) if l],
        "assignees": [a["login"] for a in node.get("assignees", {}).get("nodes", []) if a],
        "comments": [],
    }

def _normalize_discussion(node: Dict) -> Dict:
    return {
        "id": node.get("id"),
        "number": node.get("number"),
        "title": node.get("title"),
        "author": _normalize_author(node.get("author")),
        "body": node.get("body"),
        "created_at": node.get("createdAt"),
        "updated_at": node.get("updatedAt"),
        "state": None,
        "labels": [l["name"] for l in node.get("labels", {}).get("nodes", []) if l],
        "assignees": [],
        "comments": [],
    }

def _process_discussion_comment(comment_node: Dict, item_map: Dict[str, Dict[str, Any]], queue: deque[PaginationTask], parent_item_id: str):
    if not comment_node or not comment_node.get("id"):
        return

    norm_comment = _normalize_discussion_comment(comment_node, {})
    if not norm_comment:
        return

    parent_item = item_map[parent_item_id]
    parent_item.setdefault("comments", []).append(norm_comment)
    item_map[norm_comment["id"]] = norm_comment

    replies_connection = comment_node.get("replies", {})
    if replies_connection and replies_connection.get("nodes"):
        norm_comment["replies"].extend(
            [_normalize_comment(r) for r in replies_connection["nodes"] if r]
        )

    _queue_paginated(queue, comment_node, "replies", "discussion_comment_replies", {"target_item_id": norm_comment["id"]})

def _queue_work(node: Dict, item_type: str, item_map: Dict[str, Dict[str, Any]], queue: deque[PaginationTask]):
    if item_type == "discussions":
        item = _normalize_discussion(node)
    else:
        item = _normalize_base(node)
    item_map[item["id"]] = item

    if item_type in ("issues", "pull_requests"):
        item["comments"].extend([_normalize_comment(c) for c in node.get("comments", {}).get("nodes", []) if c])
        _queue_paginated(queue, node, "comments", f"{'issue' if item_type == 'issues' else 'pr'}_comments", {"target_item_id": item["id"]})

    if item_type == "pull_requests":
        item.update({
            "merged_at": node.get("mergedAt"),
            "is_draft": node.get("isDraft"),
            "head_ref": node.get("headRefName"),
            "base_ref": node.get("baseRefName"),
            "reviews": [],
            "review_comments": [],
        })
        item["reviews"].extend([_normalize_review(r) for r in node.get("reviews", {}).get("nodes", []) if r])
        _queue_paginated(queue, node, "reviews", "pr_reviews", {"target_item_id": item["id"]})

        for thread in node.get("reviewThreads", {}).get("nodes", []):
            if not thread: continue
            item["review_comments"].extend([_normalize_review_comment(c, {"path": thread.get("path")}) for c in thread.get("comments", {}).get("nodes", []) if c])
            _queue_paginated(queue, thread, "comments", "pr_review_thread_comments", {"target_item_id": item["id"], "path": thread.get("path")})
        _queue_paginated(queue, node, "reviewThreads", "pr_review_threads", {"target_item_id": item["id"]})

    if item_type == "discussions":
        for comment_node in node.get("comments", {}).get("nodes", []):
            _process_discussion_comment(comment_node, item_map, queue, item["id"])
        _queue_paginated(queue, node, "comments", "discussion_comments", {"target_item_id": item["id"]})

    return item

def _paginator(session: requests.Session, query: str, variables: Dict, path: List[str], session_stats: Dict):
    variables.pop("after", None)
    has_next_page = True
    while has_next_page:
        time.sleep(INTER_REQUEST_DELAY)
        response_json = _perform_request(session, {"query": query, "variables": variables}, session_stats)
        connection = response_json.get("data", {})
        for key in path: connection = connection.get(key) if connection else None
        if not connection: break

        nodes = connection.get("nodes", [])
        if not nodes: break
        yield nodes

        page_info = connection.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        if has_next_page: variables["after"] = page_info.get("endCursor")

def fetch_process(session: requests.Session, owner: str, repo: str, item_type: str, session_stats: Dict):
    logging.info(f"Fetching all {item_type.replace('_', ' ')}...")
    item_map: Dict[str, Dict[str, Any]] = {}
    pagination_queue: deque[PaginationTask] = deque()
    all_processed_items = []
    config = ITEM_TYPE_CONFIG[item_type]

    def get_query(graphql_name: str, fields_fragment: str, initial_fragments: Set[str]) -> str:
        all_required_fragments = _resolve_fragments(initial_fragments)
        fragments_str = "\n".join(ALL_FRAGMENTS[name]["query"] for name in all_required_fragments)
        return f"""query Get{graphql_name.capitalize()}($owner: String!, $repo: String!, $perPage: Int!, $after: String) {{
            repository(owner: $owner, name: $repo) {{
                {graphql_name}(first: $perPage, after: $after, orderBy: {{field: CREATED_AT, direction: ASC}}) {{
                    pageInfo {{ ...PageInfoFragment }}
                    nodes {{ {fields_fragment} }}
                }}
            }}
        }} {fragments_str}"""

    query = get_query(config["graphql_name"], config["fields"], config["required_fragments"])
    variables = {"owner": owner, "repo": repo, "perPage": PER_PAGE}
    paginator = _paginator(session, query, variables, config["path"], session_stats)

    count = 0
    for page_of_nodes in paginator:
        for node in page_of_nodes:
            if node and node.get("id"):
                item = _queue_work(node, item_type, item_map, pagination_queue)
                all_processed_items.append(item)
        count += len(page_of_nodes)
        logging.info(f"Processed {count} {item_type}...")

    if not all_processed_items:
        logging.info(f"No {item_type} found for this repository.")
        return []

    logging.info(f"Fetching all nested data for {len(all_processed_items)} items ({len(pagination_queue)} pages)...")
    _process_pagination_queue(session, pagination_queue, item_map, session_stats)

    logging.info(f"Finished {item_type}: Found and processed {len(all_processed_items)} items.")
    return all_processed_items

def main():
    parser = argparse.ArgumentParser(description="Fetch issues, PRs, and discussions from a GitHub repository.")
    parser.add_argument("repository", help="the repository in 'owner/name' format.")
    parser.add_argument("-o", "--output", help="output JSON file name (default: owner__repo.json).")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose/debug logging.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s", stream=sys.stdout)

    if not GITHUB_TOKEN:
        logging.error("Error: GITHUB_TOKEN environment variable not set.")
        sys.exit(1)
    logging.getLogger().addFilter(SecretFilter(GITHUB_TOKEN))

    try:
        owner, repo_name = args.repository.split("/")
        if not re.fullmatch(r"^[a-zA-Z0-9.\-_]{1,100}$", owner) or \
           not re.fullmatch(r"^[a-zA-Z0-9.\-_]{1,100}$", repo_name):
            raise ValueError("Invalid characters or length in owner/repo name.")
        if owner.startswith('-') or owner.endswith('-') or \
           repo_name.startswith('-') or repo_name.endswith('-'):
            raise ValueError("Owner or repository name cannot begin or end with a hyphen.")
    except (ValueError, IndexError):
        logging.error("Error: Repository name must be in 'owner/name' format with valid characters.")
        sys.exit(1)

    output_file = args.output or f"{owner}__{repo_name}.json"
    start_time = time.time()
    session_stats = {"requests_made": 0}

    scraped_at_utc = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    repo_data = {"repository": f"{owner}/{repo_name}", "scraped_at": scraped_at_utc}
    with requests.Session() as session:
        for item_type in ITEM_TYPE_CONFIG.keys():
            try:
                data = fetch_process(
                    session, owner, repo_name, item_type, session_stats
                )
                repo_data[item_type] = data
            except FatalError as e:
                logging.critical(f"A fatal error occurred: {e}. Aborting.")
                sys.exit(1)
            except GitHubError as e:
                logging.error(f"Could not fetch {item_type} due to an error: {e}. Skipping this section.")
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
