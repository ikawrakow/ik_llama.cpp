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
from typing import Dict, List, Any, Optional, TypedDict, Set, Callable, Tuple, cast, Union

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GRAPHQL_URL = "https://api.github.com/graphql"

PER_PAGE = 100
PER_PAGE_NESTED = 100
PER_PAGE_REVIEW_COMMENTS = 45
INITIAL_REPLIES = 45
BATCH_SIZE = 500
MAX_RETRIES = 5
INITIAL_BACKOFF = 2
REQUEST_TIMEOUT = 60
REQUEST_DELAY = 0.05
RATE_LIMIT_BUFFER = 100

PAGE_INFO_FRAG = "fragment PageInfoFragment on PageInfo { endCursor, hasNextPage }"
AUTHOR_FRAG = "fragment AuthorFragment on Actor { login }"
COMMENT_FRAG = """
    fragment CommentFields on Comment {
        id, author { ...AuthorFragment }, body, createdAt,
        ... on DiscussionComment { replyTo { id } }
        ... on PullRequestReviewComment { replyTo { id } }
    }
"""
BASE_ITEM_FRAG = """
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
DISCUSSION_FRAG = """
    fragment DiscussionFields on Discussion {
        id, number, title, body, createdAt, updatedAt, closed, isAnswered
        author { ...AuthorFragment }
        labels(first: 20) { nodes { name } }
    }
"""

class GitHubError(Exception): pass
class FatalError(GitHubError): pass
class RetriableError(GitHubError): pass

class PaginationTask(TypedDict):
    node_id: str
    cursor: Optional[str]
    task_type: str
    context: Dict[str, Any]

class ReviewThread(TypedDict):
    id: str
    path: str
    comments: List[Dict[str, Any]]

class BaseItem(TypedDict):
    id: str
    number: int
    title: str
    author: str
    body: str
    created_at: str
    updated_at: str
    labels: List[str]
    comments: List[Dict[str, Any]]

class IssueItem(BaseItem):
    state: str
    assignees: List[str]

class DiscussionItem(BaseItem):
    state: str
    assignees: List[str]

class PullRequestItem(IssueItem):
    merged_at: Optional[str]
    is_draft: bool
    head_ref: str
    base_ref: str
    reviews: List[Dict[str, Any]]
    review_threads: List[ReviewThread]

FinalItem = Union[IssueItem, PullRequestItem, DiscussionItem]
NestedData = Dict[str, Any]
AllNestedData = Dict[str, NestedData]

class ItemConfig(TypedDict):
    graphql_name: str
    path: Tuple[str, ...]
    fields: str
    fragment_deps: Set[str]
    parser_func: Callable[[Dict[str, Any], Optional[Set[str]]], Tuple[Dict[str, Any], List[PaginationTask]]]

class PagingConfigBase(TypedDict):
    conn_name: str
    frags: Set[str]
    q: Callable[[int, str], str]
    extract_func: Callable[[Dict[str, Any], PaginationTask], Tuple[Any, List[PaginationTask]]]
    target_key: str

class PagingConfigWithNormalizer(PagingConfigBase):
    normalizer: Callable[[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]

class TokenFilter(logging.Filter):
    def __init__(self, secret: str):
        super().__init__()
        self.secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        if self.secret:
            record.msg = str(record.msg).replace(self.secret, "[REDACTED_TOKEN]")
            if isinstance(record.args, dict):
                record.args = {
                    k: str(v).replace(self.secret, "[REDACTED_TOKEN]")
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(
                    str(arg).replace(self.secret, "[REDACTED_TOKEN]")
                    for arg in record.args
                )
        return True

class GQLFragment(TypedDict):
    query: str
    deps: Set[str]

class StatsTracker:
    def __init__(self):
        self.requests_made: int = 0
    def increment(self):
        self.requests_made += 1
    def get_count(self) -> int:
        return self.requests_made

ALL_FRAGMENTS: Dict[str, GQLFragment] = {
    "PageInfoFragment": {"query": PAGE_INFO_FRAG, "deps": set()},
    "AuthorFragment": {"query": AUTHOR_FRAG, "deps": set()},
    "CommentFields": {"query": COMMENT_FRAG, "deps": {"AuthorFragment"}},
    "BaseItemFields": {"query": BASE_ITEM_FRAG, "deps": {"AuthorFragment"}},
    "DiscussionFields": {"query": DISCUSSION_FRAG, "deps": {"AuthorFragment"}},
}

def resolve_frags(required: Set[str]) -> Set[str]:
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

def check_rate_limit(response: requests.Response, stats: StatsTracker):
    remaining_str = response.headers.get("X-RateLimit-Remaining")
    reset_str = response.headers.get("X-RateLimit-Reset")
    if not remaining_str or not reset_str:
        return

    remaining = int(remaining_str)
    reset_time = int(reset_str)
    reset_delay = reset_time - int(time.time())
    minutes, seconds = divmod(max(0, reset_delay), 60)
    reset_info = f"resets in {minutes}m {seconds}s"
    req_num = stats.get_count()
    logging.info(
        f"API Rate Limit (Req #{req_num}): {remaining} points remaining, {reset_info}."
    )
    if remaining < RATE_LIMIT_BUFFER:
        sleep_duration = max(0, reset_delay) + 5
        logging.warning(
            f"Approaching GraphQL rate limit. Sleeping for {sleep_duration:.2f}s..."
        )
        time.sleep(sleep_duration)

def run_gql(
    session: requests.Session, json_payload: Dict[str, Any], stats: StatsTracker
) -> Dict[str, Any]:
    stats.increment()
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            while True:
                response = session.post(
                    GRAPHQL_URL,
                    json=json_payload,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
                if response.status_code == 403 and "Retry-After" in response.headers:
                    retry_after = min(int(response.headers["Retry-After"]), 60)
                    logging.warning(
                        "Secondary rate limit hit. Retrying after "
                        f"{retry_after + 2}s."
                    )
                    time.sleep(retry_after + 2)
                    continue
                break
            if response.status_code >= 500:
                raise RetriableError(
                    f"Server Error: {response.status_code} {response.reason}"
                )
            if response.status_code in (401, 404):
                raise FatalError(
                    f"Fatal HTTP Error: {response.status_code} - {response.text}"
                )
            if response.status_code == 403:
                if "rate limit" not in response.text.lower():
                    msg = (
                        "Fatal HTTP Error: 403 Forbidden. This is likely a token "
                        "permission issue (e.g., no 'read:discussion' scope) "
                        f"or access problem. Response: {response.text}"
                    )
                    raise FatalError(msg)
            response.raise_for_status()
            resp_data = response.json()
            if not isinstance(resp_data, dict):
                raise RetriableError(f"Unexpected API response type: {type(resp_data)}")

            if "errors" in resp_data:
                err_summary = "; ".join(
                    f"({e.get('type')}) {e.get('message')}" for e in resp_data["errors"]
                )
                error_types = {e.get("type") for e in resp_data["errors"]}
                if "NOT_FOUND" in error_types and resp_data.get("data"):
                    logging.warning(
                        f"GraphQL returned non-fatal NOT_FOUND errors: {err_summary}"
                    )
                else:
                    raise RetriableError(f"GraphQL API errors: {err_summary}")
            check_rate_limit(response, stats)
            return resp_data
        except (requests.RequestException, RetriableError, json.JSONDecodeError) as e:
            last_err = e
            logging.warning(
                f"Retriable error ({type(e).__name__}) on attempt "
                f"{attempt + 1}/{MAX_RETRIES}: {e}. Retrying..."
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_BACKOFF * (2**attempt))
    raise FatalError(f"Max retries reached. Last error: {last_err}")

def get_author(actor: Optional[Dict[str, Any]]) -> str:
    return actor.get("login", "ghost") if actor else "ghost"

def format_comment(node: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not node: return None
    return {
        "id": node.get("id"),
        "author": get_author(node.get("author")),
        "body": node.get("body"),
        "created_at": node.get("createdAt"),
        "reply_to_id": (node.get("replyTo") or {}).get("id"),
    }

def format_review(node: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not node: return None
    return {
        "id": node.get("id"),
        "author": get_author(node.get("author")),
        "body": node.get("body"),
        "state": node.get("state"),
        "submitted_at": node.get("submittedAt"),
    }

def nest_comments(flat_comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not flat_comments:
        return []
    comment_map = {
        c["id"]: {**c, "replies": []} for c in flat_comments if "id" in c
    }
    root_comments = []
    for comment_id, comment in comment_map.items():
        parent_id = comment.get("reply_to_id")
        if parent_id and parent_id in comment_map:
            comment_map[parent_id]["replies"].append(comment)
        else:
            root_comments.append(comment)
    for comment in comment_map.values():
        comment["replies"].sort(key=lambda x: x.get("created_at", ""))
    root_comments.sort(key=lambda x: x.get("created_at", ""))
    return root_comments

def new_page_task(
    node_id: str, cursor: str, task_type: str, context: Dict[str, Any]
) -> PaginationTask:
    return {
        "node_id": node_id, "cursor": cursor,
        "task_type": task_type, "context": context
    }

PAGING_QUERIES = {
    "issue_comments": lambda p, c: (
        f'... on Issue {{ comments(first: {p}, after: "{c}") {{ '
        f"pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}} }} }}"
    ),
    "pr_comments": lambda p, c: (
        f'... on PullRequest {{ comments(first: {p}, after: "{c}") {{ '
        f"pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}} }} }}"
    ),
    "pr_reviews": lambda p, c: (
        f'... on PullRequest {{ reviews(first: {p}, after: "{c}") {{ '
        f"pageInfo {{...PageInfoFragment}}, nodes {{ id, author {{...AuthorFragment}}, "
        f"body, state, submittedAt }} }} }}"
    ),
    "pr_review_threads": lambda p, c: (
        f'... on PullRequest {{ reviewThreads(first: {p}, after: "{c}") {{ '
        f"pageInfo {{...PageInfoFragment}}, nodes {{ id, path, "
        f"comments(first: {PER_PAGE_REVIEW_COMMENTS}) {{ "
        f"pageInfo {{...PageInfoFragment}}, nodes {{ ...CommentFields }} }} }} }} }}"
    ),
    "pr_thread_comments": lambda p, c: (
        f'... on PullRequestReviewThread {{ comments(first: {p}, after: "{c}") {{ '
        f"pageInfo {{...PageInfoFragment}}, nodes {{ ...CommentFields }} }} }}"
    ),
    "discussion_comments": lambda p, c: (
        f'... on Discussion {{ comments(first: {p}, after: "{c}") {{ '
        f"pageInfo {{...PageInfoFragment}}, nodes {{ ...CommentFields, "
        f"replies(first: {INITIAL_REPLIES}) {{ pageInfo {{...PageInfoFragment}}, "
        f"nodes {{ ...CommentFields }} }} }} }} }}"
    ),
    "discussion_replies": lambda p, c: (
        f'... on DiscussionComment {{ replies(first: {p}, after: "{c}") {{ '
        f"pageInfo {{...PageInfoFragment}}, nodes {{ ...CommentFields, "
        f"replies(first: {INITIAL_REPLIES}) {{ pageInfo {{...PageInfoFragment}}, "
        f"nodes {{ ...CommentFields }} }} }} }} }}"
    ),
}

def get_nodes(
    conn: Optional[Dict[str, Any]], normalizer: Callable
) -> List[Dict[str, Any]]:
    return [normalizer(n) for n in (conn or {}).get("nodes", []) if n]

def get_simple_nodes(
    item_data: Dict[str, Any], task: PaginationTask
) -> Tuple[List[Dict[str, Any]], List[PaginationTask]]:
    config = cast(PagingConfigWithNormalizer, PAGING_CONFIG[task["task_type"]])
    conn = item_data.get(config["conn_name"])
    nodes = get_nodes(conn, config["normalizer"])
    return nodes, []

def parse_comment_tree(
    nodes: List[Dict[str, Any]], context: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[PaginationTask]]:
    new_tasks, formatted_comments = [], []
    processed_ids = context.get("processed_ids")
    comment_queue = deque(nodes)
    while comment_queue:
        comment_node = comment_queue.popleft()
        if not comment_node or (processed_ids and comment_node.get("id") in processed_ids):
            if comment_node:
                msg = f"Skipping already processed discussion comment ID: {comment_node.get('id')}"
                logging.debug(msg)
            continue

        if formatted := format_comment(comment_node):
            formatted_comments.append(formatted)

        if processed_ids is not None and (cid := comment_node.get("id")):
            processed_ids.add(cid)

        if replies_conn := comment_node.get("replies"):
            if replies_conn.get("pageInfo", {}).get("hasNextPage"):
                new_tasks.append(
                    new_page_task(
                        comment_node["id"],
                        replies_conn["pageInfo"]["endCursor"],
                        "discussion_replies",
                        context,
                    )
                )
            if replies_conn.get("nodes"):
                comment_queue.extend(replies_conn["nodes"])
    return formatted_comments, new_tasks

def parse_discussion_replies(
    item_data: Dict[str, Any], task: PaginationTask
) -> Tuple[List[Dict[str, Any]], List[PaginationTask]]:
    config = cast(PagingConfigBase, PAGING_CONFIG[task["task_type"]])
    conn_name = config["conn_name"]
    if not (conn := item_data.get(conn_name)) or not conn.get("nodes"):
        return [], []
    return parse_comment_tree(conn["nodes"], task["context"])

def get_review_threads(
    item_data: Dict[str, Any], task: PaginationTask
) -> Tuple[Dict[str, Any], List[PaginationTask]]:
    new_tasks, threads_map = [], {}
    if not (conn := item_data.get("reviewThreads")) or not conn.get("nodes"):
        return {}, []
    for thread_node in conn["nodes"]:
        if not thread_node: continue
        comments_conn = thread_node.get("comments", {})
        thread_id = thread_node["id"]
        threads_map[thread_id] = {
            "id": thread_id,
            "path": thread_node["path"],
            "comments": get_nodes(comments_conn, format_comment),
        }
        if comments_conn.get("pageInfo", {}).get("hasNextPage"):
            context = {**task["context"], "thread_id": thread_id}
            new_tasks.append(
                new_page_task(
                    thread_id,
                    comments_conn["pageInfo"]["endCursor"],
                    "pr_thread_comments",
                    context,
                )
            )
    return threads_map, new_tasks

PAGING_CONFIG: Dict[str, Dict[str, Any]] = {
    "issue_comments": {
        "conn_name": "comments", "normalizer": format_comment,
        "frags": {"PageInfoFragment", "CommentFields"},
        "q": PAGING_QUERIES["issue_comments"],
        "extract_func": get_simple_nodes, "target_key": "comments",
    },
    "pr_comments": {
        "conn_name": "comments", "normalizer": format_comment,
        "frags": {"PageInfoFragment", "CommentFields"},
        "q": PAGING_QUERIES["pr_comments"],
        "extract_func": get_simple_nodes, "target_key": "comments",
    },
    "pr_reviews": {
        "conn_name": "reviews", "normalizer": format_review,
        "frags": {"PageInfoFragment", "AuthorFragment"},
        "q": PAGING_QUERIES["pr_reviews"],
        "extract_func": get_simple_nodes, "target_key": "reviews",
    },
    "pr_review_threads": {
        "conn_name": "reviewThreads", "frags": {"PageInfoFragment", "CommentFields"},
        "q": PAGING_QUERIES["pr_review_threads"],
        "extract_func": get_review_threads, "target_key": "review_threads_map",
    },
    "pr_thread_comments": {
        "conn_name": "comments", "normalizer": format_comment,
        "frags": {"PageInfoFragment", "CommentFields"},
        "q": PAGING_QUERIES["pr_thread_comments"],
        "extract_func": get_simple_nodes, "target_key": "pr_thread_comments",
    },
    "discussion_comments": {
        "conn_name": "comments", "frags": {"PageInfoFragment", "CommentFields"},
        "q": PAGING_QUERIES["discussion_comments"],
        "extract_func": parse_discussion_replies, "target_key": "comments",
    },
    "discussion_replies": {
        "conn_name": "replies", "frags": {"PageInfoFragment", "CommentFields"},
        "q": PAGING_QUERIES["discussion_replies"],
        "extract_func": parse_discussion_replies, "target_key": "comments",
    },
}

def get_comments_frag(page_size: int) -> str:
    return (
        f"comments(first: {page_size}) {{ "
        f"pageInfo {{...PageInfoFragment}}, nodes {{...CommentFields}} }}"
    )

def get_reviews_frag(page_size: int) -> str:
    return (
        f"reviews(first: {page_size}) {{ pageInfo {{...PageInfoFragment}}, "
        f"nodes {{ id, author {{...AuthorFragment}}, body, state, submittedAt }} }}"
    )

def get_review_threads_frag(page_size: int, nested_size: int) -> str:
    return (
        f"reviewThreads(first: {page_size}) {{ "
        f"pageInfo {{...PageInfoFragment}}, nodes {{ id, path, "
        f"comments(first: {nested_size}) {{ pageInfo {{...PageInfoFragment}}, "
        f"nodes {{ ...CommentFields }} }} }} }}"
    )

def get_discussion_comments_frag(page_size: int, nested_size: int) -> str:
    return (
        f"comments(first: {page_size}) {{ pageInfo {{...PageInfoFragment}}, "
        f"nodes {{ ...CommentFields, replies(first: {nested_size}) {{ "
        f"pageInfo {{...PageInfoFragment}}, nodes {{ ...CommentFields }} }} }} }}"
    )

def parse_issue(
    node: Dict[str, Any], _: Optional[Set[str]]
) -> Tuple[Dict[str, Any], List[PaginationTask]]:
    item = {"comments": get_nodes(node.get("comments"), format_comment)}
    tasks = []
    if (conn := node.get("comments")) and conn.get("pageInfo", {}).get("hasNextPage"):
        context = {"target_item_id": node["id"]}
        tasks.append(
            new_page_task(
                node["id"], conn["pageInfo"]["endCursor"], "issue_comments", context
            )
        )
    return item, tasks

def parse_pr(
    node: Dict[str, Any], _: Optional[Set[str]]
) -> Tuple[Dict[str, Any], List[PaginationTask]]:
    item = {
        "merged_at": node.get("mergedAt"), "is_draft": node.get("isDraft"),
        "head_ref": node.get("headRefName"), "base_ref": node.get("baseRefName"),
        "comments": get_nodes(node.get("comments"), format_comment),
        "reviews": get_nodes(node.get("reviews"), format_review),
    }
    context = {"target_item_id": node["id"]}
    synthetic_task = PaginationTask(
        node_id=node["id"], cursor=None, task_type="pr_review_threads", context=context
    )
    initial_threads_map, thread_tasks = get_review_threads(node, synthetic_task)
    item["review_threads_map"] = initial_threads_map

    tasks = thread_tasks
    if (conn := node.get("comments")) and conn.get("pageInfo", {}).get("hasNextPage"):
        tasks.append(
            new_page_task(node["id"], conn["pageInfo"]["endCursor"], "pr_comments", context)
        )
    if (conn := node.get("reviews")) and conn.get("pageInfo", {}).get("hasNextPage"):
        tasks.append(
            new_page_task(node["id"], conn["pageInfo"]["endCursor"], "pr_reviews", context)
        )
    if (conn := node.get("reviewThreads")) and conn.get("pageInfo", {}).get("hasNextPage"):
        tasks.append(
            new_page_task(node["id"], conn["pageInfo"]["endCursor"], "pr_review_threads", context)
        )
    return item, tasks

def parse_discussion(
    node: Dict[str, Any], processed_ids: Optional[Set[str]]
) -> Tuple[Dict[str, Any], List[PaginationTask]]:
    assert processed_ids is not None, "processed_ids should be a set for discussions"

    state = "Answered" if node.get("isAnswered") else "Closed" if node.get("closed") else "Open"
    context = {"target_item_id": node["id"], "processed_ids": processed_ids}
    synthetic_task = PaginationTask(
        node_id=node["id"], cursor=None, task_type="discussion_comments", context=context
    )
    initial_comments, tasks = parse_discussion_replies(node, synthetic_task)

    item = {"state": state, "assignees": [], "comments": initial_comments}
    if (conn := node.get("comments")) and conn.get("pageInfo", {}).get("hasNextPage"):
        tasks.append(
            new_page_task(
                node["id"], conn["pageInfo"]["endCursor"], "discussion_comments", context
            )
        )
    return item, tasks

ITEM_CONFIG: Dict[str, ItemConfig] = {
    "issues": {
        "graphql_name": "issues",
        "path": ("repository", "issues"),
        "fields": (
            f"...BaseItemFields, {get_comments_frag(PER_PAGE_NESTED)}"
        ),
        "fragment_deps": {"PageInfoFragment", "CommentFields", "BaseItemFields"},
        "parser_func": parse_issue,
    },
    "pull_requests": {
        "graphql_name": "pullRequests",
        "path": ("repository", "pullRequests"),
        "fields": (
            "...BaseItemFields, "
            f"{get_comments_frag(PER_PAGE_NESTED)}, "
            f"{get_reviews_frag(PER_PAGE_NESTED)}, "
            f"{get_review_threads_frag(PER_PAGE_NESTED, PER_PAGE_REVIEW_COMMENTS)}"
        ),
        "fragment_deps": {"PageInfoFragment", "CommentFields", "BaseItemFields"},
        "parser_func": parse_pr,
    },
    "discussions": {
        "graphql_name": "discussions",
        "path": ("repository", "discussions"),
        "fields": (
            "...DiscussionFields, "
            f"{get_discussion_comments_frag(PER_PAGE_NESTED, INITIAL_REPLIES)}"
        ),
        "fragment_deps": {
            "PageInfoFragment", "CommentFields", "DiscussionFields", "AuthorFragment"
        },
        "parser_func": parse_discussion,
    },
}

def fetch_page_batch(
    session: requests.Session, batch: List[PaginationTask], stats: StatsTracker
) -> Dict[str, Any]:
    query_parts: List[str] = []
    req_fragments: Set[str] = set()
    for i, task in enumerate(batch):
        task_config = cast(PagingConfigBase, PAGING_CONFIG[task["task_type"]])
        query_node = task_config["q"](PER_PAGE_NESTED, task["cursor"] or "")
        query_parts.append(f'item{i}: node(id: "{task["node_id"]}") {{ {query_node} }}')
        req_fragments.update(task_config["frags"])
    if not query_parts:
        return {}
    all_frags = resolve_frags(req_fragments)
    fragments_str = " ".join(ALL_FRAGMENTS[name]["query"] for name in all_frags)
    query = f"query GetBatchedNestedData {{ {' '.join(query_parts)} }} {fragments_str}"
    data = run_gql(session, {"query": query}, stats).get("data", {})
    return cast(Dict[str, Any], data)

def fetch_paginated_data(
    session: requests.Session, queue: deque[PaginationTask], stats: StatsTracker
) -> AllNestedData:
    results: AllNestedData = {}
    while queue:
        batch = [queue.popleft() for _ in range(min(BATCH_SIZE, len(queue)))]
        if not batch: continue
        time.sleep(REQUEST_DELAY)
        batched_data = fetch_page_batch(session, batch, stats)
        tasks_to_add = []
        for i, task in enumerate(batch):
            if not (item_data := batched_data.get(f"item{i}")):
                continue
            target_item_id = task["context"].get("target_item_id")
            if not target_item_id:
                continue

            config = cast(PagingConfigBase, PAGING_CONFIG[task["task_type"]])
            target_key = config["target_key"]
            new_nodes, new_tasks = config["extract_func"](item_data, task)
            tasks_to_add.extend(new_tasks)

            if new_nodes:
                item_results = results.setdefault(target_item_id, {})
                if task["task_type"] == "pr_review_threads":
                    threads_map = item_results.setdefault(target_key, {})
                    threads_map.update(new_nodes)
                elif task["task_type"] == "pr_thread_comments":
                    if thread_id := task["context"].get("thread_id"):
                        threads_map = item_results.setdefault("review_threads_map", {})
                        thread = threads_map.setdefault(thread_id, {})
                        thread_comments = thread.setdefault("comments", [])
                        thread_comments.extend(new_nodes)
                else:
                    item_results.setdefault(target_key, []).extend(new_nodes)
            if (conn := item_data.get(config["conn_name"])):
                if conn.get("pageInfo", {}).get("hasNextPage"):
                    task["cursor"] = conn["pageInfo"]["endCursor"]
                    tasks_to_add.append(task)

        if tasks_to_add:
            queue.extend(tasks_to_add)
        logging.info(f"Processed batch of {len(batch)} pages. {len(queue)} pages remaining.")
    return results

def parse_item(
    node: Dict[str, Any], item_type: str, processed_ids: Optional[Set[str]]
) -> Tuple[Dict[str, Any], List[PaginationTask]]:
    config = ITEM_CONFIG[item_type]
    labels_nodes = node.get("labels", {}).get("nodes", [])
    assignees_nodes = node.get("assignees", {}).get("nodes", [])
    base_item: Dict[str, Any] = {
        "id": node.get("id"), "number": node.get("number"), "title": node.get("title"),
        "author": get_author(node.get("author")), "body": node.get("body"),
        "created_at": node.get("createdAt"), "updated_at": node.get("updatedAt"),
        "labels": [l["name"] for l in labels_nodes if l],
    }
    if item_type in ("issues", "pull_requests"):
        base_item.update({
            "state": node.get("state"),
            "assignees": [a["login"] for a in assignees_nodes if a],
        })
    item_specific_data, tasks = config["parser_func"](node, processed_ids)
    base_item.update(item_specific_data)
    return base_item, tasks

def paginator(
    session: requests.Session, query: str, variables: Dict[str, Any],
    path: Tuple[str, ...], stats: StatsTracker
):
    variables.pop("after", None)
    has_next_page = True
    while has_next_page:
        time.sleep(REQUEST_DELAY)
        resp_data = run_gql(session, {"query": query, "variables": variables}, stats)
        connection = resp_data.get("data", {})
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
    session: requests.Session, owner: str, repo: str, item_type: str, stats: StatsTracker
) -> List[FinalItem]:
    logging.info(f"Fetching all {item_type.replace('_', ' ')}...")
    page_queue: deque[PaginationTask] = deque()
    items: List[Dict[str, Any]] = []
    config = ITEM_CONFIG[item_type]
    all_frags = resolve_frags(config["fragment_deps"])
    fragments_str = "\n".join(ALL_FRAGMENTS[name]["query"] for name in all_frags)
    query = f"""
    query GetAllItems($owner: String!, $repo: String!, $perPage: Int!, $after: String) {{
        repository(owner: $owner, name: $repo) {{
            {config["graphql_name"]}(
                first: $perPage,
                after: $after,
                orderBy: {{field: CREATED_AT, direction: ASC}}
            ) {{
                pageInfo {{ ...PageInfoFragment }}
                nodes {{ {config["fields"]} }}
            }}
        }}
    }}
    {fragments_str}
    """
    variables = {"owner": owner, "repo": repo, "perPage": PER_PAGE}
    count = 0
    processed_ids: Optional[Set[str]] = set() if item_type == "discussions" else None

    for page in paginator(session, query, variables, config["path"], stats):
        for node in page:
            if not (node and node.get("id")):
                continue
            item, new_tasks = parse_item(node, item_type, processed_ids)
            items.append(item)
            page_queue.extend(new_tasks)
        count += len(page)
        logging.info(f"Processed {count} {item_type}...")

    if not items:
        logging.info(f"No {item_type} found for this repository.")
        return []

    msg = f"Fetching all nested data for {len(items)} items ({len(page_queue)} pages)..."
    logging.info(msg)
    nested_data_map = fetch_paginated_data(session, page_queue, stats)
    logging.info(f"Structuring final items for {item_type}...")
    final_items: List[FinalItem] = []
    for item in items:
        nested_data = nested_data_map.get(item["id"], {})
        all_comments = item.get("comments", []) + nested_data.get("comments", [])
        item["comments"] = nest_comments(all_comments)

        if item_type == "pull_requests":
            all_reviews = item.get("reviews", []) + nested_data.get("reviews", [])
            threads_map = item.get("review_threads_map", {})
            threads_map.update(nested_data.get("review_threads_map", {}))
            final_review_threads = []
            for thread_data in threads_map.values():
                thread_comments = thread_data.get("comments", [])
                thread_data["comments"] = nest_comments(thread_comments)
                final_review_threads.append(cast(ReviewThread, thread_data))
            item["reviews"] = all_reviews
            item["review_threads"] = final_review_threads
            item.pop("review_threads_map", None)
        final_items.append(cast(FinalItem, item))
    logging.info(f"Finished {item_type}: Found and processed {len(items)} items.")
    return final_items

def main():
    parser = argparse.ArgumentParser(
        description="Fetch issues, PRs, and discussions from a GitHub repo."
    )
    parser.add_argument("repository", help="the repository in 'owner/name' format.")
    parser.add_argument(
        "-o", "--output", help="output JSON file name (default: owner__repo.json)."
    )
    args = parser.parse_args()

    log_level = logging.INFO
    logging.addLevelName(logging.INFO, "I")
    logging.addLevelName(logging.WARNING, "W")
    logging.addLevelName(logging.ERROR, "E")
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s", stream=sys.stdout)

    if not GITHUB_TOKEN:
        logging.error("Error: GITHUB_TOKEN environment variable not set.")
        sys.exit(1)
    logging.getLogger().addFilter(TokenFilter(GITHUB_TOKEN))

    try:
        owner, repo_name = args.repository.split("/")
        is_valid_owner = re.fullmatch(r"^[a-zA-Z0-9.\-_]{1,100}$", owner)
        is_valid_repo = re.fullmatch(r"^[a-zA-Z0-9.\-_]{1,100}$", repo_name)
        if not (is_valid_owner and is_valid_repo):
            raise ValueError("Invalid characters or length in owner/repo name.")
    except (ValueError, IndexError):
        logging.error("Error: Repository must be in 'owner/name' format with valid chars.")
        sys.exit(1)

    output_file = args.output or f"{owner}__{repo_name}.json"
    start_time = time.time()
    stats = StatsTracker()
    repo_data: Dict[str, Any] = {
        "repository": f"{owner}/{repo_name}",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    with requests.Session() as session:
        for item_type in ITEM_CONFIG.keys():
            try:
                data = fetch_items(session, owner, repo_name, item_type, stats)
                repo_data[f"{item_type}"] = [dict(d) for d in data]
            except (GitHubError, requests.RequestException) as e:
                logging.error(f"Could not fetch {item_type} due to an error: {e}. Skipping.")
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
