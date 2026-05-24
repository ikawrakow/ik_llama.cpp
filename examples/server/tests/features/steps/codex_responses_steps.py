"""
Behave step definitions for Codex CLI Responses API compatibility tests.
Tests that /v1/responses accepts Codex-shaped payloads with mixed tool types.
"""

import json

from behave import step  # pyright: ignore[reportAttributeAccessIssue]
from behave.api.async_step import async_run_until_complete

import aiohttp


# Codex CLI 0.133.0 sends mixed tool types: function + namespace + web_search + image_generation
CODEX_MIXED_TOOLS_PAYLOAD = {
    "model": "test",
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Reply exactly local-ok"}]
        }
    ],
    "tools": [
        {
            "type": "function",
            "name": "exec_command",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"]
            },
            "strict": False
        },
        {
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Sub-agent tools",
            "tools": []
        },
        {
            "type": "web_search",
            "external_web_access": True
        },
        {
            "type": "image_generation",
            "output_format": "png"
        }
    ],
    "tool_choice": "auto",
    "parallel_tool_calls": False,
    "stream": False,
    "max_output_tokens": 8,
}

PROBE_EMPTY_INPUT_PAYLOAD = {
    "model": "test",
    "input": "",
    "stream": False,
    "max_output_tokens": 1,
}

INVALID_PREVIOUS_RESPONSE_PAYLOAD = {
    "model": "test",
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "test"}]
        }
    ],
    "previous_response_id": "invalid-id-12345",
    "stream": False,
}


@step("an OAI compatible responses request with mixed Codex tool types")
@async_run_until_complete
async def step_oai_responses_mixed_tools(context):
    """
    Send a Responses API request with mixed tool types from Codex CLI:
    - function (should be converted)
    - namespace, web_search, image_generation (should be skipped, not rejected)
    """
    if context.debug:
        print("Submitting Responses API request with mixed Codex tool types...")

    payload = CODEX_MIXED_TOOLS_PAYLOAD.copy()
    if hasattr(context, "model") and context.model:
        payload["model"] = context.model

    async with aiohttp.ClientSession() as session:
        url = f"{context.base_url}/v1/responses"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {context.user_api_key if hasattr(context, 'user_api_key') else 'test'}",
        }

        async with session.post(url, json=payload, headers=headers) as response:
            context.responses_status = response.status
            context.responses_text = await response.text()


@step("a probe responses request with empty input and max_output_tokens=1")
@async_run_until_complete
async def step_probe_empty_input(context):
    """Send a probe request with empty input and minimal tokens."""
    if context.debug:
        print("Submitting probe Responses API request with empty input...")

    payload = PROBE_EMPTY_INPUT_PAYLOAD.copy()
    if hasattr(context, "model") and context.model:
        payload["model"] = context.model

    async with aiohttp.ClientSession() as session:
        url = f"{context.base_url}/v1/responses"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {context.user_api_key if hasattr(context, 'user_api_key') else 'test'}",
        }

        async with session.post(url, json=payload, headers=headers) as response:
            context.probe_status = response.status
            context.probe_text = await response.text()


@step("a responses request with invalid previous_response_id")
@async_run_until_complete
async def step_invalid_previous_response_id(context):
    """Send a request with an invalid previous_response_id to trigger an error."""
    if context.debug:
        print("Submitting Responses API request with invalid previous_response_id...")

    payload = INVALID_PREVIOUS_RESPONSE_PAYLOAD.copy()
    if hasattr(context, "model") and context.model:
        payload["model"] = context.model

    async with aiohttp.ClientSession() as session:
        url = f"{context.base_url}/v1/responses"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {context.user_api_key if hasattr(context, 'user_api_key') else 'test'}",
        }

        async with session.post(url, json=payload, headers=headers) as response:
            context.prev_resp_status = response.status
            context.prev_resp_text = await response.text()


@step("the mixed Codex tools response succeeds")
def step_mixed_tools_response_succeeds(context):
    """Assert HTTP 200, valid JSON, and required Responses fields."""
    status = getattr(context, "responses_status", None)
    text = getattr(context, "responses_text", None)

    assert status == 200, f"Mixed tools request failed with status {status}. Expected 200. Response: {text[:200] if text else '(empty)'}"
    assert text is not None, "No response body received"

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON response: {e}") from e

    assert "id" in data, "Expected 'id' in Responses response"
    assert "output" in data, "Expected 'output' in Responses response"


@step("the probe response is accepted")
def step_probe_response_accepted(context):
    """Assert probe request succeeds with HTTP 200, valid JSON, id, and output."""
    status = getattr(context, "probe_status", None)
    text = getattr(context, "probe_text", None)

    assert status is not None, "No probe response status"
    assert text is not None, "No probe response body"

    assert status == 200, f"Probe request failed with status {status}. Expected 200. Response: {text[:200]}"

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON probe response: {e}") from e

    assert "id" in data, "Expected 'id' in probe response"
    assert "output" in data, "Expected 'output' in probe response"


@step("the previous_response_id request returns an error")
def step_previous_response_id_returns_error(context):
    """Assert that invalid previous_response_id returns an error response (4xx or 5xx)."""
    status = getattr(context, "prev_resp_status", None)
    text = getattr(context, "prev_resp_text", None)

    assert status is not None, "No previous_response_id response status"
    assert text is not None, "No previous_response_id response body"

    assert status >= 400, f"Expected error status for invalid previous_response_id, got {status}"

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON error response: {e}") from e

    assert "error" in data, f"Expected 'error' field in error response. Got: {list(data.keys())}"
    error_msg = data["error"].get("message", "") if isinstance(data["error"], dict) else str(data["error"])
    assert "previous_response_id" in error_msg, f"Expected 'previous_response_id' in error message. Got: {error_msg}"
