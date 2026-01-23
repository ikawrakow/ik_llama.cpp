import json
from typing import Any

import aiohttp
from behave import step  # pyright: ignore[reportAttributeAccessIssue]
from behave.api.async_step import async_run_until_complete

import steps


@step("an OAI compatible responses request with {api_error} api error")
@async_run_until_complete
async def step_oai_responses(context, api_error):
    if context.debug:
        print("Submitting OAI compatible responses request...")
    expect_api_error = api_error == "raised"
    seeds = await steps.completions_seed(context, num_seeds=1)
    completion = await oai_responses(
        context.prompts.pop(),
        seeds[0] if seeds is not None else seeds,
        context.system_prompt,
        context.base_url,
        debug=context.debug,
        model=context.model if hasattr(context, "model") else None,
        n_predict=context.n_predict if hasattr(context, "n_predict") else None,
        enable_streaming=context.enable_streaming
        if hasattr(context, "enable_streaming")
        else None,
        user_api_key=context.user_api_key if hasattr(context, "user_api_key") else None,
        temperature=context.temperature,
        expect_api_error=expect_api_error,
    )
    context.tasks_result.append(completion)
    if context.debug:
        print(f"Responses completion response: {completion}")
    if expect_api_error:
        assert completion == 401, f"completion must be an 401 status code: {completion}"


def extract_responses_output_text(
    response_json: dict[str, Any],
) -> tuple[str, str | None]:
    output_text = ""
    message_id = None
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        message_id = item.get("id")
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                output_text += part.get("text", "")
    return output_text, message_id


async def oai_responses(
    user_prompt,
    seed,
    system_prompt,
    base_url: str,
    debug=False,
    temperature=None,
    model=None,
    n_predict=None,
    enable_streaming=None,
    user_api_key=None,
    expect_api_error=None,
) -> int | dict[str, Any]:
    if debug:
        print(f"Sending OAI responses request: {user_prompt}")
    user_api_key = user_api_key if user_api_key is not None else "nope"
    seed = seed if seed is not None else 42
    enable_streaming = enable_streaming if enable_streaming is not None else False
    payload = {
        "input": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "model": model,
        "stream": enable_streaming,
        "temperature": temperature if temperature is not None else 0.0,
        "seed": seed,
    }
    if n_predict is not None:
        payload["max_output_tokens"] = n_predict
    completion_response = {
        "content": "",
        "timings": {
            "predicted_n": 0,
            "prompt_n": 0,
        },
    }
    origin = "llama.cpp"
    headers = {"Authorization": f"Bearer {user_api_key}", "Origin": origin}
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/v1/responses", json=payload, headers=headers
        ) as response:
            if expect_api_error is not None and expect_api_error:
                if response.status == 401:
                    return 401
                assert False, f"unexpected status code: {response.status}"

            assert response.status == 200
            assert response.headers["Access-Control-Allow-Origin"] == origin
            if enable_streaming:
                assert response.headers["Content-Type"] == "text/event-stream"
                resp_id = ""
                msg_id = ""
                gathered_text = ""
                event_name = None
                completed_response = None
                async for line_in_bytes in response.content:
                    line = line_in_bytes.decode("utf-8").strip()
                    if not line:
                        continue
                    if line.startswith("event: "):
                        event_name = line.split(": ", 1)[1]
                        continue
                    if not line.startswith("data: "):
                        continue
                    if event_name is None:
                        continue
                    chunk_raw = line.split(": ", 1)[1]
                    data = json.loads(chunk_raw)

                    if event_name == "response.created":
                        resp_id = data["response"]["id"]
                        assert resp_id.startswith("resp_")
                    elif event_name == "response.in_progress":
                        assert data["response"]["id"] == resp_id
                    elif event_name == "response.output_item.added":
                        item = data["item"]
                        if item.get("type") == "message":
                            msg_id = item["id"]
                            assert msg_id.startswith("msg_")
                    elif event_name in (
                        "response.content_part.added",
                        "response.output_text.delta",
                        "response.output_text.done",
                        "response.content_part.done",
                    ):
                        assert data["item_id"] == msg_id
                    elif event_name == "response.output_item.done":
                        item = data["item"]
                        if item.get("type") == "message":
                            assert item["id"] == msg_id
                    if event_name == "response.output_text.delta":
                        gathered_text += data["delta"]
                    if event_name == "response.completed":
                        completed_response = data["response"]

                assert completed_response is not None
                output_text, completed_msg_id = extract_responses_output_text(
                    completed_response
                )
                assert completed_msg_id is not None
                assert completed_msg_id.startswith("msg_")
                assert output_text == gathered_text
                completion_response = {
                    "content": output_text,
                    "timings": {
                        "predicted_n": completed_response["usage"]["output_tokens"],
                        "prompt_n": completed_response["usage"]["input_tokens"],
                    },
                }
            else:
                assert (
                    response.headers["Content-Type"]
                    == "application/json; charset=utf-8"
                )
                response_json = await response.json()
                assert response_json["id"].startswith("resp_")
                output_text, message_id = extract_responses_output_text(response_json)
                assert message_id is not None
                assert message_id.startswith("msg_")
                completion_response = {
                    "content": output_text,
                    "timings": {
                        "predicted_n": response_json["usage"]["output_tokens"],
                        "prompt_n": response_json["usage"]["input_tokens"],
                    },
                }
    if debug:
        print("OAI response formatted to llama.cpp:", completion_response)
    return completion_response
