@llama.cpp
@server
@codex
Feature: Codex CLI Responses API Compatibility

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a model file test-model.gguf
    And   a model alias tinyllama-2
    And   42 as server seed
    And   256 KV cache size
    And   32 as batch size
    And   2 slots
    And   64 server max tokens to predict
    And   Jinja templating enabled
    Then  the server is starting
    Then  the server is healthy

  Scenario: Responses API accepts mixed tool types from Codex
    Given a model test
    And   an OAI compatible responses request with mixed Codex tool types
    Then  the mixed Codex tools response succeeds

  Scenario: Probe request with empty input and max_output_tokens=1 is accepted
    Given a model test
    And   a probe responses request with empty input and max_output_tokens=1
    Then  the probe response is accepted

  Scenario: previous_response_id returns a controlled error
    Given a model test
    And   a responses request with invalid previous_response_id
    Then  the previous_response_id request returns an error
