@llama.cpp
@memory
Feature: Per-device memory use

  Background: Server setup
    Given a server listening on localhost:8080
    And   a model file test-model.gguf
    And   4096 KV cache size
    And   32 as batch size
    And   2 slots
    And   64 server max tokens to predict

  Scenario: The memory endpoint is off by default
    Then  the server is starting
    Then  the server is healthy
    And   the memory endpoint answers 501

  Scenario: The memory endpoint reports the host
    Given memory endpoint exposed
    Then  the server is starting
    Then  the server is healthy
    Given a memory report is fetched
    Then  the memory report has a host row and the layer count
    And   the memory report counts at least 1 bytes of model
    And   the memory report counts at least 2000000 bytes of context
    And   the memory report counts at least 1 bytes of compute
    And   the model memory is within 0.5 and 1.5 of the model file size

  Scenario: The metrics carry a gauge for each memory figure
    Given prometheus compatible metrics exposed
    And   memory endpoint exposed
    Then  the server is starting
    Then  the server is healthy
    Given a memory report is fetched
    Then  every memory row has an exact gauge in the metrics

  Scenario: The metrics carry the gauges without the memory endpoint
    Given prometheus compatible metrics exposed
    Then  the server is starting
    Then  the server is healthy
    And   prometheus metrics are exposed
    And   metric llamacpp:memory_model_bytes is positive on the host

  Scenario: The projector memory is reported
    Given a model file tinygemma3-Q8_0.gguf
    And   a multimodal projector file mmproj-tinygemma3.gguf
    And   memory endpoint exposed
    Then  the server is starting
    Then  the server is healthy
    Given a memory report is fetched
    Then  the memory report counts at least 1 bytes of mmproj
