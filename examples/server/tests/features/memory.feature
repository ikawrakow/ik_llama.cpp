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

  Scenario: The device buffers are not folded into the host row
    Given 99 GPU offloaded layers
    And   memory endpoint exposed
    Then  the server is starting
    Then  the server is healthy
    Given a memory report is fetched
    Then  a device row carries model memory

  Scenario: The output buffer is counted in the context memory
    Given 1 slots
    And   2048 KV cache size
    And   memory endpoint exposed
    Then  the server is starting
    Then  the server is healthy
    Given a memory report is fetched
    Then  the context memory is recorded
    Then  shutting the server down
    And   4096 KV cache size
    Then  the server is starting
    Then  the server is healthy
    Given a memory report is fetched
    Then  the output buffer is counted in the context memory

  Scenario: The MoE expert cache is counted in the context memory
    Given a model file stories15M_MOE-Q8_0.gguf
    And   1 slots
    And   99 GPU offloaded layers
    And   the MoE weights kept in CPU memory
    And   memory endpoint exposed
    Then  the server is starting
    Then  the server is healthy
    Given a memory report is fetched
    Then  a device row carries model memory
    And   the context memory is recorded
    Then  shutting the server down
    And   8 MiB of MoE expert cache
    Then  the server is starting
    Then  the server is healthy
    Given a memory report is fetched
    # the cache holds whole expert slots, one slot of this model is under 1 MiB
    Then  the context memory grows by at least 7 MiB

  Scenario: The draft model is counted in the memory report
    Given a model file stories15M_MOE-Q8_0.gguf
    And   1 slots
    And   memory endpoint exposed
    Then  the server is starting
    Then  the server is healthy
    Given a memory report is fetched
    Then  the memory figures are kept as target
    Then  shutting the server down
    Given a model file stories15M-q4_0.gguf
    Then  the server is starting
    Then  the server is healthy
    Given a memory report is fetched
    Then  the memory figures are kept as draft
    Then  shutting the server down
    Given a model file stories15M_MOE-Q8_0.gguf
    And   a draft model file stories15M-q4_0.gguf
    Then  the server is starting
    Then  the server is healthy
    Given a memory report is fetched
    Then  the memory figures are kept as both
    # the weights do not depend on the context, so the draft adds exactly the bytes it holds when loaded alone
    And   the model memory of both is that of target plus that of draft
    # the draft context has its own KV cache and compute buffers
    And   the context memory of both is larger than that of target
    And   the compute memory of both is larger than that of target
