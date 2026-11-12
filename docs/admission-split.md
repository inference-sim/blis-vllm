# Admission: the render-thread part and the delay

Contents: [Logic](#logic) | [Inputs](#inputs) | [Reproduce](#reproduce) | [Cost](#cost) | [Limits](#limits)

## Logic

A request's admission overhead (`LatencyModel.QueueingTime`) has two parts with different
physics, and the engine now prices them apart.

- `LatencyModel.RenderTime`: the time the request holds one render thread of its API server.
  This is chat-template rendering and tokenization, which scale with the prompt. vLLM v0.32.0
  runs both on the renderer's thread pool, `ThreadPoolExecutor(max_workers=
  renderer_num_workers)`, default 1 (`vllm/renderers/base.py:139-158`,
  `vllm/renderers/hf.py:965`, `vllm/config/model.py:408`). Two requests on one API server
  contend for it, so the host queues requests for a thread for this time.
- The rest of `QueueingTime` (`QueueingTime - RenderTime`): a latency the request waits out
  without holding a thread. In v0.32.0 only the tokenizer and template calls are handed to
  the pool; the rest of a request's handling in the API server is asyncio code that overlaps
  across requests.

Before the split the whole `QueueingTime` was the thread's service time. With the kernel's
fixed 40 ms per request, one thread started at most 1 / 0.0404 s = 24.8 requests a second, so a
closed-loop client population saturated the front end and not the GPUs.

`Model.RenderTime` (`compat/kernelmodel`) forwards the substrate kernel's
`AdmissionRenderTime`, the per-token admission term. No coefficient changes: the 40 ms and the
per-token rate are still the registry's, with the registry's methods.

`EngineCore.QueuedAt` composes them: a request arriving at `t` is queued at
`Renderer.Admit(t, RenderTime) + (QueueingTime - RenderTime)`, or `t + QueueingTime` when there
is no render stage. One request alone therefore waits exactly what it waited before.

The cap that remains is the thread's own work and it scales with the front end's width, as
vLLM's does: `renderer_num_workers` threads per API server, and `api_server_count` servers,
each with its own pool (`vllm/entrypoints/cli/serve.py:107-137` at v0.32.0). Two threads, or
two servers, halve the time to render a burst (`TestRenderCapScalesWithRendererNumWorkers
AndAPIServerCount`). Under vLLM's Rust front end `api_server_count` is forced to 1 and the
front end is multithreaded (`serve.py:107-137`, `serve.py:274-280`); the model does not
represent that case separately, and a deployment on it is priced with the Rust per-token rate
(`host_admission_per_token_rust_frontend`) on the same queue.

Both settings are still `not_modelled` in `fields.json`. `modelled` needs a recorded vLLM trace
that sets the field, and none exists: the oracle's boundary replaces the tokenizer, so the
render pool's contention is outside what any trace in this repository shows. The status is left
as it is, and the maintainer decides what evidence would change it.

## Inputs

The substrate commit that adds `AdmissionRenderTime` to `contracts/kernel.Kernel`
(branch `m2/17-admission-split`). This module's `go.mod` pin moves to that commit when the
substrate layer is published.

## Reproduce

```
go test ./compat/engine -run 'FixedAdmission|SingleRequestAdmission|RenderThreadStill'
go test ./compat/kernelmodel -run RenderTimeIs
```

Expected: pass. Reverting `QueuedAt` to charge `QueueingTime` on the thread fails the first
and third (the last of 100 requests is queued at 4.0 s instead of 75 ms).

## Cost

Constant time per request; no memory.

## Limits

Whether the per-request 40 ms is a delay and not thread work is the registry's reading of its
own rationale (the tokenizer and message costs it measured are 370 times smaller), not a
measurement of the 40 ms. The split removes an artefact of how the constant was charged; it
does not validate the constant. This is a development-split result, not held-out.
