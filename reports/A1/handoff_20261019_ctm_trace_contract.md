# Handoff: A1 / CTM trace contract

## Status

- status: ready-for-review
- branch: `ctm-ucl-experiment-20261019` (shared integration workspace; no commit made)
- commit: uncommitted A1-owned files only
- owner: A1 CTM probe engineer
- date: 2026-10-19

## Question answered

Can the local official CTM provide a frozen, read-only, multi-tick trace with
real synchronization shapes, without modifying the upstream source? **Yes.**
The upstream `forward(x, track=True)` already exposes tick-wise post-activation
and both official synchronization representations. The new wrapper only selects
the pre-registered ticks and preserves those exported shapes.

## Source and version lock

- local upstream checkout: `../ctm-official`
- upstream Git commit inspected: `4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466`
- inspected upstream files:
  - `../ctm-official/models/ctm.py` (`compute_synchronisation`, `forward`)
  - `../ctm-official/models/modules.py`
  - `../ctm-official/ctm_analysis/probe.py`
  - `../ctm-official/ctm_analysis/README.md`
  - `../ctm-official/tasks/image_classification/analysis/run_imagenet_analysis.py`
- upstream checkout was already dirty. It was read only and not modified.
- relevant upstream API: `model(inputs, track=True)` returns
  `(predictions, certainties, (sync_out, sync_action), pre, post, attention)`.
- verification environment: Python 3.9, PyTorch 2.8.0, NumPy; CPU synthetic
  batch, model and input seeds both set to `7`.

## Changes

- `ucl_ctm_probe/probe/ctm_trace.py`
  - adds `CTMTraceWrapper(model, tick_ids)` with default model behavior when
    `return_trace=False` and a read-only trace route when `return_trace=True`.
  - adds `trace_ctm(...)`, which selects the fixed 1-based ticks
    `[1, 2, 4, 8, 12, 16, 20]` from the official tracked arrays.
  - rejects uninitialized lazy modules; puts the model in eval mode only for
    extraction; restores both module training flags and the upstream decay
    parameters afterwards. This is needed because the current upstream forward
    clamps decay parameters in-place.
  - returns `z_history`, `sync_history.out`, `sync_history.action`,
    `post_state`, tick IDs and actual shapes/semantics metadata.
- `ucl_ctm_probe/probe/smoke_test.py`
  - creates a minimal local CTM with 20 ticks, performs two equal probes and
    asserts exact equality plus unchanged `state_dict`.
  - writes the compact evidence JSON and the shape-flow SVG below.
- `ucl_ctm_probe/__init__.py`, `ucl_ctm_probe/probe/__init__.py`
  - minimal package markers only.
- training-logic changes: none. No upstream CTM, UCL model, shared schema or
  server file was changed.

## Evidence

Run from repository root:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ctm-a1-pycache python3 -m py_compile \
  ucl_ctm_probe/probe/ctm_trace.py ucl_ctm_probe/probe/smoke_test.py
PYTHONPYCACHEPREFIX=/private/tmp/ctm-a1-pycache python3 -m \
  ucl_ctm_probe.probe.smoke_test
git diff --check
```

All commands exited `0` on 2026-10-19. The runtime emits an unrelated local
`urllib3` LibreSSL warning but does not affect the assertions.

- raw result: `artifacts/A1/tables/ctm_trace_smoke_shapes.json`
- shape-flow figure: `artifacts/A1/figures/ctm_trace_shape_smoke.svg`
  (the checked PNG preview is retained beside it as
  `ctm_trace_shape_smoke.png`). The boxes show the synthetic input, all 20
  upstream ticks, and the selected trace shapes. It is a contract diagram,
  not a scientific result.
- reproducibility seed: `7` for NumPy and PyTorch.

Observed result (synthetic `first-last` test model):

| field | actual shape | meaning |
| --- | --- | --- |
| `z_history` | `(7, 2, 8)` | selected ticks, batch, CTM neuron state |
| `sync_history.out` | `(7, 2, 10)` | exact output sync vector |
| `sync_history.action` | `(7, 2, 10)` | exact action/query sync vector |
| `post_state` | `(2, 8)` | final tick activation |

The two executions were exactly equal and the initialized model `state_dict`
was unchanged after both wrapper calls.

## Findings

1. **The official sync output is not always a matrix.** For
   `neuron_select_type="first-last"` (the smoke setting), it is a flattened
   upper-triangle vector. With `n_synch=4`, its dimension is `4*(4+1)/2 = 10`.
   For `random-pairing`, upstream instead exports a sampled-pair vector of
   length `n_synch`. The wrapper records the selection type and calls these
   vectors; it never labels them a synchronization matrix.
2. **There are two semantically distinct sync traces.** `sync_action` is
   computed before each recurrent state update and feeds the attention query;
   `sync_out` is computed after that update and feeds the output projector.
   Both are returned rather than silently picking one. The downstream analysis
   owner must pre-register which is the primary measure.
3. **Post activation is the valid `z_history`.** The tracked `post` array
   contains the activated state after each internal update, with native shape
   `(iterations, batch, d_model)`; the wrapper samples this at the registered
   ticks. `post_state` is explicitly the final full tick, not necessarily the
   final selected tick under a future tick schedule.
4. **Probe initialization must precede tracing.** CTM uses lazy layers. A real
   loaded checkpoint is already initialized; a synthetic/new CTM needs one
   non-tracked warm-up call (shown in the smoke test) before `trace_ctm`.

## Limitations and unresolved issues

- This is an interface/determinism validation, not a trained CTM calibration or
  evidence about UCL forgetting.
- The upstream tracked arrays are converted to CPU NumPy arrays; at real batch
  and model sizes, raw traces must remain in ignored server storage as specified
  by the experiment plan.
- The wrapper assumes the currently inspected six-item official tracked return.
  It intentionally fails loudly if an upstream revision changes that API.
- `CTMTraceWrapper(..., return_trace=True)` is an external wrapper, not a
  modification of upstream `ContinuousThoughtMachine.forward`; callers must
  supply a loaded/initialized, frozen model and record its checkpoint hash.
- The wrapper does not yet add the CTM checkpoint hash or adapter hash because
  those are owned by the A0 manifest/schema contract, not A1's directory.

## Next agent request

- **A0:** review and freeze the choice of primary synchronization trace
  (`out` is the natural default because it is post-update and feeds the output
  projector); preserve the other trace in records for sensitivity analysis.
  Add CTM checkpoint/adaptor hashes in the central manifest rather than in
  this wrapper.
- **A2:** initialize/load the frozen CTM once after adapter calibration, then
  call `CTMTraceWrapper(ctm)(adapter_features, return_trace=True)` on the fixed
  old probe batch. The actual CTM input must match the loaded CTM backbone/input
  expectations; the smoke tensor only validates the official `backbone_type`
  `none` path.
- **A5:** consume `sync_history["out"]` and `sync_history["action"]` as
  separate vector representations. Use cosine/L2 drift for vectors; apply a
  Frobenius metric only if an explicitly separately-exported square diagnostic
  is present and labelled as such.
- **A6:** audit the restored decay parameters/training flags and confirm that
  model checkpoint plus adapter normalization are frozen before the first UCL
  update.
