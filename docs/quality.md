# Research-code quality criteria

Quality is demonstrated by reproducible behavior and evidence, not by a paper-award label.

| Criterion | Evidence required |
|---|---|
| Reproducible experiment | Resolved config, source/split manifests, weight identity and software/code versions |
| Correct training | Gradient/freeze tests, paired geometry, accumulation and optimizer-step semantics |
| Honest evaluation | Explicit preprocessing, image alignment, metric protocol and held-out selection |
| Reliable persistence | Interrupted-save tests, strict key/shape validation and condition/EMA round trips |
| Local extension | A new data format/loss/backend does not duplicate the train or evaluation loop |
| Usable documentation | Offline smoke run, runnable commands and accurate support boundaries |

The September 12 refactor preserves legacy data output and RNG behavior through
exact regression comparisons. It adds tested manifests, preprocessing strategies,
configuration composition, checkpoint publication, optimizer/pixel-loss choices
and backend primitives.

Remaining work must not be hidden behind configuration options: full SD3/FLUX
restoration adapters, real SD-Turbo restoration validation, native persisted early
stopping/best selection, exact batch/RNG replay, distributed evaluation, streaming
datasets, and full environment/version-matrix CI. Some dependencies currently
have broad allowed ranges; passing tests in the installed environment does not
prove every version combination in those ranges works.

Never silently change an existing benchmark protocol to improve its numbers.
Keep research ideas in separate configurations and compare them under matched
data, initialization and training budgets.
