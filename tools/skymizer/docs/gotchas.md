# Experiment checks

Use these checks when freezing a comparison. Details belong to the linked guide.

| Check | Why it matters | Contract |
| --- | --- | --- |
| Same base checkpoint and target vocabulary | A family is one checkpoint and its quantizations, not every size in a model series. | [Collection inputs](collect.md#inputs) |
| Hold unmeasured components constant | For LLM quantization, share the projector; for projector quantization, share the LLM. Effects need not be additive. | [What gets compared](compare.md#what-gets-compared) |
| Freeze all runtime and executable settings | Batch shape, scoring horizon, cache policy and backend can change logits. | [Runtime identity](collect.md#runtime-identity) |
| Preserve source image order and complete tile groups | Image count is not tile count. Similar image dimensions do not prove equal native context use. | [Image and prefix failures](troubleshooting.md#image-and-prefix-failures) |
| Use identical eligible rows and targets | One-sided exclusions or failures change the paired sample. | [Completion and artifacts](collect.md#completion-and-artifacts) |
| Pick grouping before comparing candidates | Tokens within an answer or corpus block are correlated. Articles and blocks can also remain dependent. | [Corpus groups](compare.md#fixed-text-corpora-and-grouped-inference) |
| Distinguish original and saved-base PPL | llama-perplexity's uint16 saved reference can clip low log probabilities. | [Text bridge](perplexity-llm-kld-aws-handover.md) |
| Keep pilot planning separate from prospective power | Observed pilot effects do not supply an externally chosen meaningful effect. | [Planning](../knowledge/seq-power-analysis.md) |

Dataset row indices are positions after the selected sort. Use stable item IDs and pinned dataset content when reconciling runs. Pilot overlap with a later collection must be stated; it is not an independent replication.
