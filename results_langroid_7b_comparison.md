# Langroid / Qwen 2.5 7B Comparison

## Protocol

- 500 agents represent 4,000 vehicles, or 8 vehicles per agent.
- Each Langroid agent queries Qwen 2.5 7B through Ollama for non-forced choices.
- `llm_checked` corrects a valid but non-minimal LLM response.
- `llm_raw` accepts every valid LLM response without correction.
- The references come from `Multiagenten_Braess.pdf`.

## Results

| Scenario | Reference | LLM checked | LLM raw |
| --- | ---: | ---: | ---: |
| Original network | 65.0000 min | 65.0000 min | 78.0573 min |
| Extended network without toll | 82.0000 min | 82.0000 min | 85.0000 min |
| Extended network with toll 40 | 64.8948 min | 64.8876 min | 79.7920 min |

## Agent Distribution

| Scenario | LLM checked | LLM raw |
| --- | --- | --- |
| Original | 250 route A, 250 route B | 452 route A, 48 route B |
| Extended without toll | 500 via A-B | 500 via A-Z |
| Toll 40 | 37 via A-B, 232 via A-Z, 231 route B | 465 via A-Z, 35 route B, 0 via A-B |

## LLM Decision Quality

| Scenario | Corrected non-minimal choices | Accepted non-minimal choices |
| --- | ---: | ---: |
| Original | 127 | 448 |
| Extended without toll | 496 | 500 |
| Toll 40 | 105 | 893 |

## Interpretation

The `llm_checked` mode reproduces the theoretical results: the Braess paradox increases average travel time from 65 to 82 minutes when A-B is free, and toll 40 restores a near-equilibrium distribution with 64.8876 minutes.

The `llm_raw` mode measures the actual behaviour of the 7B model. It predominantly selects S_A at the start and A_Z at node A, even when the total cost of A_B is lower. Traffic therefore concentrates on the upper route. Without correction, toll 40 does not create the expected theoretical effect: no agent uses A-B and the average travel time remains 79.7920 minutes.

Conclusion: Qwen 2.5 7B produces valid JSON output, but it is not reliable enough to perform local numerical comparisons on its own. `llm_checked` should be presented as an LLM simulation audited by deterministic computation, while `llm_raw` evaluates autonomous LLM behaviour.

## Source Files

- `results_langroid_7b_toll40_checked/comparison.csv`
- `results_langroid_7b_toll40_raw/comparison.csv`
- `results_langroid_tolls_heuristic/comparison.csv`