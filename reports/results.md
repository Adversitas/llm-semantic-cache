# Results

LLM verifier alone (Qwen2.5-1.5B-Instruct) PAWS-X fr test ROC-AUC: 0.796

## Pair discrimination

| Embedder | PAWS-X fr test ROC-AUC | STS-B fr test Spearman | Embed ms/query |
|---|---|---|---|
| char2-4-hashing | 0.549 | 0.653 | 0.9 |
| multilingual-e5-small | 0.550 | 0.792 | 34.5 |

## Cache decisions on PAWS-X fr test (thresholds chosen on dev)

| Embedder | Policy | Target precision | Served % | Precision % | Recall % | Wrong answers / 1000 req. | Verifier calls % |
|---|---|---|---|---|---|---|---|
| char2-4-hashing | threshold only | 0.90 | 0.2 | 75.0 | 0.3 | 0.5 | 0.0 |
| char2-4-hashing | threshold only | 0.95 | 0.2 | 75.0 | 0.3 | 0.5 | 0.0 |
| char2-4-hashing | threshold only | 0.99 | 0.2 | 75.0 | 0.3 | 0.5 | 0.0 |
| char2-4-hashing | cascade | 0.90 | 4.2 | 92.9 | 8.6 | 3.0 | 18.4 |
| char2-4-hashing | cascade | 0.95 | 0.2 | 75.0 | 0.3 | 0.5 | 0.0 |
| char2-4-hashing | cascade | 0.99 | 0.2 | 75.0 | 0.3 | 0.5 | 0.0 |
| multilingual-e5-small | threshold only | 0.90 | 1.5 | 93.3 | 3.1 | 1.0 | 0.0 |
| multilingual-e5-small | threshold only | 0.95 | 1.5 | 93.1 | 3.0 | 1.0 | 0.0 |
| multilingual-e5-small | threshold only | 0.99 | 1.4 | 100.0 | 3.0 | 0.0 | 0.0 |
| multilingual-e5-small | cascade | 0.90 | 6.2 | 93.6 | 13.0 | 4.0 | 24.9 |
| multilingual-e5-small | cascade | 0.95 | 3.1 | 96.8 | 6.8 | 1.0 | 9.4 |
| multilingual-e5-small | cascade | 0.99 | 1.6 | 100.0 | 3.4 | 0.0 | 2.2 |

## Replay through the real cache (all test s1 cached, every s2 queried, 0.95 threshold)

| Embedder | Entries | Served % | Precision % | Wrong / 1000 | Lookup ms (excl. embed) |
|---|---|---|---|---|---|
| char2-4-hashing | 1952 | 5.7 | 98.2 | 1.0 | 4.86 |
| multilingual-e5-small | 1952 | 5.7 | 97.4 | 1.5 | 0.66 |

## Cost of a miss

Generating an answer with Qwen2.5-1.5B-Instruct: 24.38 s, 150 tokens on average (5 insurance questions).
One verifier call: 185.7 ms.
