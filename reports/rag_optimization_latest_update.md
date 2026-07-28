# RAG Optimization: Latest Validation Update

## Rejected experiment

The Korean-English dual-query experiment reduced F1 to `0.7495`. Although it
retrieved some English CATIA terms successfully, the additional context often
produced verbose and contradictory answers. It was not adopted.

## Current leading experiment

The retrieval configuration was kept fixed and only answer normalization was
added. The postprocessor extracts an explicitly marked final answer and reduces
safe refusal responses to the required one sentence.

| Configuration | F1 | Faithfulness | Mean response time | Manual hallucination rate |
|---|---:|---:|---:|---:|
| BGE-M3 + Chroma MMR 500 | 0.7555 | 0.7738 | 0.88s | 7.5% |
| BGE-M3 + Chroma MMR 500 + answer normalization | **0.7637** | **0.8371** | **0.76s** | 7.5% |

All 40 answers of the normalized configuration were manually reviewed. Three
answers were marked as unsupported or incorrect: two drawing-extension answers
that stated `.CATPart`, and one IGES/STEP answer that used the wrong purpose.

## Decision

The answer-normalization configuration is the current leading RAG candidate.
The next validation will compare this fixed RAG configuration against LLM-only
answers after the team's final LLM is selected.
