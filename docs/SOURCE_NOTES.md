# Source Notes

Verified on **2026-10-08**. This file records external facts used to define protocols; it is not a bibliography of project results.

## TypeSafe Jev

- Models: <https://docs.typesafe.ai/models>
  - current pinned model used by configs: `jev-1.13.0`;
  - text-only input;
  - aliases may move, so final experiments pin the versioned ID;
  - official docs advise decomposing broad judgments and combining probabilities downstream.
- API: <https://docs.typesafe.ai/api>
  - endpoint: `POST /v1/systemone`;
  - question types: `noul`, `choice`, `score`;
  - Choice returns selected option, full probability map and confidence;
  - Noul returns a probability in `[0,1]`.

Because service pricing/rate limits can change, the README intentionally does not use those values as stable project requirements.

## Existing Jev recommendation work

- “Decision-Oriented Recommendation Reranking: An Empirical Study of Jev”:
  <https://arxiv.org/abs/2609.40241>
- Uses Amazon Reviews 2023 domains, SASRec-generated candidate sets and multiple K values.
- This project treats it as a compatibility baseline, not as proof that Jev is production-latency competitive with specialized rankers.

## EMER / CMR

- EMER: <https://arxiv.org/abs/2508.05093>
  - request-level candidate comparison;
  - RAS-style posterior signal;
  - Transformer reranker;
  - self-evolving Advantage Evaluator.
- CMR: <https://arxiv.org/abs/2306.05118>
  - preference-conditioned controllable multi-objective reranking.

ParetoJev borrows problem framing, not code or claimed results.

## KuaiRand

- Repository: <https://github.com/chongminggao/KuaiRand>
- Logs include click/valid-play, like, follow, comment, forward, hate, long-view, watch time, duration and random-exposure indicators.
- Supplementary captions/categories are documented by the official repository.
- The dataset does not expose every original request candidate slate, hence the explicit pseudo-slate terminology.

## MicroLens and community JEV demo

- MicroLens: <https://github.com/westlake-repl/MicroLens>
- JEV-27B demo: <https://github.com/yuhai-china/JEV-27B-DEMO>
- JEV-27B-VL model card: <https://huggingface.co/autotrust/JEV-27B-VL>

The public demo protocol and metrics are upstream self-reports. ParetoJev implements the protocol independently and will label reproduced values separately.
