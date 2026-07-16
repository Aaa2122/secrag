# Evals changelog

- `2026-07-09` **vector-baseline** (4b8b50b, mode=vector): recall@5 0.484 · recall@10 0.580 · MRR@10 0.424 · p95 70ms · $0.0000/query
- `2026-07-09` **hybrid-rrf** (4b8b50b, mode=hybrid): recall@5 0.535 · recall@10 0.638 · MRR@10 0.473 · p95 69ms · $0.0000/query
- `2026-07-09` **hybrid-rerank-v2m3** (657e921, mode=hybrid): recall@5 0.702 · recall@10 0.747 · MRR@10 0.622 · p95 57315ms · $0.0000/query
- `2026-07-10` **generation-hybrid-rerank-haiku** (f6525e3, claude-haiku-4-5): faithful 0.978 · fabricated figures 0.077 · refusal-correct 1.00 · $0.2537 total
- `2026-07-13` **hybrid-decomposed** (9bb9f5d, mode=hybrid): recall@5 0.606 · recall@10 0.721 · MRR@10 0.531 · p95 236ms · $0.0000/query
- `2026-07-13` **hybrid-decomposed-rerank-v2m3** (9bb9f5d, mode=hybrid): recall@5 0.708 · recall@10 0.763 · MRR@10 0.643 · p95 140320ms · $0.0000/query
- `2026-07-14` **hybrid-decomposed-minilm-l6** (ebdc089, mode=hybrid): recall@5 0.702 · recall@10 0.766 · MRR@10 0.559 · p95 6195ms · $0.0000/query
