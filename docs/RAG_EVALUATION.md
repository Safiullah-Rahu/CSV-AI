# RAG evaluation

How retrieval and answer quality were measured, what the numbers are, and what
they do and do not show. Raw results (per question) are in
`backend/eval/results/*.json`; the harness is `backend/eval/run_eval.py`.

## Method

`run_eval.py` runs the real application in-process against an embedded
PostgreSQL 16 + pgvector, uploads the corpus **through the API** (so parsing,
chunking, embedding and indexing are the production code paths), asks every
question through `POST /chat/ask`, and scores the ranked sources and the answer.

- **Corpus** (`backend/eval/corpus.py`): 7 documents across policy, IT security,
  finance and tabular sales data (Markdown, plain text and CSV).
- **Questions**: 36, of which 30 are answerable (each with its relevant
  document(s) and expected key phrases) and 6 are deliberately unanswerable
  from the corpus.
- **Retrieval metrics** (document-level relevance, k = 5): precision@5,
  recall@5, MRR, nDCG@5.
- **Answer metrics**: answer relevance (expected key-phrase recall), citation
  correctness (every cited source is a relevant document), faithfulness
  (share of answer sentences lexically supported by the cited passages, from
  the grounding verifier), false-refusal rate on answerable questions, refusal
  accuracy on unanswerable questions.
- **Latency**: retrieval (ms, mean and p95) and generation.

Every question has exactly one or two relevant documents, so **precision@5 is
capped near 0.2** by construction; recall, MRR and nDCG are the informative
retrieval numbers here.

```bash
cd backend
python eval/run_eval.py                       # hashing embedder, extractive answers
python eval/run_eval.py --embedder local      # fastembed BAAI/bge-small-en-v1.5
python eval/run_eval.py --llm ollama:qwen2.5:7b-instruct --limit 10
python eval/run_eval.py --embedder local --llm anthropic:claude-sonnet-5-5 \
  --configs hybrid+rerank --env-file ../.env.eval   # keys read, never printed
```

## Results

### Default pipeline (hybrid dense + BM25, reranking, PRF expansion), extractive answers

| Metric | Hashing embedder (768-d, no model) | Local `bge-small-en-v1.5` |
|---|---|---|
| precision@5 | 0.20 | 0.20 |
| recall@5 | 1.00 | 1.00 |
| MRR | 1.00 | 1.00 |
| nDCG@5 | 1.00 | 1.00 |
| answer relevance | 0.967 | 1.00 |
| citation correctness | 0.967 | 1.00 |
| faithfulness | 1.00 | 1.00 |
| false refusals (answerable) | 0.033 | 0.00 |
| refusal accuracy (unanswerable) | 0.833 | 0.667 |
| retrieval latency, mean / p95 | 18.6 ms / 20.2 ms | 30.4 ms / 36.2 ms |

Sources: `20261102-021653-hashing.json`, `20261102-021812-local.json`.

The two embedders trade off differently: the semantic model answers every
answerable question and cites perfectly, but its similarity floor lets two of
the six unanswerable questions through (it finds loosely related passages).
The refusal threshold is configurable per knowledge base
(`retrieval_config.similarity_floor`), so this is a tuning choice rather than a
fixed property.

### Ablations (local `bge-small-en-v1.5`, extractive)

| Configuration | recall@5 | MRR | answer rel. | citation | false refusals | refusal acc. | retrieval mean |
|---|---|---|---|---|---|---|---|
| hybrid + rerank (default) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.667 | 30.4 ms |
| hybrid, no rerank | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.667 | 28.1 ms |
| dense only | 1.00 | 1.00 | 1.00 | 0.933 | 0.00 | 0.667 | 25.4 ms |
| sparse (BM25) only | 1.00 | 1.00 | 0.967 | 0.967 | 0.033 | 0.833 | 21.2 ms |
| hybrid, no PRF expansion | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.667 | 27.8 ms |

On a corpus this small every configuration retrieves the right document;
hybrid search mainly improves citation precision over dense-only and answer
coverage over BM25-only. Reranking and PRF expansion cost 2–3 ms and show no
measurable gain here; they are kept because they matter on larger, noisier
corpora, which this benchmark does not contain.

### What changed during development

The first full run (`20261102-021243-hashing.json`) exposed weak grounding:
citation correctness 0.17, faithfulness 0.43, unanswerable-refusal accuracy
0.50. The fixes were IDF-weighted query coverage in the reranker, citation
normalisation and line-aware sentence splitting in the verifier, stricter
extractive sentence selection, prefix matching in BM25, and a per-embedder
similarity floor. The same benchmark afterwards gave the numbers above.

## Generated answers with hosted models (live APIs)

Same 36 questions, local `bge-small-en-v1.5` embeddings, default pipeline,
strict grounding policy, answers generated through the platform's own
provider code against the Anthropic and OpenAI APIs.

| Metric | Claude Sonnet 5.5 | Claude Opus 5.5 | GPT-5.4 mini | GPT-4.1 mini |
|---|---|---|---|---|
| answered by the model / refused before generation | 32 / 4 | 32 / 4 | 32 / 4 | 31 / 4 (+1 withheld) |
| recall@5 / MRR / nDCG@5 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| answer relevance | 1.00 | 1.00 | 1.00 | 1.00 |
| citation correctness | 1.00 | 0.933 | 1.00 | 1.00 |
| faithfulness | 1.00 | 1.00 | 1.00 | 1.00 |
| false refusals (answerable) | 0.00 | 0.00 | 0.00 | 0.00 |
| refusal accuracy (unanswerable) | 1.00 (0.667 first scored) | 1.00 (0.667 first scored) | 1.00 | 0.833 |
| generation latency, mean | 1.94 s | 1.96 s | 1.45 s | 1.02 s |
| tokens in / out (all 36) | 29,376 / 1,570 | 29,376 / 2,381 | 18,384 / 817 | 18,416 / 801 |
| estimated cost | $0.07 | $0.17 | under $0.05 | about $0.01 |

Sources: `20261102-034813-local-anthropic_claude-sonnet-5-5.json`,
`20261102-034945-local-anthropic_claude-opus-5-5.json`,
`20261102-035641-local-openai_gpt-5.4-mini.json`,
`20261102-035451-local-openai_gpt-4.1-mini.json` (answers included). Claude
costs use Anthropic list prices; OpenAI costs are rough estimates, so check
current pricing. Token counts differ between vendors because their tokenizers
differ.

How to read this:

- **Refusals.** Both Claude models and GPT-5.4 mini declined all six unanswerable questions, e.g.
  *"The provided sources don't include a stock ticker symbol for the company."*
  The first scoring only recognised the phrase "couldn't find", so it counted
  two of these correct declines as answers (0.667). That narrow check also
  existed in the engine, where it was a real bug: under the strict policy an
  uncited "the sources don't contain …" would have been withheld and replaced
  with unrelated passages. Refusal detection is now one shared function
  (`grounding.is_refusal`, tested in `test_refusal_detection_covers_common_phrasings`
  and `test_uncited_refusal_is_not_withheld_in_strict_mode`), and the saved
  answers were re-scored offline with it (`refusal_accuracy_unanswerable_rescored`
  in each file). No answers were regenerated.
- **GPT-4.1 mini 0.833.** On one unanswerable question (parental leave) its
  answer carried no citation and its wording was not recognised as a decline,
  so the strict policy withheld it and showed the closest passages instead: a
  conservative failure, but not a correct refusal. The harness did not yet
  record withheld text; a targeted re-run (`--grep parental`) produced a
  recognised decline (*"The provided sources do not contain any information
  about the company's parental leave policy."*). The detector now also covers
  openings such as "There is no information …" and "None of the sources …",
  and the harness records `rejected_model_answer` for future runs.
- **Opus citation correctness 0.933.** In two answers Opus added a second,
  correctly cited fact from another document (the one-hour data-loss rule next
  to the 24-hour lost-equipment rule; the MFA requirement next to account
  creation). The metric only allows the document labelled relevant, so it
  counts both as misses. The extra statements are accurate and supported.
- **Effort.** The Claude runs used each model's default effort (Sonnet 5.5: high,
  Opus 5.5: medium) because the provider factory did not yet forward the new
  `effort` setting. That is fixed; `EVAL_EFFORT` (default `low`) now applies.
- **Latency.** Retrieval latency in these runs (44–187 ms mean) is inflated:
  the machine was running several builds and browser sessions at the time.
  Use the extractive runs above for retrieval timing.
- **Two live bugs found on the way, both now fixed.** The first Claude
  request returned HTTP 400 (`temperature is deprecated for this model`): the
  provider still sent sampling parameters on API version `2023-06-01`. It now
  uses `2026-10-21`, which rejects them on every model, and no longer sends
  them. The first GPT-5.4 mini request returned HTTP 400 (`'max_tokens' is not
  supported with this model`): the OpenAI provider now sends
  `max_completion_tokens` to OpenAI's own API (compatible servers keep
  `max_tokens`) and retries once without `temperature` for models that accept
  only the default. In both cases the engine fell back to an extractive answer,
  as designed, until the fix landed.

## Generated answers with a local model: not credibly measured on this machine

Two runs with a local Ollama `qwen2.5:7b-instruct` (10 questions each,
`20261102-024233-…` and `20261102-025249-…`) scored 0 on answer relevance,
citations and faithfulness, with empty answers recorded. The log
(`llm_run.log`) shows why: Ollama repeatedly aborted generation with
`prediction aborted, token repeat limit reached`, meaning the model fell into
repetition loops. In manual checks on the same machine it also produced garbled
text for trivial prompts. Inference was CPU-only (30–49 s per attempt). These
runs therefore measure a malfunctioning local model, not the RAG pipeline, and
they are reported only for completeness.

They also exposed a boundary in the engine's fallback: if a model fails
**before** emitting any text, the engine answers extractively with a notice
(`test_model_failure_before_output_falls_back_to_extractive`); if it fails
**mid-answer**, as here, the request ends with an error event and the partial
answer is kept. Uncited model answers under the strict policy are withheld and
replaced by an extractive answer
(`test_uncited_model_answer_is_withheld_in_strict_mode`). (The run
`20261102-021832-…-ollama…` never reached the model, as its generation time of
0 ms shows, and is equivalent to an extractive run.)

To evaluate another hosted model, pass its key through the environment or a
gitignored env file, e.g.
`python eval/run_eval.py --embedder local --llm openai:gpt-4.1-mini --configs hybrid+rerank --env-file ../.env.eval`.

## Limitations of this benchmark

- Small, clean, English-only corpus; one relevant document per question.
- Lexical faithfulness (support of cited sentences) is a proxy; it does not
  judge semantic entailment.
- No multi-hop or cross-document synthesis questions; graph retrieval is not
  implemented.
- No load or concurrency testing; latencies are single-user, warm cache, on a
  developer workstation.
