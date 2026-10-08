
# Annual Report RAG — Guardrails, Reranking and Human-in-the-Loop

A teaching project. A LangGraph pipeline that answers **year-over-year comparison
questions** about annual reports, with guardrails on the way in and out, a
cross-encoder reranker in the middle, and a human review step that pauses the
graph when an answer cannot be verified.

The use case is deliberately plain. The point is not the answers — it is being
able to see three mechanisms working:

| Mechanism                   | Where you see it                                                                                           |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
|                             |                                                                                                            |
| **Reranking**         | 20 candidates in, 5 out, with visible movement in the rankings                                             |
| **Guardrails**        | An off-topic question stopped before it costs anything; an ungrounded answer caught before anyone reads it |
| **Human-in-the-loop** | The graph genuinely stops mid-run and waits                                                                |

---

## The documents

Two companies, two financial years each — Infosys and Eternal (formerly Zomato),
FY2024-25 and FY2025-26.

That set is chosen for one reason: **the two reports of the same company are
written in nearly identical language.** Same headings, same boilerplate, only the
numbers differ.

So searching by meaning genuinely cannot tell the years apart, and no threshold
fixes it. That failure is real, unavoidable, and it is what the metadata
filtering and the reranker exist to solve.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then add your OpenAI key
```

Download the four annual reports from each company's investor relations page and
save them in `data/raw/` under **exactly** these names:

```
data/raw/infosys_fy2025.pdf
data/raw/infosys_fy2026.pdf
data/raw/eternal_fy2025.pdf
data/raw/eternal_fy2026.pdf
```

The PDFs are not in this repository — they are large and publicly available.

---

## Running it

**Build the index. Once, first. Takes a few minutes.**

```bash
python run_ingestion.py
```

Check the chunks-per-document table it prints at the end. The two years of one
company should be broadly comparable. If one is a fraction of the other, that
report has scanned pages and extraction partly failed — open the markdown in
`data/processed/` and look.

**Ask a question.**

```bash
python run_query.py
python run_query.py "How did the risk disclosures change?"
python run_query.py "How did revenue change?" Eternal
```

---

## Running the pieces separately

Every step runs on its own, which is how the project is meant to be taught —
one mechanism at a time.

```bash
python graph/state.py            # what flows through the pipeline (free)
python graph/build_graph.py      # compiles and draws the graph (free)

python retrieval/search.py       # vector search, with and without year filtering
python retrieval/rerank.py       # before and after reranking, side by side

python guardrails/topic.py       # what gets refused, and why
python guardrails/factual.py     # what gets flagged for review
```

`graph/build_graph.py` prints a Mermaid diagram. Paste it into
[mermaid.live](https://mermaid.live) for a picture of the pipeline generated from
the code rather than drawn by hand.

---

## Structure

```
config.py                 every setting, one file
run_ingestion.py          build the index
run_query.py              ask a question

ingestion/
  load_pdfs.py            PDF  -> markdown
  chunk.py                markdown -> labelled chunks
  build_index.py          chunks -> vector database

retrieval/
  search.py               fast, imprecise, filtered by year
  rerank.py               slow, careful, cuts 20 to 5

guardrails/
  topic.py                input:  is this our subject?
  factual.py              output: is this supported by the sources?

graph/
  state.py                what flows through
  nodes.py                what each step does
  build_graph.py          what order they run in
```

---

## The pipeline

```
                    input guardrail
                          |
                +---------+---------+
             blocked              allowed
                |                    |
                |                retrieve      20 candidates per year
                |                    |
                |                 rerank       cut to 5 per year
                |                    |
                |                generate
                |                    |
                |             output guardrail
                |                    |
                |          +---------+---------+
                |      grounded            needs review
                |          |                    |
                |      finalise         [ GRAPH PAUSES ]
                |          |                    |
                |          |             human review
                +----------+---------+----------+
                                |
                               END
```

---

## Three ideas worth taking away

**Filtering and similarity do different jobs.** A filter is a hard rule — fail it
and you cannot appear in the results no matter how similar you look. Similarity
is a soft ranking over whatever survives. Use filtering to decide *which
documents* are eligible, similarity to decide *which parts* are relevant. Asking
similarity to do both is where this corpus breaks.

**Reranking discards as much as it reorders.** The cut from 20 to 5 matters as
much as the new ordering. Fifteen loosely relevant chunks are not harmless — they
cost money, crowd the real evidence, and give the model plausible material to
reach a wrong conclusion from.

**The graph does not wait — it returns.** `interrupt_before` does not block or
sleep. It hands control back with the state saved, so the pause costs nothing
while it lasts. A wait of one second and a wait of three days are the same thing
to the graph. That is what makes human review workable when the reviewer is a
person with a queue, rather than someone sitting at your terminal.

---

## A note on the guardrails

The guardrails here are specific to *this* application. An earlier version of
these files, written for a mutual fund customer assistant, used entirely
different vocabulary — NAV, SIP, folio, redemption.

The mechanism transfers between projects. **The content does not.** Copying a
guardrail from elsewhere and leaving its examples in place produces something
that blocks your real users while letting through exactly what you did not want.

### Some more queries to try : 

python3 run_query.py "What new risks appeared in FY2026 that were not mentioned in FY2025?"

python3 run_query.py "By what percentage did revenue grow between the two years?"

python3 run_query.py "Did management sound more cautious about hiring in FY2026?"

python3 run_query.py "How did the company's approach to quantum computing change?"

If none of them pause:

Then your threshold is too permissive for this corpus. In `config.py`: 

set : 

```python
GROUNDING_THRESHOLD = 0.95
```
