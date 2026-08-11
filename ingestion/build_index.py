"""
================================================================================
INGESTION STEP 3 - BUILDING THE SEARCH INDEX
================================================================================

WHAT THIS FILE DOES
-------------------
Takes the chunks from step 2, turns each one into a list of numbers called an
embedding, and stores both the chunk and its embedding in a database that can
search them.

After this file runs, the pipeline is ready to answer questions.

WHAT AN EMBEDDING IS
--------------------
An embedding is a long list of numbers -- 1,536 of them for the model we use --
that represents the MEANING of a piece of text.

The useful property is this: texts with similar meaning produce similar lists,
even when they share no words at all. "The company faces currency risk" and
"exchange rate movements could affect us" have almost nothing in common
letter-by-letter, but their embeddings sit close together.

That is what lets us search by meaning instead of by keyword. A question gets
turned into an embedding by the same model, and we look for the chunks whose
embeddings sit nearest to it.

WHY WE EMBED EXPLICITLY HERE
----------------------------
Many tutorials let the database handle embedding invisibly. We call the embedding
model ourselves and hand the resulting numbers to the database.

It is a few more lines, and it makes the step visible. You can see the API call
happen, see how many chunks go in each batch, and see the size of what comes
back. When something is slow or expensive, you know exactly which line is
responsible.

WHAT CHROMA IS
--------------
Chroma is the database. It stores three things per chunk -- the text, the
embedding, and the metadata -- and it can find the chunks whose embeddings are
closest to a given embedding.

It runs entirely on your machine and saves to a folder on disk. Nothing to sign
up for, nothing to configure, and it survives between runs so ingestion only has
to happen once.

HOW TO RUN
----------
Normally called by run_ingestion.py. To run this step alone:

    python ingestion/build_index.py

================================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List

import chromadb
from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent))

import config


# The client we use to call the embedding model.
client = OpenAI(api_key=config.OPENAI_API_KEY)


# The name of the collection inside Chroma. A collection is roughly what a table
# is in an ordinary database: a named place where related records live.
COLLECTION_NAME = "annual_reports"


# How many chunks to embed in a single API call.
#
# Sending them one at a time would mean thousands of separate network round
# trips, which is slow and wasteful. Sending all of them at once exceeds the
# request size the API accepts.
#
# 100 is a comfortable middle. It is well inside every limit and reduces a few
# thousand calls down to a few dozen.
EMBEDDING_BATCH_SIZE = 100


# ==============================================================================
# SECTION 1 - TURNING TEXT INTO EMBEDDINGS
# ==============================================================================

def embed_texts(texts: List[str], verbose: bool = True) -> List[List[float]]:
    """
    Convert a list of texts into their embeddings, working in batches.

    PARAMETERS
    ----------
    texts
        The chunk texts to embed.

    RETURNS
    -------
    A list of embeddings, in the same order as the texts that went in. Each
    embedding is itself a list of floating point numbers.

    ORDER MATTERS ENORMOUSLY HERE
    -----------------------------
    The API returns embeddings in the same order it received the texts, and we
    rely on that completely: embedding number 7 must belong to chunk number 7.

    If the order were ever scrambled, nothing would crash. The database would
    fill up perfectly happily, and every search would return chunks that had
    nothing to do with the question -- with no error message anywhere to explain
    why. Silent misalignment is the worst kind of bug, so it is worth knowing
    that this is where it would happen.
    """
    all_embeddings = []

    # range with a step of EMBEDDING_BATCH_SIZE gives us the starting index of
    # each batch: 0, 100, 200, and so on.
    for batch_start in range(0, len(texts), EMBEDDING_BATCH_SIZE):

        # Python slicing does not complain when the end index runs past the end
        # of the list -- it simply stops at the last item. So the final batch
        # being smaller than the rest needs no special handling.
        batch = texts[batch_start:batch_start + EMBEDDING_BATCH_SIZE]

        if verbose:
            batch_number = batch_start // EMBEDDING_BATCH_SIZE + 1
            total_batches = (len(texts) - 1) // EMBEDDING_BATCH_SIZE + 1
            print(f"    embedding batch {batch_number} of {total_batches} "
                  f"({len(batch)} chunks)", flush=True)

        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=batch,
        )

        # response.data holds one item per input text, each with an .embedding
        # attribute containing the numbers. We pull them out in order.
        all_embeddings.extend(item.embedding for item in response.data)

    return all_embeddings


# ==============================================================================
# SECTION 2 - PREPARING THE METADATA
# ==============================================================================

def build_metadata(chunk: Dict) -> Dict:
    """
    Extract the metadata fields Chroma should store alongside a chunk.

    WHY THIS IS A SEPARATE FUNCTION RATHER THAN AN INLINE DICTIONARY
    ---------------------------------------------------------------
    Chroma has a restriction worth knowing: metadata values must be simple
    types -- a string, a number, or a true/false value. Lists and nested
    dictionaries are rejected.

    That is fine for us, since all our fields are simple. But keeping the
    conversion in one named function means there is one obvious place to look
    when someone later adds a field that Chroma refuses, rather than the
    restriction being buried inside a larger function.

    Note also that we deliberately do NOT copy the whole chunk dictionary. It
    contains the text itself, which Chroma stores separately -- copying it into
    the metadata as well would double the storage for no benefit.
    """
    return {
        "company": chunk["company"],
        "fiscal_year": chunk["fiscal_year"],
        "year_label": chunk["year_label"],
        "source_file": chunk["source_file"],
        "position": chunk["position"],
    }


# ==============================================================================
# SECTION 3 - BUILDING THE INDEX
# ==============================================================================

def build_index(chunks: List[Dict], verbose: bool = True) -> int:
    """
    Embed every chunk and store it in Chroma. Returns how many were stored.

    THIS FUNCTION REBUILDS FROM SCRATCH EVERY TIME
    ----------------------------------------------
    If a collection already exists, we delete it and start again rather than
    adding to it.

    That is deliberate. Adding to an existing collection means running ingestion
    twice leaves two copies of every chunk, which quietly degrades search --
    duplicate results crowd out genuinely different ones, and nothing looks
    broken.

    Rebuilding is slightly slower and completely predictable. For a project of
    this size the rebuild takes a minute or two, which is a good trade for never
    having to wonder what is actually in the database.
    """
    # PersistentClient saves to a folder on disk, so the database survives after
    # the script ends. The alternative, an in-memory client, would vanish when
    # the process exits and force a rebuild before every single question.
    chroma_client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR))

    # Remove any previous version of the collection.
    #
    # The try/except is needed because delete_collection raises an error if the
    # collection does not exist -- which is exactly the situation on a first run.
    # We do not want a first run to fail, so we catch that and carry on.
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        if verbose:
            print("  removed the previous collection")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        # Cosine similarity compares the DIRECTION of two embeddings and ignores
        # their length. That is what we want: a short chunk and a long chunk
        # about the same subject should count as similar, and it is the direction
        # that carries the meaning.
        #
        # Chroma's default is squared euclidean distance, which is affected by
        # length. Setting this explicitly avoids a subtle quality loss that would
        # be very hard to notice.
        metadata={"hnsw:space": "cosine"},
    )

    if verbose:
        print(f"  embedding {len(chunks):,} chunks")

    # Pull the texts out into their own list, in order, to send for embedding.
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts, verbose=verbose)

    # Give every chunk an identifier. Chroma requires one per record, and it must
    # be a string.
    #
    # We build it from the company, year and position -- for example
    # "Infosys_FY2026_0042" -- rather than using a random value.
    #
    # A readable identifier makes debugging far easier: when a search returns
    # something odd, the id alone tells you where in which document it came from,
    # with nothing else to look up.
    ids = [
        f"{chunk['company']}_{chunk['fiscal_year']}_{chunk['position']:04d}"
        for chunk in chunks
    ]

    metadatas = [build_metadata(chunk) for chunk in chunks]

    # Hand everything to Chroma in one call. The four lists must be the same
    # length and in the same order -- item 5 of each belongs to the same chunk.
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection.count()


# ==============================================================================
# SECTION 4 - OPENING THE INDEX LATER
# ==============================================================================

def get_collection():
    """
    Open the existing collection so it can be searched.

    This is what the retrieval code calls. It does not build anything -- it
    assumes ingestion has already run.

    The error message below matters more than it looks. Without the check, a
    missing collection produces a Chroma error that mentions the collection name
    but gives no hint that the fix is simply to run ingestion first. That is a
    confusing five minutes for someone who has just cloned the project.
    """
    chroma_client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR))

    try:
        return chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        raise RuntimeError(
            f"No search index found at {config.VECTOR_STORE_DIR}\n"
            f"Run 'python run_ingestion.py' first to build it."
        )


# ==============================================================================
# SECTION 5 - RUNNING THIS FILE DIRECTLY
# ==============================================================================

if __name__ == "__main__":
    from load_pdfs import load_all_pdfs
    from chunk import chunk_all_documents

    print("=" * 70)
    print("BUILDING THE SEARCH INDEX")
    print("=" * 70)
    print()

    documents = load_all_pdfs(verbose=False)
    chunks = chunk_all_documents(documents, verbose=False)

    stored_count = build_index(chunks)

    print()
    print(f"Stored {stored_count:,} chunks.")
    print(f"Index saved to: {config.VECTOR_STORE_DIR}")

    # Show how the chunks are distributed across the four documents.
    #
    # This is worth checking before moving on. The two years of one company
    # should hold roughly comparable numbers. A large gap means one document
    # extracted poorly back in step 1 -- and since the whole project compares one
    # year against the other, a thin year would produce weak comparisons with no
    # obvious cause.
    collection = get_collection()
    print()
    print("  chunks per document:")

    for company in config.COMPANIES:
        for fiscal_year in config.FISCAL_YEARS:
            # A "where" filter asks Chroma to count only records whose metadata
            # matches. $and requires both conditions to hold. This same filtering
            # mechanism is what the year-by-year search will use later.
            result = collection.get(
                where={"$and": [
                    {"company": {"$eq": company}},
                    {"fiscal_year": {"$eq": fiscal_year}},
                ]},
                include=[],          # we only want the count, not the content
            )
            print(f"    {company:<10} {fiscal_year}   {len(result['ids']):>5}")