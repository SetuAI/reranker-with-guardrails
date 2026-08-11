"""
================================================================================
RUN INGESTION
================================================================================

WHAT THIS FILE DOES
-------------------
Runs the three ingestion steps in order, start to finish:

    1. load_pdfs    the four PDFs           ->  markdown text
    2. chunk        markdown text           ->  small labelled pieces
    3. build_index  pieces                  ->  a searchable database

WHEN TO RUN IT
--------------
Once, before anything else in the project works. Then again only if you change
the documents, the chunk settings, or the embedding model.

It is not part of answering a question. Ingestion happens ahead of time; asking
questions happens afterwards, against the database this leaves behind.

WHY IT IS A SEPARATE FILE FROM THE THREE STEPS
----------------------------------------------
Each step file knows how to do its own job and nothing else. This file knows the
ORDER. Keeping those apart means you can run any single step on its own while
debugging, without the others getting in the way.

It also gives the project one obvious front door. Somebody cloning the repository
does not need to work out which of eight files to run first.

HOW TO RUN
----------
From the project root:

    python run_ingestion.py

Expect it to take a few minutes. Most of that is the embedding calls in step 3.

================================================================================
"""

import time

# These imports work without any sys.path adjustment, because this file already
# sits in the project root -- the same folder as config.py, and the parent of the
# ingestion folder.
#
# The step files needed that adjustment; this one does not. That is worth
# noticing, because it explains why the awkward line at the top of the ingestion
# files is not simply copied everywhere.
import config
from ingestion.load_pdfs import load_all_pdfs
from ingestion.chunk import chunk_all_documents
from ingestion.build_index import build_index, get_collection


def print_heading(text: str) -> None:
    """
    Print a section heading.

    A small helper purely so the three step headings are formatted identically
    without the formatting being written out three times. When the output of a
    long-running script is the only thing you can see, making it readable is
    worth a few lines.
    """
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def check_pdfs_present() -> bool:
    """
    Confirm all four PDFs are in place before starting.

    WHY CHECK UPFRONT INSTEAD OF LETTING IT FAIL NATURALLY
    ------------------------------------------------------
    Without this, a missing fourth PDF is only discovered after the first three
    have already been extracted -- several minutes of work thrown away, and the
    user has to go and download a file and start over.

    Checking everything first costs milliseconds and reports every missing file
    at once, so one trip to the download page fixes all of them.

    This is a general habit worth forming: in any pipeline with an expensive
    middle, validate the inputs before the expensive part begins.
    """
    missing = []

    for document in config.DOCUMENTS:
        pdf_path = config.RAW_PDF_DIR / document["filename"]
        if not pdf_path.exists():
            missing.append(document)

    if not missing:
        return True

    print()
    print("Cannot start -- some PDFs are missing.")
    print(f"Expected them in: {config.RAW_PDF_DIR}")
    print()

    for document in missing:
        print(f"  missing: {document['filename']}")
        print(f"           ({document['company']} annual report, "
              f"{document['year_label']})")

    print()
    print("Download each report from the company's investor relations page and")
    print("save it under exactly the filename shown above.")

    return False


def main() -> None:
    """Run the full ingestion pipeline."""

    # Record the start time so we can report how long the whole thing took.
    # time.time() returns the number of seconds since a fixed point in the past;
    # subtracting two readings gives an elapsed duration.
    started_at = time.time()

    print_heading("INGESTION")
    print()
    print(f"  documents   {len(config.DOCUMENTS)}")
    print(f"  companies   {', '.join(config.COMPANIES)}")
    print(f"  years       {', '.join(config.FISCAL_YEARS)}")
    print(f"  chunk size  {config.CHUNK_SIZE} characters, "
          f"{config.CHUNK_OVERLAP} overlap")

    # Stop here if anything is missing, before doing any real work.
    if not check_pdfs_present():
        return

    # ---------------------------------------------------------------------
    # STEP 1 - extract text from the PDFs
    # ---------------------------------------------------------------------
    print_heading("STEP 1 OF 3 - READING THE PDFS")
    print()
    documents = load_all_pdfs()

    # ---------------------------------------------------------------------
    # STEP 2 - cut the text into chunks
    # ---------------------------------------------------------------------
    print_heading("STEP 2 OF 3 - CHUNKING")
    print()
    chunks = chunk_all_documents(documents)
    print()
    print(f"  {len(chunks):,} chunks in total")

    # ---------------------------------------------------------------------
    # STEP 3 - embed the chunks and store them
    # ---------------------------------------------------------------------
    print_heading("STEP 3 OF 3 - BUILDING THE SEARCH INDEX")
    print()
    stored_count = build_index(chunks)

    # ---------------------------------------------------------------------
    # Report what was built
    # ---------------------------------------------------------------------
    elapsed_seconds = time.time() - started_at

    print_heading("DONE")
    print()
    print(f"  stored      {stored_count:,} chunks")
    print(f"  location    {config.VECTOR_STORE_DIR}")
    print(f"  took        {elapsed_seconds:.0f} seconds")

    # THE MOST USEFUL PART OF THIS OUTPUT
    # -----------------------------------
    # The breakdown below is the one thing worth reading carefully before moving
    # on.
    #
    # The two years of the same company should hold broadly comparable numbers of
    # chunks. They are similar documents describing similar years.
    #
    # If one year comes out at a fraction of the other, extraction went wrong for
    # that document back in step 1 -- most often because parts of it are scanned
    # images rather than real text.
    #
    # That matters here more than in most projects, because the whole point is
    # comparing one year against the other. A thin year produces weak, one-sided
    # comparisons, and the cause sits three steps upstream where nobody thinks to
    # look.
    collection = get_collection()

    print()
    print("  chunks per document:")
    print()

    for company in config.COMPANIES:
        for fiscal_year in config.FISCAL_YEARS:
            # Ask Chroma to return only records matching both conditions.
            # $and means both must hold; $eq means "equals".
            #
            # This is the same filtering mechanism the year-by-year search will
            # use later, which is why it is worth seeing it here first in a
            # simple form.
            result = collection.get(
                where={"$and": [
                    {"company": {"$eq": company}},
                    {"fiscal_year": {"$eq": fiscal_year}},
                ]},
                include=[],       # we want the count only, not the text
            )
            print(f"    {company:<10} {fiscal_year}   {len(result['ids']):>6}")

    print()
    print("  If the two years of a company differ wildly, open the markdown in")
    print(f"  {config.PROCESSED_DIR} and check the thinner one extracted properly.")
    print()


if __name__ == "__main__":
    main()