"""
================================================================================
INGESTION STEP 1 - LOADING THE PDFS
================================================================================

WHAT THIS FILE DOES
-------------------
Takes the four annual report PDFs sitting in data/raw/, converts each one into
markdown text, and saves the result into data/processed/.

That is the whole job. No cutting into chunks, no embeddings, no database. Those
come in the next two files.

WHY IT IS A SEPARATE STEP
-------------------------
Two reasons, and the second one is the important one.

First, one file doing one job is easier to read than one file doing three.

Second, and far more useful: because this step writes its output to disk, you can
OPEN the result and read it.

That sounds obvious and it is routinely skipped. Most tutorials extract the PDF
in memory, chunk it in memory, embed it, and store it -- so if the extraction was
bad, nobody ever finds out. Retrieval starts returning nonsense, and the search
for the cause begins at the wrong end of the pipeline.

Extraction quality sets a ceiling on everything downstream. No reranker rescues a
badly parsed document, because the information was already lost before search
ever ran. Writing the markdown to disk turns an invisible failure into a visible
one.

WHY pymupdf4llm AND NOT SOMETHING ELSE
--------------------------------------
Plenty of libraries read PDFs. They differ mainly in how hard they try to
understand the LAYOUT of a page.

A PDF does not store text as a document. It stores instructions for painting
marks at coordinates on a page. There is no "this is a paragraph" or "this is a
table" -- only positions. Every library has to reconstruct the structure from
those positions, and they vary enormously in how well they do it.

  - The simplest tools read text in the order it happens to be stored, which is
    often not reading order. A two-column page comes out with the columns
    interleaved, line by line, producing text that looks like words but reads
    like nonsense.

  - The most capable tools run machine learning models to detect layout regions
    and rebuild tables properly. Much better on complex pages, at the cost of a
    1-2 GB model download and a slow first run.

pymupdf4llm sits usefully in between. It handles reading order and headings,
returns markdown rather than flat text, installs in seconds, and needs no models.

For this project that is the right trade-off, because our questions are about
NARRATIVE -- how a company describes its risks, what changed in its strategy
language between two years. That text is ordinary flowing prose in the reports.

If you later want answers that live inside financial tables, this is the file to
revisit, and a table-aware library starts being worth its cost.

HOW TO RUN
----------
Normally this runs as part of run_ingestion.py in the project root.

To run just this step on its own:

    python ingestion/load_pdfs.py

================================================================================
"""

# sys lets us modify where Python looks for modules to import. Used just below.
import sys

# Path is Python's modern way of handling file locations. Explained further in
# config.py, which uses it heavily.
from pathlib import Path

# Dict and List are type hints. They do not change how the code runs; they
# describe what a function expects and returns, so anyone reading the function
# signature knows what they are dealing with without reading the body.
from typing import Dict, List

# The library that does the actual PDF reading.
import pymupdf4llm


# ------------------------------------------------------------------------------
# MAKING config.py IMPORTABLE FROM HERE
# ------------------------------------------------------------------------------
# This file lives in the ingestion/ folder. config.py lives one level up, in the
# project root.
#
# By default Python only looks for imports in the folder of the script being run
# and in the places where installed packages live. It does not look upward into
# parent folders. So a plain "import config" from inside ingestion/ fails.
#
# The line below fixes that by adding the project root to the list of places
# Python searches.
#
# Reading it from the inside out:
#   __file__               this file, ingestion/load_pdfs.py
#   Path(__file__).parent  the ingestion/ folder
#   .parent again          the project root, one level up
#   str(...)               sys.path holds plain strings, not Path objects
#   sys.path.append(...)   add it to the search list
#
# This must happen BEFORE "import config", which is why it sits here between the
# imports rather than at the top with them.
# ------------------------------------------------------------------------------
sys.path.append(str(Path(__file__).parent.parent))

import config


# ==============================================================================
# SECTION 1 - LOADING ONE PDF
# ==============================================================================

def load_single_pdf(document: Dict) -> Dict:
    """
    Convert one PDF into markdown text and attach what we know about it.

    PARAMETERS
    ----------
    document
        One entry from config.DOCUMENTS. A dictionary holding the filename to
        look for, plus the company name and financial year.

    RETURNS
    -------
    A dictionary containing the extracted text and its metadata, ready to be
    passed to the chunking step.

    WHY THE METADATA TRAVELS WITH THE TEXT FROM THE VERY START
    ----------------------------------------------------------
    Notice that company and fiscal_year are carried alongside the text right from
    this first step, rather than being worked out later.

    The alternative would be to extract all four documents into one pile and
    afterwards try to determine which is which -- by reading each one and hoping
    it states its own year somewhere findable. Every report does say so, in a
    place and format that differs by company and by year.

    Carrying the labels forward from a table we control is reliable and costs
    nothing. Detecting them later is unreliable and costs an afternoon.

    And these labels are not optional decoration. The entire project depends on
    them: comparing FY2025 with FY2026 requires being able to search within one
    year at a time, and that requires every chunk knowing which year it belongs
    to.
    """
    # Build the full path to the PDF by joining the raw folder with the filename.
    # The / operator on Path objects joins path parts using whatever separator
    # the operating system uses.
    pdf_path = config.RAW_PDF_DIR / document["filename"]

    # Check the file exists before trying to read it.
    #
    # Without this check, pymupdf4llm fails with a low-level error about being
    # unable to open a file, which does not make it obvious that the real problem
    # is a missing download or a filename typo. Checking first lets us say
    # exactly what is wrong and exactly how to fix it.
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Could not find '{document['filename']}' in {config.RAW_PDF_DIR}\n"
            f"Download the {document['company']} {document['year_label']} annual "
            f"report and save it under exactly that filename."
        )

    # The actual extraction.
    #
    # to_markdown reads the whole PDF and returns one long markdown string, with
    # headings preserved as lines beginning with # and ##.
    #
    # Those headings matter for the next step. The chunker uses them as preferred
    # places to cut, so that chunks tend to align with sections of the report
    # rather than falling arbitrarily in the middle of one.
    #
    # str() is needed because to_markdown expects a plain string path, not a Path
    # object.
    markdown_text = pymupdf4llm.to_markdown(str(pdf_path))

    # Return the text together with everything we know about where it came from.
    return {
        "text": markdown_text,
        "company": document["company"],
        "fiscal_year": document["fiscal_year"],
        "year_label": document["year_label"],
        "source_file": document["filename"],
    }


# ==============================================================================
# SECTION 2 - SAVING THE RESULT SO IT CAN BE INSPECTED
# ==============================================================================

def save_markdown(loaded_document: Dict) -> Path:
    """
    Write the extracted markdown into data/processed/ and return where it went.

    This is the step that makes extraction quality visible. Open these files
    before moving on -- read a page or two of each. It takes two minutes and it
    is the cheapest debugging you will ever do on this project.
    """
    # Create the processed folder if it is not already there.
    #
    #   parents=True   also create any missing parent folders (data/ itself)
    #   exist_ok=True  do nothing if the folder already exists, instead of
    #                  raising an error. Without this, running ingestion twice
    #                  would fail the second time.
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Name the output file after the company and year rather than after the
    # original PDF, so the processed folder is readable at a glance:
    #
    #     Eternal_FY2025.md
    #     Eternal_FY2026.md
    #     Infosys_FY2025.md
    #     Infosys_FY2026.md
    output_path = (
        config.PROCESSED_DIR
        / f"{loaded_document['company']}_{loaded_document['fiscal_year']}.md"
    )

    # Write the text to disk.
    #
    # encoding="utf-8" is not optional here, and leaving it out is a real bug
    # rather than a style choice.
    #
    # Annual reports contain the rupee symbol, curly quotation marks, em dashes,
    # and occasionally accented names. On systems whose default encoding is not
    # UTF-8 -- Windows in particular -- writing those characters fails with a
    # UnicodeEncodeError whose message does not make the cause obvious.
    output_path.write_text(loaded_document["text"], encoding="utf-8")

    return output_path


# ==============================================================================
# SECTION 3 - RUNNING THE WHOLE STEP
# ==============================================================================

def load_all_pdfs(verbose: bool = True) -> List[Dict]:
    """
    Load every document listed in config.DOCUMENTS.

    PARAMETERS
    ----------
    verbose
        When True, prints progress as each file is read. Set it to False when
        calling this from other code that has its own output.

    RETURNS
    -------
    A list of loaded documents, ready for chunking.

    WATCH THE CHARACTER COUNTS
    --------------------------
    The counts printed as this runs are worth a moment's attention.

    Two annual reports from the same company should come out broadly comparable
    in size -- they are similar documents about similar years. If one comes out
    at a small fraction of the other, extraction has silently failed on part of
    it.

    That is enormously easier to notice here, as two numbers side by side, than
    two hours later when retrieval keeps failing to find anything from one year
    and the cause is four steps upstream.
    """
    # Collect the results here as we go.
    loaded = []

    # Walk through the four documents defined in config.
    for document in config.DOCUMENTS:

        if verbose:
            # end=" " keeps the cursor on the same line so the character count
            # can be printed after it, giving one tidy line per file.
            #
            # flush=True forces the text to appear immediately. Python normally
            # holds printed output in a buffer and writes it in batches, which
            # would mean the "reading..." message only appearing AFTER the file
            # had finished loading -- exactly backwards for a progress message.
            print(f"  reading {document['filename']} ...", end=" ", flush=True)

        # Extract the text.
        loaded_document = load_single_pdf(document)

        # Write it to disk so it can be inspected.
        output_path = save_markdown(loaded_document)

        if verbose:
            character_count = len(loaded_document["text"])
            # The :, in the format string inserts thousands separators, turning
            # 1284302 into 1,284,302 -- much easier to compare at a glance.
            print(f"{character_count:,} characters  ->  {output_path.name}")

        loaded.append(loaded_document)

    return loaded


# ==============================================================================
# SECTION 4 - RUNNING THIS FILE DIRECTLY
# ==============================================================================
# Everything below only runs when this file is started directly, with:
#
#     python ingestion/load_pdfs.py
#
# It does NOT run when another file imports this one.
#
# HOW THAT WORKS
# --------------
# Python sets a hidden variable called __name__ in every file. When a file is run
# directly, __name__ is set to the string "__main__". When it is imported by
# another file, __name__ is set to the module's name instead.
#
# So this check means "only do this if I am the file being run". It lets a file
# work both as an importable module and as a runnable script.
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LOADING PDFS")
    print("=" * 70)
    print()

    documents = load_all_pdfs()

    print()
    print(f"Loaded {len(documents)} documents.")
    print(f"Markdown written to: {config.PROCESSED_DIR}")
    print()
    print("Before going further, open those files and read a page or two of")
    print("each. Confirm the text is readable and the two years look comparable.")