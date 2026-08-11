"""
================================================================================
INGESTION STEP 2 - CHUNKING
================================================================================

WHAT THIS FILE DOES
-------------------
Takes the long markdown text produced by step 1 and cuts it into small pieces
called chunks. Every chunk keeps a copy of the company name and financial year it
came from.

WHY WE CUT DOCUMENTS UP AT ALL
------------------------------
An annual report runs to several hundred pages. Two problems stop us handing the
whole thing to the model:

  1. It will not fit. Models accept a limited amount of text at once, and a full
     annual report is well beyond it -- four of them, far beyond.

  2. Even if it fitted, it would work badly. Burying two relevant sentences
     inside three hundred pages of other text makes the model's job harder, not
     easier, and costs a great deal per question.

So instead we cut the document into pieces, find the few pieces relevant to the
question, and send only those. That is what the "retrieval" in retrieval
augmented generation means.

WHY THIS STEP DECIDES HOW GOOD THE PROJECT CAN BE
-------------------------------------------------
Chunking is the least glamorous step and the one that most determines quality.

Whatever information gets destroyed here cannot be recovered later. If a chunk is
cut so that a number is separated from the sentence explaining what it measures,
then no amount of clever searching or reranking will put them back together. The
retriever can only return chunks that exist.

It is worth being unhurried here for that reason.

HOW THE CUTTING ACTUALLY WORKS
------------------------------
We use a recursive character splitter. Despite the name, the idea is simple.

We give it a list of places we would PREFER to cut, in order of preference:

    1. at a top-level heading      (a line starting with "# ")
    2. at a sub-heading            (a line starting with "## ")
    3. at a blank line             (a paragraph break)
    4. at a single line break
    5. at a sentence end
    6. at a space between words
    7. anywhere at all

It tries the first option. If the resulting pieces are still too big, it takes
each oversized piece and tries the second option on it, and so on down the list.
Cutting mid-word only happens if nothing else worked.

The effect is that chunks tend to line up with real sections of the report rather
than falling at arbitrary points, which is exactly what we want.

HOW TO RUN
----------
Normally called by run_ingestion.py. To run this step alone:

    python ingestion/chunk.py

(It expects step 1 to have run first, so data/processed/ contains markdown.)

================================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List

# The splitter that does the cutting. It comes from LangChain, but this is the
# only piece of LangChain used in the ingestion pipeline -- everything else here
# is plain Python.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Make config.py importable from this folder. Explained in detail in
# ingestion/load_pdfs.py.
sys.path.append(str(Path(__file__).parent.parent))

import config


# ==============================================================================
# SECTION 1 - WHERE WE PREFER TO CUT
# ==============================================================================
# The list of separators, in order of preference. The splitter works down this
# list, only moving to the next option when the current one leaves pieces that
# are still too large.
#
# The order matters and is worth reading carefully.
#
# Headings come first because a heading marks a genuine change of subject in the
# document. Cutting there produces chunks that are about one thing.
#
# Paragraph breaks come next, for the same reason at a smaller scale.
#
# The last entry is an empty string, which means "cut anywhere". It is the
# fallback of last resort, used only when a single unbroken run of text is longer
# than the chunk size all by itself -- which does happen in reports, usually in a
# badly extracted table.
# ==============================================================================

MARKDOWN_SEPARATORS = [
    "\n# ",      # top-level heading  ("# Risk Factors")
    "\n## ",     # sub-heading        ("## Currency Risk")
    "\n### ",    # sub-sub-heading
    "\n\n",      # blank line, meaning a paragraph break
    "\n",        # a single line break
    ". ",        # the end of a sentence
    " ",         # a space between words
    "",          # anywhere -- last resort
]


# ==============================================================================
# SECTION 2 - BUILDING THE SPLITTER
# ==============================================================================

def build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create the text splitter, configured from the values in config.py.

    ABOUT length_function=len
    -------------------------
    This tells the splitter HOW to measure the size of a piece of text. We pass
    Python's built-in len, so size is measured in characters.

    The alternative would be to measure in tokens -- the units models actually
    count. Tokens are more accurate for staying inside a model's limits, but they
    require an extra library, they are slower to compute, and the relationship
    between them and characters is roughly constant anyway (about four characters
    per token for English).

    For our purposes characters are simpler and close enough. If you later needed
    to fit chunks precisely into a tight context budget, this is the setting you
    would change.

    ABOUT is_separator_regex=False
    ------------------------------
    Our separators above are plain text like "\\n## ", not regular expression
    patterns. Setting this to False tells the splitter to treat them literally.

    If it were True, characters such as . and * inside our separators would be
    interpreted as regex symbols with special meanings, and the cutting would go
    somewhere unexpected.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=MARKDOWN_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


# ==============================================================================
# SECTION 3 - CHUNKING ONE DOCUMENT
# ==============================================================================

def chunk_document(loaded_document: Dict) -> List[Dict]:
    """
    Cut one loaded document into chunks, copying its metadata onto each one.

    PARAMETERS
    ----------
    loaded_document
        One entry produced by ingestion/load_pdfs.py -- a dictionary holding the
        markdown text plus the company and fiscal year.

    RETURNS
    -------
    A list of chunk dictionaries, each carrying its own text and metadata.

    THE METADATA COPYING IS THE WHOLE POINT OF THIS FUNCTION
    --------------------------------------------------------
    Cutting the text is the easy part -- one line of code does it. The important
    work is that every single chunk leaves here knowing which company and which
    year it came from.

    Here is why that is not optional.

    Infosys FY2025 and Infosys FY2026 are written in nearly identical language.
    Same section headings, same sentence structures, same boilerplate. Only the
    numbers differ, and often not by much.

    A search that works by comparing MEANING therefore cannot reliably separate
    them, because in terms of meaning they genuinely are almost the same text.
    This is not a problem you can tune away.

    So we do not ask search to figure out the year. We label every chunk with its
    year here, and later we FILTER by that label -- searching only within FY2025,
    then only within FY2026, and comparing the two results.

    Everything the project does with year-over-year comparison rests on these few
    lines.
    """
    splitter = build_splitter()

    # split_text takes one long string and returns a list of shorter strings.
    # This single call is the actual cutting.
    text_pieces = splitter.split_text(loaded_document["text"])

    chunks = []

    # enumerate walks through the list AND gives us a counter at the same time.
    # Without it we would have to track the position with a separate variable.
    for position, text in enumerate(text_pieces):

        # Remove leading and trailing whitespace. Splitting frequently leaves a
        # stray newline at the start or end of a piece, and those affect the
        # length check below.
        text = text.strip()

        # Throw away anything too short to be useful.
        #
        # PDF extraction produces a lot of debris: page numbers alone on a line,
        # a heading whose body landed in the next chunk, fragments of table
        # borders, single stray words.
        #
        # These are never the answer to anything. But if kept, they still get
        # embedded, still get searched, and occasionally rank high -- because a
        # very short text can look deceptively similar to a very short question.
        # Removing them keeps the database clean.
        if len(text) < config.MIN_CHUNK_LENGTH:
            continue

        chunks.append({
            "text": text,

            # --- the metadata, copied onto every chunk ---------------------
            "company": loaded_document["company"],
            "fiscal_year": loaded_document["fiscal_year"],
            "year_label": loaded_document["year_label"],
            "source_file": loaded_document["source_file"],

            # Where this chunk sat in the original document. Not used for
            # searching, but very useful when inspecting results: it tells you
            # whether a chunk came from the front of the report or the back.
            "position": position,
        })

    return chunks


# ==============================================================================
# SECTION 4 - CHUNKING EVERYTHING
# ==============================================================================

def chunk_all_documents(loaded_documents: List[Dict], verbose: bool = True) -> List[Dict]:
    """
    Chunk every loaded document and return one combined list.

    All four documents' chunks end up in a single flat list. They do not need
    keeping apart, because each chunk carries its own company and year labels --
    which is exactly what makes filtering possible later.

    WATCH THE COUNTS AS THEY PRINT
    ------------------------------
    The two reports for one company should produce broadly similar numbers of
    chunks. A large mismatch means one document extracted badly in step 1, and
    catching that here is far cheaper than discovering it during retrieval.
    """
    all_chunks = []

    for loaded_document in loaded_documents:
        chunks = chunk_document(loaded_document)

        if verbose:
            label = f"{loaded_document['company']} {loaded_document['fiscal_year']}"

            # Work out the average chunk length. Most should land close to the
            # configured chunk size; a much smaller average suggests the document
            # is full of short fragments, which usually points back to extraction.
            #
            # The if/else guards against dividing by zero when a document
            # produced no chunks at all.
            average_length = (
                sum(len(c["text"]) for c in chunks) // len(chunks)
                if chunks else 0
            )

            # :<18 pads the label to 18 characters so the columns line up.
            print(f"  {label:<18} {len(chunks):>5} chunks   "
                  f"average {average_length:,} characters")

        # extend adds every item from the list, whereas append would add the list
        # itself as a single nested item.
        all_chunks.extend(chunks)

    return all_chunks


# ==============================================================================
# SECTION 5 - RUNNING THIS FILE DIRECTLY
# ==============================================================================

if __name__ == "__main__":
    # Import here rather than at the top, because this is only needed when
    # running standalone. When run_ingestion.py calls this file's functions, it
    # has already loaded the documents itself.
    from load_pdfs import load_all_pdfs

    print("=" * 70)
    print("CHUNKING")
    print("=" * 70)
    print()

    documents = load_all_pdfs(verbose=False)
    chunks = chunk_all_documents(documents)

    print()
    print(f"Produced {len(chunks):,} chunks in total.")
    print()

    # Print one chunk in full. Reading an actual chunk tells you more about
    # whether the settings are right than any statistic does -- you can see
    # immediately whether it holds a coherent idea or stops mid-thought.
    if chunks:
        sample = chunks[len(chunks) // 2]
        print("-" * 70)
        print(f"SAMPLE CHUNK  ({sample['company']} {sample['fiscal_year']}, "
              f"position {sample['position']})")
        print("-" * 70)
        print(sample["text"][:600])
        print("-" * 70)