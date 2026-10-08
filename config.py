"""
================================================================================
CONFIGURATION
================================================================================

WHAT THIS FILE IS
-----------------
Every setting the project uses lives here, and nowhere else.

Other files import this one and read values from it. No other file contains a
model name, a threshold, or a folder path written directly into the code.

WHY DO IT THIS WAY
------------------
The alternative is what most projects do by accident: the chunk size is defined
inside the chunker, the model name appears in four different files, and a
threshold sits buried on line 200 of something.

That works fine until you want to change something. Then "let me try a bigger
chunk size" turns into ten minutes of searching, and you find three of the four
places the model name appears and miss the fourth.

Putting everything in one file means tuning takes seconds and you can see all
your choices side by side.

WHAT YOU WILL ACTUALLY CHANGE
-----------------------------
CHUNK_SIZE, CHUNK_OVERLAP           how the documents are cut up
VECTOR_SEARCH_TOP_K, RERANK_TOP_K   how many chunks are found and kept
GROUNDING_THRESHOLD                 how strict the fact check is

The rest is set once and left alone.

================================================================================
"""

# os lets us read environment variables -- values that live outside the code,
# such as the API key.
import os

# Path is Python's modern way of handling file and folder locations. It is nicer
# to work with than plain strings, for reasons explained in the PATHS section.
from pathlib import Path

# load_dotenv reads a file named .env sitting in this folder and copies whatever
# is inside it into the environment variables.
from dotenv import load_dotenv

# Calling it here, at the top of the config file, means the key is loaded before
# anything else in the project runs. Every other file imports config, so this
# line runs first no matter which script is started.
#
# If there is no .env file, this does nothing and raises no error.
load_dotenv(override=True)


# ==============================================================================
# SECTION 1 - PATHS
# ==============================================================================
# Where things live on disk.
#
# WHY NOT JUST WRITE "data/raw"
# -----------------------------
# A path like "data/raw" is a RELATIVE path. It means "starting from wherever I
# am standing right now, look for a folder called data".
#
# Where you are standing is the folder you ran the command from, which is not
# necessarily the folder the script lives in. Run the project from inside its own
# folder and everything works. Run it from one folder up and Python looks in the
# wrong place, finds nothing, and gives you an error that does not explain why.
#
# So instead we build every path outward from the location of THIS FILE, which is
# always the same no matter where you run things from.
# ==============================================================================

# __file__ is a built-in variable holding the path to the file currently being
# run -- in this case, config.py itself.
#
# Path(__file__) turns that into a Path object, and .parent gives the folder
# containing it. Since config.py sits at the top of the project, its parent
# folder IS the project root.
PROJECT_ROOT = Path(__file__).parent

# Path objects join with a forward slash. DATA_DIR / "raw" reads naturally and
# produces the correct result on Windows, Mac and Linux alike -- Python inserts
# whichever separator the operating system uses.
DATA_DIR = PROJECT_ROOT / "data"

# The four annual report PDFs. You put these here by hand.
RAW_PDF_DIR = DATA_DIR / "raw"

# The markdown extracted from those PDFs. Written by the ingestion step, and
# meant to be opened and read -- see ingestion/load_pdfs.py for why that matters.
PROCESSED_DIR = DATA_DIR / "processed"

# The vector database. Chroma creates and manages this folder itself; you never
# need to look inside it.
VECTOR_STORE_DIR = DATA_DIR / "chroma"


# ==============================================================================
# SECTION 2 - THE DOCUMENT SET
# ==============================================================================
# The four documents we are working with: two companies, two financial years
# each.
#
# THIS TABLE IS THE MOST IMPORTANT THING IN THE FILE
# --------------------------------------------------
# It is a list of dictionaries. Each dictionary describes one PDF: the filename
# to look for, and what we know about it.
#
# Everything except the filename becomes METADATA -- extra information attached
# to every chunk that comes out of that document.
#
# Here is why that matters so much.
#
# Our central question is "what changed between FY2025 and FY2026?". Answering it
# means retrieving text from one year, then from the other, and comparing them.
#
# But the two annual reports of the same company are written in almost identical
# language. Same section titles, same phrasing, same boilerplate -- only the
# numbers differ. To a search system that compares meaning, an FY2025 paragraph
# and its FY2026 equivalent look like the same paragraph.
#
# So searching by meaning alone CANNOT reliably tell the years apart. It is not a
# tuning problem; the text genuinely is nearly the same.
#
# The fix is to label every chunk with its year as it goes into the database, so
# we can filter by year rather than hoping search figures it out. That labelling
# starts right here, in this table.
#
# WHY WRITE IT DOWN INSTEAD OF DETECTING IT
# -----------------------------------------
# We could try to read each PDF and work out which year it covers. Each report
# does say so, somewhere, in a format that differs by company and by year.
#
# Writing it in a table we control takes thirty seconds and is always right.
# Detecting it takes an afternoon and is sometimes wrong. When a small amount of
# manual setup removes a whole category of bug, do the manual setup.
# ==============================================================================

DOCUMENTS = [
    {
        # The exact filename that must exist inside data/raw/
        "filename": "infosys_fy2025.pdf",

        # Which company this report belongs to. Used to filter searches to one
        # company, so an Infosys question never retrieves Eternal text.
        "company": "Infosys",

        # The financial year, written in a fixed machine-friendly form. Every
        # entry uses exactly this format so filtering is a simple exact match.
        # FY2025 means the year that ended on 31 March 2025.
        "fiscal_year": "FY2025",

        # The same year written the way a person writes it. Used only when
        # displaying results, never for filtering.
        "year_label": "FY 2024-25",
    },
    {
        "filename": "infosys_fy2026.pdf",
        "company": "Infosys",
        "fiscal_year": "FY2026",       # year ended 31 March 2026
        "year_label": "FY 2025-26",
    },
    {
        "filename": "eternal_fy2025.pdf",
        "company": "Eternal",
        "fiscal_year": "FY2025",
        "year_label": "FY 2024-25",
    },
    {
        "filename": "eternal_fy2026.pdf",
        "company": "Eternal",
        "fiscal_year": "FY2026",
        "year_label": "FY 2025-26",
    },
]


# The two lines below produce the list of companies and the list of years, worked
# out from the table above rather than typed separately.
#
# Doing it this way means adding a third company to DOCUMENTS automatically
# updates both lists. Typing them by hand would mean remembering to update three
# places instead of one, and forgetting is the normal outcome.
#
# READING THE SYNTAX
# ------------------
#   {doc["company"] for doc in DOCUMENTS}
#
# The curly braces make this a SET COMPREHENSION. It walks through every entry in
# DOCUMENTS, pulls out the "company" value, and collects them into a set.
#
# A set automatically discards duplicates. Both Infosys entries produce the word
# "Infosys", and the set keeps only one.
#
# sorted() then turns the set into a list in alphabetical order. Without it the
# order would vary between runs, which makes printed output annoying to compare.
COMPANIES = sorted({doc["company"] for doc in DOCUMENTS})
FISCAL_YEARS = sorted({doc["fiscal_year"] for doc in DOCUMENTS})


# ==============================================================================
# SECTION 3 - MODELS
# ==============================================================================
# The four models this project uses, and why each one was chosen.
# ==============================================================================

# The model that writes the answers.
#
# This is the expensive, capable one, and it runs once per question. Comparison
# answers are the hardest thing we ask for -- the model has to read text from two
# different years and describe what changed without inventing anything -- so this
# is where the capability is worth paying for.
GENERATION_MODEL = "gpt-4o"

# The model that runs the guardrail checks.
#
# Deliberately a smaller and cheaper model. These checks only ever classify:
# they answer IN_SCOPE or OUT_OF_SCOPE, SAFE or UNSAFE. That is a much easier job
# than writing an answer.
#
# Using the large model here would cost several times more, add delay to every
# single question, and decide no better. Match the model to the difficulty of the
# job, not to what is newest.
GUARDRAIL_MODEL = "gpt-4o-mini"

# The model that turns text into numbers so it can be searched.
#
# An embedding model converts a piece of text into a long list of numbers, in
# such a way that texts with similar meaning produce similar lists. That is what
# makes it possible to search by meaning instead of by exact words.
EMBEDDING_MODEL = "text-embedding-3-small"

# The reranking model.
#
# THIS ONE IS DIFFERENT FROM THE OTHERS
# -------------------------------------
# It is not an OpenAI model and it does not run over the internet. It runs on
# your own machine, and the first time you use it, roughly 90 MB is downloaded
# and cached. After that it works offline.
#
# WHAT A CROSS-ENCODER IS, AND WHY WE NEED A SECOND KIND OF MODEL
# ---------------------------------------------------------------
# The embedding model above converts the question into numbers, converts each
# chunk into numbers separately, and then compares the two sets of numbers.
#
# Because the question and the chunk are never seen together, a lot of detail is
# lost. It is like describing two people to a third party and asking whether they
# would get along, instead of introducing them.
#
# A cross-encoder reads the question and the chunk TOGETHER, as one input, and
# scores how well they match. Far more accurate -- and far too slow to run
# against thousands of chunks, because it needs a separate pass for every pair.
#
# So the two are used one after the other. Embeddings quickly narrow thousands of
# chunks down to twenty candidates; the cross-encoder then reads those twenty
# properly and picks the best five. Fast and rough, then slow and careful.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# The API key, read from the environment rather than written here.
#
# os.getenv looks for a variable called OPENAI_API_KEY. load_dotenv() at the top
# of this file will have already copied it out of the .env file if one exists.
#
# It returns None if the variable is not set, rather than raising an error, which
# is why the check below exists.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Failing here, at import time, with a clear message is much kinder than failing
# later inside an API call with an authentication error that does not tell you
# the key was simply missing.
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        "Copy .env.example to .env and put your key in it, or set the "
        "environment variable directly."
    )


# ==============================================================================
# SECTION 4 - CHUNKING
# ==============================================================================
# An annual report runs to several hundred pages. We cannot hand all of it to the
# model at once, so each document is cut into small pieces called chunks, and we
# retrieve only the pieces relevant to the question.
#
# These two numbers control how that cutting happens, and they matter more than
# almost anything else in the project. Bad chunking cannot be rescued by a better
# model or a better reranker.
# ==============================================================================

# How large each chunk is, measured in CHARACTERS -- not words, not tokens.
#
# Roughly 1,200 characters is about 200 words, which is a healthy paragraph.
#
# WHY NOT SMALLER
# ---------------
# A chunk that is too small loses its own context. A sentence reading "this
# increased by 12% during the year" is useless on its own, because the sentence
# naming what "this" refers to was left behind in the previous chunk.
#
# WHY NOT LARGER
# --------------
# A chunk that is too large covers several unrelated topics at once. Retrieval
# then returns a page of text where only two sentences are relevant, and the
# model has to find them. Worse, the chunk's embedding becomes an average of
# several topics, which makes it match everything vaguely and nothing precisely.
CHUNK_SIZE = 1200

# How much consecutive chunks share with each other.
#
# With an overlap of 200, the last 200 characters of one chunk are also the first
# 200 characters of the next.
#
# WHY THIS IS NEEDED
# ------------------
# Cutting at a fixed size means the cut sometimes lands in the middle of a
# sentence. Without overlap, that sentence is split in half, and neither half
# makes sense on its own -- so a question about it matches neither chunk.
#
# Overlap guarantees that every sentence appears complete in at least one chunk.
# The cost is a slightly larger database, which is a very cheap price.
CHUNK_OVERLAP = 200

# Chunks shorter than this are thrown away.
#
# PDF extraction produces a lot of debris: page numbers on their own line,
# section headings with no body, fragments of table borders, single stray words.
#
# These are never the answer to anything, but they still get embedded, still get
# searched, and still occasionally rank highly because a short text can look
# deceptively similar to a short question. Removing them keeps the database clean.
MIN_CHUNK_LENGTH = 100


# ==============================================================================
# SECTION 5 - RETRIEVAL
# ==============================================================================
# Two numbers, and the gap between them is exactly where reranking lives.
# ==============================================================================

# How many chunks the fast vector search returns.
#
# We deliberately ask for a generous number rather than trusting the top few.
# Vector search is quick but imprecise, so the genuinely best chunk is often
# sitting at position 11 rather than position 1. Asking for 20 makes sure it is
# somewhere in the pile.
VECTOR_SEARCH_TOP_K = 20

# How many chunks survive after reranking and actually reach the model.
#
# The cross-encoder reads all 20 candidates properly and keeps the best 5.
#
# WHY THESE TWO NUMBERS MUST DIFFER
# ---------------------------------
# If both were 20, reranking would only reorder a list without removing anything
# from it, and the model would still receive all 20 chunks. The reranker's value
# comes from being able to DISCARD -- to say that 15 of those candidates were not
# actually relevant, so the model never sees them and cannot be misled by them.
RERANK_TOP_K = 5


# ==============================================================================
# SECTION 6 - GUARDRAIL THRESHOLDS
# ==============================================================================

# TOPIC GUARDRAIL
# ---------------
# How close in meaning a question must be to our subject before we accept it.
#
# Scores run from roughly 0.0 (completely unrelated) to 1.0 (identical meaning).
# Above 0.40 we treat the question as in scope.
#
# There is nothing magic about 0.40 -- it is a sensible starting point for the
# embedding model named above. A different embedding model produces a different
# spread of scores and would need its own value. The way to set it properly is to
# run real questions through and move the number until the split matches your
# judgement.
TOPIC_SIMILARITY_THRESHOLD = 0.40

# FACTUAL GUARDRAIL
# -----------------
# What proportion of the numbers in an answer must be traceable back to the
# retrieved chunks before we accept it without human review.
#
# WHY NOT 1.0
# -----------
# Because answers legitimately contain numbers that are not in the source text: a
# year the user mentioned in their question, a difference the model correctly
# calculated between two figures, an ordinal like "the three points below".
#
# Demanding that every number be traceable would send nearly every answer to
# human review, and a review queue that contains everything is a review queue
# nobody reads.
GROUNDING_THRESHOLD = 0.90


# ==============================================================================
# SECTION 7 - HUMAN IN THE LOOP
# ==============================================================================
# When the pipeline should stop running and wait for a person.
#
# THE TEMPTATION TO REVIEW EVERYTHING
# -----------------------------------
# It feels safer to have a human check every answer. In practice it is worse than
# useless. A reviewer facing every single answer stops reading carefully within an
# hour and begins approving on reflex -- which looks like oversight while
# providing none, and gives everyone false confidence.
#
# So we interrupt only where the automated checks are genuinely unsure. In this
# project that means comparison answers, where the model has drawn a conclusion
# across two documents rather than repeating a fact found in one.
#
# Setting this to False turns the interrupt off entirely, which is useful when
# you want to run a batch of questions without stopping.
INTERRUPT_ON_LOW_GROUNDING = True