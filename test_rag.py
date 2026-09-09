import os
import time
import statistics
from pathlib import Path
from dotenv import load_dotenv

from crewai_tools import PDFSearchTool

# configuration

PDF_PATH = "doc.pdf"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Number of retrieval results requested
TOP_K = 5

# Number of repeated runs for latency measurement
WARMUP_RUNS = 2
TIMED_RUNS = 3


# TEST DATASET #
# Each question is manually labelled:
# vectorstore = should be answerable from the PDF
# websearch   = should require external information
# For vectorstore questions, keywords are used as a lightweight
# relevance check. This is NOT a semantic benchmark; it gives us
# a transparent, reproducible baseline.


TEST_QUERIES = [

 # pdf questions

    {
        "id": 1,
        "question": "What does Sporo Health do?",
        "expected_route": "vectorstore",
        "expected_terms": ["sporo", "clinical", "summar"],
    },

    {
        "id": 2,
        "question": "Does Sporo streamline patient chart reviews?",
        "expected_route": "vectorstore",
        "expected_terms": ["sporo", "chart", "review"],
    },

    {
        "id": 3,
        "question": "How much time can a thorough patient chart review take?",
        "expected_route": "vectorstore",
        "expected_terms": ["30", "minute", "chart", "review"],
    },

    {
        "id": 4,
        "question": "What percentage of physicians' daily time with EHRs is spent using the EHR?",
        "expected_route": "vectorstore",
        "expected_terms": ["49.2", "ehr"],
    },

    {
        "id": 5,
        "question": "How much physician time is spent face-to-face with patients?",
        "expected_route": "vectorstore",
        "expected_terms": ["27", "face", "face"],
    },

    {
        "id": 6,
        "question": "What percentage of medical errors in one family medicine case were associated with processing patient information?",
        "expected_route": "vectorstore",
        "expected_terms": ["29", "error", "patient", "information"],
    },

    {
        "id": 7,
        "question": "How many primary care physicians receive more information than they can handle?",
        "expected_route": "vectorstore",
        "expected_terms": ["69.6", "physician", "information"],
    },

    {
        "id": 8,
        "question": "What accuracy was reported for correctly identifying major patient issues?",
        "expected_route": "vectorstore",
        "expected_terms": ["93.8", "patient", "issue"],
    },

    {
        "id": 9,
        "question": "What proportion of cases required correction during a second review?",
        "expected_route": "vectorstore",
        "expected_terms": ["36.6", "correction", "review"],
    },

    {
        "id": 10,
        "question": "How can AI and NLP help with clinical summarization?",
        "expected_route": "vectorstore",
        "expected_terms": ["ai", "nlp", "clinical", "summar"],
    },


# web search questions
    {
        "id": 11,
        "question": "What is the current market share of Epic Systems in 2026?",
        "expected_route": "websearch",
        "expected_terms": [],
    },

    {
        "id": 12,
        "question": "What is the current price of ChatGPT Plus?",
        "expected_route": "websearch",
        "expected_terms": [],
    },

    {
        "id": 13,
        "question": "What are the latest FDA regulations for clinical AI systems?",
        "expected_route": "websearch",
        "expected_terms": [],
    },

    {
        "id": 14,
        "question": "What is the latest version of Python?",
        "expected_route": "websearch",
        "expected_terms": [],
    },

    {
        "id": 15,
        "question": "What are the latest developments in medical LLMs in 2026?",
        "expected_route": "websearch",
        "expected_terms": [],
    },
]




def percentile(values, p):
    """Calculate a percentile without requiring NumPy."""

    if not values:
        return 0.0

    values = sorted(values)

    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    fraction = index - lower

    return values[lower] + (
        values[upper] - values[lower]
    ) * fraction


def contains_expected_information(result, expected_terms):
    """
    Lightweight retrieval relevance check.

    A retrieval is considered a hit if at least two expected terms
    appear in the retrieved text.

    For very short expected-term lists, all available terms must appear.
    """

    if not expected_terms:
        return True

    text = str(result).lower()

    matches = sum(
        1 for term in expected_terms
        if term.lower() in text
    )

    required = min(2, len(expected_terms))

    return matches >= required


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

# startup
print_header("AGENTIC RAG EVALUATION")

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY not found.")
    print("This script does not require Groq for retrieval testing.")

pdf = Path(PDF_PATH)

if not pdf.exists():
    raise FileNotFoundError(
        f"Could not find {PDF_PATH}. "
        "Make sure test_rag.py is in the same folder as doc.pdf."
    )


print_header("DOCUMENT METRICS")

file_size_bytes = pdf.stat().st_size
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024

print(f"Document:          {PDF_PATH}")
print(f"File size:         {file_size_kb:.1f} KB")
print(f"File size:         {file_size_mb:.2f} MB")


print_header("INITIALISING RAG PIPELINE")

start_init = time.perf_counter()

rag_tool = PDFSearchTool(
    pdf=PDF_PATH,
    config={
        "embedding_model": {
            "provider": "sentence-transformer",
            "config": {
                "model_name": MODEL_NAME,
            },
        },
        "vectordb": {
            "provider": "chromadb",
            "config": {},
        },
    },
)

initialisation_time = time.perf_counter() - start_init

print(
    f"RAG initialisation/indexing time: "
    f"{initialisation_time:.3f} seconds"
)

print(f"Embedding model: {MODEL_NAME}")
print("Vector database: ChromaDB")


print_header("WARMING UP")

warmup_question = "What does Sporo Health do?"

for i in range(WARMUP_RUNS):

    start = time.perf_counter()

    try:
        rag_tool.run(warmup_question)

        elapsed = time.perf_counter() - start

        print(
            f"Warmup {i + 1}/{WARMUP_RUNS}: "
            f"{elapsed:.3f}s"
        )

    except Exception as e:

        print(
            f"Warmup {i + 1} failed: {type(e).__name__}: {e}"
        )



print_header("RETRIEVAL EVALUATION")

retrieval_results = []

successful_queries = 0
failed_queries = 0
relevant_hits = 0

latencies = []

for item in TEST_QUERIES:

    question_id = item["id"]
    question = item["question"]
    expected_route = item["expected_route"]
    expected_terms = item["expected_terms"]

    print()
    print(f"[{question_id:02d}] {question}")

    start = time.perf_counter()

    try:

        result = rag_tool.run(question)

        elapsed = time.perf_counter() - start

        latencies.append(elapsed)

        successful_queries += 1

        # Only perform relevance checking for questions
        # that are supposed to come from the PDF.
        if expected_route == "vectorstore":

            hit = contains_expected_information(
                result,
                expected_terms
            )

            if hit:
                relevant_hits += 1
                relevance = "HIT"
            else:
                relevance = "MISS"

        else:

            # These questions are deliberately outside
            # the document's intended scope.
            relevance = "N/A"

        print(f"Expected route: {expected_route}")
        print(f"Latency:        {elapsed:.3f}s")
        print(f"Retrieval:      {relevance}")

        retrieval_results.append({
            "id": question_id,
            "question": question,
            "expected_route": expected_route,
            "latency": elapsed,
            "relevance": relevance,
            "success": True,
        })

    except Exception as e:

        failed_queries += 1

        print(
            f"FAILED: {type(e).__name__}: {e}"
        )

        retrieval_results.append({
            "id": question_id,
            "question": question,
            "expected_route": expected_route,
            "latency": None,
            "relevance": "ERROR",
            "success": False,
        })



print_header("RETRIEVAL METRICS")

pdf_queries = [
    x for x in TEST_QUERIES
    if x["expected_route"] == "vectorstore"
]

successful_pdf_queries = [
    x for x in retrieval_results
    if x["expected_route"] == "vectorstore"
    and x["success"]
]

pdf_query_count = len(pdf_queries)

if pdf_query_count:

    retrieval_hit_rate = (
        relevant_hits / pdf_query_count
    ) * 100

else:

    retrieval_hit_rate = 0


if latencies:

    median_latency = statistics.median(latencies)

    mean_latency = statistics.mean(latencies)

    p95_latency = percentile(
        latencies,
        0.95
    )

    min_latency = min(latencies)

    max_latency = max(latencies)

else:

    median_latency = 0
    mean_latency = 0
    p95_latency = 0
    min_latency = 0
    max_latency = 0


success_rate = (
    successful_queries / len(TEST_QUERIES)
) * 100


print(f"Test queries:             {len(TEST_QUERIES)}")
print(f"Successful queries:       {successful_queries}")
print(f"Failed queries:           {failed_queries}")
print(f"Retrieval success rate:   {success_rate:.1f}%")

print()

print(
    f"PDF queries evaluated:    {pdf_query_count}"
)

print(
    f"Relevant retrievals:      {relevant_hits}/{pdf_query_count}"
)

print(
    f"Retrieval hit rate:       {retrieval_hit_rate:.1f}%"
)

print()

print(
    f"Mean retrieval latency:   {mean_latency:.3f}s"
)

print(
    f"Median retrieval latency: {median_latency:.3f}s"
)

print(
    f"P95 retrieval latency:    {p95_latency:.3f}s"
)

print(
    f"Minimum latency:          {min_latency:.3f}s"
)

print(
    f"Maximum latency:          {max_latency:.3f}s"
)



# This evaluates the built deterministic router_tool rather than spending Groq tokens.
# We reproduce its logic here so the evaluation is cheap and repeatable.

print_header("ROUTER EVALUATION")

PDF_KEYWORDS = [
    "sporo",
    "patient chart",
    "chart review",
    "clinical summarization",
    "clinical summary",
    "medical management",
    "diagnostic accuracy",
    "physician burnout",
    "ehr",
    "electronic health record",
    "clinical workflow",
    "medical errors",
    "ai clinical",
]


def local_router(question):

    question_lower = question.lower()

    for keyword in PDF_KEYWORDS:

        if keyword in question_lower:
            return "vectorstore"

    return "websearch"


router_correct = 0

router_results = []

for item in TEST_QUERIES:

    predicted = local_router(
        item["question"]
    )

    expected = item["expected_route"]

    correct = predicted == expected

    if correct:
        router_correct += 1

    router_results.append({
        "question": item["question"],
        "expected": expected,
        "predicted": predicted,
        "correct": correct,
    })

    print()
    print(item["question"])
    print(f"Expected:  {expected}")
    print(f"Predicted: {predicted}")
    print(
        "Result:    "
        + ("CORRECT" if correct else "INCORRECT")
    )


router_accuracy = (
    router_correct / len(TEST_QUERIES)
) * 100


print()
print(
    f"Router accuracy: "
    f"{router_correct}/{len(TEST_QUERIES)} "
    f"({router_accuracy:.1f}%)"
)


# Router breakdown
print_header("ROUTER BREAKDOWN")

vector_queries = [
    x for x in router_results
    if x["expected"] == "vectorstore"
]

web_queries = [
    x for x in router_results
    if x["expected"] == "websearch"
]

vector_correct = sum(
    x["correct"] for x in vector_queries
)

web_correct = sum(
    x["correct"] for x in web_queries
)

print(
    f"Vectorstore routing: "
    f"{vector_correct}/{len(vector_queries)} "
    f"({(vector_correct / len(vector_queries) * 100):.1f}%)"
)

print(
    f"Web-search routing:  "
    f"{web_correct}/{len(web_queries)} "
    f"({(web_correct / len(web_queries) * 100):.1f}%)"
)

# CrewAI workflow consumes Groq TPM. Running 15 questions
# through all 5 agents would be unnecessarily expensive.
# You can enable this with:
# python test_rag.py --full


import sys

if "--full" in sys.argv:

    print_header("FULL AGENTIC PIPELINE TEST")

    print(
        "WARNING: This will execute app.py and therefore "
        "consume Groq API tokens."
    )

    print(
        "Run this only when you specifically want "
        "end-to-end agent latency."
    )

    full_question = (
        "Does Sporo streamline patient chart reviews?"
    )

    start = time.perf_counter()

    try:

        # Importing app executes its current workflow.
        import app

        elapsed = time.perf_counter() - start

        print()
        print(
            f"Full 5-agent pipeline latency: "
            f"{elapsed:.3f}s"
        )

        print()
        print("Full pipeline completed successfully.")

    except Exception as e:

        elapsed = time.perf_counter() - start

        print()
        print(
            f"Full pipeline failed after "
            f"{elapsed:.3f}s"
        )

        print(
            f"{type(e).__name__}: {e}"
        )




print_header("METRICS")

print(
    f"""
RAG SYSTEM
----------
Embedding model:             {MODEL_NAME}
Vector database:             ChromaDB
Evaluation queries:          {len(TEST_QUERIES)}
PDF retrieval queries:       {pdf_query_count}

Retrieval hit rate:          {retrieval_hit_rate:.1f}%
Median retrieval latency:    {median_latency:.3f}s
P95 retrieval latency:       {p95_latency:.3f}s

Router accuracy:             {router_accuracy:.1f}%
Router test cases:           {len(TEST_QUERIES)}

Successful retrievals:       {successful_queries}/{len(TEST_QUERIES)}
Retrieval success rate:      {success_rate:.1f}%

Initialisation time:         {initialisation_time:.3f}s
"""
)

print("=" * 70)
print("Evaluation complete.")
print("=" * 70)