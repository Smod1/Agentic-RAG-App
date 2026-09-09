from crewai_tools import PDFSearchTool

print("1. Starting PDFSearchTool...", flush=True)

rag_tool = PDFSearchTool(
    pdf="doc.pdf",
    config={
        "embedding_model": {
            "provider": "sentence-transformer",
            "config": {
                "model_name": "BAAI/bge-small-en-v1.5",
            },
        },
        "vectordb": {
            "provider": "chromadb",
            "config": {},
        },
    },
)

print("2. PDFSearchTool CREATED", flush=True)

print("3. Running search...", flush=True)

result = rag_tool.run(
    "What does Sporo Health do?"
)

print("4. SEARCH FINISHED", flush=True)
print(result, flush=True)