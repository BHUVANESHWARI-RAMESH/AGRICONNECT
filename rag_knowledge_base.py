"""Ingest sourced crop documents and retrieve matching knowledge chunks."""

import argparse
import re
from pathlib import Path
from typing import Any

import chromadb
from pypdf import PdfReader


PROJECT_DIRECTORY = Path(__file__).resolve().parent
KNOWLEDGE_DIRECTORY = PROJECT_DIRECTORY / "knowledge"
VECTOR_DATABASE_DIRECTORY = KNOWLEDGE_DIRECTORY / "vector_store"
COLLECTION_NAME = "agricultural_knowledge"
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
CROP_ALIASES = {
    "paddy": {"paddy", "rice"},
    "cotton": {"cotton"},
    "banana": {"banana"},
    "groundnut": {"groundnut", "peanut", "peanuts"},
}


def extract_text(document_path: Path) -> str:
    """Read plain text/Markdown or extract selectable text from a PDF."""
    extension = document_path.suffix.lower()
    if extension in {".md", ".txt"}:
        return document_path.read_text(encoding="utf-8")
    if extension == ".pdf":
        pdf = PdfReader(str(document_path))
        return "\n".join(page.extract_text() or "" for page in pdf.pages)
    raise ValueError(f"Unsupported document type: {extension or '(no extension)'}")


def split_text(text: str, chunk_size: int = 180, overlap: int = 30) -> list[str]:
    """Split text into overlapping word-based chunks for search."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be zero or more and smaller than chunk_size.")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def _read_source_metadata(text: str, document_path: Path, knowledge_directory: Path) -> dict[str, str]:
    """Keep the crop, document title, and source link alongside each chunk."""
    title = document_path.stem.replace("_", " ").title()
    source_url = ""
    source_organization = "Not specified"
    content_lines = []

    for line in text.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("# "):
            title = stripped_line[2:].strip()
        elif stripped_line.lower().startswith("source url:"):
            source_url = stripped_line.split(":", 1)[1].strip()
        elif stripped_line.lower().startswith("source organization:"):
            source_organization = stripped_line.split(":", 1)[1].strip()
        elif stripped_line.lower().startswith("topics:"):
            continue
        else:
            content_lines.append(line)

    relative_path = document_path.relative_to(knowledge_directory)
    return {
        "crop": relative_path.parts[0].casefold(),
        "source_document": relative_path.as_posix(),
        "title": title,
        "source_url": source_url,
        "source_organization": source_organization,
        "content": "\n".join(content_lines).strip(),
    }


def _crop_filter(question: str) -> str | None:
    """Narrow searches that clearly name one crop; keep comparisons unfiltered."""
    normalized_question = f" {re.sub(r'[^a-z0-9]+', ' ', question.casefold()).strip()} "
    matches = [
        crop
        for crop, aliases in CROP_ALIASES.items()
        if any(f" {alias} " in normalized_question for alias in aliases)
    ]
    return matches[0] if len(matches) == 1 else None


def ingest_documents(
    knowledge_directory: Path = KNOWLEDGE_DIRECTORY,
    vector_database_directory: Path = VECTOR_DATABASE_DIRECTORY,
) -> int:
    """Extract, chunk, embed, and store all supported crop documents."""
    knowledge_directory = Path(knowledge_directory)
    vector_database_directory = Path(vector_database_directory)
    document_paths = sorted(
        path
        for path in knowledge_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not document_paths:
        raise FileNotFoundError(f"No supported documents found in {knowledge_directory}")

    chunk_texts = []
    chunk_metadata = []
    chunk_ids = []
    for document_path in document_paths:
        text = extract_text(document_path)
        source = _read_source_metadata(text, document_path, knowledge_directory)
        for chunk_index, chunk in enumerate(split_text(source.pop("content"))):
            metadata = {**source, "chunk_index": chunk_index}
            chunk_texts.append(chunk)
            chunk_metadata.append(metadata)
            chunk_ids.append(f"{metadata['source_document']}::{chunk_index}")

    if not chunk_texts:
        raise ValueError("The supported documents did not contain any searchable text.")

    client = chromadb.PersistentClient(path=str(vector_database_directory))
    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Rebuild this collection so removed or shortened source files do not leave stale chunks.
        old_ids = collection.get()["ids"]
        if old_ids:
            collection.delete(ids=old_ids)

        # Chroma's default local embedding function creates and stores the text vectors.
        collection.add(ids=chunk_ids, documents=chunk_texts, metadatas=chunk_metadata)
        return len(chunk_texts)
    finally:
        client.close()


def retrieve_relevant_chunks(
    question: str,
    top_k: int = 5,
    vector_database_directory: Path = VECTOR_DATABASE_DIRECTORY,
) -> list[dict[str, Any]]:
    """Return the most relevant chunks with their original source metadata."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please provide a non-empty question.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive whole number.")

    vector_database_directory = Path(vector_database_directory)
    if not vector_database_directory.exists():
        raise FileNotFoundError("Knowledge base is not built yet. Run the ingest command first.")

    client = chromadb.PersistentClient(path=str(vector_database_directory))
    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        result_count = min(top_k, collection.count())
        if result_count == 0:
            return []

        crop = _crop_filter(question)
        query_options: dict[str, Any] = {
            "query_texts": [question.strip()],
            "n_results": result_count,
        }
        if crop:
            query_options["where"] = {"crop": crop}
        results = collection.query(**query_options)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        return [
            {"chunk": chunk, "source": metadata, "distance": distance}
            for chunk, metadata, distance in zip(documents, metadatas, distances)
        ]
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the agricultural knowledge base.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="Build or refresh the local vector database.")
    ask_parser = commands.add_parser("ask", help="Retrieve sourced chunks for a question.")
    ask_parser.add_argument("question", help="A crop question to search for.")
    args = parser.parse_args()

    if args.command == "ingest":
        chunk_count = ingest_documents()
        print(f"Ingested {chunk_count} chunks into {VECTOR_DATABASE_DIRECTORY}")
        return

    for result in retrieve_relevant_chunks(args.question):
        source = result["source"]
        print(f"{source['title']} ({source['crop']})")
        print(f"Document: {source['source_document']}")
        print(f"Source: {source['source_url']}")
        print(f"Distance: {result['distance']:.4f}")
        print(result["chunk"])
        print()


if __name__ == "__main__":
    main()