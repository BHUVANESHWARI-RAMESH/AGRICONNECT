"""End-to-end tests for agricultural document ingestion and retrieval."""

import tempfile
import unittest
from pathlib import Path

from rag_knowledge_base import ingest_documents, retrieve_relevant_chunks


class AgriculturalKnowledgeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.vector_database_directory = Path(cls.temporary_directory.name) / "vectors"
        cls.chunk_count = ingest_documents(
            vector_database_directory=cls.vector_database_directory
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_ingestion_creates_searchable_chunks(self) -> None:
        self.assertGreater(self.chunk_count, 0)

    def test_crop_questions_return_source_attributed_chunks(self) -> None:
        questions = {
            "What are the irrigation considerations for paddy?": "paddy",
            "When should cotton be irrigated and harvested?": "cotton",
            "What irrigation stages matter for groundnut?": "groundnut",
            "How is banana irrigation scheduled?": "banana",
        }

        for question, expected_crop in questions.items():
            with self.subTest(crop=expected_crop):
                results = retrieve_relevant_chunks(
                    question,
                    top_k=5,
                    vector_database_directory=self.vector_database_directory,
                )
                matching_sources = [
                    result
                    for result in results
                    if result["source"]["crop"] == expected_crop
                ]
                self.assertTrue(matching_sources, f"No {expected_crop} source retrieved")
                self.assertTrue(
                    all(result["source"]["crop"] == expected_crop for result in results),
                    f"A question naming {expected_crop} returned another crop",
                )

                for result in matching_sources:
                    source = result["source"]
                    self.assertTrue(result["chunk"].strip())
                    self.assertTrue(source["source_document"])
                    self.assertTrue(source["source_url"].startswith("https://"))
                    self.assertTrue(source["source_organization"])

                best_match = matching_sources[0]["source"]
                print(
                    f"Retrieved for {expected_crop}: "
                    f"{best_match['source_document']} | {best_match['source_url']}"
                )

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty question"):
            retrieve_relevant_chunks(
                " ", vector_database_directory=self.vector_database_directory
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)