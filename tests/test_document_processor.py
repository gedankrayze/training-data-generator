"""
Tests for document processor module.
"""

import os
import tempfile
import unittest

from src.document_processor import load_document, chunk_document, chunk_document_with_overlap


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for document processor functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary test files
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a test file
        self.test_file_path = os.path.join(self.temp_dir.name, "test_doc.txt")
        with open(self.test_file_path, "w") as f:
            f.write("This is a test document.\n\n"
                    "It has multiple paragraphs.\n\n"
                    "Each paragraph should be processed correctly.\n\n"
                    "The paragraphs should be split correctly.")

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def test_load_document(self):
        """Test loading a document."""
        doc = load_document(self.test_file_path)
        self.assertEqual(doc["file_name"], "test_doc.txt")
        self.assertEqual(doc["extension"], ".txt")
        self.assertIn("This is a test document.", doc["content"])
        self.assertNotIn("error", doc)

    def test_load_nonexistent_document(self):
        """Test loading a nonexistent document."""
        doc = load_document("nonexistent_file.txt")
        self.assertEqual(doc["file_name"], "nonexistent_file.txt")
        self.assertEqual(doc["content"], "")
        self.assertIn("error", doc)

    def test_chunk_document(self):
        """Test chunking a document."""
        doc = load_document(self.test_file_path)
        # Test with max_chunk_size larger than document
        chunks = chunk_document(doc, max_chunk_size=1000, min_chunk_size=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_total"], 1)

        # Test with small max_chunk_size to force multiple chunks
        chunks = chunk_document(doc, max_chunk_size=50, min_chunk_size=10)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk["chunk_total"], len(chunks))
            self.assertLessEqual(len(chunk["content"]), 50)

    def test_chunk_document_with_overlap(self):
        """Test chunking a document with overlap."""
        doc = load_document(self.test_file_path)

        # Test with max_chunk_size larger than document
        chunks = chunk_document_with_overlap(doc, max_chunk_size=1000, min_chunk_size=10, overlap_size=20)
        self.assertEqual(len(chunks), 1)

        # Test with small max_chunk_size to force multiple chunks with overlap
        chunks = chunk_document_with_overlap(doc, max_chunk_size=50, min_chunk_size=10, overlap_size=20)
        self.assertGreater(len(chunks), 1)

        # Check if chunks have the expected overlap
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                # Get the end of the current chunk
                current_end = chunks[i]["content"][-20:]
                # Get the start of the next chunk
                next_start = chunks[i + 1]["content"][:20]
                # Check for some overlap (might not be exact due to paragraph boundaries)
                self.assertTrue(
                    current_end in chunks[i + 1]["content"] or next_start in chunks[i]["content"],
                    f"No overlap found between chunk {i} and {i + 1}"
                )


if __name__ == "__main__":
    unittest.main()
