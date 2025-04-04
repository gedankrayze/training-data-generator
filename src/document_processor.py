"""
Functions for document loading and chunking.
"""

import logging
import os
import random
import re
from typing import Dict, Any, List, Optional

from tqdm import tqdm

from .table_processor import preprocess_tables

# Configure logging
logger = logging.getLogger("document_processor")


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove ZWNBSP characters (U+FEFF)
    text = text.replace('\ufeff', '')
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


async def load_document(
        file_path: str,
        api_key: Optional[str] = None,
        preprocess_md_tables: bool = False,
        model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Load a document from file and extract content.

    Args:
        file_path: Path to the document file
        api_key: OpenAI API key for table preprocessing
        preprocess_md_tables: Whether to preprocess markdown tables
        model: LLM model to use for preprocessing

    Returns:
        Dict containing document content and metadata
    """
    try:
        extension = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract content based on file type
        if extension == '.md' or extension == '.txt':
            # For markdown and text files, keep content as is
            content = content.replace('\ufeff', '')

            # Preprocess markdown tables if enabled and API key provided
            if preprocess_md_tables and api_key and extension == '.md':
                logger.info(f"Preprocessing tables in {file_name}")
                content = await preprocess_tables(content, api_key, model)
        else:
            # For other file types, try to extract plain text
            content = clean_text(content)

        return {
            "file_path": file_path,
            "file_name": file_name,
            "content": content,
            "extension": extension
        }

    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "content": "",
            "extension": os.path.splitext(file_path)[1].lower(),
            "error": str(e)
        }


def chunk_document(doc: Dict[str, Any], max_chunk_size: int = 2000, min_chunk_size: int = 200) -> List[Dict[str, Any]]:
    """
    Split document into smaller chunks for processing.

    Args:
        doc: Document dictionary with content
        max_chunk_size: Maximum chunk size in characters
        min_chunk_size: Minimum chunk size in characters

    Returns:
        List of chunk dictionaries
    """
    content = doc["content"]
    chunks = []

    # If content is short enough, keep as single chunk
    if len(content) <= max_chunk_size:
        if len(content) >= min_chunk_size:
            chunk = doc.copy()
            chunk["chunk_id"] = 0
            chunk["chunk_total"] = 1
            chunks.append(chunk)
        return chunks

    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', content)
    current_chunk = ""
    chunk_id = 0

    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue

        # If adding this paragraph exceeds max size, start new chunk
        if len(current_chunk) + len(paragraph) > max_chunk_size:
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunk = doc.copy()
                chunk["content"] = current_chunk
                chunk["chunk_id"] = chunk_id
                chunks.append(chunk)
                chunk_id += 1
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

    # Add the last chunk if not empty
    if current_chunk and len(current_chunk) >= min_chunk_size:
        chunk = doc.copy()
        chunk["content"] = current_chunk
        chunk["chunk_id"] = chunk_id
        chunks.append(chunk)

    # Update total chunks count
    for chunk in chunks:
        chunk["chunk_total"] = len(chunks)

    return chunks


def chunk_document_with_overlap(
        doc: Dict[str, Any],
        max_chunk_size: int = 2000,
        min_chunk_size: int = 200,
        overlap_size: int = 200
) -> List[Dict[str, Any]]:
    """
    Split document into smaller chunks with overlap for processing.
    
    Args:
        doc: Document dictionary with content
        max_chunk_size: Maximum chunk size in characters
        min_chunk_size: Minimum chunk size in characters
        overlap_size: Number of characters to overlap between chunks
        
    Returns:
        List of chunk dictionaries
    """
    content = doc["content"]
    chunks = []

    # If content is short enough, keep as single chunk
    if len(content) <= max_chunk_size:
        if len(content) >= min_chunk_size:
            chunk = doc.copy()
            chunk["chunk_id"] = 0
            chunk["chunk_total"] = 1
            chunks.append(chunk)
        return chunks

    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', content)

    # Create chunks with overlapping paragraphs
    current_chunk = ""
    current_paragraphs = []
    chunk_id = 0

    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue

        # If adding this paragraph exceeds max size, start new chunk
        if len(current_chunk) + len(paragraph) > max_chunk_size and len(current_chunk) >= min_chunk_size:
            # Create a chunk with the current content
            chunk = doc.copy()
            chunk["content"] = current_chunk
            chunk["chunk_id"] = chunk_id
            chunk["paragraphs"] = current_paragraphs.copy()  # Store paragraph info for debugging
            chunks.append(chunk)
            chunk_id += 1

            # Start a new chunk with overlap
            # Find paragraphs to include in the overlap
            overlap_chars = 0
            overlap_paragraphs = []

            # Add paragraphs from the end until we reach desired overlap
            for p in reversed(current_paragraphs):
                if overlap_chars + len(p) <= overlap_size:
                    overlap_paragraphs.insert(0, p)
                    overlap_chars += len(p)
                else:
                    # If we need part of this paragraph to reach overlap_size
                    if overlap_chars < overlap_size:
                        # Take as much as needed from the end of the paragraph
                        needed_chars = overlap_size - overlap_chars
                        # Make sure we don't truncate in the middle of a word
                        split_point = len(p) - needed_chars
                        while split_point > 0 and p[split_point] != ' ':
                            split_point -= 1
                        if split_point > 0:
                            overlap_paragraphs.insert(0, p[split_point:])
                    break

            # Start new chunk with overlap content
            current_chunk = "\n\n".join(overlap_paragraphs)
            current_paragraphs = overlap_paragraphs.copy()

            # Add the current paragraph
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
            current_paragraphs.append(paragraph)
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
            current_paragraphs.append(paragraph)

    # Add the last chunk if not empty
    if current_chunk and len(current_chunk) >= min_chunk_size:
        chunk = doc.copy()
        chunk["content"] = current_chunk
        chunk["chunk_id"] = chunk_id
        chunk["paragraphs"] = current_paragraphs  # Store paragraph info for debugging
        chunks.append(chunk)

    # Update total chunks count
    for chunk in chunks:
        chunk["chunk_total"] = len(chunks)

    return chunks


async def process_directory(
        input_dir: str,
        max_files: Optional[int] = None,
        max_chunk_size: int = 2000,
        api_key: Optional[str] = None,
        preprocess_md_tables: bool = False,
        model: str = "gpt-4o-mini"
) -> List[Dict[str, Any]]:
    """
    Process all documents in a directory and its subdirectories.

    Args:
        input_dir: Directory containing documents
        max_files: Maximum number of files to process (None for all)
        max_chunk_size: Maximum chunk size in characters
        api_key: OpenAI API key for table preprocessing
        preprocess_md_tables: Whether to preprocess markdown tables
        model: LLM model to use for preprocessing

    Returns:
        List of processed document chunks
    """
    all_chunks = []
    file_count = 0

    # Walk through directory and collect files
    file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.md', '.txt', '.facts')):
                file_paths.append(os.path.join(root, file))

    # Limit files if specified
    if max_files is not None:
        random.shuffle(file_paths)
        file_paths = file_paths[:max_files]

    logger.info(f"Processing {len(file_paths)} files from {input_dir}")

    # Process each file
    for file_path in tqdm(file_paths, desc="Processing files"):
        # Load document
        doc = await load_document(
            file_path,
            api_key=api_key,
            preprocess_md_tables=preprocess_md_tables,
            model=model
        )

        # Skip empty or error documents
        if not doc["content"] or "error" in doc:
            continue

        # Chunk document
        chunks = chunk_document(doc, max_chunk_size=max_chunk_size)
        all_chunks.extend(chunks)
        file_count += 1

    logger.info(f"Processed {file_count} files into {len(all_chunks)} chunks")
    return all_chunks
