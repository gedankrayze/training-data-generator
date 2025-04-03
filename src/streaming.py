"""
Memory-efficient streaming processing for large document collections.
"""

import asyncio
import json
import logging
import os
import random
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

from .checkpoint import CheckpointManager
from .document_processor import load_document, chunk_document_with_overlap
from .error_logging import ErrorLogger
from .example_generator import generate_examples_with_openai

# Configure logging
logger = logging.getLogger("streaming")


async def process_chunks_with_recovery(
        chunks: List[Dict[str, Any]],
        api_key: str,
        checkpoint_mgr: CheckpointManager,
        model: str = "gpt-4o",
        example_count: int = 2,
        negative_count: int = 2,
        max_concurrent: int = 5,
        temperature: float = 0.7,
        seed: int = 42,
        checkpoint_frequency: int = 10,
        error_file: str = "errors.json"
) -> List[Dict[str, Any]]:
    """
    Process chunks with error recovery through checkpointing.
    
    Args:
        chunks: List of document chunks
        api_key: OpenAI API key
        checkpoint_mgr: Checkpoint manager instance
        model: OpenAI model to use
        example_count: Number of examples to generate per chunk
        negative_count: Number of negative examples per positive example
        max_concurrent: Maximum number of concurrent tasks
        temperature: Temperature for generation
        seed: Random seed for reproducibility
        checkpoint_frequency: How often to save checkpoints (number of chunks)
        
    Returns:
        List of generated training examples
    """
    # Create AsyncOpenAI client
    client = AsyncOpenAI(api_key=api_key)

    # Set random seed
    random.seed(seed)

    # Initialize error logger
    error_logger = ErrorLogger(error_file)

    # Create a unique key for this run
    run_key = f"chunks_{len(chunks)}_{model}_{example_count}_{negative_count}_{seed}"

    # Check if we have a checkpoint for this run
    checkpoint_data = checkpoint_mgr.load_checkpoint(run_key)
    if checkpoint_data is not None:
        processed_chunks = checkpoint_data.get("processed_chunks", [])
        all_examples = checkpoint_data.get("examples", [])
        # Find unprocessed chunks
        processed_ids = {chunk["file_path"] + str(chunk["chunk_id"]) for chunk in processed_chunks}
        chunks_to_process = [c for c in chunks if c["file_path"] + str(c["chunk_id"]) not in processed_ids]
        logger.info(
            f"Resuming from checkpoint: {len(processed_chunks)} chunks already processed, {len(chunks_to_process)} remaining")
    else:
        processed_chunks = []
        all_examples = []
        chunks_to_process = chunks

    # If all chunks are processed, return the examples
    if not chunks_to_process:
        logger.info(f"All {len(chunks)} chunks already processed")
        return all_examples

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    # Setup progress tracking
    other_chunks = chunks.copy()

    async def process_chunk(chunk, chunk_index):
        # Use semaphore to limit concurrent requests
        async with semaphore:
            # Get sample of other chunks for negative examples
            # Remove current chunk from candidates
            negative_candidates = [c for c in other_chunks if c != chunk]
            # Sample a subset if there are many chunks
            if len(negative_candidates) > 10:
                negative_candidates = random.sample(negative_candidates, 10)

            # Generate examples
            try:
                examples = await generate_examples_with_openai(
                    chunk,
                    negative_candidates,
                    client,
                    model,
                    example_count,
                    negative_count,
                    temperature,
                    error_logger=error_logger
                )

                # Add this chunk to processed chunks
                processed_chunks.append(chunk)

                # Save checkpoint periodically
                if chunk_index % checkpoint_frequency == 0:
                    checkpoint_mgr.save_checkpoint(run_key, {
                        "processed_chunks": processed_chunks,
                        "examples": all_examples + examples
                    })

                return examples
            except Exception as e:
                logger.error(f"Error processing chunk {chunk['file_path']} (chunk {chunk['chunk_id']}): {e}")
                return []

    # Process chunks with progress tracking
    tasks = []
    for i, chunk in enumerate(chunks_to_process):
        tasks.append(process_chunk(chunk, i))

    for future in async_tqdm.as_completed(tasks, desc="Processing chunks", total=len(tasks)):
        examples = await future
        all_examples.extend(examples)

        # Save final checkpoint when done
        checkpoint_mgr.save_checkpoint(run_key, {
            "processed_chunks": processed_chunks,
            "examples": all_examples
        })

    logger.info(f"Generated {len(all_examples)} examples from {len(chunks)} chunks")
    return all_examples


async def process_directory_streaming(
        input_dir: str,
        api_key: str,
        output_file: str,
        checkpoint_mgr: CheckpointManager,
        max_files: Optional[int] = None,
        max_chunk_size: int = 2000,
        overlap_size: int = 200,
        batch_size: int = 10,
        model: str = "gpt-4o",
        example_count: int = 2,
        negative_count: int = 2,
        max_concurrent: int = 5,
        temperature: float = 0.7,
        seed: int = 42,
        error_file: str = "errors.json"
) -> int:
    """
    Process documents in a streaming fashion to minimize memory usage.
    Instead of loading all documents into memory, this function processes
    documents in batches and incrementally writes results.
    
    Args:
        input_dir: Directory containing documents
        api_key: OpenAI API key
        output_file: Path to output file
        checkpoint_mgr: Checkpoint manager for recovery
        max_files: Maximum number of files to process (None for all)
        max_chunk_size: Maximum chunk size in characters
        overlap_size: Number of characters to overlap between chunks
        batch_size: Number of files to process in each batch
        model: OpenAI model to use
        example_count: Number of examples to generate per chunk
        negative_count: Number of negative examples per positive example
        max_concurrent: Maximum number of concurrent API calls
        temperature: Temperature for generation
        seed: Random seed for reproducibility
        
    Returns:
        int: Total number of examples generated
    """
    # Collect file paths
    file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.md', '.txt', '.facts')):
                file_paths.append(os.path.join(root, file))

    # Shuffle and limit files if specified
    random.seed(seed)
    random.shuffle(file_paths)
    if max_files is not None:
        file_paths = file_paths[:max_files]

    logger.info(f"Processing {len(file_paths)} files from {input_dir}")

    # Setup for incremental JSON writing
    total_examples = 0
    run_key = f"streaming_{input_dir}_{len(file_paths)}_{model}"

    # Check if we have a checkpoint
    checkpoint_data = checkpoint_mgr.load_checkpoint(run_key)
    if checkpoint_data is not None:
        processed_files = checkpoint_data.get("processed_files", [])
        total_examples = checkpoint_data.get("total_examples", 0)
        # Find unprocessed files
        file_paths = [f for f in file_paths if f not in processed_files]
        logger.info(
            f"Resuming from checkpoint: {len(processed_files)} files already processed, {len(file_paths)} remaining")

        # Initialize or append to output file
        if os.path.exists(output_file):
            # File exists, we'll append to it
            file_mode = 'a'  # Append mode
        else:
            # Create new file with opening bracket
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('[\n')
            file_mode = 'a'  # Append mode for future writes
    else:
        # Start fresh
        processed_files = []
        # Create new output file with opening bracket
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
        file_mode = 'a'  # Append mode for future writes

    # Process files in batches
    for batch_start in range(0, len(file_paths), batch_size):
        batch_files = file_paths[batch_start:batch_start + batch_size]
        logger.info(
            f"Processing batch of {len(batch_files)} files ({batch_start + 1}-{batch_start + len(batch_files)} of {len(file_paths)})")

        # Load and process this batch of files
        all_chunks = []
        for file_path in batch_files:
            # Load document
            doc = load_document(file_path)

            # Skip empty or error documents
            if not doc["content"] or "error" in doc:
                processed_files.append(file_path)
                continue

            # Chunk document with overlap
            chunks = chunk_document_with_overlap(
                doc,
                max_chunk_size=max_chunk_size,
                overlap_size=overlap_size
            )
            all_chunks.extend(chunks)

        # Process chunks for this batch
        if all_chunks:
            examples = await process_chunks_with_recovery(
                all_chunks,
                api_key,
                checkpoint_mgr,
                model,
                example_count,
                negative_count,
                max_concurrent,
                temperature,
                seed,
                error_file=error_file
            )

            # Convert to format expected by SPLADE training script
            formatted_examples = []
            for example in examples:
                formatted_example = {
                    "query": example["query"],
                    "positive_document": example["positive_document"],
                    "negative_documents": [doc["document"] for doc in example["negative_documents"]],
                    "explanations": [doc["explanation"] for doc in example["negative_documents"]]
                }
                formatted_examples.append(formatted_example)

            # Write this batch to the output file
            with open(output_file, file_mode, encoding='utf-8') as f:
                for i, example in enumerate(formatted_examples):
                    # Add comma if not the first entry
                    if total_examples > 0 or i > 0:
                        f.write(',\n')
                    # Write the example as JSON
                    f.write(json.dumps(example, ensure_ascii=False, indent=2))

            total_examples += len(formatted_examples)

        # Mark these files as processed
        processed_files.extend(batch_files)

        # Update checkpoint
        checkpoint_mgr.save_checkpoint(run_key, {
            "processed_files": processed_files,
            "total_examples": total_examples
        })

        logger.info(f"Batch complete. Total examples so far: {total_examples}")

    # Close the JSON array
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n]')

    logger.info(f"Processing complete. Generated {total_examples} examples from {len(processed_files)} files")
    return total_examples
