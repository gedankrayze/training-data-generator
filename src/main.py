"""
Main entry point for training data generation.
"""

import argparse
import asyncio
import json
import logging
import os
import random

import nltk

from .checkpoint import CheckpointManager
from .document_processor import process_directory
from .example_generator import process_chunks_async
from .streaming import process_directory_streaming

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

# Try to load NLTK data, download if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate training data for SPLADE model fine-tuning")

    parser.add_argument('--input-dir', '-i', required=True,
                        help='Directory containing document files')

    parser.add_argument('--output-file', '-o', required=True,
                        help='Output file for training data (JSON)')

    parser.add_argument('--validation-file',
                        help='Output file for validation data (JSON). If not specified, no separate validation file is created.')

    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Ratio of validation examples (default: 0.1)')

    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (default: all)')

    parser.add_argument('--max-chunk-size', type=int, default=2000,
                        help='Maximum chunk size in characters (default: 2000)')

    parser.add_argument('--overlap-size', type=int, default=200,
                        help='Number of characters to overlap between chunks (default: 200)')

    parser.add_argument('--example-count', type=int, default=2,
                        help='Number of examples to generate per chunk (default: 2)')

    parser.add_argument('--negative-count', type=int, default=2,
                        help='Number of negative examples per positive example (default: 2)')

    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of files to process in each batch (default: 10)')

    parser.add_argument('--max-concurrent', type=int, default=5,
                        help='Maximum number of concurrent API calls (default: 5)')

    parser.add_argument('--model', default="gpt-4o",
                        help='OpenAI model to use (default: gpt-4o)')

    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for generation (default: 0.7)')

    parser.add_argument('--api-key',
                        help='OpenAI API key (default: from OPENAI_API_KEY environment variable)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--streaming', action='store_true',
                        help='Use memory-efficient streaming processing (recommended for large datasets)')

    parser.add_argument('--checkpoint-dir', default="./checkpoints",
                        help='Directory for checkpoint files (default: ./checkpoints)')

    parser.add_argument('--error-file', default="errors.json",
                        help='File to store error logs (default: errors.json)')

    return parser.parse_args()


def split_train_val(data: list, val_ratio: float = 0.1) -> tuple:
    """
    Split data into training and validation sets.

    Args:
        data: List of training examples
        val_ratio: Ratio of validation examples

    Returns:
        Tuple of (training_data, validation_data)
    """
    # Shuffle data
    data_copy = data.copy()
    random.shuffle(data_copy)

    # Split into train and validation sets
    val_size = int(len(data_copy) * val_ratio)
    train_data = data_copy[val_size:]
    val_data = data_copy[:val_size]

    logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation examples")
    return train_data, val_data


async def main_streaming(args):
    """Main function using memory-efficient streaming approach."""
    # Set random seed for reproducibility
    random.seed(args.seed)

    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key")
        return

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)

    # Process directory in streaming fashion
    total_examples = await process_directory_streaming(
        args.input_dir,
        api_key,
        args.output_file,
        checkpoint_mgr,
        args.max_files,
        args.max_chunk_size,
        args.overlap_size,
        args.batch_size,
        args.model,
        args.example_count,
        args.negative_count,
        args.max_concurrent,
        args.temperature,
        args.seed,
        error_file=args.error_file
    )

    # If validation file is specified, split the data
    if args.validation_file and total_examples > 0:
        # Since we're using streaming, we need to load the data again to split it
        with open(args.output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Split into train and validation sets
        val_size = int(len(data) * args.val_ratio)
        random.shuffle(data)  # Shuffle again before splitting
        train_data = data[val_size:]
        val_data = data[:val_size]

        # Save training data (overwrite original file)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        # Save validation data
        with open(args.validation_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation examples")

    logger.info("Training data generation completed")


async def main_standard(args):
    """Main function using standard approach (loads all data into memory)."""
    # Set random seed for reproducibility
    random.seed(args.seed)

    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key")
        return

    # Process input directory
    logger.info(f"Processing directory: {args.input_dir}")
    chunks = process_directory(
        args.input_dir,
        args.max_files,
        args.max_chunk_size
    )

    if not chunks:
        logger.error("No valid document chunks found. Check input directory.")
        return

    # Generate training examples
    logger.info(f"Generating training examples using {args.model}")
    examples = await process_chunks_async(
        chunks,
        api_key,
        args.model,
        args.example_count,
        args.negative_count,
        args.max_concurrent,
        args.temperature,
        args.seed,
        args.error_file
    )

    if not examples:
        logger.error("Failed to generate training examples")
        return

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

    # Split into train and validation sets if needed
    if args.validation_file:
        train_data, val_data = split_train_val(formatted_examples, args.val_ratio)

        # Save training data
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        # Save validation data
        with open(args.validation_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
    else:
        # Save all data as training
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_examples, f, ensure_ascii=False, indent=2)

    logger.info("Training data generation completed")


def main():
    """Entry point function."""
    args = parse_arguments()

    # Choose between streaming and standard approach
    if args.streaming:
        asyncio.run(main_streaming(args))
    else:
        asyncio.run(main_standard(args))


if __name__ == "__main__":
    main()
