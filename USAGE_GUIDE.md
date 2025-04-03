# Training Data Generator - Usage Guide

This guide will help you use the training data generator effectively to create high-quality training data for SPLADE
model fine-tuning.

## Project Structure

The code has been structured into the following modules:

```
training-data-generator/
├── generate.py              # Main entry point script
├── src/                     # Source code
│   ├── __init__.py          # Package definition
│   ├── main.py              # Main orchestration logic
│   ├── models.py            # Pydantic data models
│   ├── document_processor.py # Document loading and chunking
│   ├── checkpoint.py        # Checkpoint management for recovery
│   ├── example_generator.py # LLM-based example generation
│   └── streaming.py         # Memory-efficient streaming processing
├── tests/                   # Test modules
│   ├── __init__.py
│   └── test_document_processor.py
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## Basic Usage

The simplest way to use the tool is through the `generate.py` script:

```bash
python generate.py --input-dir /path/to/documents --output-file training_data.json
```

This will:

1. Process all supported documents (.md, .txt, .facts) in the input directory
2. Generate examples using the OpenAI API
3. Save the resulting training data to the specified output file

## Advanced Usage

### Memory-Efficient Processing for Large Document Collections

For very large document collections, use the streaming mode:

```bash
python generate.py --input-dir /path/to/large_corpus --output-file training_data.json --streaming
```

Streaming mode:

- Processes files in batches rather than all at once
- Incrementally writes to the output file
- Can resume from where it left off if interrupted
- Uses much less memory

### Customizing Example Generation

You can control various aspects of example generation:

```bash
python generate.py --input-dir ./docs --output-file train.json \
    --example-count 3 \        # Generate 3 examples per chunk
    --negative-count 4 \       # Include 4 negative examples per positive
    --max-concurrent 10 \      # Process 10 chunks concurrently
    --model gpt-4 \            # Use GPT-4 instead of GPT-3.5-turbo
    --temperature 0.4          # Lower temperature for more focused output
```

### Creating Validation Sets

To create separate training and validation sets:

```bash
python generate.py --input-dir ./docs --output-file train.json \
    --validation-file val.json --val-ratio 0.2
```

This creates a validation set with 20% of the examples.

## Extending the Code

### Adding New Document Types

To support additional document types, modify the `load_document` function in `document_processor.py` to handle the new
file extensions and content formats.

### Using a Different LLM Provider

The code currently uses the OpenAI API. To use a different LLM provider:

1. Create a new function in `example_generator.py` similar to `generate_examples_with_openai`
2. Update the `process_chunks_async` and `process_chunks_with_recovery` functions to use your new function
3. Update the argument parser in `main.py` to accept parameters for your LLM provider

### Customizing Chunking

The default chunking algorithm splits documents by paragraphs. To use a different chunking strategy:

1. Create a new function in `document_processor.py`
2. Update the relevant functions in `main.py` and `streaming.py` to use your new chunking function

## Running Tests

To run the tests:

```bash
python -m unittest discover tests
```

## Troubleshooting

### API Key Issues

If you get authentication errors, make sure your OpenAI API key is:

- Set in the OPENAI_API_KEY environment variable, or
- Passed via the --api-key parameter

### Resource Limitations

If you encounter memory issues:

- Use the --streaming mode for large datasets
- Reduce --batch-size to process fewer files at once
- Reduce --max-concurrent to limit parallel API calls

### Error Logging and Analysis

The tool logs detailed error information to help debug issues:

```bash
python generate.py --input-dir ./docs --output-file train.json --error-file errors-detailed.json
```

The error log file contains:

- Timestamp of when the error occurred
- Error type (API error, parsing error, validation error, etc.)
- Error message
- Snippet of the content being processed
- Raw API response for debugging

You can analyze the error log to:

- Identify patterns in failures
- Refine your documents for better processing
- Debug API issues
- Fix prompt engineering problems

Example of analyzing error logs:

```python
import json

# Load error log
with open("errors.json", "r") as f:
    errors = json.load(f)

# Count errors by type
error_counts = {}
for error in errors:
    error_type = error["error_type"]
    error_counts[error_type] = error_counts.get(error_type, 0) + 1

print("Error counts by type:", error_counts)
```

### Checkpoint Recovery

If processing is interrupted, simply run the same command again. The tool will load the latest checkpoint and continue
from where it left off.

## Examples

### Process Technical Documentation

```bash
python generate.py --input-dir ./technical_docs --output-file technical_train.json \
    --streaming --example-count 3 --batch-size 20 --max-concurrent 8
```

### Process Educational Content

```bash
python generate.py --input-dir ./courses --output-file education_train.json \
    --model gpt-4 --temperature 0.5 --negative-count 3
```

### Process Multiple Directories with Different Settings

You can run the tool multiple times with different settings for different types of content:

```bash
# Technical content - more examples, lower temperature
python generate.py --input-dir ./technical --output-file technical.json \
    --example-count 4 --temperature 0.3

# Creative content - fewer examples, higher temperature
python generate.py --input-dir ./creative --output-file creative.json \
    --example-count 2 --temperature 0.8
```
