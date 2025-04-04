# Training Data Generator for SPLADE Model Fine-tuning

This tool uses LLMs (like OpenAI's GPT models) to generate high-quality training data for SPLADE model fine-tuning. It
processes documents, chunks them, and generates query-document pairs that can be used for training.

## Features

- Processes documents from a directory and its subdirectories
- Chunks documents with controlled overlap to maintain context
- Generates natural search queries for document content
- Creates challenging negative examples for better training
- Memory-efficient streaming mode for large document collections
- Robust error recovery with checkpointing
- Detailed error logging for debugging and analysis
- Asyncio-based processing for high throughput
- Table preprocessing for better handling of markdown tables

## Installation

```bash
# Clone the repository
git clone https://github.com/gedankrayze/training-data-generator.git
cd training-data-generator

# Install requirements
pip install -r requirements.txt
```

## Usage

```bash
python generate.py --input-dir /path/to/documents --output-file training_data.json
```

### Command-line Arguments

```
--input-dir, -i       Directory containing document files (required)
--output-file, -o     Output file for training data in JSON format (required)
--validation-file     Output file for validation data (optional)
--val-ratio           Ratio of validation examples (default: 0.1)
--max-files           Maximum number of files to process (default: all)
--max-chunk-size      Maximum chunk size in characters (default: 2000)
--overlap-size        Number of characters to overlap between chunks (default: 200)
--example-count       Number of examples to generate per chunk (default: 2)
--negative-count      Number of negative examples per positive example (default: 2)
--batch-size          Number of files to process in each batch (default: 10)
--max-concurrent      Maximum number of concurrent API calls (default: 5)
--model               OpenAI model to use (default: gpt-4o)
--temperature         Temperature for generation (default: 0.7)
--api-key             OpenAI API key (default: from OPENAI_API_KEY environment variable)
--seed                Random seed for reproducibility (default: 42)
--streaming           Use memory-efficient streaming processing (recommended for large datasets)
--checkpoint-dir      Directory for checkpoint files (default: ./checkpoints)
--error-file          File to store error logs (default: errors.json)
--preprocess-tables   Preprocess markdown tables to convert them to descriptive text
--table-model         OpenAI model to use for table preprocessing (default: gpt-4o-mini)
```

## Examples

### Generate training data with default settings

```bash
python generate.py --input-dir ./documents --output-file training_data.json
```

### Generate training and validation data

```bash
python generate.py --input-dir ./documents --output-file train.json --validation-file val.json --val-ratio 0.2
```

### Process large document collection with streaming mode

```bash
python generate.py --input-dir ./large_corpus --output-file training_data.json --streaming --batch-size 20
```

### Use a different OpenAI model with custom parameters

```bash
python generate.py --input-dir ./documents --output-file training_data.json --model gpt-4o-mini --temperature 0.5 --example-count 3 --negative-count 3
```

### Process markdown files with table preprocessing

```bash
python generate.py --input-dir ./technical_docs --output-file training_data.json --preprocess-tables --table-model gpt-4o-mini
```

## Output Format

The generated data is saved in JSON format with the following structure:

```json
[
  {
    "query": "A natural search query",
    "positive_document": "The document content that answers the query",
    "negative_documents": [
      "Content that doesn't answer the query but looks relevant",
      "Another document that doesn't answer the query"
    ],
    "explanations": [
      "Why this document was selected as a negative example",
      "Why this document was selected as a negative example"
    ]
  },
  ...
]
```

## License

[MIT License](LICENSE)
