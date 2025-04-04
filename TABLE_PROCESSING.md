# Table Preprocessing for Markdown Documents

This feature adds support for preprocessing markdown tables before generating training examples. The preprocessing
converts markdown tables into a more natural language format that is better suited for LLM processing.

## How It Works

1. When enabled, the table preprocessing step detects markdown tables in the documents.
2. Each table is sent to the LLM (by default GPT-3.5-Turbo), which converts it to a natural language description.
3. The original table in the document is replaced with this natural language description.
4. The modified document is then processed normally through chunking and example generation.

## Benefits

- **Better Handling of Tabular Data**: Converts complex tables into a format that is more natural for LLMs to
  understand.
- **Prevents Table Splitting**: By converting tables to text paragraphs, it reduces the risk of tables being split
  across chunks.
- **Improved Query Generation**: The narrative format allows the LLM to generate more natural queries about information
  that was in tables.
- **Better Context Integration**: The flattened tables integrate more smoothly with surrounding text.

## Usage

Enable table preprocessing by adding the `--preprocess-tables` flag to your command:

```bash
python generate.py --input-dir ./documents --output-file train.json --preprocess-tables
```

### Additional Options

- `--table-model`: Specify the model to use for table preprocessing (default: gpt-4o-mini)

```bash
python generate.py --input-dir ./documents --output-file train.json --preprocess-tables --table-model gpt-4o-mini
```

## Example Transformation

Original markdown table:

```
Symbol| Signalwort| Erläuterung  
---|---|---  
⚠️| GEFAHR| Bedeutet schwere bis lebensgefährliche Personenschäden  
⚠️| WARNUNG| Bedeutet mögliche schwere Personenschäden
```

Transformed to narrative text:

```
In der Dokumentation werden verschiedene Signalwörter mit entsprechenden Symbolen verwendet. 
Das Warnsymbol ⚠️ kombiniert mit dem Signalwort "GEFAHR" bedeutet, dass schwere bis 
lebensgefährliche Personenschäden auftreten werden. Dasselbe Warnsymbol ⚠️ mit dem 
Signalwort "WARNUNG" bedeutet, dass schwere bis lebensgefährliche Personenschäden 
auftreten können.
```

## Implementation Details

The table preprocessing is implemented in the following files:

- `src/table_processor.py`: Contains functions for detecting tables and converting them to text.
- `src/document_processor.py`: Modified to support table preprocessing during document loading.
- `src/main.py`: Added command-line arguments for enabling table preprocessing.
- `src/streaming.py`: Updated to support table preprocessing in streaming mode.

## Notes and Limitations

- Table preprocessing adds additional API calls, which may increase processing time and costs.
- The quality of the transformation depends on the LLM model used.
- For very large documents with many tables, consider using a less expensive model like gpt-3.5-turbo.
- Not all markdown tables may be detected correctly, especially if they have non-standard formatting.
- The preprocessing is optimized for tables that contain information that benefits from narrative description rather
  than purely numerical data tables.
