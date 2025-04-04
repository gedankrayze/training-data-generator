"""
Functions for preprocessing markdown tables using LLMs.
"""

import logging
import re
from typing import List, Dict, Any

from openai import AsyncOpenAI

# Configure logging
logger = logging.getLogger("table_processor")


def detect_tables(content: str) -> List[Dict[str, Any]]:
    """
    Detect markdown tables in content.
    
    Args:
        content: The markdown content to analyze
        
    Returns:
        List of dictionaries with table info (start, end positions and content)
    """
    # This regex pattern matches markdown tables:
    # - First line with pipes and text
    # - Second line with pipes and dashes/colons for alignment
    # - Subsequent lines with pipes and text
    table_pattern = r'(\|[^\n]+\|\n\s*\|[\s*:?-]+\|\s*\n(?:\s*\|[^\n]+\|\s*\n)+)'

    tables = []
    for match in re.finditer(table_pattern, content, re.MULTILINE):
        tables.append({
            'start': match.start(),
            'end': match.end(),
            'content': match.group(0)
        })

    logger.info(f"Detected {len(tables)} markdown tables in content")
    return tables


async def preprocess_tables(content: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """
    Preprocess markdown tables in content, converting them to descriptive text.
    
    Args:
        content: The markdown content to process
        api_key: OpenAI API key
        model: The LLM model to use
        
    Returns:
        Processed content with tables converted to descriptive text
    """
    # If no tables found, return the original content
    tables = detect_tables(content)
    if not tables:
        return content

    # Create OpenAI client
    client = AsyncOpenAI(api_key=api_key)

    # Process the content in reverse order to preserve positions
    for table in reversed(tables):
        try:
            # Convert table to descriptive text
            flattened_table = await convert_table_with_llm(
                table['content'],
                client,
                model
            )

            # Replace the table with its flattened version
            content = (
                    content[:table['start']] +
                    flattened_table +
                    content[table['end']:]
            )

        except Exception as e:
            logger.error(f"Error processing table: {e}")
            # Continue with the original table if there's an error

    logger.info(f"Preprocessed {len(tables)} tables in content")
    return content


async def convert_table_with_llm(
        table: str,
        client: AsyncOpenAI,
        model: str = "gpt-4o-mini"
) -> str:
    """
    Convert a markdown table to descriptive text using LLM.
    
    Args:
        table: The markdown table to convert
        client: AsyncOpenAI client
        model: The model to use
        
    Returns:
        Descriptive text representation of the table
    """
    # Prepare the prompt for the LLM
    system_prompt = """You are an expert at converting markdown tables to natural language descriptions.
Your task is to convert a markdown table into a descriptive paragraph format that:
1. Preserves all information and relationships in the table
2. Maintains the semantic structure of the data
3. Creates a natural, easy-to-read format
4. Ensures no information is lost in the conversion
5. Uses clear language to describe headers, rows, and relationships
6. Returns ONLY the converted text without explaining what you did
"""

    user_prompt = f"""
Convert this markdown table to descriptive text that preserves all information and relationships:

{table}

Use a natural paragraph format that clearly explains the table structure and content.
"""

    try:
        # Call the LLM API
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more deterministic results
            max_tokens=1000,  # Adjust based on expected table size
        )

        # Get the generated description
        table_description = response.choices[0].message.content.strip()

        # Add a newline before and after to maintain paragraph separation
        return f"\n\n{table_description}\n\n"

    except Exception as e:
        logger.error(f"LLM conversion error: {e}")
        raise
