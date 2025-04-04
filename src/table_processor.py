"""
Functions for preprocessing markdown tables using LLMs.
"""

import logging
import re
from typing import List, Dict, Any

from openai import AsyncOpenAI

# Configure logging
# logging.basicConfig(level=logging.DEBUG)  # Temporarily set to DEBUG for more detailed logs
logger = logging.getLogger("table_processor")


def detect_tables(content: str) -> List[Dict[str, Any]]:
    """
    Detect markdown tables in content.
    
    Args:
        content: The markdown content to analyze
        
    Returns:
        List of dictionaries with table info (start, end positions and content)
    """
    tables = []

    # Pattern 1: Numbered format like "1| text| 2| text"
    pattern1 = r'(\d+\|[^\n]+\|\s*\d*\|[^\n]*\n[-|:\s]+\n(?:(?:\d*\|[^\n]+\|[^\n]*\n)+|(?:[^\n]+\|\s*\|[^\n]*\n)+))'

    # Pattern 2: Standard format like "|text|text|"
    pattern2 = r'(\|[^\n]+\|\n[-|:\s]+\n(?:\|[^\n]+\|\n)+)'

    # Alternative format like "Symbol| Meaning" (no leading pipe)
    pattern3 = r'([A-Za-z0-9]+\|[^\n]+\n[-|:\s]+\n(?:[A-Za-z0-9\!\[\]]+\|[^\n]+\n)+)'

    # Apply each pattern
    for i, pattern in enumerate([pattern1, pattern2, pattern3]):
        try:
            for match in re.finditer(pattern, content, re.MULTILINE):
                table_content = match.group(0)

                # Basic validation - ensure it has multiple lines and pipe characters
                lines = table_content.strip().split('\n')
                if len(lines) >= 3 and all('|' in line for line in lines):
                    tables.append({
                        'start': match.start(),
                        'end': match.end(),
                        'content': table_content,
                        'pattern_used': i + 1
                    })
                    logger.debug(f"Detected table with pattern {i + 1}: {table_content[:50]}...")
        except re.error as e:
            logger.error(f"Error with pattern {i + 1}: {e}")

    # Remove any duplicate tables (might be detected by multiple patterns)
    unique_tables = []
    positions = set()

    for table in tables:
        pos = (table['start'], table['end'])
        if pos not in positions:
            positions.add(pos)
            unique_tables.append(table)

    logger.info(f"Detected {len(unique_tables)} unique markdown tables in content")

    # Sort tables by their position in the document
    unique_tables.sort(key=lambda x: x['start'])

    return unique_tables


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
    # Detect tables in the content
    tables = detect_tables(content)

    # If no tables found, return the original content
    if not tables:
        return content

    # Create OpenAI client
    client = AsyncOpenAI(api_key=api_key)

    # Process the content in reverse order to preserve positions
    # (from end to beginning so position changes don't affect other matches)
    for table in reversed(tables):
        try:
            # Log some useful info about the table being processed
            logger.info(
                f"Processing table at position {table['start']}-{table['end']} (pattern {table.get('pattern_used', 'unknown')})")
            
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

    logger.info(f"Successfully preprocessed {len(tables)} tables in content")
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
        # Log the table being sent to the LLM (truncated for brevity)
        logger.debug(f"Sending table to LLM: {table[:100]}...")
        
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

        # Log a preview of the result
        logger.debug(f"Received response from LLM: {table_description[:100]}...")

        # Add a newline before and after to maintain paragraph separation
        return f"\n\n{table_description}\n\n"

    except Exception as e:
        logger.error(f"LLM conversion error: {e}")
        raise
