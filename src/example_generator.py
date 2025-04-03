"""
Functions for generating training examples using LLMs.
"""

import asyncio
import json
import logging
import random
from typing import List, Dict, Any

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

from .error_logging import ErrorLogger

# Configure logging
logger = logging.getLogger("example_generator")


async def generate_examples_with_openai(
        chunk: Dict[str, Any],
        other_chunks: List[Dict[str, Any]],
        client: AsyncOpenAI,
        model: str = "gpt-4o",
        example_count: int = 2,
        negative_count: int = 2,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        error_logger: ErrorLogger = None
) -> List[Dict[str, Any]]:
    """
    Generate training examples using OpenAI API with asyncio.

    Args:
        chunk: Document chunk to generate examples from
        other_chunks: Other document chunks for negative examples
        client: AsyncOpenAI client
        model: OpenAI model to use
        example_count: Number of examples to generate per chunk
        negative_count: Number of negative examples per positive example
        temperature: Temperature for generation
        max_tokens: Maximum tokens for response
        retry_count: Number of retries on error
        retry_delay: Delay between retries
        error_logger: Error logger instance for logging errors

    Returns:
        List of generated training examples
    """
    # Extract content and metadata
    content = chunk["content"]
    file_name = chunk["file_name"]

    # Prepare system prompt
    system_prompt = f"""You are an expert at creating training data for information retrieval models. 
Your task is to create {example_count} realistic search queries that someone might use to find specific information in the provided document.

For each query:
1. Create a natural, specific question someone might search for
2. Identify the exact text passage that answers this query

The document content is technical documentation about heating systems, heat pumps and related topics.
"""

    # Prepare user prompt
    user_prompt = f"""Here is a document chunk to create training examples from:

DOCUMENT: {content}

Create EXACTLY {example_count} training examples based on this content. Each should have:
1. A natural search query someone might ask
2. The positive document passage that answers the query

I need EXACTLY {example_count} examples, no more and no less.
Make sure the query is specific enough that it can be answered by the document, but general enough that a user might actually search for it.

Provide your response in JSON format.
"""

    # Define JSON schema for the response
    json_schema = {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "positive_document": {"type": "string"}
                    },
                    "required": ["query", "positive_document"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["examples"],
        "additionalProperties": False
    }

    # Retry loop for API calls
    for attempt in range(retry_count):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": "I'll create the examples and format them as JSON."}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "example_response",
                        "strict": True,
                        "schema": json_schema
                    }

                },
                seed=42 + attempt  # Different seed for each retry
            )

            # Extract and parse response
            response_text = response.choices[0].message.content

            # Parse JSON response
            try:
                response_data = json.loads(response_text)

                examples = response_data["examples"]

                # Check if we got the right number of examples
                if len(examples) != example_count:
                    logger.warning(f"Expected {example_count} examples, but got {len(examples)}. Continuing anyway.")

                # Generate negative examples for each example
                complete_examples = []
                for ex in examples:
                    # Select negative examples for this query
                    negative_docs = await select_negative_examples(
                        ex["query"],
                        ex["positive_document"],
                        other_chunks,
                        client,
                        negative_count,
                        model,
                        temperature=0.4
                    )

                    complete_examples.append({
                        "query": ex["query"],
                        "positive_document": ex["positive_document"],
                        "negative_documents": negative_docs
                    })

                return complete_examples

            except Exception as e:
                error_msg = f"Failed to parse response: {e}"
                logger.warning(f"{error_msg}\nResponse: {response_text}")

                # Log the error if error_logger is provided
                if error_logger:
                    error_logger.log_error(
                        error_type="parsing_error",
                        error_message=str(e),
                        chunk_info=chunk,
                        response_text=response_text,
                        additional_info={
                            "attempt": attempt + 1,
                            "model": model,
                            "example_count": example_count
                        }
                    )

                if attempt < retry_count - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            error_msg = f"API error: {e}"
            logger.warning(error_msg)

            # Log the error if error_logger is provided
            if error_logger:
                error_logger.log_error(
                    error_type="api_error",
                    error_message=str(e),
                    chunk_info=chunk,
                    additional_info={
                        "attempt": attempt + 1,
                        "model": model,
                        "example_count": example_count
                    }
                )

            if attempt < retry_count - 1:
                await asyncio.sleep(retry_delay)
            continue

    # Return empty list if all retries failed
    error_msg = f"Failed to generate examples for chunk {chunk.get('file_name')}"
    logger.error(error_msg)

    # Log the error if error_logger is provided
    if error_logger:
        error_logger.log_error(
            error_type="generation_failure",
            error_message=f"All {retry_count} attempts failed to generate examples",
            chunk_info=chunk,
            additional_info={
                "model": model,
                "example_count": example_count,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )

    return []


async def select_negative_examples(
        query: str,
        positive_document: str,
        candidate_chunks: List[Dict[str, Any]],
        client: AsyncOpenAI,
        count: int = 2,
        model: str = "gpt-4o",
        temperature: float = 0.4,
        max_tokens: int = 1000
) -> List[Dict[str, str]]:
    """
    Select high-quality negative examples that are challenging but don't answer the query.
    
    Args:
        query: The search query
        positive_document: The positive document that answers the query
        candidate_chunks: List of candidate chunks for negative examples
        client: AsyncOpenAI client
        count: Number of negative examples to select
        model: OpenAI model to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens for response
        
    Returns:
        List of negative example dictionaries with document and explanation
    """
    # First, extract keywords from the query to find semantically related chunks
    system_prompt = """You are an expert at information retrieval and relevance judgement.
Your task is to select negative examples that are challenging but don't answer the query.
A good negative example should:
1. Share keywords or topics with the query
2. Look relevant at first glance
3. Actually fail to provide the specific information requested in the query
"""

    # If we have too few candidate chunks, just use them all
    if len(candidate_chunks) <= count:
        return [{"document": chunk["content"],
                 "explanation": "Selected as a negative example because it doesn't answer the query"}
                for chunk in candidate_chunks[:count]]

    # If we have many candidates, use the LLM to select the most challenging negative examples
    if len(candidate_chunks) > 10:
        # Select a subset of candidates to keep the prompt smaller
        candidate_chunks = random.sample(candidate_chunks, 10)

    # Create a numbered list of candidate chunks for the LLM to select from
    candidates_text = ""
    for i, chunk in enumerate(candidate_chunks):
        candidates_text += f"\nCANDIDATE {i + 1}:\n{chunk['content']}\n"

    user_prompt = f"""QUERY: {query}

POSITIVE DOCUMENT (answers the query):
{positive_document}

CANDIDATE NEGATIVE DOCUMENTS:
{candidates_text}

Select EXACTLY {count} best negative examples from the candidates above. 
These should be documents that might be retrieved for the query but actually don't answer it well.
For each selection, provide:
1. The candidate number
2. Why it's a good negative example (explanation)

I need EXACTLY {count} selections, no more and no less.
Return your answer in JSON format with an array of objects, each with "candidate_num", "document", and "explanation" fields.

Provide your response in JSON format.
"""

    # Define JSON schema for the response
    json_schema = {
        "type": "object",
        "properties": {
            "selections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_num": {"type": "integer"},
                        "document": {"type": "string"},
                        "explanation": {"type": "string"}
                    },
                    "required": ["candidate_num", "explanation", "document"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["selections"],
        "additionalProperties": False
    }

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "I'll select the negative examples and format them as JSON."}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "negative_example_response",
                    "strict": True,
                    "schema": json_schema
                }
            }
        )

        response_text = response.choices[0].message.content
        response_data = json.loads(response_text)

        negative_examples = []

        # Check if the response has the expected format
        if "selections" in response_data:
            selections = response_data["selections"]

            # Check if we got the right number of selections
            if len(selections) != count:
                logger.warning(f"Expected {count} negative examples, but got {len(selections)}. Will adjust as needed.")

            for selection in selections[:count]:
                if "candidate_num" in selection:
                    candidate_idx = int(selection["candidate_num"]) - 1
                    if 0 <= candidate_idx < len(candidate_chunks):
                        negative_examples.append({
                            "document": candidate_chunks[candidate_idx]["content"],
                            "explanation": selection.get("explanation", "Selected as a challenging negative example")
                        })

        # If we didn't get enough examples, just use random ones to fill in
        while len(negative_examples) < count and candidate_chunks:
            random_chunk = random.choice(candidate_chunks)
            candidate_chunks.remove(random_chunk)
            negative_examples.append({
                "document": random_chunk["content"],
                "explanation": "Randomly selected as a negative example"
            })

        return negative_examples

    except Exception as e:
        logger.warning(f"Error selecting negative examples: {e}")
        # Fallback: just select random chunks
        return [{"document": chunk["content"],
                 "explanation": "Randomly selected as a negative example"}
                for chunk in random.sample(candidate_chunks, min(count, len(candidate_chunks)))]


async def process_chunks_async(
        chunks: List[Dict[str, Any]],
        api_key: str,
        model: str = "gpt-4o",
        example_count: int = 2,
        negative_count: int = 2,
        max_concurrent: int = 5,
        temperature: float = 0.7,
        seed: int = 42,
        error_file: str = "errors.json"
) -> List[Dict[str, Any]]:
    """
    Process chunks in parallel using asyncio.

    Args:
        chunks: List of document chunks
        api_key: OpenAI API key
        model: OpenAI model to use
        example_count: Number of examples to generate per chunk
        negative_count: Number of negative examples per positive example
        max_concurrent: Maximum number of concurrent tasks
        temperature: Temperature for generation
        seed: Random seed for reproducibility

    Returns:
        List of generated training examples
    """
    # Create AsyncOpenAI client
    client = AsyncOpenAI(api_key=api_key)

    # Set random seed
    random.seed(seed)

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    # Initialize error logger
    error_logger = ErrorLogger(error_file)

    # Setup progress tracking
    all_examples = []
    other_chunks = chunks.copy()

    async def process_chunk(chunk):
        # Use semaphore to limit concurrent requests
        async with semaphore:
            # Get sample of other chunks for negative examples
            # Remove current chunk from candidates
            negative_candidates = [c for c in other_chunks if c != chunk]
            # Sample a subset if there are many chunks
            if len(negative_candidates) > 10:
                negative_candidates = random.sample(negative_candidates, 10)

            # Generate examples
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
            return examples

    # Create tasks for all chunks
    tasks = [process_chunk(chunk) for chunk in chunks]

    # Process chunks with progress tracking
    for task in async_tqdm.as_completed(tasks, desc="Processing chunks", total=len(tasks)):
        examples = await task
        all_examples.extend(examples)

    logger.info(f"Generated {len(all_examples)} examples from {len(chunks)} chunks")
    return all_examples
