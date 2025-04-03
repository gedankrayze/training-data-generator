# Changes Made to Implement JSON Schema Response Format

## Latest Updates (2025-04-03)

### Fixed JSON Response Format Issues

1. **Added JSON Reference in Messages**

   When using `response_format={"type": "json_object"}`, the API requires that the word "json" appears somewhere in the
   messages. We've fixed this by adding an assistant message:

   ```python
   messages=[
       {"role": "system", "content": system_prompt},
       {"role": "user", "content": user_prompt},
       {"role": "assistant", "content": "I'll create the examples and format them as JSON."}
   ]
   ```

2. **Simplified Schema Implementation**

   We've simplified the JSON schema implementation by using the basic `{"type": "json_object"}` format instead of the
   more complex JSON schema approach that was causing errors.

3. **Added Runtime Validation**

   Instead of relying on schema validation, we now check the counts post-response to ensure we received the expected
   number of examples:

   ```python
   if len(examples) != example_count:
       logger.warning(f"Expected {example_count} examples, but got {len(examples)}.")
   ```

4. **Enhanced User Prompts**

   The prompts now explicitly request EXACTLY the specified number of examples to guide the model's output format.

### Previous Fixes

## Initial Changes

We have made significant improvements to the training data generator by implementing the JSON schema response format for
all API calls. These changes should solve the validation errors shown in the error logs.

## Main Changes

1. **JSON Schema Response Format**

   Instead of using Pydantic for validation after receiving responses, we now enforce the structure directly in the API
   call using JSON schema. This ensures the LLM returns data in exactly the format we expect.

   ```python
   response_format={
       "type": "json_schema",
       "json_schema": {
           "name": "training_examples",
           "schema": json_schema,
           "strict": True
       }
   }
   ```

2. **Strict Schema Definition**

   We've defined strict schemas for both example generation and negative example selection:

    - Required `query` and `positive_document` fields
    - No additional properties allowed
    - Minimum and maximum item counts enforced
    - Type constraints

3. **Model Update**

   Changed default model from `gpt-3.5-turbo` to `gpt-4o` for better response quality and improved schema adherence.

4. **Response Handling Simplified**

    - Removed Pydantic validation step since schema validation happens at the API level
    - Changed response handling to directly use the structured JSON data
    - Eliminated field name mismatches that were causing most errors

5. **Error Logging Enhancement**

    - Added comprehensive error logging system
    - Captures all API errors, parsing issues, and validation problems
    - Creates a structured JSON file of errors for analysis

## Benefits

1. **Reduced Format Errors**

   The JSON schema approach eliminates the most common errors observed in the logs where the model would use
   inconsistent field names like `passage`, `answer`, or `document_passage` instead of the expected `positive_document`.

2. **Improved Success Rate**

   By enforcing the structure at the API level, the success rate of example generation should significantly increase,
   resulting in more efficient data generation.

3. **Better Error Diagnostics**

   The error logging system provides detailed information about failures, making it easier to troubleshoot and improve
   the system.

4. **More Consistent Data Format**

   The generated data will have a consistent format, making downstream processing more reliable.

## Implementation Details

1. **Example Generator**

   Updated the `generate_examples_with_openai()` function to:
    - Define a strict JSON schema
    - Use the JSON schema response format
    - Process response data according to the enforced structure

2. **Negative Example Selection**

   Updated the `select_negative_examples()` function to:
    - Use a JSON schema for selections
    - Enforce a consistent structure for negative examples

3. **Default Model**

   Changed all default model parameters from `gpt-3.5-turbo` to `gpt-4o` in:
    - `generate_examples_with_openai()`
    - `select_negative_examples()`
    - `process_chunks_async()`
    - `process_chunks_with_recovery()`
    - `process_directory_streaming()`
    - Command-line arguments

4. **Documentation**

   Updated README.md and other documentation to reflect the model change and error logging options.

## Testing Recommendations

To ensure the changes are effective, we recommend testing with various inputs and monitoring the error logs. The error
rate should decrease significantly, especially the validation errors related to field names.

## Future Improvements

1. **Enhanced Negative Example Selection**

   Consider adding more sophisticated algorithms for hard negative selection.

2. **Adaptive Temperature**

   Implement adaptive temperature based on failure rates to optimize for diversity vs. accuracy.

3. **Schema Evolution**

   Add support for schema evolution to handle different data formats as requirements change.
