"""
Pydantic models for data validation and structure.
"""

from typing import List

from pydantic import BaseModel, Field, model_validator


class NegativeExample(BaseModel):
    """A negative document example that doesn't answer the query."""
    document: str = Field(..., description="The negative document content that doesn't answer the query")
    explanation: str = Field(..., description="Why this document was selected as a negative example")


class TrainingExample(BaseModel):
    """A single training example for SPLADE model training."""
    query: str = Field(..., description="A natural, specific query someone might search for")
    positive_document: str = Field(..., description="The document content that answers the query")
    negative_documents: List[NegativeExample] = Field(
        ...,
        description="List of negative examples that don't answer the query"
    )

    @model_validator(mode='after')
    def check_different_documents(self) -> 'TrainingExample':
        """Validate that positive and negative documents are different."""
        for neg_doc in self.negative_documents:
            if neg_doc.document == self.positive_document:
                raise ValueError("Negative document must be different from positive document")

        # Validate number of negative documents
        if len(self.negative_documents) < 1:
            raise ValueError("At least one negative document is required")
        if len(self.negative_documents) > 5:
            raise ValueError("No more than five negative documents are allowed")

        return self


class TrainingData(BaseModel):
    """Collection of training examples."""
    examples: List[TrainingExample]


class Example(BaseModel):
    """Basic example structure for API responses."""
    query: str = Field(..., description="A natural, specific query someone might search for")
    positive_document: str = Field(..., description="The document content that answers the query")


class ExampleResponse(BaseModel):
    """Model for API response validation."""
    examples: List[Example] = Field(..., description="List of training examples")
