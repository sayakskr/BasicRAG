from pydantic import BaseModel, Field

class EmbeddingContext(BaseModel):
    """
    Represents the context for an embedding operation, including the text to be embedded
    and any additional metadata.
    """
    text: str = Field(..., description="The text to be embedded.")
    corpus_name: str = Field(..., description="The name of the corpus the text belongs to.")
    metadata: dict = Field(default_factory=dict, description="Additional metadata for the embedding.")