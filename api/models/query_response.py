from pydantic import BaseModel, Field

class QueryResponse(BaseModel):
    """
    Represents the response from a search query, including the matched text and its similarity score.
    """
    text: str = Field(..., description="The matched text from the corpus.")
    score: float = Field(..., description="The similarity score of the matched text.")