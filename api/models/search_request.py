from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    """
    Represents a search request containing the query text and the corpus name to search within.
    """
    query: str = Field(..., description="The search query text.")
    corpus_name: str = Field(..., description="The name of the corpus to search within.")