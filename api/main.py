from fastapi import FastAPI, HTTPException, Query
from typing import Optional
from .embedding_service import create_ollama_embedding_service, EmbeddingService
from .models.embedding_context import EmbeddingContext
from .models.query_response import QueryResponse
from .responder import Responder

app = FastAPI(
    title="RAG API",
    description="A simple API for document retrieval and corpus management",
    version="1.0.0",
    docs_url="/swagger",            # Serve Swagger UI at /swagger
    redoc_url="/redoc",             # (Optional) serve ReDoc at /redoc
    openapi_url="/openapi.json"
)

try:
    embedding_service: Optional[EmbeddingService] = create_ollama_embedding_service(
        persist_directory="./vector_db",
        ollama_base_url="http://localhost:11434",
        embedding_model="all-minilm:latest"
    )
    responder = Responder(embedding_service)
    print("✅ OLLAMA Embedding Service initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize OLLAMA Embedding Service: {str(e)}")
    print("Make sure OLLAMA is running and all-minilm model is pulled")
    embedding_service = None
    embedding_service = None

@app.get("/")
def read_root():
    """Root endpoint with OLLAMA status"""
    if embedding_service:
        connection_test = embedding_service.test_ollama_connection()
        return {
            "message": "FastAPI RAG service with OLLAMA embeddings is running!",
            "ollama_status": connection_test
        }
    else:
        return {
            "message": "FastAPI service is running",
            "error": "OLLAMA embedding service not available"
        }

@app.post("/corpus")
def add_corpus(embedding_context: EmbeddingContext):
    """Add new corpus content with OLLAMA embeddings"""
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not available")
    
    try:
        result = embedding_service.add_corpus(
            corpus_name=embedding_context.corpus_name,
            content=embedding_context.text,
            metadata={
                "timestamp": "2024-01-01",
                "content_type": "user_uploaded"
            }
        )
        
        if result["status"] == "success":
            return {
                "message": f"Corpus '{embedding_context.corpus_name}' added successfully with OLLAMA embeddings",
                "chunks_created": result["chunks_added"],
                "total_content_length": result["total_content_length"],
                "embedding_model": result["embedding_model"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add corpus: {str(e)}")


@app.get("/query", response_model=str)
def search_query(
    q: str = Query(..., description="Search query string"),
    corpus: str = Query(None, description="Specific corpus to search"),
    k: int = Query(4, description="Number of results to return"),
    min_score: float = Query(None, description="Minimum similarity score")
):
    """Search using OLLAMA vector similarity"""
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not available")
    
    try:
        results = responder.generate_response(
            query=q,
            corpus_name=corpus,
            k=k
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")