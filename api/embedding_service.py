import os
import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import uuid

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Import OLLAMA embedding model
from langchain_ollama import OllamaEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    A comprehensive embedding service that uses LangChain with ChromaDB 
    and OLLAMA's all-minilm model for storing and retrieving text content 
    with vector embeddings.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "all-minilm:latest",
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        collection_prefix: str = "corpus"
    ):
        """
        Initialize the embedding service with OLLAMA all-minilm model.
        
        Args:
            persist_directory: Directory to store ChromaDB data
            ollama_base_url: Base URL for OLLAMA service
            embedding_model: OLLAMA embedding model name
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            collection_prefix: Prefix for collection names
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_prefix = collection_prefix
        self.ollama_base_url = ollama_base_url
        self.embedding_model_name = embedding_model
        
        # Initialize OLLAMA embedding model
        try:
            self.embedding_model = OllamaEmbeddings(
                model=self.embedding_model_name,
                base_url=self.ollama_base_url
            )
            logger.info(f"Initialized OLLAMA embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OLLAMA embedding model: {str(e)}")
            raise ValueError(
                f"Could not connect to OLLAMA at {ollama_base_url}. "
                f"Make sure OLLAMA is running and the model '{embedding_model}' is pulled."
            )
            
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Store vector stores for different collections
        self._vector_stores: Dict[str, Chroma] = {}
        
        logger.info(f"EmbeddingService initialized with OLLAMA model: {embedding_model}")
        logger.info(f"Persist directory: {self.persist_directory}")
        logger.info(f"OLLAMA base URL: {ollama_base_url}")
    
    def _get_collection_name(self, corpus_name: str) -> str:
        """Generate a valid collection name."""
        # ChromaDB collection names must be 3-63 characters, alphanumeric and hyphens only
        safe_name = f"{self.collection_prefix}_{corpus_name}".lower()
        safe_name = "".join(c if c.isalnum() or c == "-" else "-" for c in safe_name)
        safe_name = safe_name[:63]  # Limit to 63 characters
        return safe_name
    
    def _get_or_create_vector_store(self, corpus_name: str) -> Chroma:
        """Get or create a vector store for the given corpus."""
        collection_name = self._get_collection_name(corpus_name)
        
        if collection_name not in self._vector_stores:
            try:
                # Try to load existing collection
                self._vector_stores[collection_name] = Chroma(
                    client=self.chroma_client,
                    collection_name=collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=str(self.persist_directory),
                    collection_metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Loaded existing collection: {collection_name}")
            except Exception as e:
                # Create new collection if loading fails
                self._vector_stores[collection_name] = Chroma(
                    client=self.chroma_client,
                    collection_name=collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=str(self.persist_directory),
                    collection_metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {collection_name}")
        
        return self._vector_stores[collection_name]
    
    def test_ollama_connection(self) -> Dict[str, Any]:
        """Test the connection to OLLAMA and the embedding model."""
        try:
            # Test embedding generation
            test_text = "This is a test sentence for OLLAMA embedding generation."
            embeddings = self.embedding_model.embed_query(test_text)
            
            return {
                "status": "success",
                "model": self.embedding_model_name,
                "base_url": self.ollama_base_url,
                "embedding_dimensions": len(embeddings),
                "test_embedding_preview": embeddings[:5]  # Show first 5 dimensions
            }
        except Exception as e:
            return {
                "status": "error",
                "model": self.embedding_model_name,
                "base_url": self.ollama_base_url,
                "error": str(e)
            }
    
    def add_corpus(
        self, 
        corpus_name: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add text content to a corpus collection using OLLAMA embeddings.
        
        Args:
            corpus_name: Name of the corpus/collection
            content: Text content to be embedded and stored
            metadata: Optional metadata to associate with the content
            
        Returns:
            Dictionary with operation details
        """
        try:
            logger.info(f"Adding corpus '{corpus_name}' with {len(content)} characters")
            
            # Split the content into chunks
            documents = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[metadata or {"corpus_name": corpus_name}]
            )
            
            # Enhance metadata with additional information
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "corpus_name": corpus_name,
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "total_chunks": len(documents),
                    "content_length": len(doc.page_content),
                    "embedding_model": self.embedding_model_name
                })
            
            # Get or create vector store for this corpus
            vector_store = self._get_or_create_vector_store(corpus_name)
            
            # Add documents to the vector store (OLLAMA will generate embeddings)
            logger.info(f"Generating embeddings for {len(documents)} chunks using {self.embedding_model_name}")
            ids = vector_store.add_documents(documents)
            
            logger.info(f"Successfully added {len(documents)} chunks to corpus '{corpus_name}'")
            
            return {
                "status": "success",
                "corpus_name": corpus_name,
                "chunks_added": len(documents),
                "document_ids": ids,
                "total_content_length": len(content),
                "embedding_model": self.embedding_model_name
            }
            
        except Exception as e:
            logger.error(f"Error adding corpus '{corpus_name}': {str(e)}")
            return {
                "status": "error",
                "corpus_name": corpus_name,
                "error": str(e),
                "embedding_model": self.embedding_model_name
            }
    
    def search_corpus(
        self, 
        query: str, 
        corpus_name: Optional[str] = None,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar content using OLLAMA embeddings.
        
        Args:
            query: Search query
            corpus_name: Specific corpus to search (None for all)
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with content and metadata
        """
        try:
            logger.info(f"Searching with query: '{query[:50]}...' using {self.embedding_model_name}")
            results = []
            
            if corpus_name:
                collections_to_search = [corpus_name]
            else:
                collections_to_search = self.list_collections()
            
            for corpus in collections_to_search:
                try:
                    vector_store = self._get_or_create_vector_store(corpus)
                    
                    if score_threshold is not None:
                        # Use similarity search with score threshold
                        docs_with_scores = vector_store.similarity_search_with_score(
                            query, k=k
                        )
                        corpus_results = [
                            {
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                                "score": score,
                                "corpus_name": corpus,
                                "embedding_model": self.embedding_model_name
                            }
                            for doc, score in docs_with_scores
                            if score >= score_threshold
                        ]
                    else:
                        # Standard similarity search
                        docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
                        corpus_results = [
                            {
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                                "score": score,
                                "corpus_name": corpus,
                                "embedding_model": self.embedding_model_name
                            }
                            for doc, score in docs_with_scores
                        ]
                    
                    results.extend(corpus_results)
                    
                except Exception as e:
                    logger.warning(f"Error searching corpus '{corpus}': {str(e)}")
                    continue
            
            # Sort by score if available
            if score_threshold is not None:
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            logger.info(f"Search returned {len(results)} results")
            return results[:k]  # Limit to k results
            
        except Exception as e:
            logger.error(f"Error searching with query '{query}': {str(e)}")
            return []
    
    def get_corpus_info(self, corpus_name: str) -> Dict[str, Any]:
        """Get information about a specific corpus."""
        try:
            collection_name = self._get_collection_name(corpus_name)
            collection = self.chroma_client.get_collection(collection_name)
            
            count = collection.count()
            
            return {
                "corpus_name": corpus_name,
                "collection_name": collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model_name,
                "ollama_base_url": self.ollama_base_url,
                "status": "exists" if count > 0 else "empty"
            }
        except Exception as e:
            return {
                "corpus_name": corpus_name,
                "status": "not_found",
                "error": str(e),
                "embedding_model": self.embedding_model_name
            }
    
    def list_collections(self) -> List[str]:
        """List all available corpus collections."""
        try:
            collections = self.chroma_client.list_collections()
            corpus_names = []
            
            for collection in collections:
                name = collection.name
                if name.startswith(f"{self.collection_prefix}_"):
                    corpus_name = name[len(f"{self.collection_prefix}_"):]
                    corpus_names.append(corpus_name)
            
            return corpus_names
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []
    
    def delete_corpus(self, corpus_name: str) -> Dict[str, Any]:
        """Delete a corpus collection."""
        try:
            collection_name = self._get_collection_name(corpus_name)
            
            # Remove from cache
            if collection_name in self._vector_stores:
                del self._vector_stores[collection_name]
            
            # Delete from ChromaDB
            self.chroma_client.delete_collection(collection_name)
            
            logger.info(f"Deleted corpus '{corpus_name}'")
            return {
                "status": "success",
                "corpus_name": corpus_name,
                "message": "Corpus deleted successfully",
                "embedding_model": self.embedding_model_name
            }
        except Exception as e:
            logger.error(f"Error deleting corpus '{corpus_name}': {str(e)}")
            return {
                "status": "error",
                "corpus_name": corpus_name,
                "error": str(e)
            }
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the current embedding setup."""
        return {
            "embedding_model": self.embedding_model_name,
            "ollama_base_url": self.ollama_base_url,
            "model_dimensions": 384,  # all-minilm produces 384-dimensional embeddings
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "persist_directory": str(self.persist_directory)
        }


# Factory function for OLLAMA embedding service
def create_ollama_embedding_service(
    persist_directory: str = "./chroma_db",
    ollama_base_url: str = "http://localhost:11434",
    embedding_model: str = "all-minilm:latest",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> EmbeddingService:
    """
    Create an embedding service using OLLAMA all-minilm model.
    
    Args:
        persist_directory: Directory for ChromaDB persistence
        ollama_base_url: OLLAMA service URL
        embedding_model: OLLAMA embedding model name
        chunk_size: Text chunk size for splitting
        chunk_overlap: Overlap between chunks
    
    Returns:
        Configured EmbeddingService instance
    """
    return EmbeddingService(
        persist_directory=persist_directory,
        ollama_base_url=ollama_base_url,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

# Example usage and testing
if __name__ == "__main__":
    # Create OLLAMA embedding service
    service = create_ollama_embedding_service()
    
    # Test OLLAMA connection
    connection_test = service.test_ollama_connection()
    print("OLLAMA Connection Test:", connection_test)
    
    if connection_test["status"] == "success":
        # Add sample content
        result = service.add_corpus(
            corpus_name="sample_docs",
            content="""
            This is a sample document about machine learning and artificial intelligence.
            Machine learning is a subset of artificial intelligence that focuses on algorithms
            and statistical models that computer systems use to effectively perform tasks
            without explicit instructions. These systems rely on patterns and inference instead.
            Deep learning is a part of machine learning based on artificial neural networks.
            """,
            metadata={
                "topic": "machine_learning", 
                "author": "example",
                "source": "test_document"
            }
        )
        print("Add result:", result)
        
        # Search the corpus
        search_results = service.search_corpus(
            query="What is machine learning and how does it work?",
            corpus_name="sample_docs",
            k=2
        )
        print("Search results:", search_results)
        
        # Get embedding info
        embedding_info = service.get_embedding_info()
        print("Embedding info:", embedding_info)
    else:
        print("Failed to connect to OLLAMA. Please check:")
        print("1. OLLAMA is running: ollama serve")
        print("2. all-minilm model is pulled: ollama pull all-minilm:latest")
