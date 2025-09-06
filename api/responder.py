from .embedding_service import EmbeddingService
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class Responder:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.model_name = "llama3.2:3b"
        self.ollama_base_url = "http://localhost:11434"
        self.llm = self._get_llm()

    def _get_llm(self):
        llm = Ollama(
            model=self.model_name,
            base_url=self.ollama_base_url,
            temperature=0.0
        )
        return llm

    def generate_response(
        self,
        query: str,
        corpus_name: str,
        k: int = 5
    ) -> str:
        """
        Generate a response based on the query by retrieving relevant documents from the embedding service.

        Args:
            query (str): The search query.
            collection_name (str): The name of the collection to search in.
            k (int): The number of top results to retrieve.
            score_threshold (Optional[float]): Minimum similarity score threshold for filtering results.

        Returns:
            List[QueryResponse]: A list of QueryResponse objects containing matched text and scores.
        """
        # Define a custom prompt template for RAG
        PROMPT_TEMPLATE = PromptTemplate(
            input_variables=["question", "context"],
            template=(
                "You are a knowledge assistant. Use the following context to answer the question.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            ),
        )

        rag_chain = self._build_rag_chain(corpus_name, PROMPT_TEMPLATE, k)
        results = rag_chain({"query": query})
        answer = results.get("result", "I'm sorry, I couldn't find an answer.")
        return answer
        

    # Build the RetrievalQA chain
    def _build_rag_chain(self, corpus_name: str, prompt_template: PromptTemplate, k:int) -> RetrievalQA:
        vector_store = self.embedding_service._get_or_create_vector_store(corpus_name)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": k}
        )
        chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=retriever,
            prompt=prompt_template,
            return_source_documents=True,
        )
        return chain