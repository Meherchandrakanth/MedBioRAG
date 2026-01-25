import os
from retriever import SemanticRetriever, LexicalRetriever, HybridRetriever, CrossEncoderReRanker
from generator import AnswerGenerator
from typing import List

class MedBioRAGPipeline:
    def __init__(self, documents: List[str] = None, github_token: str = None, storage_folder: str = "storage"):
        # Initialize Retrievers
        self.sem_retriever = SemanticRetriever()
        self.lex_retriever = LexicalRetriever()
        self.reranker = CrossEncoderReRanker()
        self.storage_folder = storage_folder
        
        # Build or Load indices
        if documents:
            print("Building new indices...")
            self.sem_retriever.build_index(documents)
            self.lex_retriever.build_index(documents)
            self.save_to_disk()
        elif os.path.exists(os.path.join(storage_folder, "docs.pkl")):
            print("Loading indices from disk...")
            self.load_from_disk()
        
        self.hybrid_retriever = HybridRetriever(self.sem_retriever, self.lex_retriever, reranker=self.reranker)
        
        # Initialize Generator
        self.generator = AnswerGenerator(api_key=github_token)

    def save_to_disk(self):
        print(f"Saving indices to {self.storage_folder}...")
        self.sem_retriever.save_index(self.storage_folder)
        self.lex_retriever.save_index(self.storage_folder)

    def load_from_disk(self):
        self.sem_retriever.load_index(self.storage_folder)
        self.lex_retriever.load_index(self.storage_folder)

    def add_new_documents(self, new_docs: List[str]):
        print(f"Adding {len(new_docs)} new documents...")
        self.sem_retriever.add_documents(new_docs)
        # BM25 is harder to update incrementally, rebuilding for consistency
        all_docs = self.sem_retriever.documents
        self.lex_retriever.build_index(all_docs)
        self.save_to_disk()

    def run(self, query: str, task_type: str = "long-form") -> str:
        # 1. Retrieve
        print(f"Retrieving for query: {query}")
        retrieved_docs = self.hybrid_retriever.retrieve(query, k=3)
        context = [d['content'] for d in retrieved_docs]
        
        # 2. Generate
        print(f"Generating answer using {task_type} format...")
        answer = self.generator.generate_answer(query, context, task_type=task_type)
        
        return answer

if __name__ == "__main__":
    # Sample Mock Documents
    mock_docs = [
        "Insulin is a hormone produced by the pancreas that regulates glucose levels in the blood.",
        "Type 1 diabetes occurs when the body destroys insulin-producing beta cells in the pancreas.",
        "Metformin is a common medication used to treat Type 2 diabetes by improving insulin sensitivity.",
        "High blood sugar, or hyperglycemia, can lead to long-term health complications if not managed.",
        "The ketogenic diet is a high-fat, low-carb diet that some people use to manage blood sugar."
    ]
    
    # Simple test run (Requires GOOGLE_API_KEY in environment)
    # pipeline = MedBioRAGPipeline(mock_docs)
    # response = pipeline.run("How does insulin work?")
    # print(response)
    print("Pipeline ready. Mock data loaded.")
