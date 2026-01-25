import numpy as np
import pickle
import os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any
import faiss

class CrossEncoderReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        
        results = []
        for i, score in enumerate(scores):
            results.append({
                "content": documents[i],
                "score": float(score)
            })
            
        # Sort by Cross-Encoder score
        return sorted(results, key=lambda x: x["score"], reverse=True)

class SemanticRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, documents: List[str]):
        self.documents = documents
        embeddings = self.model.encode(documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def add_documents(self, new_docs: List[str]):
        embeddings = self.model.encode(new_docs)
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(new_docs)

    def save_index(self, folder: str):
        if not os.path.exists(folder):
            os.makedirs(folder)
        faiss.write_index(self.index, os.path.join(folder, "semantic.index"))
        with open(os.path.join(folder, "docs.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load_index(self, folder: str):
        self.index = faiss.read_index(os.path.join(folder, "semantic.index"))
        with open(os.path.join(folder, "docs.pkl"), "rb") as f:
            self.documents = pickle.load(f)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None: return []
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "content": self.documents[idx],
                    "score": float(1 / (1 + distances[0][i])),
                    "type": "semantic"
                })
        return results

class LexicalRetriever:
    def __init__(self):
        self.bm25 = None
        self.documents = []

    def build_index(self, documents: List[str]):
        self.documents = documents
        tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save_index(self, folder: str):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, "lexical.pkl"), "wb") as f:
            pickle.dump((self.documents, self.bm25), f)

    def load_index(self, folder: str):
        with open(os.path.join(folder, "lexical.pkl"), "rb") as f:
            self.documents, self.bm25 = pickle.load(f)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_n:
            results.append({
                "content": self.documents[idx],
                "score": float(scores[idx]),
                "type": "lexical"
            })
        return results

class HybridRetriever:
    def __init__(self, semantic_retriever: SemanticRetriever, lexical_retriever: LexicalRetriever, reranker: CrossEncoderReRanker = None):
        self.semantic = semantic_retriever
        self.lexical = lexical_retriever
        self.reranker = reranker

    def retrieve(self, query: str, k: int = 5, use_reranker: bool = True) -> List[Dict[str, Any]]:
        # Initial candidates (retrieve more than needed for re-ranking)
        fetch_k = k * 4 if self.reranker and use_reranker else k
        
        semantic_results = self.semantic.retrieve(query, k=fetch_k)
        lexical_results = self.lexical.retrieve(query, k=fetch_k)
        
        # Combine and deduplicate
        combined_docs = list(set([r['content'] for r in semantic_results] + [r['content'] for r in lexical_results]))
        
        if self.reranker and use_reranker:
            # Stage 2: Re-ranking
            reranked = self.reranker.rerank(query, combined_docs)
            return reranked[:k]
        else:
            # Fallback to simple fusion if no reranker
            combined = {}
            max_sem = max([r['score'] for r in semantic_results]) if semantic_results else 1
            max_lex = max([r['score'] for r in lexical_results]) if lexical_results else 1
            
            for r in semantic_results:
                combined[r['content']] = combined.get(r['content'], 0) + (r['score'] / max_sem)
            for r in lexical_results:
                combined[r['content']] = combined.get(r['content'], 0) + (r['score'] / max_lex)
                
            sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            return [{"content": content, "score": score} for content, score in sorted_results[:k]]
