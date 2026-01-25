import time
from typing import List, Dict, Any
from retriever import SemanticRetriever, LexicalRetriever, HybridRetriever, CrossEncoderReRanker
from generator import AnswerGenerator
from metrics import EvaluationMetrics
import pandas as pd

class MedBioEvaluator:
    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.metrics = EvaluationMetrics()

    def evaluate_retrieval(self, dataset: List[Dict[str, Any]], retrievers: Dict[str, Any], k: int = 10):
        """
        Evaluates different retrievers on a given dataset (e.g., NFCorpus).
        dataset format: [{'query': str, 'relevant_docs': List[str], 'all_docs': List[str]}]
        """
        results = []
        for name, retriever in retrievers.items():
            print(f"Evaluating retriever: {name}")
            scores = {"ndcg": [], "mrr": [], "recall": [], "precision": []}
            
            for item in dataset:
                query = item['query']
                relevant_docs = item['relevant_docs']
                all_docs = item['all_docs']
                
                # Retrieve
                retrieved = retriever.retrieve(query, k=k)
                retrieved_contents = [r['content'] for r in retrieved]
                
                # Prep metrics data
                y_true = np.zeros(len(all_docs))
                for doc in relevant_docs:
                    if doc in all_docs:
                        y_true[all_docs.index(doc)] = 1
                
                # For simplicity in this demo, match by exact content
                y_score = np.zeros(len(all_docs))
                retrieved_indices = []
                for r_content in retrieved_contents:
                    if r_content in all_docs:
                        idx = all_docs.index(r_content)
                        retrieved_indices.append(idx)
                        y_score[idx] = 1 # Simplified score
                
                scores['ndcg'].append(self.metrics.calculate_ndcg(y_true, y_score, k=k))
                # MRR, Recall, Precision logic would go here using retrieved_indices
            
            results.append({
                "Retriever": name,
                "Avg NDCG@10": np.mean(scores['ndcg'])
            })
            
        return pd.DataFrame(results)

    def evaluate_qa(self, dataset: List[Dict[str, Any]], task_type: str = "close-ended"):
        """
        Evaluates QA performance (Close-ended or Long-form).
        dataset format: [{'query': str, 'reference': str, 'context_docs': List[str]}]
        """
        predictions = []
        references = [item['reference'] for item in dataset]
        
        print(f"Running {task_type} QA Evaluation...")
        for item in dataset:
            # Generate answer using the pipeline
            ans = self.pipeline.generator.generate_answer(item['query'], item['context_docs'], task_type=task_type)
            predictions.append(ans)
            
        if task_type == "close-ended":
            acc = self.metrics.calculate_accuracy(predictions, references)
            return {"accuracy": acc}
        else:
            rouge = self.metrics.calculate_rouge(predictions, references)
            # bscore = self.metrics.calculate_bertscore(predictions, references)
            return {**rouge}

if __name__ == "__main__":
    import numpy as np
    # Mock data for demonstration
    mock_retrieval_data = [{
        "query": "symptoms of insulin resistance",
        "relevant_docs": ["High blood sugar levels and weight gain."],
        "all_docs": ["High blood sugar levels and weight gain.", "The sky is blue.", "Exercise is good for health."]
    }]
    
    # Initialize components for evaluation
    sem = SemanticRetriever()
    sem.build_index(mock_retrieval_data[0]['all_docs'])
    lex = LexicalRetriever()
    lex.build_index(mock_retrieval_data[0]['all_docs'])
    
    evaluator = MedBioEvaluator()
    retrieval_results = evaluator.evaluate_retrieval(mock_retrieval_data, {"Semantic": sem, "Lexical": lex})
    print("\nRetrieval Evaluation Results:")
    print(retrieval_results)
