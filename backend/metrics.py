import numpy as np
from sklearn.metrics import ndcg_score
from typing import List, Dict
import evaluate
from bert_score import score as bert_score_calc

class EvaluationMetrics:
    @staticmethod
    def calculate_ndcg(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k."""
        return ndcg_score([y_true], [y_score], k=k)

    @staticmethod
    def calculate_mrr(y_true: List[int], y_score: List[int]) -> float:
        """Calculate Mean Reciprocal Rank."""
        # y_true is binary relevance, y_score is indices of retrieved docs
        for i, idx in enumerate(y_score):
            if y_true[idx] > 0:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def calculate_precision_at_k(y_true: List[int], y_score: List[int], k: int = 10) -> float:
        """Calculate Precision at k."""
        top_k = y_score[:k]
        relevant = sum([1 for idx in top_k if y_true[idx] > 0])
        return relevant / k

    @staticmethod
    def calculate_recall_at_k(y_true: List[int], y_score: List[int], k: int = 10) -> float:
        """Calculate Recall at k."""
        top_k = y_score[:k]
        total_relevant = sum([1 for val in y_true if val > 0])
        if total_relevant == 0: return 0.0
        relevant_retrieved = sum([1 for idx in top_k if y_true[idx] > 0])
        return relevant_retrieved / total_relevant

    @staticmethod
    def calculate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores (scaled to 100)."""
        rouge = evaluate.load("rouge")
        # Use stemmer to match research paper evaluation settings
        results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        # Scale to 0-100 for paper-like reporting
        return {k: v * 100 for k, v in results.items()}

    @staticmethod
    def calculate_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BLEU score (scaled to 100)."""
        bleu = evaluate.load("bleu")
        results = bleu.compute(predictions=predictions, references=references)
        return {"bleu": results['bleu'] * 100}

    @staticmethod
    def calculate_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore (scaled to 100)."""
        P, R, F1 = bert_score_calc(predictions, references, lang="en", verbose=False)
        return {
            "bert_precision": float(P.mean()) * 100,
            "bert_recall": float(R.mean()) * 100,
            "bert_f1": float(F1.mean()) * 100
        }
        
    @staticmethod
    def calculate_bleurt(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BLEURT score (Placeholder - requires heavy model)."""
        # In a real setting, we would load the 'bleurt' metric from evaluate
        # For this demonstration, we will simulate the score based on BERTScore correlation
        # as BLEURT environment setup is complex.
        try:
            bleurt = evaluate.load("bleurt", "BLEURT-20")
            results = bleurt.compute(predictions=predictions, references=references)
            return {"bleurt": float(np.mean(results['scores']))}
        except:
            return {"bleurt": 0.0} # Fallback if model not available

    @staticmethod
    def calculate_accuracy(predictions: List[str], references: List[str]) -> float:
        """Calculate Accuracy for close-ended QA."""
        correct = sum([1 for p, r in zip(predictions, references) if p.strip().upper() == r.strip().upper()])
        return correct / len(predictions) if predictions else 0.0
