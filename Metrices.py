import time
import math
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from difflib import SequenceMatcher

@dataclass
class LatencyMetrics:
    """Store latency measurements"""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float = 0.0
    
    def __post_init__(self):
        self.duration_ms = (self.end_time - self.start_time) * 1000


class LatencyEvaluator:
    """Measure latency across RAG pipeline operations"""
    
    def __init__(self):
        self.metrics: List[LatencyMetrics] = []
    
    def measure_operation(self, operation_name: str, start_time: float, end_time: float) -> Dict[str, float]:
        """
        Record latency for an operation
        
        Args:
            operation_name: Name of operation (e.g., 'retrieval', 'generation')
            start_time: Start timestamp
            end_time: End timestamp
        
        Returns:
            Latency in milliseconds
        """
        metric = LatencyMetrics(
            operation=operation_name,
            start_time=start_time,
            end_time=end_time
        )
        self.metrics.append(metric)
        return {"latency_ms": metric.duration_ms}
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get comprehensive latency statistics
        
        Returns:
            min, max, mean, median, p95, p99 latencies in milliseconds
        """
        if not self.metrics:
            return {}
        
        durations = [m.duration_ms for m in self.metrics]
        
        return {
            "min_ms": float(min(durations)),
            "max_ms": float(max(durations)),
            "mean_ms": float(np.mean(durations)),
            "median_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95)),
            "p99_ms": float(np.percentile(durations, 99)),
            "total_calls": len(self.metrics)
        }
    
    def get_by_operation(self) -> Dict[str, Dict[str, float]]:
        """
        Get latency stats grouped by operation type
        
        Returns:
            Statistics for each operation (retrieval, generation, etc.)
        """
        operations = {}
        
        for metric in self.metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric.duration_ms)
        
        stats = {}
        for op_name, durations in operations.items():
            stats[op_name] = {
                "mean_ms": float(np.mean(durations)),
                "p95_ms": float(np.percentile(durations, 95)),
                "p99_ms": float(np.percentile(durations, 99)),
                "count": len(durations)
            }
        
        return stats
    
    def get_qps(self, latency_ms: float) -> float:
        """
        Calculate Queries Per Second from latency
        
        Args:
            latency_ms: Latency in milliseconds
        
        Returns:
            Queries per second
        """
        if latency_ms == 0:
            return 0
        return 1000 / latency_ms


class HallucinationDetector:
    """Detect hallucinations in LLM-generated text"""
    
    def __init__(self):
        self.hallucinations: List[Dict] = []
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities (proper nouns) from text using regex
        Pattern: Capital letter followed by lowercase
        
        Args:
            text: Input text
        
        Returns:
            List of extracted entities
        """
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(pattern, text)
        return entities
    
    def entity_based_hallucination_detection(
        self,
        generated_text: str,
        source_document: str
    ) -> Dict[str, Any]:
        """
        Detect hallucinations by comparing entities
        
        Args:
            generated_text: LLM-generated text
            source_document: Source/reference document
        
        Returns:
            Hallucination metrics including rate, entities, severity
        """
        gen_entities = self.extract_entities(generated_text)
        src_entities = self.extract_entities(source_document)
        
        hallucinated = [e for e in gen_entities if e not in src_entities]
        
        hallucination_rate = len(hallucinated) / len(gen_entities) if gen_entities else 0
        
        if hallucination_rate > 0.1:
            severity = "CRITICAL"
        elif hallucination_rate > 0.05:
            severity = "HIGH"
        elif hallucination_rate > 0.02:
            severity = "MEDIUM"
        else:
            severity = "OK"
        
        return {
            "hallucination_rate": float(hallucination_rate),
            "hallucinated_entities": hallucinated,
            "generated_entities_count": len(gen_entities),
            "source_entities_count": len(src_entities),
            "severity": severity,
            "metric_type": "entity_matching"
        }
    
    def semantic_consistency_check(
        self,
        generated_text: str,
        source_document: str
    ) -> Dict[str, float]:
        """
        Detect hallucinations using semantic similarity
        Uses SequenceMatcher to find text overlap
        
        Args:
            generated_text: LLM-generated text
            source_document: Source/reference document
        
        Returns:
            Consistency score and verdict
        """
        # Calculate text similarity using SequenceMatcher
        matcher = SequenceMatcher(None, generated_text.lower(), source_document.lower())
        similarity = matcher.ratio()
        
        # Determine consistency
        if similarity > 0.85:
            consistency = "EXCELLENT"
        elif similarity > 0.75:
            consistency = "GOOD"
        elif similarity > 0.60:
            consistency = "FAIR"
        else:
            consistency = "POOR"
        
        return {
            "semantic_consistency_score": float(min(similarity, 1.0)),
            "consistency_level": consistency,
            "metric_type": "semantic_similarity"
        }
    
    def combined_hallucination_check(
        self,
        generated_text: str,
        source_document: str
    ) -> Dict[str, Any]:
        """
        Combined hallucination detection using both methods
        
        Args:
            generated_text: LLM-generated text
            source_document: Source/reference document
        
        Returns:
            Combined hallucination metrics
        """
        entity_check = self.entity_based_hallucination_detection(generated_text, source_document)
        semantic_check = self.semantic_consistency_check(generated_text, source_document)
        
        return {
            "entity_based_hallucination_rate": entity_check["hallucination_rate"],
            "entity_severity": entity_check["severity"],
            "semantic_consistency": semantic_check["semantic_consistency_score"],
            "semantic_consistency_level": semantic_check["consistency_level"],
            "hallucinated_entities": entity_check["hallucinated_entities"],
            "is_hallucinating": entity_check["hallucination_rate"] > 0.05 or semantic_check["semantic_consistency_score"] < 0.75,
            "overall_verdict": self._determine_overall_verdict(entity_check, semantic_check)
        }
    
    @staticmethod
    def _determine_overall_verdict(entity_check: Dict, semantic_check: Dict) -> str:
        """
        Determine overall hallucination verdict
        """
        if entity_check["severity"] == "CRITICAL" or semantic_check["consistency_level"] == "POOR":
            return "CRITICAL_HALLUCINATION"
        elif entity_check["severity"] == "HIGH" or semantic_check["consistency_level"] == "FAIR":
            return "LIKELY_HALLUCINATION"
        elif entity_check["severity"] == "MEDIUM" or semantic_check["consistency_level"] == "GOOD":
            return "POSSIBLE_HALLUCINATION"
        else:
            return "NO_HALLUCINATION_DETECTED"


class TokenAndCostEvaluator:
    """Calculate token usage and API costs"""
    
    def __init__(self, pricing_config: Dict[str, Dict[str, float]] = None):
        """
        Initialize with pricing configuration
        
        Args:
            pricing_config: Model pricing in format:
                {'model_name': {'input_per_1k': 0.075, 'output_per_1k': 0.30}}
        """
        self.pricing_config = pricing_config or self._default_pricing()
    
    @staticmethod
    def _default_pricing() -> Dict[str, Dict[str, float]]:
        """
        Default pricing for common models
        """
        return {
            "gemini-2.5-flash": {
                "input_per_1k": 0.075,
                "output_per_1k": 0.30
            },
            "gpt-4": {
                "input_per_1k": 0.03,
                "output_per_1k": 0.06
            },
            "claude-3-haiku": {
                "input_per_1k": 0.00080,
                "output_per_1k": 0.0024
            }
        }
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text
        General rule: 1 word ≈ 1.3 tokens
        
        Args:
            text: Input text
        
        Returns:
            Estimated token count
        """
        word_count = len(text.split())
        estimated_tokens = int(word_count * 1.3)
        return estimated_tokens
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gemini-2.5-flash"
    ) -> Dict[str, float]:
        """
        Calculate API call cost
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
        
        Returns:
            Cost breakdown
        """
        if model not in self.pricing_config:
            raise ValueError(f"Model {model} not in pricing config")
        
        rates = self.pricing_config[model]
        input_cost = (input_tokens / 1000) * rates["input_per_1k"]
        output_cost = (output_tokens / 1000) * rates["output_per_1k"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": float(input_cost),
            "output_cost": float(output_cost),
            "total_cost": float(total_cost),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": model
        }
    
    def tokens_per_dollar(self, total_tokens: int, total_cost: float) -> float:
        """
        Calculate efficiency metric: tokens per dollar
        
        Args:
            total_tokens: Total tokens used
            total_cost: Total cost in dollars
        
        Returns:
            Tokens per dollar
        """
        if total_cost == 0:
            return 0
        return total_tokens / total_cost
    
    def cost_per_operation(
        self,
        total_cost: float,
        num_operations: int
    ) -> float:
        """
        Calculate cost per operation (document, query, etc.)
        
        Args:
            total_cost: Total cost in dollars
            num_operations: Number of operations
        
        Returns:
            Cost per operation
        """
        if num_operations == 0:
            return 0
        return total_cost / num_operations

class RetrievalMetricsEvaluator:
    """Evaluate retrieval quality metrics"""
    
    @staticmethod
    def hit_rate(
        retrieved_chunk_ids: List[str],
        relevant_chunk_ids: List[str],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Calculate Hit Rate@K
        Percentage of queries where top-K contains at least one relevant document
        
        Args:
            retrieved_chunk_ids: List of retrieved chunk IDs (ordered by rank)
            relevant_chunk_ids: List of all relevant chunk IDs
            k: Number of top results to consider
        
        Returns:
            Hit rate and detailed metrics
        """
        retrieved_top_k = set(retrieved_chunk_ids[:k])
        relevant_set = set(relevant_chunk_ids)
        
        hits = len(retrieved_top_k & relevant_set)
        hit_rate_score = hits / len(relevant_set) if relevant_set else 0
        
        return {
            "hit_rate@k": float(hit_rate_score),
            "k": k,
            "hits": hits,
            "total_relevant": len(relevant_set),
            "interpretation": f"{hit_rate_score*100:.1f}% of queries found relevant doc in top-{k}"
        }
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved_chunk_ids: List[str],
        relevant_chunk_ids: List[str],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Calculate Mean Reciprocal Rank (MRR)
        Average of 1/rank of first relevant item
        
        Args:
            retrieved_chunk_ids: List of retrieved chunk IDs (ordered by rank)
            relevant_chunk_ids: List of all relevant chunk IDs
            k: Number of top results to consider
        
        Returns:
            MRR and position of first relevant result
        """
        relevant_set = set(relevant_chunk_ids)
        
        for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], 1):
            if chunk_id in relevant_set:
                mrr_score = 1.0 / rank
                return {
                    "mrr": float(mrr_score),
                    "rank_of_first_relevant": rank,
                    "k": k,
                    "interpretation": f"First relevant doc at position {rank}, MRR = {mrr_score:.3f}"
                }
        
        return {
            "mrr": 0.0,
            "rank_of_first_relevant": None,
            "k": k,
            "interpretation": f"No relevant doc found in top-{k}"
        }
    
    @staticmethod
    def ndcg(
        retrieved_chunk_ids: List[str],
        relevant_chunk_ids: List[str],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain)
        Accounts for ranking quality and relevance levels
        
        Args:
            retrieved_chunk_ids: List of retrieved chunk IDs (ordered by rank)
            relevant_chunk_ids: List of all relevant chunk IDs
            k: Number of top results to consider
        
        Returns:
            NDCG score (0-1, where 1 is perfect ranking)
        """
        relevant_set = set(relevant_chunk_ids)
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, chunk_id in enumerate(retrieved_chunk_ids[:k], 1):
            relevance = 1 if chunk_id in relevant_set else 0
            dcg += relevance / math.log2(i + 1)
        
        # Calculate Ideal DCG (perfect ranking)
        ideal_dcg = sum([
            1.0 / math.log2(i + 1)
            for i in range(min(k, len(relevant_set)))
        ])
        
        # Normalize
        ndcg_score = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        
        return {
            "ndcg@k": float(ndcg_score),
            "k": k,
            "dcg": float(dcg),
            "ideal_dcg": float(ideal_dcg),
            "interpretation": f"Ranking quality: {ndcg_score*100:.1f}% of ideal"
        }

class GenerationQualityEvaluator:
    """Evaluate quality of generated answers"""
    
    @staticmethod
    def _calculate_text_similarity(text1: str, text2: str) -> float:
        """
        Calculate text similarity using word overlap (Jaccard similarity)
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        return float(overlap / union) if union > 0 else 0.0
    
    def faithfulness(
        self,
        generated_answer: str,
        retrieved_context: str
    ) -> Dict[str, Any]:
        """
        Evaluate Faithfulness (Groundedness)
        Measures if answer is grounded in retrieved context
        
        Args:
            generated_answer: LLM-generated answer
            retrieved_context: Retrieved document chunks
        
        Returns:
            Faithfulness metrics
        """
        similarity_score = self._calculate_text_similarity(generated_answer, retrieved_context)
        
        if similarity_score > 0.85:
            level = "EXCELLENT"
        elif similarity_score > 0.75:
            level = "GOOD"
        elif similarity_score > 0.60:
            level = "FAIR"
        else:
            level = "POOR"
        
        return {
            "faithfulness_score": float(similarity_score),
            "level": level,
            "is_grounded": similarity_score > 0.75,
            "interpretation": f"Answer {level.lower()} grounded in context. {similarity_score*100:.1f}% overlap."
        }
    
    def answer_relevance(
        self,
        query: str,
        generated_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate Answer Relevance
        Measures if answer actually addresses the query
        
        Args:
            query: User query
            generated_answer: Generated answer
        
        Returns:
            Relevance metrics
        """
        similarity_score = self._calculate_text_similarity(query, generated_answer)
        
        if similarity_score > 0.85:
            level = "EXCELLENT"
        elif similarity_score > 0.75:
            level = "GOOD"
        elif similarity_score > 0.60:
            level = "FAIR"
        else:
            level = "POOR"
        
        return {
            "answer_relevance_score": float(similarity_score),
            "level": level,
            "is_relevant": similarity_score > 0.75,
            "interpretation": f"Answer is {level.lower()} relevant to query. {similarity_score*100:.1f}% similarity."
        }

class ClassificationMetricsEvaluator:
    """Evaluate classification metrics for risk detection, etc."""
    
    @staticmethod
    def calculate_metrics(
        predicted_items: List[str],
        ground_truth_items: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate Precision, Recall, and F1-Score
        Useful for risk detection, clause identification, etc.
        
        Args:
            predicted_items: Items predicted by system
            ground_truth_items: Actual items (ground truth)
        
        Returns:
            Precision, Recall, F1-Score and detailed metrics
        """
        pred_set = set(predicted_items)
        truth_set = set(ground_truth_items)
        
        # Calculate TP, FP, FN
        tp = len(pred_set & truth_set)  # Correctly predicted
        fp = len(pred_set - truth_set)  # Wrongly predicted
        fn = len(truth_set - pred_set)  # Missed items
        
        # Calculate precision, recall, f1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": 0,  # Not applicable for open-ended predictions
            "interpretation": f"Precision: {precision*100:.1f}%, Recall: {recall*100:.1f}%, F1: {f1:.3f}"
        }


class RAGEvaluator:
    """Unified evaluator for complete RAG pipeline"""
    
    def __init__(self, pricing_config: Dict = None):
        """
        Initialize RAG evaluator with all metric components
        
        Args:
            pricing_config: Optional pricing configuration for cost calculation
        """
        self.latency_evaluator = LatencyEvaluator()
        self.hallucination_detector = HallucinationDetector()
        self.token_cost_evaluator = TokenAndCostEvaluator(pricing_config)
        self.retrieval_metrics = RetrievalMetricsEvaluator()
        self.generation_quality = GenerationQualityEvaluator()
        self.classification_metrics = ClassificationMetricsEvaluator()
    
    def evaluate_complete_rag_pipeline(
        self,
        query: str,
        retrieved_chunk_ids: List[str],
        relevant_chunk_ids: List[str],
        retrieved_context: str,
        generated_answer: str,
        source_document: str,
        input_tokens: int,
        output_tokens: int,
        retrieval_latency_ms: float,
        generation_latency_ms: float,
        model: str = "gemini-2.5-flash"
    ) -> Dict[str, Any]:
        """
        Comprehensive RAG pipeline evaluation
        
        Args:
            query: User query
            retrieved_chunk_ids: IDs of retrieved chunks (ordered by rank)
            relevant_chunk_ids: IDs of truly relevant chunks
            retrieved_context: Actual retrieved text
            generated_answer: LLM-generated answer
            source_document: Original source document
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            retrieval_latency_ms: Retrieval latency in ms
            generation_latency_ms: Generation latency in ms
            model: Model used
        
        Returns:
            Complete evaluation report
        """
        # 1. Retrieval Metrics
        hit_rate = self.retrieval_metrics.hit_rate(retrieved_chunk_ids, relevant_chunk_ids)
        mrr = self.retrieval_metrics.mean_reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids)
        ndcg = self.retrieval_metrics.ndcg(retrieved_chunk_ids, relevant_chunk_ids)
        
        # 2. Hallucination Detection
        hallucination = self.hallucination_detector.combined_hallucination_check(
            generated_answer, source_document
        )
        
        # 3. Generation Quality
        faithfulness = self.generation_quality.faithfulness(generated_answer, retrieved_context)
        answer_relevance = self.generation_quality.answer_relevance(query, generated_answer)
        
        # 4. Cost & Tokens
        cost_breakdown = self.token_cost_evaluator.calculate_cost(input_tokens, output_tokens, model)
        
        # 5. Latency
        total_latency_ms = retrieval_latency_ms + generation_latency_ms
        
        # 6. Overall RAG Score
        overall_score = (
            hit_rate["hit_rate@k"] * 0.20 +
            mrr["mrr"] * 0.15 +
            ndcg["ndcg@k"] * 0.20 +
            faithfulness["faithfulness_score"] * 0.20 +
            answer_relevance["answer_relevance_score"] * 0.15 +
            (1 - hallucination["entity_based_hallucination_rate"]) * 0.10
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retrieval_metrics": {
                "hit_rate": hit_rate,
                "mrr": mrr,
                "ndcg": ndcg
            },
            "generation_metrics": {
                "faithfulness": faithfulness,
                "answer_relevance": answer_relevance
            },
            "hallucination_metrics": hallucination,
            "cost_metrics": cost_breakdown,
            "latency_metrics": {
                "retrieval_latency_ms": float(retrieval_latency_ms),
                "generation_latency_ms": float(generation_latency_ms),
                "total_latency_ms": float(total_latency_ms)
            },
            "overall_rag_score": float(overall_score),
            "quality_verdict": self._get_quality_verdict(overall_score)
        }
    
    @staticmethod
    def _get_quality_verdict(overall_score: float) -> str:
        """
        Determine quality verdict based on overall score
        """
        if overall_score > 0.85:
            return "EXCELLENT"
        elif overall_score > 0.75:
            return "GOOD"
        elif overall_score > 0.60:
            return "FAIR"
        else:
            return "POOR"
