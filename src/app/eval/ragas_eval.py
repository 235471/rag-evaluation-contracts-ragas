"""
RAGAS evaluation module.
Builds datasets, runs evaluation metrics, and exports results.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset

from src.app.logging_conf import get_logger

logger = get_logger(__name__)


def build_ragas_dataset(results: List[Dict[str, Any]]) -> Dataset:
    """
    Build a RAGAS-compatible Dataset from RAG results.

    Each result dict should contain:
    - question: The query
    - answer: The generated answer
    - contexts: List of context strings
    - ground_truth: The expected answer

    Args:
        results: List of result dictionaries

    Returns:
        HuggingFace Dataset in RAGAS format
    """
    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results],
        }
    )

    logger.info(f"Built RAGAS dataset with {len(results)} samples")
    return dataset


def build_single_sample_dataset(
    query: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> Dataset:
    """
    Build a RAGAS dataset from a single sample.

    Args:
        query: The question
        answer: The generated answer
        contexts: List of context strings
        ground_truth: The expected answer

    Returns:
        HuggingFace Dataset with single sample
    """
    return Dataset.from_dict(
        {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
    )


def get_default_metrics():
    """
    Get the default RAGAS metrics from notebook2.

    Returns:
        List of metric instances
    """
    from ragas.metrics import (
        faithfulness,
        answer_correctness,
        context_precision,
        context_recall,
    )

    return [
        faithfulness,
        answer_correctness,
        context_precision,
        context_recall,
    ]


def run_ragas_evaluation(
    dataset: Dataset,
    llm=None,
    embeddings=None,
    metrics: Optional[List] = None,
):
    """
    Run RAGAS evaluation on a dataset.

    Args:
        dataset: RAGAS-formatted Dataset
        llm: LLM for evaluation (defaults to get_eval_llm())
        embeddings: Embeddings for evaluation (defaults to get_eval_embeddings())
        metrics: List of metrics (defaults to get_default_metrics())

    Returns:
        RAGAS EvaluationResult
    """
    from ragas import evaluate

    if llm is None:
        from src.app.config import get_eval_llm

        llm = get_eval_llm()

    if embeddings is None:
        from src.app.config import get_eval_embeddings

        embeddings = get_eval_embeddings()

    if metrics is None:
        metrics = get_default_metrics()

    logger.info(f"Running RAGAS evaluation with {len(metrics)} metrics")

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )
        logger.info("RAGAS evaluation completed successfully")
        return result

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        raise


def export_results(
    dataset: Dataset,
    eval_result,
    output_prefix: str = "eval_run",
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Export evaluation results to JSON and CSV files.

    Args:
        dataset: The evaluated dataset
        eval_result: RAGAS EvaluationResult
        output_prefix: Prefix for output filenames
        output_dir: Output directory (defaults to current directory)

    Returns:
        Dict with paths to exported files
    """
    import pandas as pd

    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export dataset to JSON for auditability
    json_path = output_dir / f"{output_prefix}.json"
    dataset.to_json(str(json_path))
    logger.info(f"Exported dataset to: {json_path}")

    # Export metrics to CSV
    csv_path = output_dir / f"{output_prefix}_metrics.csv"
    df = eval_result.to_pandas()
    df.to_csv(csv_path, index=False)
    logger.info(f"Exported metrics to: {csv_path}")

    return {
        "dataset_json": json_path,
        "metrics_csv": csv_path,
    }


def format_metrics_summary(eval_result) -> str:
    """
    Format evaluation metrics as a human-readable summary.

    Args:
        eval_result: RAGAS EvaluationResult

    Returns:
        Formatted string summary
    """
    import pandas as pd

    df = eval_result.to_pandas()

    # Get metric columns (exclude common non-metric columns in RAGAS 0.1 and 0.2)
    excluded_cols = [
        "question",
        "answer",
        "contexts",
        "ground_truth",
        "user_input",
        "retrieved_contexts",
        "response",
        "reference",
    ]
    metric_cols = [c for c in df.columns if c not in excluded_cols]

    lines = ["📊 RAGAS Evaluation Summary", "=" * 40]

    for col in metric_cols:
        try:
            # Ensure we only try to average numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_val = df[col].mean()
                if pd.isna(mean_val):
                    lines.append(f"{col}: N/A (all timed out)")
                else:
                    lines.append(f"{col}: {mean_val:.4f}")
            else:
                logger.debug(f"Skipping non-numeric column in summary: {col}")
        except Exception as e:
            logger.warning(f"Could not calculate mean for column {col}: {e}")

    return "\n".join(lines)
