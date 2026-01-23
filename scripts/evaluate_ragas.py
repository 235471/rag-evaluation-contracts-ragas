#!/usr/bin/env python3
"""
CLI script for RAGAS evaluation.
Uses existing vectorstore - does NOT run ingestão or bootstrap.

Usage:
    python scripts/evaluate_ragas.py --question "..." --ground-truth "..."
    python scripts/evaluate_ragas.py --input-file eval_cases.json --output-prefix my_eval
"""

import argparse
import json
import sys
from pathlib import Path

# Fix for psycopg3 on Windows
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.logging_conf import setup_logging, get_logger
from src.app.config import get_settings, get_llm, get_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on RAG responses"
    )

    # Single question mode
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to evaluate",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Expected answer for single question mode",
    )

    # Batch mode
    parser.add_argument(
        "--input-file",
        type=str,
        help="JSON file with test cases [{question, ground_truth}, ...]",
    )

    # Output options
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="eval_run",
        help="Prefix for output files (default: eval_run)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default: current directory)",
    )

    # Backend options
    parser.add_argument(
        "--backend",
        type=str,
        choices=["postgres", "pinecone"],
        default=None,
        help="Vectorstore backend (default: from env)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Number of documents to retrieve (default: 2)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.question and not args.input_file:
        parser.error("Either --question or --input-file is required")

    if args.question and not args.ground_truth:
        parser.error("--ground-truth is required when using --question")

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = get_logger(__name__)

    settings = get_settings()
    backend = args.backend or settings.VECTORSTORE_BACKEND

    try:
        # Build test cases
        if args.input_file:
            with open(args.input_file, "r", encoding="utf-8") as f:
                test_cases = json.load(f)
            logger.info(f"Loaded {len(test_cases)} test cases from {args.input_file}")
        else:
            test_cases = [
                {"question": args.question, "ground_truth": args.ground_truth}
            ]

        # Get vectorstore
        embeddings = get_embeddings()

        if backend == "postgres":
            from src.app.vectorstores.postgres_pgvector import get_pgvector_store

            vector_store = get_pgvector_store(embeddings=embeddings)
        else:
            from src.app.vectorstores.pinecone_store import get_pinecone_store

            vector_store = get_pinecone_store(embeddings=embeddings)

        retriever = vector_store.as_retriever(search_kwargs={"k": args.k})
        llm = get_llm()

        # Build RAG chain
        from src.app.rag.chains import build_rag_chain_with_context_full

        rag_chain = build_rag_chain_with_context_full(retriever, llm)

        # Run RAG on all test cases
        logger.info(f"Running RAG on {len(test_cases)} test cases...")
        results_list = []

        for i, test in enumerate(test_cases):
            logger.info(
                f"Processing {i + 1}/{len(test_cases)}: {test['question'][:50]}..."
            )

            try:
                result = rag_chain.invoke(test["question"])
                results_list.append(
                    {
                        "question": result["query"],
                        "answer": result["answer"],
                        "contexts": result["contexts"],
                        "ground_truth": test["ground_truth"],
                    }
                )
            except Exception as e:
                logger.error(f"Error on question {i + 1}: {e}")
                results_list.append(
                    {
                        "question": test["question"],
                        "answer": f"ERROR: {e}",
                        "contexts": [],
                        "ground_truth": test["ground_truth"],
                    }
                )

        # Build dataset and run evaluation
        from src.app.eval.ragas_eval import (
            build_ragas_dataset,
            run_ragas_evaluation,
            export_results,
            format_metrics_summary,
        )

        dataset = build_ragas_dataset(results_list)

        logger.info("Running RAGAS evaluation...")
        eval_result = run_ragas_evaluation(dataset)

        # Export results
        output_dir = Path(args.output_dir)
        exported = export_results(
            dataset=dataset,
            eval_result=eval_result,
            output_prefix=args.output_prefix,
            output_dir=output_dir,
        )

        # Display summary
        print("\n" + "=" * 60)
        print(format_metrics_summary(eval_result))
        print("\n" + "-" * 60)
        print(f"📁 Dataset exported to: {exported['dataset_json']}")
        print(f"📊 Metrics exported to: {exported['metrics_csv']}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
