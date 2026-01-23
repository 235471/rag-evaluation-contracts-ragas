#!/usr/bin/env python3
"""
CLI script to generate synthetic Q/A pairs and optionally evaluate them.

Usage:
    python scripts/generate_synthetic.py --sample-size 20
    python scripts/generate_synthetic.py --sample-size 10 --evaluate
    python scripts/generate_synthetic.py --documents-dir ./documents --output qa_pairs.json
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
        description="Generate synthetic Q/A pairs from document chunks"
    )
    parser.add_argument(
        "--documents-dir",
        type=str,
        default="documents",
        help="Directory containing PDF documents (default: documents)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of chunks to sample for Q/A generation (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_qa.json",
        help="Output file for generated Q/A pairs (default: synthetic_qa.json)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run RAGAS evaluation on generated pairs",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Limit number of pairs to evaluate (default: all)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["postgres", "pinecone"],
        default=None,
        help="Vectorstore backend for evaluation (default: from env)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = get_logger(__name__)

    settings = get_settings()

    try:
        # Load and chunk documents
        from src.app.eval.synthetic_qa import (
            load_chunks_from_documents,
            generate_synthetic_qa,
            run_synthetic_evaluation,
        )

        logger.info(f"Loading documents from: {args.documents_dir}")
        chunks = load_chunks_from_documents(args.documents_dir)

        if len(chunks) == 0:
            print(f"❌ No chunks found in {args.documents_dir}")
            sys.exit(1)

        # Generate synthetic Q/A
        logger.info(f"Generating synthetic Q/A for {args.sample_size} chunks...")
        qa_pairs = generate_synthetic_qa(
            chunks=chunks,
            sample_size=args.sample_size,
            random_seed=args.seed,
        )

        # Save Q/A pairs
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Generated {len(qa_pairs)} Q/A pairs")
        print(f"📁 Saved to: {output_path}")

        # Optional evaluation
        if args.evaluate:
            print("\n" + "=" * 60)
            print("🔬 Running evaluation...")

            backend = args.backend or settings.VECTORSTORE_BACKEND
            embeddings = get_embeddings()

            if backend == "postgres":
                from src.app.vectorstores.postgres_pgvector import get_pgvector_store

                vector_store = get_pgvector_store(embeddings=embeddings)
            else:
                from src.app.vectorstores.pinecone_store import get_pinecone_store

                vector_store = get_pinecone_store(embeddings=embeddings)

            retriever = vector_store.as_retriever(search_kwargs={"k": 10})
            llm = get_llm()

            from src.app.rag.chains import build_rag_chain_with_context_full

            rag_chain = build_rag_chain_with_context_full(retriever, llm)

            # Run synthetic evaluation
            results = run_synthetic_evaluation(
                qa_pairs=qa_pairs,
                rag_chain=rag_chain,
                limit=args.eval_limit,
            )

            # Run RAGAS evaluation
            from src.app.eval.ragas_eval import (
                build_ragas_dataset,
                run_ragas_evaluation,
                export_results,
                format_metrics_summary,
            )

            dataset = build_ragas_dataset(results)
            eval_result = run_ragas_evaluation(dataset)

            # Export
            output_prefix = output_path.stem + "_eval"
            exported = export_results(
                dataset=dataset,
                eval_result=eval_result,
                output_prefix=output_prefix,
                output_dir=output_path.parent,
            )

            print("\n" + format_metrics_summary(eval_result))
            print(f"\n📁 Dataset: {exported['dataset_json']}")
            print(f"📊 Metrics: {exported['metrics_csv']}")

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
