#!/usr/bin/env python3
"""
CLI script to bootstrap Pinecone infrastructure.
Creates index if it doesn't exist.

Usage:
    python scripts/bootstrap_pinecone.py
    python scripts/bootstrap_pinecone.py --dimension 3072 --metric cosine
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.logging_conf import setup_logging, get_logger
from src.app.config import get_settings


def main():
    parser = argparse.ArgumentParser(description="Bootstrap Pinecone index")
    parser.add_argument(
        "--index-name",
        type=str,
        default=None,
        help="Index name (default: from PINECONE_INDEX env)",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=None,
        help="Vector dimension (default: from PINECONE_DIMENSION env)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["cosine", "euclidean", "dotproduct"],
        default=None,
        help="Similarity metric (default: from PINECONE_METRIC env)",
    )
    parser.add_argument(
        "--cloud",
        type=str,
        default=None,
        help="Cloud provider (default: from PINECONE_CLOUD env)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Region (default: from PINECONE_REGION env)",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=15,
        help="Seconds to wait after creating index (default: 15)",
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

    index_name = args.index_name or settings.PINECONE_INDEX
    dimension = args.dimension or settings.PINECONE_DIMENSION
    metric = args.metric or settings.PINECONE_METRIC
    cloud = args.cloud or settings.PINECONE_CLOUD
    region = args.region or settings.PINECONE_REGION

    print(f"\n🔧 Pinecone Bootstrap")
    print("=" * 60)
    print(f"📊 Index: {index_name}")
    print(f"📐 Dimension: {dimension}")
    print(f"📏 Metric: {metric}")
    print(f"☁️ Cloud: {cloud}")
    print(f"🌍 Region: {region}")
    print("=" * 60)
    print()

    try:
        from src.app.vectorstores.pinecone_store import (
            index_exists,
            bootstrap_pinecone_index,
        )

        # Check current state
        if index_exists(index_name):
            logger.info(f"Index '{index_name}' already exists")
            print(f"ℹ️ Index '{index_name}' already exists")

        # Bootstrap
        bootstrap_pinecone_index(
            index_name=index_name,
            dimension=dimension,
            metric=metric,
            cloud=cloud,
            region=region,
            wait_seconds=args.wait_seconds,
        )

        print("\n" + "=" * 60)
        print("✅ Bootstrap completed successfully!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"  1. Run ingestion: python scripts/ingest_pinecone.py")
        print(
            f'  2. Ask questions: python scripts/ask.py "Your question here" --backend pinecone'
        )

    except Exception as e:
        logger.error(f"Bootstrap failed: {e}")
        print(f"\n❌ Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
