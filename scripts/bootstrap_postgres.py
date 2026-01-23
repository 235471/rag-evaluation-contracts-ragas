#!/usr/bin/env python3
"""
CLI script to bootstrap PostgreSQL/PGVector infrastructure.
Creates table and unique index for content_hash.

Usage:
    python scripts/bootstrap_postgres.py
    python scripts/bootstrap_postgres.py --table my_table --vector-size 3072
    python scripts/bootstrap_postgres.py --no-index
"""

import argparse
import sys
from pathlib import Path

# Fix for psycopg3 on Windows
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.logging_conf import setup_logging, get_logger
from src.app.config import get_settings


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap PostgreSQL/PGVector table and index"
    )
    parser.add_argument(
        "--table",
        type=str,
        default=None,
        help="Table name (default: from POSTGRES_TABLE_NAME env)",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=3072,
        help="Vector dimension size (default: 3072 for Gemini embeddings)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="public",
        help="Schema name (default: public)",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip creating unique index on content_hash",
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
    table_name = args.table or settings.POSTGRES_TABLE_NAME

    print(f"\n🔧 PostgreSQL Bootstrap")
    print("=" * 60)
    print(f"📊 Table: {args.schema}.{table_name}")
    print(f"📐 Vector size: {args.vector_size}")
    print(f"🔑 Create unique index: {not args.no_index}")
    print("=" * 60)
    print()

    try:
        from src.app.vectorstores.postgres_pgvector import (
            table_exists,
            bootstrap_pgvector_table,
        )

        # Check current state
        if table_exists(table_name, schema=args.schema):
            logger.info(f"Table '{table_name}' already exists")
            print(f"ℹ️ Table '{table_name}' already exists")

        # Bootstrap
        bootstrap_pgvector_table(
            table_name=table_name,
            vector_size=args.vector_size,
            schema=args.schema,
            create_unique_index=not args.no_index,
        )

        print("\n" + "=" * 60)
        print("✅ Bootstrap completed successfully!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(
            f"  1. Run ingestion: python scripts/ingest_postgres.py --table {table_name}"
        )
        print(f'  2. Ask questions: python scripts/ask.py "Your question here"')

    except Exception as e:
        logger.error(f"Bootstrap failed: {e}")
        print(f"\n❌ Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
