#!/usr/bin/env python3
"""
CLI script to bootstrap PostgreSQL/PGVector infrastructure.
Creates embeddings table, cache table, or both.

Usage:
    python scripts/bootstrap_postgres.py                    # Embeddings table only (default)
    python scripts/bootstrap_postgres.py --table my_table   # Custom embeddings table name
    python scripts/bootstrap_postgres.py --cache-table      # Cache table only
    python scripts/bootstrap_postgres.py --all              # Both embeddings + cache tables
    python scripts/bootstrap_postgres.py --clear-cache      # Clear cache entries
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
        help="Embeddings table name (default: from POSTGRES_TABLE_NAME env)",
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
        "--cache-table",
        action="store_true",
        help="Create only the semantic cache table (skips embeddings table)",
    )
    parser.add_argument(
        "--cache-name",
        type=str,
        default=None,
        help="Cache table name (default: from CACHE_TABLE_NAME env)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create both embeddings and cache tables",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all entries from the cache table",
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
    cache_name = args.cache_name or settings.CACHE_TABLE_NAME

    # Determine what to create
    create_embeddings = not args.cache_table or args.all
    create_cache = args.cache_table or args.all

    # Handle --clear-cache
    if args.clear_cache:
        print(f"\n🗑️  Clearing Cache")
        print("=" * 60)
        print(f"📊 Cache table: {cache_name}")
        print("=" * 60)

        try:
            from src.app.cache.semantic_cache import SemanticCache

            cache = SemanticCache(table_name=cache_name)
            count_before = cache.count()
            cache.clear()
            print(f"\n✅ Cleared {count_before} entries from cache '{cache_name}'")
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            print(f"\n❌ Erro: {e}")
            sys.exit(1)
        return

    print(f"\n🔧 PostgreSQL Bootstrap")
    print("=" * 60)
    if create_embeddings:
        print(f"📊 Embeddings table: {args.schema}.{table_name}")
        print(f"📐 Vector size: {args.vector_size}")
        print(f"🔑 Create unique index: {not args.no_index}")
    if create_cache:
        print(f"💾 Cache table: {args.schema}.{cache_name}")
        print(f"📐 Cache vector dim: {settings.CACHE_EMBEDDING_DIMENSION}")
    print("=" * 60)
    print()

    try:
        from src.app.vectorstores.postgres_pgvector import (
            table_exists,
            bootstrap_pgvector_table,
            bootstrap_cache_table,
        )

        # Bootstrap embeddings table
        if create_embeddings:
            if table_exists(table_name, schema=args.schema):
                logger.info(f"Table '{table_name}' already exists")
                print(f"ℹ️ Table '{table_name}' already exists")

            bootstrap_pgvector_table(
                table_name=table_name,
                vector_size=args.vector_size,
                schema=args.schema,
                create_unique_index=not args.no_index,
            )
            print(f"✅ Embeddings table '{table_name}' ready")

        # Bootstrap cache table
        if create_cache:
            if table_exists(cache_name, schema=args.schema):
                logger.info(f"Cache table '{cache_name}' already exists")
                print(f"ℹ️ Cache table '{cache_name}' already exists")

            bootstrap_cache_table(
                table_name=cache_name,
                vector_size=settings.CACHE_EMBEDDING_DIMENSION,
                schema=args.schema,
            )
            print(f"✅ Cache table '{cache_name}' ready")

        print("\n" + "=" * 60)
        print("✅ Bootstrap completed successfully!")
        print("=" * 60)
        print(f"\nNext steps:")
        if create_embeddings:
            print(
                f"  1. Run ingestion: python scripts/ingest_postgres.py --table {table_name}"
            )
            print(f'  2. Ask questions: python scripts/ask.py "Your question here"')
        if create_cache:
            print(f"  📦 Cache is ready to use automatically in ask.py and Streamlit")

    except Exception as e:
        logger.error(f"Bootstrap failed: {e}")
        print(f"\n❌ Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
