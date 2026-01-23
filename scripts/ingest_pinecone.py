#!/usr/bin/env python3
"""
CLI script to ingest documents into Pinecone.
Loads PDFs, chunks, embeds, and adds to vectorstore with retry/dedup.

Usage:
    python scripts/ingest_pinecone.py
    python scripts/ingest_pinecone.py --documents-dir ./my_docs
    python scripts/ingest_pinecone.py --batch-size 20 --base-delay 5
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.logging_conf import setup_logging, get_logger
from src.app.config import get_settings, get_embeddings


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone")
    parser.add_argument(
        "--documents-dir",
        type=str,
        default="documents",
        help="Directory containing PDF documents (default: documents)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=380,
        help="Chunk size in tokens (default: 380)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for insertion (default: 20)",
    )
    parser.add_argument(
        "--base-delay",
        type=float,
        default=5.0,
        help="Base delay for retry backoff in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per batch (default: 5)",
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

    print(f"\n📂 Documents directory: {args.documents_dir}")
    print(f"📊 Target index: {settings.PINECONE_INDEX}")
    print(f"📐 Chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")
    print(f"📦 Batch size: {args.batch_size}")
    print()

    try:
        # Load documents
        from langchain_community.document_loaders import PyPDFDirectoryLoader

        logger.info(f"Loading documents from: {args.documents_dir}")
        loader = PyPDFDirectoryLoader(args.documents_dir)
        pdfs = loader.load()

        if not pdfs:
            print(f"❌ No PDF documents found in {args.documents_dir}")
            sys.exit(1)

        logger.info(f"Loaded {len(pdfs)} PDF pages")

        # Get adaptive chunking configuration based on embedding model
        from src.app.utils.chunking import (
            get_chunking_config_from_env,
            create_text_splitter,
        )

        chunking_config = get_chunking_config_from_env()
        logger.info(
            f"Chunking config: {chunking_config.chunk_size} tokens "
            f"(embedding: {chunking_config.embedding_dims}D, max_tokens: {chunking_config.max_tokens})"
        )

        # Allow CLI overrides for chunk_size/overlap
        if args.chunk_size != 380 or args.chunk_overlap != 50:
            chunking_config.chunk_size = args.chunk_size
            chunking_config.chunk_overlap = args.chunk_overlap
            logger.info(
                f"Using CLI override: chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}"
            )

        splitter = create_text_splitter(chunking_config)
        chunks = splitter.split_documents(pdfs)
        logger.info(f"Created {len(chunks)} chunks")

        # Add content hashes
        from src.app.utils.dedup import add_content_hash_to_docs, get_content_hash_ids

        docs = add_content_hash_to_docs(chunks)

        # Get IDs for Pinecone (uses content_hash as ID)
        ids = get_content_hash_ids(docs)
        logger.info("Added content_hash to all documents")

        # Get vectorstore
        from src.app.vectorstores.pinecone_store import get_pinecone_store

        embeddings = get_embeddings()
        vector_store = get_pinecone_store(embeddings=embeddings)

        # Insert with retry
        from src.app.utils.retry import add_documents_with_retry

        logger.info(f"Starting ingestion of {len(docs)} documents...")
        result = add_documents_with_retry(
            vector_store=vector_store,
            docs=docs,
            ids=ids,  # Use content_hash as Pinecone IDs
            batch_size=args.batch_size,
            base_delay=args.base_delay,
            max_retries=args.max_retries,
        )

        print("\n" + "=" * 60)
        print("📊 INGESTION SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully inserted: {result['success_count']} documents")
        print(f"❌ Failed: {result['failed_count']} documents")

        if result["failed_batches"]:
            print(f"⚠️ Failed batches: {result['failed_batches']}")

        print("=" * 60)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"\n❌ Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
