"""
Retry utilities with exponential backoff and jitter.
Used for resilient batch operations against vector stores and APIs.
"""

import random
import time
from typing import List, Optional, Callable, Any

from src.app.logging_conf import get_logger

logger = get_logger(__name__)


def add_documents_with_retry(
    vector_store,
    docs: List,
    ids: Optional[List[str]] = None,
    max_retries: int = 5,
    base_delay: float = 15.0,
    batch_size: int = 20,
    on_batch_success: Optional[Callable[[int, int], None]] = None,
    on_batch_error: Optional[Callable[[int, Exception], None]] = None,
) -> dict:
    """
    Add documents to a vector store in batches with exponential backoff retry.

    Args:
        vector_store: Vector store instance with add_documents method
        docs: List of Document objects to add
        ids: Optional list of IDs (for Pinecone). Must match length of docs if provided.
        max_retries: Maximum retry attempts per batch
        base_delay: Base delay in seconds for exponential backoff
        batch_size: Number of documents per batch
        on_batch_success: Optional callback(batch_num, total_batches) on success
        on_batch_error: Optional callback(batch_num, exception) on final failure

    Returns:
        Dict with 'success_count', 'failed_count', 'failed_batches'
    """
    total_batches = (len(docs) + batch_size - 1) // batch_size
    success_count = 0
    failed_count = 0
    failed_batches = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        batch_ids = ids[i : i + batch_size] if ids else None
        batch_num = i // batch_size + 1
        attempt = 0

        while attempt <= max_retries:
            try:
                if batch_ids:
                    vector_store.add_documents(batch, ids=batch_ids)
                else:
                    vector_store.add_documents(batch)

                success_count += len(batch)
                logger.info(f"✅ Inserted batch {batch_num}/{total_batches}")

                if on_batch_success:
                    on_batch_success(batch_num, total_batches)
                break

            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    logger.error(f"❌ Failed batch {batch_num} permanently: {e}")
                    failed_count += len(batch)
                    failed_batches.append(batch_num)

                    if on_batch_error:
                        on_batch_error(batch_num, e)
                    break

                delay = base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, 1)
                sleep_time = delay + jitter

                logger.warning(
                    f"⚠️ Rate limit or transient error on batch {batch_num}. "
                    f"Retry {attempt}/{max_retries} in {sleep_time:.1f}s - Error: {e}"
                )
                time.sleep(sleep_time)

    return {
        "success_count": success_count,
        "failed_count": failed_count,
        "failed_batches": failed_batches,
    }


def retry_with_backoff(
    func: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """
    Execute a function with exponential backoff retry on exceptions.

    Args:
        func: Callable to execute
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s - Error: {e}"
                )
                time.sleep(delay)

    raise last_exception  # type: ignore
