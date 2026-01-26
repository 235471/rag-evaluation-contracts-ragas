"""
Synthetic Q/A generation module.
Generates question/answer pairs from document chunks for evaluation.
"""

import random
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.app.rag.prompts import get_synthetic_qa_prompt
from src.app.logging_conf import get_logger

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = get_logger(__name__)


class SyntheticQA(BaseModel):
    """Schema for synthetic Q/A pairs."""

    question: str = Field(description="Uma pergunta baseada estritamente no contexto")
    ground_truth: str = Field(description="A resposta exata encontrada no contexto")


def generate_synthetic_qa(
    chunks: List["Document"],
    llm=None,
    sample_size: int = 20,
    random_seed: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Generate synthetic Q/A pairs from document chunks.

    Args:
        chunks: List of Document chunks
        llm: LLM for generation (defaults to Gemini gemma-3-27b-it)
        sample_size: Number of chunks to sample
        random_seed: Random seed for reproducibility

    Returns:
        List of dicts with 'question' and 'ground_truth' keys
    """
    if random_seed is not None:
        random.seed(random_seed)

    if llm is None:
        from src.app.config import get_llm

        # Use Gemini for synthetic generation as in notebook2
        llm = get_llm(provider="gemini", model="gemma-3-27b-it", temperature=0.15)

    parser = JsonOutputParser(pydantic_object=SyntheticQA)
    prompt = get_synthetic_qa_prompt()

    gen_chain = prompt | llm | parser

    # Sample chunks
    actual_sample_size = min(sample_size, len(chunks))
    sample_chunks = random.sample(chunks, actual_sample_size)

    logger.info(f"Generating synthetic Q/A for {actual_sample_size} chunks")

    synthetic_dataset = []

    for i, doc in enumerate(sample_chunks):
        try:
            result = gen_chain.invoke(
                {
                    "context": doc.page_content,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            synthetic_dataset.append(result)
            logger.info(f"✅ Generated pair {i + 1}/{actual_sample_size}")

        except Exception as e:
            logger.warning(f"❌ Error on chunk {i}: {e}")
            continue

    logger.info(f"Generated {len(synthetic_dataset)} synthetic Q/A pairs")
    return synthetic_dataset


def run_synthetic_evaluation(
    qa_pairs: List[Dict[str, str]],
    rag_chain,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run RAG chain on synthetic Q/A pairs and collect results.

    Args:
        qa_pairs: List of dicts with 'question' and 'ground_truth'
        rag_chain: RAG chain that returns {query, answer, contexts}
        limit: Optional limit on number of pairs to evaluate

    Returns:
        List of result dicts with question, answer, contexts, ground_truth
    """
    test_cases = qa_pairs[:limit] if limit else qa_pairs
    results_list = []

    logger.info(f"Running RAG on {len(test_cases)} synthetic test cases")

    for i, test in enumerate(test_cases):
        logger.info(f"Executing question {i + 1}/{len(test_cases)}")

        try:
            # Invoke RAG chain
            result = rag_chain.invoke(test["question"])

            # Collect results
            results_list.append(
                {
                    "question": result.get("query", test["question"]),
                    "answer": result.get("answer", ""),
                    "contexts": result.get("contexts", []),
                    "ground_truth": test["ground_truth"],
                }
            )

        except Exception as e:
            logger.error(f"Error on question {i + 1}: {e}")
            # Still add with empty answer for tracking
            results_list.append(
                {
                    "question": test["question"],
                    "answer": f"ERROR: {e}",
                    "contexts": [],
                    "ground_truth": test["ground_truth"],
                }
            )

    logger.info(f"Completed {len(results_list)} synthetic evaluations")
    return results_list


def load_chunks_from_documents(
    documents_dir: str = "documents",
    chunk_size: int = 380,
    chunk_overlap: int = 50,
) -> List["Document"]:
    """
    Load and chunk documents from a directory.

    Uses the same tokenizer and settings as notebook2.

    Args:
        documents_dir: Path to documents directory
        chunk_size: Chunk size in tokens
        chunk_overlap: Chunk overlap in tokens

    Returns:
        List of chunked Documents
    """
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from src.app.utils.chunking import (
        get_chunking_config_from_env,
        create_text_splitter,
    )

    logger.info(f"Loading documents from: {documents_dir}")

    # Load PDFs
    loader = PyPDFDirectoryLoader(documents_dir)
    pdfs = loader.load()
    logger.info(f"Loaded {len(pdfs)} PDF pages")

    # Get chunking configuration
    chunking_config = get_chunking_config_from_env()

    # Apply overrides if provided
    if chunk_size is not None:
        chunking_config.chunk_size = chunk_size
    if chunk_overlap is not None:
        chunking_config.chunk_overlap = chunk_overlap

    logger.info(
        f"Chunking config: {chunking_config.chunk_size} tokens, "
        f"overlap: {chunking_config.chunk_overlap}"
    )

    splitter = create_text_splitter(chunking_config)
    chunks = splitter.split_documents(pdfs)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks
