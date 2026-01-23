"""
Prompt templates for RAG chains.
All templates extracted from notebook2.md.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# System prompt for RAG responses
SYSTEM_PROMPT = """Responda usando exclusivamente o conteúdo fornecido, seja conciso em sua resposta.

Context:
{context}"""


def get_rag_prompt() -> ChatPromptTemplate:
    """
    Get the main RAG prompt template.

    Uses system message with context and human message with query.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )


# Query rewriter prompt template
REWRITER_PROMPT_TEMPLATE = """
Gere consulta de pesquisa para o banco de dados de vetores (Vector DB) a partir de uma pergunta do usuário,
permitindo uma resposta mais precisa por meio da busca semantica.
Basta retornar a consulta revisada do Vector DB, entre aspas.

Pergunta do usuário: {user_question}

Consulta revisada do Vector DB:
"""


def get_rewriter_prompt() -> PromptTemplate:
    """Get the query rewriter prompt template."""
    return PromptTemplate.from_template(REWRITER_PROMPT_TEMPLATE)


# Multi-query prompt template
MULTI_QUERY_PROMPT_TEMPLATE = """Você é um assistente de modelo de linguagem de IA. Sua tarefa é gerar cinco
versões diferentes da pergunta do usuário para recuperar documentos relevantes de um banco de dados vetorial.
Ao gerar múltiplas perspectivas sobre a pergunta do usuário, seu objetivo é ajudar
o usuário a superar algumas das limitações da busca por similaridade baseada em distância.
Forneça estas perguntas alternativas separadas por quebras de linha.
Pergunta original: {question}"""


def get_multi_query_prompt() -> PromptTemplate:
    """Get the multi-query generation prompt template."""
    return PromptTemplate.from_template(MULTI_QUERY_PROMPT_TEMPLATE)


# HyDE (Hypothetical Document Embeddings) prompt template
HYDE_PROMPT_TEMPLATE = """
Escreva um paragrafo que possa responder a pergunta abaixo, relacionada a operadora de cartão de crédito. Não adicione informações adicionais de racionalização ou explicação.
Pergunta: {user_question}
Responda:
"""


def get_hyde_prompt() -> PromptTemplate:
    """Get the HyDE (Hypothetical Document Embeddings) prompt template."""
    return PromptTemplate.from_template(HYDE_PROMPT_TEMPLATE)


# Synthetic QA generation prompt template
SYNTHETIC_QA_PROMPT_TEMPLATE = """Você é um auditor de seguros. Com base no trecho do manual abaixo, \
crie uma pergunta técnica que um cliente poderia fazer e a resposta correta.

Trecho: {context}

{format_instructions}"""


def get_synthetic_qa_prompt() -> ChatPromptTemplate:
    """Get the synthetic Q/A generation prompt template."""
    return ChatPromptTemplate.from_template(SYNTHETIC_QA_PROMPT_TEMPLATE)
