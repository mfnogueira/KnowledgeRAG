from dotenv import load_dotenv
import os

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def retrieve_context(query, db_path):
    """
    Recupera o contexto mais relevante da base de conhecimento para a consulta fornecida.

    Args:
        query (str): A pergunta ou consulta para a qual o contexto deve ser recuperado.
        db_path (str): O caminho para a base de conhecimento salva localmente.

    Returns:
        str: O conteúdo do primeiro documento mais semelhante ao `query` da base de dados.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Carrega a base de conhecimento armazenada no diretório especificado
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    # Realiza a busca de similaridade para encontrar o contexto relevante
    docs = db.similarity_search(query)
    
    # Retorna o conteúdo da primeira correspondência encontrada
    if docs:
        retrieved_context = docs[0].page_content
    else:
        retrieved_context = "Nenhum contexto encontrado para a consulta."

    return str(retrieved_context)

def create_augmeted(query, db_path):
    """
    Cria um prompt de entrada aumentado com o contexto relevante para a consulta.

    Args:
        query (str): A pergunta ou consulta a ser respondida.
        db_path (str): O caminho para a base de conhecimento salva localmente.

    Returns:
        str: O prompt formatado contendo o contexto recuperado e a pergunta original.
    """
    retrieved_context = retrieve_context(query, db_path)

    # Formata o prompt incluindo a consulta e o contexto relevante
    augmented_prompt = f"""
    Given the context below answer the question.

    Question: {query} 

    Context: {retrieved_context}

    Remember to answer only based on the context provided and not from /
    any other source. 

    If the question cannot be answered based on the provided context, /
    say I don’t know.
    """

    return str(augmented_prompt)

def create_rag(query, db_path):
    """
    Gera uma resposta usando um modelo de linguagem com base no prompt aumentado.

    Args:
        query (str): A pergunta ou consulta a ser respondida.
        db_path (str): O caminho para a base de conhecimento salva localmente.

    Returns:
        str: A resposta gerada pelo modelo de linguagem com base no contexto.
    """
    # Gera o prompt aumentado com contexto
    augmented_prompt = create_augmeted(query, db_path)

    # Importar a biblioteca OpenAI para gerar respostas
    from openai import OpenAI

    # Instanciar o cliente OpenAI
    client = OpenAI()

    # Enviar o prompt para o modelo de linguagem para gerar a resposta
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": augmented_prompt}
        ]
    )

    # Extrair a resposta do objeto de resposta retornado pela API
    answer = response.choices[0].message.content

    return answer

# Sistema de perguntas e respostas
create_rag("What was Virat Kohli's achievement in the Cup?", "./Assets/Data")
