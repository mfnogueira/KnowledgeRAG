from dotenv import load_dotenv
import os
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

def extract_transform_data(url: str):
    """
    Extrai dados de uma página da web e os transforma em texto.

    Args:
        url (str): URL da página web para extração.

    Returns:
        list: Lista de documentos transformados em texto.
    """
    # Instancia o carregador assíncrono e carrega a página da web
    loader = AsyncHtmlLoader(url)
    data = loader.load()

    # Converte HTML extraído para texto simples
    html2text = Html2TextTransformer()
    return html2text.transform_documents(data)

def chunk_text(text: str, separator: str = "\n", chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Divide o texto em pedaços para armazenamento eficiente.

    Args:
        text (str): Texto a ser dividido.
        separator (str): Caractere usado para separar o texto.
        chunk_size (int): Tamanho máximo de cada pedaço em caracteres.
        chunk_overlap (int): Número de caracteres sobrepostos entre pedaços.

    Returns:
        list: Lista de pedaços de texto.
    """
    text_splitter = CharacterTextSplitter(separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents([text])

def create_and_save_knowledge_base(chunks, model_name: str, save_path: str):
    """
    Cria uma base de conhecimento a partir dos pedaços de texto e salva localmente.

    Args:
        chunks (list): Lista de pedaços de texto para armazenar.
        model_name (str): Nome do modelo de embeddings a ser usado.
        save_path (str): Caminho para salvar a base de conhecimento.

    Returns:
        None
    """
    # Instancia o modelo de embeddings com a chave da API carregada do .env
    embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Cria e salva a base de conhecimento
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(save_path)

if __name__ == "__main__":
    # URL da página da qual queremos extrair informações
    url = "https://en.wikipedia.org/wiki/2024_ICC_Men%27s_T20_World_Cup"

    # Etapa 1: Extração e transformação de dados
    data_transformed = extract_transform_data(url)

    # Etapa 2: Divisão do texto em pedaços
    chunks = chunk_text(data_transformed[0].page_content)

    # Etapa 3: Criação e salvamento da base de conhecimento
    create_and_save_knowledge_base(chunks, model_name="text-embedding-3-large", save_path="./Assets/Data")
