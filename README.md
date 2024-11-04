# Contextual RAG and Knowledge Base Builder

Este repositório contém dois scripts principais: `knowledge_base_builder.py` e `contextual_rag.py`. O primeiro script extrai dados de uma página da web e constrói uma base de conhecimento, enquanto o segundo utiliza essa base para responder perguntas de forma contextualizada.

## Requisitos

Para executar esses scripts, você precisará das seguintes bibliotecas:

- `Python 3.x`
- `langchain`
- `langchain_openai`
- `langchain_community`
- `openai`
- `python-dotenv`
- `faiss-cpu` (ou `faiss-gpu`, dependendo da sua configuração)

Você pode instalar as dependências necessárias usando o seguinte comando:

```bash
pip install langchain langchain_openai langchain_community openai python-dotenv faiss-cpu
```
## Configuração do Ambiente
1- Crie um arquivo chamado `.env` na raiz do seu projeto.

2- Adicione a sua chave da API OpenAI no arquivo `.env`:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
3- Adicione a sua chave da API langchain no arquivo `.env`:
```bash
LANGCHAIN_API_KEY=lanchain_api_key_here
```

## Scripts
1. `knowledge_base_builder.py`
Este script extrai informações de uma página da web e cria uma base de conhecimento armazenada localmente utilizando o FAISS para pesquisa eficiente.

## Uso
```bash
python knowledge_base_builder.py
```

## Como funciona
    * O script utiliza um carregador HTML assíncrono para obter dados da página da web especificada.
    * Em seguida, transforma o HTML em texto legível.
    * Divide o texto em chunks e cria embeddings utilizando a API OpenAI.
    * Por fim, salva a base de conhecimento em um diretório local.
2. `contextual_rag.py`
Este script utiliza a base de conhecimento criada pelo `knowledge_base_builder.py` para responder a perguntas de forma contextualizada.

Uso
```bash
python contextual_rag.py
```

### Como funciona
    * O script carrega a base de conhecimento armazenada.
    * Realiza uma busca de similaridade para encontrar o contexto relevante para a pergunta.
    * Cria um prompt aumentado com o contexto e a pergunta.
    * Usa a API OpenAI para gerar uma resposta com base no prompt.

## Exemplo de Uso
Para usar o sistema de perguntas e respostas, basta chamar a função `create_rag` passando a pergunta desejada e o caminho para a base de dados:
```bash
create_rag("What was Virat Kohli's achievement in the Cup?", "./Assets/Data")
```