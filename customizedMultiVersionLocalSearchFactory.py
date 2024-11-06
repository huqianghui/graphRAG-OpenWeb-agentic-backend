import asyncio
import logging
import os

import pandas as pd
import tiktoken

# GraphRAG related imports
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

from customizedMultiVersionLocalSearch import MultiVersionLocalSearch

DOCUMENT_TABLE = "create_base_documents.parquet"


from dotenv import load_dotenv

from appConfig import (
    COMMUNITY_LEVEL,
    COMMUNITY_REPORT_TABLE,
    COMMUNITY_TABLE,
    COVARIATE_TABLE,
    ENTITY_EMBEDDING_TABLE,
    ENTITY_TABLE,
    INPUT_DIR,
    LANCEDB_URI,
    RELATIONSHIP_TABLE,
    TEXT_UNIT_TABLE,
)

load_dotenv()


log = logging.getLogger(__name__)

async def getFilteredDocumentIdsByPattern(pattern: str) -> list:
    # Load the documents
    documents = pd.read_parquet(os.path.join(INPUT_DIR, DOCUMENT_TABLE))
    documents = documents[documents['title'].str.contains(pattern, case=False, na=False)]
    return documents['id'].unique().tolist()

async def setup_llm_and_embedder():
    """
    Set up Language Model (LLM) and embedding model
    """
    log.info("Setting up LLM and embedder")

    # Get API keys and base URLs
    api_key = os.environ.get("GRAPHRAG_API_KEY", "YOUR_API_KEY")
    api_key_embedding = os.environ.get("GRAPHRAG_API_KEY_EMBEDDING", api_key)
    api_base = os.environ.get("API_BASE", "https://api.openai.com/v1")
    api_base_embedding = os.environ.get("API_BASE_EMBEDDING", "https://api.openai.com/v1")

    # Get model names
    llm_model = os.environ.get("GRAPHRAG_LLM_MODEL", "gpt-4o")
    embedding_model = os.environ.get("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small")

    # Check if API key exists
    if api_key == "YOUR_API_KEY":
        log.error("Valid GRAPHRAG_API_KEY not found in environment variables")
        raise ValueError("GRAPHRAG_API_KEY is not set correctly")

    # Initialize ChatOpenAI instance
    llm = ChatOpenAI(
        api_key=api_key,
        api_base=api_base,
        model=llm_model,
        deployment_name=llm_model,
        api_type=OpenaiApiType.AzureOpenAI,
        max_retries=20,
    )

    # Initialize token encoder
    token_encoder = tiktoken.get_encoding("cl100k_base")

    # Initialize text embedding model
    text_embedder = OpenAIEmbedding(
        api_key=api_key_embedding,
        api_base=api_base_embedding,
        api_type=OpenaiApiType.AzureOpenAI,
        deployment_name=embedding_model,
        max_retries=20,
    )


    log.info("LLM and embedder setup complete")
    return llm, token_encoder, text_embedder

async def load_context_by_fileNameParttern(fileNameParttern:str):
    """
    Load context data including entities, relationships, reports, text units, and covariates
    """
    log.info("Loading context data")
    try:
        # rebuild context_builder by 
        # step 1) get docIds by fileNameParttern
        docIds = await getFilteredDocumentIdsByPattern(fileNameParttern)
        unique_docIds_set = set(docIds)

        def is_valid_text_unit_row(document_ids):
            return any(doc_id in unique_docIds_set for doc_id in document_ids)

        # step 2) get and filter text unit by docIds
        text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
        filtered_text_unit_df = text_unit_df[text_unit_df['document_ids'].apply(is_valid_text_unit_row)]
        filtered_text_unit_ids = filtered_text_unit_df['id'].tolist()
        unique_filtered_text_unit_ids_set = set(filtered_text_unit_ids)
        text_units = read_indexer_text_units(filtered_text_unit_df)

        def is_valid_entity_embedding_row(filtered_text_unit_ids):
            return any(text_uint_id in unique_filtered_text_unit_ids_set for text_uint_id in filtered_text_unit_ids)
        
        # step 3) get and filter entities by text units ids 
        entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
        filtered_entity_embedding_df = entity_embedding_df[entity_embedding_df['text_unit_ids'].apply(is_valid_entity_embedding_row)]
        filtered_entity_embedding_ids = filtered_entity_embedding_df['id'].tolist()
        
        # step 4) get and filter node  by  entities 
        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        filtered_entity_df = entity_df[entity_df['id'].isin(filtered_entity_embedding_ids)]

        entities = read_indexer_entities(filtered_entity_df, filtered_entity_embedding_df, COMMUNITY_LEVEL)
        description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
        description_embedding_store.connect(db_uri=LANCEDB_URI)
        store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)

        # step 5) get and filter relationship by entity ids
        relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
        filtered_relationship_df = relationship_df[relationship_df['text_unit_ids'].isin(filtered_text_unit_ids)]
        relationships = read_indexer_relationships(filtered_relationship_df)

        # step 6) get and filter community reorts by text unit
        def is_valid_community_row(filtered_text_unit_ids):
            return all(text_uint_id in unique_filtered_text_unit_ids_set for text_uint_id in filtered_text_unit_ids)
        
        community_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_TABLE}.parquet")
        filtered_community_df = community_df[community_df['text_unit_ids'].apply(is_valid_community_row)]
        filtered_community_df_ids = filtered_community_df['id'].tolist()

        # reports including different documents, so we cannot use them in this case.
        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        filtered_report_df = report_df[report_df['id'].isin(filtered_community_df_ids)]

        reports = read_indexer_reports(filtered_report_df, filtered_entity_df, COMMUNITY_LEVEL)


        # TODO: without covariates
        # covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")
        # claims = read_indexer_covariates(covariate_df).filter_by_ids(docIds)
        # log.info(f"Number of claim records: {len(claims)}")
        covariates = {"claims": []}

        log.info("Context data loading complete")
        return entities, relationships, reports, text_units, description_embedding_store, covariates
    except Exception as e:
        log.error(f"Error loading context data: {str(e)}")
        raise

async def setup_search_engines_by_fileNameParttern(fileNameParttern:str):
    """
    Set up local search engines
    """
    log.info("Setting up search engines")

    llm, token_encoder, text_embedder = await setup_llm_and_embedder()
    entities, relationships, reports, text_units, description_embedding_store, covariates = await load_context_by_fileNameParttern(fileNameParttern)

    # Set up local search engine
    local_context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.3,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "use_community_summary" :  True,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 18_000
    }

    local_llm_params = {
        "max_tokens": 2_000,
        "temperature": 0.0,
    }

    local_search_engine = MultiVersionLocalSearch(
        llm=llm,
        context_builder=local_context_builder,
        token_encoder=token_encoder,
        llm_params=local_llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )

    log.info("Search engines setup complete")

    return local_search_engine


if __name__ == "__main__":
    # 示例调用
    local_search_engine = asyncio.run(setup_search_engines_by_fileNameParttern("v1"))
    print("contentByMulitModel: {}",local_search_engine.asearchWithMetaData(""))