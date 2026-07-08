from fastapi import Depends, Body, HTTPException
from loguru import logger
from schemas import TextModelRequest
from scraper import extract_urls, fetch_all
from rag.service import vector_service
from rag.transformer import embed

async def get_urls_content(body: TextModelRequest = Body(...)) -> str:
    """
    Function that sends the body from the request to the extract_urls
    to extract the url and use the fetch_all to extract all the content of a web page
    """
    urls = extract_urls(body.user_input)
    if urls:
        try:
            urls_content = await fetch_all(urls)
            return urls_content
        except Exception as e:
            logger.warning(f"Faild to fetch one or several url content. Error: {e}")
    return ""

async def get_rag_content(body: TextModelRequest = Body(...)) -> str:
    """
    Function that retrieves relevant content about the user query
    and returns it
    """
    embedding_vector = await embed(body.user_input)
    rag_content = await vector_service.search(
        "knowledge_base",
        embedding_vector,
        3,
        0.7,
    )
    logger.debug(f"rag_content -> {rag_content}")
    rag_content_str = "\n".join([c.payload["original_text"] for c in rag_content]) if rag_content else ""
    return rag_content_str

def construct_prompt(
    body: TextModelRequest = Body(...),
    urls_content: str = Depends(get_urls_content),
    rag_content: str = Depends(get_rag_content)
) -> str:
    """
    A function that creates a prompt using the user's message and the urls
    present in it. If asked about the uploaded pdf appends the retrieved context.
    Handles empty url_contents
    """
    try:
        if rag_content:
            prompt = f"User Message: {body.user_input}\nRetrieved Content: {urls_content}"
            logger.debug(f"prompt -> {prompt}")
            return prompt
        if urls_content:
            prompt = f"User Message: {body.user_input}\nUrls Content: {urls_content}"
            logger.debug(f"prompt -> {prompt}")
            return prompt
        prompt = f"User Message: {body.user_input}"
        logger.debug(f"prompt -> {prompt}")
        return prompt
    except Exception as e:
        logger.warning(f"Faild to construct prompt. Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to construct prompt")