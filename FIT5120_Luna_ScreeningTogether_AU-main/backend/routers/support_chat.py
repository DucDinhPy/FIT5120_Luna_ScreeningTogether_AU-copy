import logging

from fastapi import APIRouter, HTTPException, Response
from groq import APIConnectionError, APIError, APIStatusError

from config import settings
from services.chatbot.support_chat import ChatRequest, ChatResponse, answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/support-chat", tags=["support-chat"])


@router.post("", response_model=ChatResponse)
def support_chat(request: ChatRequest, response: Response) -> ChatResponse:
    key = settings.groq_api_key.get_secret_value().strip() if settings.groq_api_key else ""
    model = settings.groq_chat_model.strip()
    if not model or not key:
        raise HTTPException(status_code=503, detail="Support chat is unavailable")

    try:
        answer = answer_question(request, model=model, api_key=key)
    except APIStatusError as exc:
        logger.warning("Support chat Groq error: HTTP %s", exc.status_code)
        if exc.status_code == 429 or exc.status_code >= 500:
            raise HTTPException(status_code=503, detail="Support chat is temporarily unavailable") from exc
        raise HTTPException(status_code=502, detail="Support chat could not answer") from exc
    except APIConnectionError as exc:
        logger.warning("Support chat Groq connection failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Support chat is temporarily unavailable") from exc
    except (APIError, RuntimeError) as exc:
        logger.warning("Support chat failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Support chat could not answer") from exc
    response.headers["Cache-Control"] = "no-store"
    return answer
