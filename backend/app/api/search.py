from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db
from backend.app.schemas.search import SearchByDescriptionRequest, SearchByDescriptionResponse
from backend.app.services.search.description_analyzer import analyze_description
from backend.app.services.search.hybrid_retrieval_service import hybrid_search


router = APIRouter(tags=["Search"])


@router.post("/search-by-description", response_model=SearchByDescriptionResponse)
def search_by_description(payload: SearchByDescriptionRequest, db: Session = Depends(get_db)):
    """Tìm species từ mô tả của người dùng."""
    analysis = analyze_description(payload.description)

    if analysis.route == "unspecified":
        raise HTTPException(
            status_code=400,
            detail="Nội dung không chứa đủ đặc điểm để nhận dạng rắn. Hãy cung cấp chính xác thông tin nhận diện.",
        )

    results = hybrid_search(db, payload.description, analysis.focus,  payload.search_mode, payload.top_k,)
    return {"description": payload.description, "results": results}