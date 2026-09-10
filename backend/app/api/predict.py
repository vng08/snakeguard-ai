from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db
from backend.app.core.config import settings
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.media.image_service import decode_image
from backend.app.services.prediction.prediction_service import PredictionService

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_snake(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Nhận request ảnh và chuyển xử lý prediction cho service."""
    try:
        # Kiểm tra MIME type ở tầng API
        if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Unsupported image type. Allowed: JPG, JPEG, PNG, WEBP, BMP, TIFF.",
            )

        image = decode_image(await file.read())
        service = PredictionService(request.app.state.pipeline)

        return service.predict(db, image, top_k=3)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc