from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db
from backend.app.schemas.prediction import PredictionLogResponse
from backend.app.services.prediction.prediction_log_service import delete_all_prediction_logs, delete_prediction_log, get_prediction_logs


router = APIRouter(prefix="/prediction-logs", tags=["Prediction Logs"])


@router.get("", response_model=list[PredictionLogResponse])
def get_logs(db: Session = Depends(get_db)):
    """Lấy toàn bộ prediction logs, mới nhất trước."""
    return get_prediction_logs(db)


@router.delete("/{log_id}")
def delete_log(log_id: int, db: Session = Depends(get_db)):
    """Xoá một prediction log theo ID."""
    if not delete_prediction_log(db, log_id):
        raise HTTPException(status_code=404, detail="Prediction log not found.")

    return {"message": "Prediction log deleted successfully.", "id": log_id}


@router.delete("")
def delete_all_logs(db: Session = Depends(get_db)):
    """Xoá toàn bộ prediction logs."""
    deleted_count = delete_all_prediction_logs(db)
    return {
        "message": "All prediction logs deleted successfully.",
        "deleted_count": deleted_count,
    }