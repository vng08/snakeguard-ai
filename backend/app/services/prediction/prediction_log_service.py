from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.app.db.models import PredictionLog, SnakeSpecies
from backend.app.services.media.storage_service import delete_all_prediction_images, delete_prediction_image


def save_prediction_log(db: Session, result: dict | None, image_url: str, model_version: str) -> PredictionLog:
    """Lưu kết quả prediction vào database."""
    log = PredictionLog(image_url=image_url, model_version=model_version)

    if result and result.get("predictions"):
        top1 = result["predictions"][0]
        species = db.query(SnakeSpecies).filter(SnakeSpecies.binomial_name == top1["binomial_name"]).first()

        log.bbox = result["bbox"]
        log.detection_confidence = result["detection_confidence"]
        log.predicted_species_id = species.id if species else None
        log.confidence = top1["confidence"]
        log.top_k_predictions = result["predictions"]

    db.add(log)
    db.commit()
    db.refresh(log)
    return log

def get_prediction_logs(db: Session) -> list[PredictionLog]:
    """Lấy toàn bộ prediction logs, sắp xếp mới nhất trước."""
    return db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).all()

def delete_prediction_log(db: Session, log_id: int) -> bool:
    """Xoá một prediction log và ảnh tương ứng."""
    log = db.query(PredictionLog).filter(PredictionLog.id == log_id).first()

    if not log:
        return False

    delete_prediction_image(log.image_url)

    db.delete(log)
    db.commit()
    return True


def delete_all_prediction_logs(db: Session) -> int:
    """Xoá toàn bộ prediction logs, ảnh và reset ID về 1."""
    deleted_count = db.query(PredictionLog).count()

    delete_all_prediction_images()
    db.execute(text("TRUNCATE TABLE prediction_logs RESTART IDENTITY"))
    db.commit()

    return deleted_count