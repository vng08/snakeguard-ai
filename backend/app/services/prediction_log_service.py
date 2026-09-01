from sqlalchemy.orm import Session
from backend.app.db.models import PredictionLog, SnakeSpecies

def save_prediction_log(db: Session, result: dict | None, image_url: str, model_version: str):
    log = PredictionLog(image_url=image_url, model_version=model_version)

    if result:
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