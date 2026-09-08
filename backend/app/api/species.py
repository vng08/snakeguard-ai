from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db
from backend.app.db.models import SnakeSpecies
from backend.app.schemas.snake import SnakeSpeciesResponse


router = APIRouter(tags=["Species"])


@router.get("/species", response_model=list[SnakeSpeciesResponse])
def get_species(db: Session = Depends(get_db)):
    """Lấy toàn bộ danh sách loài rắn trong database."""
    return db.query(SnakeSpecies).all()