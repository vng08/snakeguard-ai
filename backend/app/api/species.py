from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db
from backend.app.db.models import SnakeSpecies
from backend.app.schemas.snake import SnakeSpeciesCreate, SnakeSpeciesResponse


router = APIRouter(prefix="/species", tags=["Species"],)

@router.post("", response_model=SnakeSpeciesResponse)
def create_species(data: SnakeSpeciesCreate, db: Session = Depends(get_db)):
    species = SnakeSpecies(**data.model_dump())

    db.add(species)
    db.commit()
    db.refresh(species)

    return species


@router.get("", response_model=list[SnakeSpeciesResponse])
def get_species(db: Session = Depends(get_db)):
    return db.query(SnakeSpecies).all()