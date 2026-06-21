import uuid
from uuid import uuid4
from typing import Any, List
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException, status
# Keeping your imports, though sqlmodel is unused here since you are using raw SQLAlchemy queries
from sqlmodel import col, delete, func, select 
from .login import oauth2_scheme
from databases.database import get_db
from databases.utils import verify, hash
from databases.oauth2 import get_current_user
from databases.models import User
from typing import List
from databases.schemas import UserCreate, UserOut, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/",response_model=List[UserOut], status_code=status.HTTP_200_OK)
def get_all_users(
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)):
    users = db.query(User).all()
    
    if not users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="There is no Registered User yet"
        )
    return users 


@router.post("/", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    new_user: UserCreate, 
    db: Session = Depends(get_db)
):
    existing_user = db.query(User).filter(User.email == new_user.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User with this email already exists"
        )
    user_data = new_user.model_dump()
    user_data["password"] = hash(user_data["password"])
    user_data["id"] = uuid4()
    db_user = User(**user_data)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# @router.patch(user_data: UserUpdate, db: Session = Depends(get_db))
