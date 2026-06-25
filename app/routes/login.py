from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from databases.database import get_db
from databases.models import User
from databases.oauth2 import create_access_token
from databases.utils import verify

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


router = APIRouter(tags=["login"], prefix="/login")


@router.post("/")
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()],db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify(user.password, form_data.password):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Invalid Credentials")
    
    user_dict = {"user_id": str(user.id)}
    jwt_token = create_access_token(user_dict)
    return {
        "access_token": jwt_token,
        "token_type": "bearer"
    }
    
