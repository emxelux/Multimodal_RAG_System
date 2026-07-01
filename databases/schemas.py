from pydantic import BaseModel, EmailStr
from typing import Optional, Any
# from sqlmodel import UUID
from uuid import UUID,uuid4

class UserBase(BaseModel):
    first_name: str
    last_name:str
    email: EmailStr
    phone_number: str
    is_verified: Optional[bool] = False


class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    created_at: Any

class UserUpdate(BaseModel):
    first_name: Optional[str]
    last_name: Optional[str]
    email:Optional[EmailStr]
    phone_number: Optional[str]


class TokenData(BaseModel):
    id:str


class QueryIn(BaseModel):
    query:str
    document_id:str
    history: Optional[list[dict[str, str]]] = None


class DocumentIn(BaseModel):
    document_id: str
    documment_hash:str
    