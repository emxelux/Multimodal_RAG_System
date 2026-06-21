from fastapi import FastAPI, Depends
from app.routes import users,login
from databases.database import Base, engine


Base.metadata.create_all(bind=engine)


app = FastAPI(title="MY RAG SYSTEM")



app.include_router(login.router)
app.include_router(users.router)



@app.get("/")
def test_api():
    return {"status": "The Backend API is working succesfully"}

