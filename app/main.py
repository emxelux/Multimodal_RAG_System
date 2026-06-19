from fastapi import FastAPI, HTTPException, status, APIRouter


app = FastAPI(
    tag = ["Full Backend Infrastructure"]
)

@app.get("/")
def test_connection():
    return {"response": "This API is working perfectly"}


