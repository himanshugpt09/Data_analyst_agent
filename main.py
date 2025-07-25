from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from agent import handle_question  # you'll write this next

app = FastAPI()

@app.post("/api/")
async def process_task(file: UploadFile = File(...)):
    contents = await file.read()
    question = contents.decode("utf-8")
    
    try:
        response = await handle_question(question)
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
