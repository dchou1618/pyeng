from fastapi import FastAPI
import uvicorn
app = FastAPI(title="model_api", description="API for concurrent tensorflow model training",
version="1.0.0")


@app.get("/")
async def root():
    return {"message": "Success"}




if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=8000)
