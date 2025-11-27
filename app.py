from fastapi import FastAPI
from pydantic import BaseModel
from ocr_main import OCRProcessor

app = FastAPI()
processor = OCRProcessor()

class OCRRequest(BaseModel):
    image_path: str
    edition_date: str | None = None
    expected_lang: str | None = None

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/process-image")
def process_image(req: OCRRequest):
    res, err = processor.process_image(req.image_path, req.edition_date, req.expected_lang)
    if err:
        return {"success": False, "error": err}
    return {"success": True, "data": res}
