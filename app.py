import os
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = FastAPI(title="ISG Finn Key", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# ANA SAYFA (UI)
# -------------------------
@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

# -------------------------
# SOHBET
# -------------------------
@app.post("/chat")
async def chat(payload: dict):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sen bir İş Güvenliği Uzmanısın. Türkçe cevap ver."},
            *payload["messages"]
        ]
    )
    return {"reply": resp.choices[0].message.content}

# -------------------------
# FOTOĞRAF → FINN KEY
# -------------------------
@app.post("/photo-finnkey")
async def photo_finnkey(
    file: UploadFile = File(...),
    note: str = Form("")
):
    img = await file.read()
    b64 = base64.b64encode(img).decode()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Not: {note}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        ]
    )
    return {"result": resp.choices[0].message.content}
