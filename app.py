import os
import base64
from typing import List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY yok")

client = OpenAI(api_key=api_key)

TEXT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"

app = FastAPI(title="ISG Finn Key", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_CHAT = """Sen bir İş Güvenliği Uzmanı asistanısın.
Türkçe cevap ver.
Sahaya uygun, net ve uygulanabilir öneriler sun.
"""

Role = Literal["user", "assistant"]

class Msg(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    messages: List[Msg]

@app.post("/chat")
def chat(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_CHAT}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages[-20:]]

    resp = client.responses.create(
        model=TEXT_MODEL,
        input=messages,
        max_output_tokens=800
    )
    return {"reply": resp.output_text.strip()}

SYSTEM_FINNKEY = """Fotoğrafı analiz et.
FINN KEY maddeleri, risk skoru ve aksiyonlar üret.
"""

FOOTER = "\n\n---\nHazırlayan: Fatih Akdeniz"

@app.post("/photo-finnkey")
async def photo_finnkey(
    file: UploadFile = File(...),
    note: str = Form("")
):
    image_bytes = await file.read()
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:{file.content_type};base64,{b64}"

    resp = client.responses.create(
        model=VISION_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_FINNKEY},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": note},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        max_output_tokens=900
    )

    return {"result": resp.output_text.strip() + FOOTER}

@app.get("/")
def home():
    return FileResponse("index.html")
