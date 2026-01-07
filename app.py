import os
import base64
from typing import List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI

# =====================
# SETTINGS
# =====================

load_dotenv()

# Render'da Environment Variables içine bunu ekliyorsun:
# OPENAI_API_KEY = "..."
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY bulunamadı. Render > Environment Variables veya .env kontrol et.")

client = OpenAI(api_key=api_key)

TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="ISG Finn Key",
    version="1.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

# CORS (UI aynı domain'den geleceği için sorun olmaz, ama yine de serbest bırakıyoruz)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# PROMPTS
# =====================

SYSTEM_CHAT = """Sen bir İş Güvenliği Uzmanı asistanısın.
Türkçe cevap ver.
Sahaya uygun, net ve uygulanabilir öneriler sun.
"GPT", "OpenAI" vb. ifadeler kullanma.
"""

SYSTEM_FINNKEY = """Sen bir İş Güvenliği Uzmanı asistanısın.

Kullanıcı bir saha / şantiye fotoğrafı yüklüyor.
Fotoğrafı analiz et ve aşağıdaki formatta cevap ver:

FINN KEY (Ana Bulgular):
1) ...
2) ...
3) ...
4) ...
5) ...

- Risk Skoru: 0-100 (kısa gerekçe)
- İlk 3 Aksiyon: (hemen yapılacak net maddeler)

Emin olmadığın konuları 'olası' diye belirt.
Kişisel veri tespiti yapma.
"GPT", "OpenAI" vb. ifadeler kullanma.
"""

FOOTER = """
----------------------------
Hazırlayan: Fatih Akdeniz – İş Güvenliği Uzmanı (B)
Not: Bu çıktı yapay zeka destekli bir ön değerlendirmedir.
----------------------------
"""

# =====================
# SCHEMAS
# =====================

Role = Literal["user", "assistant"]

class Msg(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    messages: List[Msg]

# =====================
# ROUTES
# =====================

@app.get("/")
def home():
    """
    Ana sayfa: index.html serve eder.
    """
    path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(path):
        return JSONResponse(
            {"error": "index.html bulunamadı. Repo köküne index.html koymalısın (app.py ile aynı klasör)."},
            status_code=500,
        )
    return FileResponse(path)

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Body örneği:
    { "messages": [ {"role":"user","content":"..."} , {"role":"assistant","content":"..."} ] }
    """
    messages = [{"role": "system", "content": SYSTEM_CHAT}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages[-20:]]

    try:
        resp = client.responses.create(
            model=TEXT_MODEL,
            input=messages,
            max_output_tokens=800,
        )
        return {"reply": (resp.output_text or "").strip()}
    except Exception as e:
        return JSONResponse({"error": f"Chat hatası: {str(e)}"}, status_code=500)

@app.post("/photo-finnkey")
async def photo_finnkey(
    file: UploadFile = File(...),
    note: str = Form(""),
):
    """
    multipart/form-data
    file: image
    note: optional text
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            return JSONResponse({"error": "Boş dosya"}, status_code=400)

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime = file.content_type or "image/jpeg"
        data_url = f"data:{mime};base64,{b64}"

        user_text = f"Kullanıcı notu: {note or '—'}"

        resp = client.responses.create(
            model=VISION_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_FINNKEY},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {"type": "input_image", "image_url": data_url},
                    ],
                },
            ],
            max_output_tokens=900,
        )

        return {"result": (resp.output_text or "").strip() + FOOTER}

    except Exception as e:
        return JSONResponse({"error": f"Foto analiz hatası: {str(e)}"}, status_code=500)
