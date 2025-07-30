from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predict import predict_image_from_path
import tempfile
import os
import shutil

app = FastAPI(title="Plant Disease Diagnosis API")

# إعداد CORS للسماح بالاتصال من Flutter Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # لاحقًا ضع رابط تطبيقك فقط
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # التحقق من نوع الملف
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="الملف المرفوع ليس صورة")

    # استخدام ملف مؤقت لتخزين الصورة
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = predict_image_from_path(tmp_path)
    finally:
        # حذف الملف المؤقت حتى لو حدث خطأ
        os.remove(tmp_path)

    return result

@app.get("/")
async def root():
    return {"message": "Plant Disease Diagnosis API is running ✅"}
