from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import io
from PIL import Image
import tflite_runtime.interpreter as tflite
import pathlib
import py7zr
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
import tflite_runtime.interpreter as tflite   # 若用完整 tensorflow 則改 import tensorflow as tf


MODEL_FILE = "cataract.tflite"
ARCHIVE_FILE = "cataract.7z"

def ensure_model():
    """若本地沒有 cataract.tflite，便從 cataract.7z 解壓。"""
    if pathlib.Path(MODEL_FILE).exists():
        return

    if not pathlib.Path(ARCHIVE_FILE).exists():
        raise FileNotFoundError(f"找不到 {ARCHIVE_FILE}，無法解壓模型")

    print(f"[INFO] 解壓 {ARCHIVE_FILE} → {MODEL_FILE} ...")
    with py7zr.SevenZipFile(ARCHIVE_FILE, mode="r") as z:
        z.extractall()      # 解到目前目錄
    print("[INFO] 解壓完成！")

ensure_model()

# -----------------------------------------
# 以下原本載入 TFLite Interpreter 的程式不變
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path=MODEL_FILE)
interpreter.allocate_tensors()
# -----------------------------------------

app = FastAPI()
templates = Jinja2Templates(directory="templates")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 讀取與前處理影像
    content = await file.read()
    image = Image.open(io.BytesIO(content)).resize((512, 512))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction, axis=1)[0]

    class_mapping = {0: "正常", 1: "不正常", 2: "術後"}
    result = class_mapping[predicted_class]

    return {"prediction": result}
