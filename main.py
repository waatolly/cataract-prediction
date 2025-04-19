from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import io
from PIL import Image
import tflite_runtime.interpreter as tflite

import os
import pathlib
import subprocess
import zipfile
from fastapi import FastAPI, File, UploadFile

MODEL_FILE = "cataract.tflite"          # 解壓後真正要載入的檔名
ZIP_FILE   = "cataract.7z"      # 你的壓縮檔名，若用 .7z 請改成對應檔
# 若你用 .7z，請把 ZIP_FILE 改成 "cataract.tflite.7z"

def ensure_model():
    """
    若專案根目錄沒有 cataract.tflite，
    就從 cataract.tflite.zip（或 .7z）解壓縮出來。
    """
    if pathlib.Path(MODEL_FILE).exists():
        return  # 已經有模型檔，直接回傳

    if not pathlib.Path(ZIP_FILE).exists():
        raise FileNotFoundError(
            f"找不到壓縮檔 {ZIP_FILE}，無法解壓模型")

    print(f"[INFO] 正在解壓 {ZIP_FILE} ...")
    if ZIP_FILE.endswith(".7z"):
        with zipfile.ZipFile(ZIP_FILE, "r") as zf:
            zf.extractall(".")          # 解到當前資料夾
    else:
        # 假設是 .7z，呼叫系統的 7z 指令
        # 需要系統已安裝 7-Zip CLI (Windows)或 p7zip (Linux/macOS)
        subprocess.run(["7z", "x", "-y", ZIP_FILE], check=True)
    print(f"[INFO] 解壓完成！")

# ↓↓↓ 先確保模型已就緒
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
