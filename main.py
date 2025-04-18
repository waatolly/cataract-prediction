from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import io
from PIL import Image
import tflite_runtime.interpreter as tflite

import pathlib, subprocess
if not pathlib.Path("cataract.tflite").exists():
    subprocess.run(["7z", "x", "cataract.tflite.7z"], check=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 載入TFLite模型
interpreter = tflite.Interpreter(model_path="cataract.tflite")
interpreter.allocate_tensors()

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
