from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models, layers

app=FastAPI()

BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=50
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3


Model = tf.keras.models.load_model("Tomato_model_version_1.keras")

CLASS_NAMES = ['Tomato_Bacterial_spot','Tomato_Early_blight',
 'Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot', 
 'Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

@app.get("/ping")
async def ping():
    return " Helo I am alive "


def read_image(data) ->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile =File(...)
):
    image = read_image( await file.read())
    image=np.expand_dims(image,0)
    predictions = Model.predict(image)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }



if __name__ =="__main__":
    uvicorn.run(app, host='localhost' , port=8000)


