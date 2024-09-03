from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = FastAPI()

CLASS_NAMES = [
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy'
]

model = tf.keras.models.load_model(r"E:\Grad-project\modelaaa\my_model.h5")

def read_image_from_path(image_path, target_size=(256, 256)) -> np.ndarray:
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The file at {image_path} does not exist")
        
        image = Image.open(image_path)
        image = image.resize(target_size)  # Resize the image
        image = np.array(image)
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading image from path: {e}")

class ImagePath(BaseModel):
    path: str

@app.post("/predict")
async def predict(image_data: ImagePath):
    image_path = image_data.path

    if not image_path:
        raise HTTPException(status_code=400, detail="Path parameter is required")

    try:
        image = read_image_from_path(image_path)
        img_batch = np.expand_dims(image, 0)

        predictions = model.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return {
            'class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880)
