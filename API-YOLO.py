from ultralytics import YOLO
import os
import requests
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Load your YOLO model
model = YOLO(r"D:\yolo\best (1).pt")

def download_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    local_path = "temp_image.jpg"
    img.save(local_path)
    return local_path

def get_last_image_path(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def process_image(image_url):
    # Download the image from the URL
    local_image_path = download_image(image_url)
    
    try:
        # Run YOLO model on the downloaded image
        result = model(source=local_image_path, show=False, conf=0.5, save=True, save_crop=True)

        # Get the output directory from the result
        output_directory = result[0].save_dir
        project_path="D:\\yolo"
        patsh=os.path.join(project_path,output_directory)
        # Append the desired folder name to the output directory
        desired_folder = "crops\\leaf"
        updated_output_directory = os.path.join(patsh, desired_folder)

        # Get the path of the last image saved in the directory
        last_image_path = get_last_image_path(updated_output_directory)
        
        if last_image_path is None:
            raise FileNotFoundError("No images found in the specified directory.")
        
        # Return the last image path
        return last_image_path
    finally:
        # Delete the local image file after processing
        if os.path.exists(local_image_path):
            os.remove(local_image_path)

class ImageURL(BaseModel):
    image_url: str

@app.post("/process_image")
async def handle_request(image_data: ImageURL):
    image_url = image_data.image_url
    if not image_url:
        raise HTTPException(status_code=400, detail="No image URL provided")

    try:
        cropped_image_path = process_image(image_url)
        return {"cropped_image_path": cropped_image_path}
    except Exception as e:
        print(f"Error processing image: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
