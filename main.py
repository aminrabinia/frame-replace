from ultralytics import YOLO
import uvicorn
import cv2
import gradio as gr
import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from infer import display_instances
from ultralytics.yolo.utils.ops import scale_image, scale_boxes



model = YOLO('best.pt') 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.glissai.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get('/')
def root():
    return {"message": "hello from yolov8 frame segmentation!"}


def make_prediction(input_image, conf):

    results = model.predict(input_image, imgsz=640, conf=conf, save=False, device='cpu')

    if len(results[0].numpy())==0:
        return None, None

    img_orig_shape = results[0].masks.cpu().numpy().orig_shape
    r = results[0].numpy()
    
    masks = r.masks.data
    masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)

    img_infer_shape = masks.shape[:2]
    masks = scale_image(img_infer_shape, masks, img_orig_shape)
    masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)

    boxes = r.boxes.xyxy
    # boxes = scale_boxes(img_infer_shape, boxes, img_orig_shape)
    boxes = boxes.astype(int).tolist()  # Boxes object for bbox outputs

    return boxes, masks


def resize_if_big(image):
    height, width = image.shape[:2]
    if height > 640 or width > 640:
        max_dim = max(height, width)
        scaling_factor = 640 / max_dim
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return image


def gradio_infer(myimg, pic, conf=0.25):
    if myimg is None:
      return 
    myimg = resize_if_big(myimg)
    infer_input = myimg.copy()
    boxes, masks = make_prediction(infer_input, conf)
    if boxes:
      res_img = display_instances(myimg, pic, boxes, masks)
      return res_img
    else: 
      return myimg

def cleanup():
    return None

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    with gr.Row():
        with gr.Column():
            main_img = gr.Image() 
            sub_btn = gr.Button("Submit")
            clr_btn = gr.Button("Clear")
        with gr.Column():
            pic_img = gr.Image("tiny_rosie.JPG")
        sub_btn.click(gradio_infer, inputs=[main_img, pic_img], outputs=main_img)
        clr_btn.click(cleanup, inputs=None, outputs=main_img)
    gr.Examples(examples= ["testimg.jpg"],
                        inputs=main_img, fn=gradio_infer)

gr.mount_gradio_app(app, demo, path="/gradio")


if __name__ == "__main__":

	uvicorn.run(app, host='0.0.0.0', port=8080)

