import gradio as gr
from PIL import Image
import numpy as np
import os
from inference import SegmentationModel
from utils import load_image, apply_mask
import time

# Initialize model
model = SegmentationModel()

def process_images(person_img, background_img):
    person = load_image(person_img)
    background = load_image(background_img)

    mask = model.segment_person(person)

    # Debug info
    print("Mask shape:", mask.shape)
    print("Mask min/max:", mask.min(), mask.max())

    mask_display = Image.fromarray(mask)
    result = apply_mask(person, mask, background)

    # Save result to disk for download
    os.makedirs("outputs", exist_ok=True)
    timestamp = int(time.time())
    output_path = f"outputs/result_{timestamp}.png"
    result.save(output_path)

    return mask_display, result, output_path  # Return path for download

with gr.Blocks(title="Portrait Background Replacement") as demo:
    gr.Markdown("""
    # Portrait Background Replacement  
    **High-quality human segmentation with refined hair edges**

    ## Project Team Contributions
    ### 1. Zhang Xiaotong
    ### 2. Chen Jiayi
    ### 3. Pan Yanxin 
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Portrait", type="pil")
            bg_image = gr.Image(label="Upload Background", type="pil")
            process_btn = gr.Button("Replace Background", variant="primary")

        with gr.Column():
            mask_output = gr.Image(label="Segmentation Mask")
            result_output = gr.Image(label="Result")
            download_btn = gr.File(label="Download Result")

    process_btn.click(
        fn=process_images,
        inputs=[input_image, bg_image],
        outputs=[mask_output, result_output, download_btn]
    )

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    demo.launch(debug=True)
