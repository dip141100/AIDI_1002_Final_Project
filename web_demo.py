#!/usr/bin/env python
# encoding: utf-8
import gradio as gr
from PIL import Image
import traceback
import re
import torch
import argparse
from transformers import AutoModel, AutoTokenizer

# README, How to run demo on different devices
# For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
# python web_demo.py --device cuda --dtype bf16

# For Nvidia GPUs do NOT support BF16 (like V100, T4, RTX2080)
# python web_demo.py --device cuda --dtype fp16

# For Mac with MPS (Apple silicon or AMD GPUs).
# PYTORCH_ENABLE_MPS_FALLBACK=1 python web_demo.py --device mps --dtype fp16

# Argparser
parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--device', type=str, default='cuda', help='cuda or mps')
parser.add_argument('--dtype', type=str, default='bf16', help='bf16 or fp16')
args = parser.parse_args()
device = args.device
assert device in ['cuda', 'mps']
if args.dtype == 'bf16':
    dtype = torch.bfloat16
else:
    dtype = torch.float16

# Load model
model_path = 'openbmb/MiniCPM-V-2'
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = model.to(device=device, dtype=dtype)
model.eval()

ERROR_MSG = "Error, please retry"
model_name = 'MiniCPM-V 2.0'

def chat(img, msgs, ctx, params=None, vision_hidden_states=None):
    default_params = {"num_beams":3, "repetition_penalty": 1.2, "max_new_tokens": 1024}
    if params is None:
        params = default_params
    if img is None:
        return -1, "Error, invalid image, please upload a new image", None, None
    try:
        image = img.convert('RGB')
        answer, context, _ = model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            **params
        )
        res = re.sub(r'(<box>.*</box>)', '', answer)
        res = res.replace('<ref>', '')
        res = res.replace('</ref>', '')
        res = res.replace('<box>', '')
        answer = res.replace('</box>', '')
        return -1, answer, None, None
    except Exception as err:
        print(err)
        traceback.print_exc()
        return -1, ERROR_MSG, None, None

def upload_img(image, _chatbot, _app_session):
    image = Image.fromarray(image)

    _app_session['sts']=None
    _app_session['ctx']=[]
    _app_session['img']=image 
    _chatbot.append(('', 'Image uploaded successfully, you can talk to me now'))
    return _chatbot, _app_session

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            img_upload = gr.Image(label="Upload Image", source="upload", tool="select")
            chat_bot = gr.Chatbot()
        with gr.Column(scale=2):
            txt_message = gr.Textbox(label="Your Message")
            submit_btn = gr.Button("Send")
            
    img_upload.change(upload_img, inputs=img_upload, outputs=chat_bot)
    submit_btn.click(
        fn=lambda img, msg: chat(img, msg, None),
        inputs=[img_upload, txt_message],
        outputs=chat_bot
    )

demo.launch()
