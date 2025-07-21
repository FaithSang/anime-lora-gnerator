# anime-lora-gnerator

A gradio powered app for generating anime faces using a fine-tuned LoRA model

# 🧠Anime LoRA genrator

> A final project for Week 8 of my image generation learning journey- exploring Stable Diffusion, LoRa fine-tuning and app deployment

# 👉 DEMO

Try the app here : [Anime Genrator on Hugging Face spaces] (https://huggingface.co/spaces/FaySang/LoRA-anime)

![Sample output] (images/prompt_output1)

## 🚀 Features

Custom prompt input
CFG, Steps, and Seed control
Downloadable images

# 🧠How it works

LoRa model trained using "diffusers"
Anime dataset from hugging face
Web UI built using Gradio

## 📁 Folder structure

notebooks/ Colab fine-tuning scripts
app/ > Gradio web app
images/> Downloaded and sample outputs
model/ > Link to trained weights and config

## 🧪Links

Hugging face dataset : 'ppbrown/danbooru-cleaned'
Gradio Space : [TRY App] (https://huggingface.co/spaces/FaySang/LoRA-anime)

# 🧪Training details

**Base Model**: 'runwayml/stable-diffusion-v1-5'
**Dataset**: [ppbrown/danbooru-cleaned] (https://huggingface.co/datasets/ppbrown/danbooru-cleaned)
**LoRA Adapter** Fine-tuned usng "peft" with 100 anime face images

# 🚀How it works

> Accepts a prompt (e.g 'cute girl, blue long hair, school uniform')
> Applies the trained LoRA weights
> Uses 'StableDiffusionPipeline' from Diffusres to generate naime images
> All run from a simple and intutive Gradio interface

# 👾Set up locally

''bash
got clone https://github.com/FaithSang/anime-lora-gnerator.git
cd lora-anime
pip install -r requirements.txt
python app.py

## Tools ussed

Hugging Face Diffusers
PEFT LoRA Adapter
Gradio Web App
Google Colab for training
Danbooru-style Anime Dataset

## Author

Faith Sang
