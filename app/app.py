import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import get_peft_model, LoraConfig
from safetensors.torch import load_file
import os

# Paths
base_model = "runwayml/stable-diffusion-v1-5"
lora_path = "./"  # Local path inside the repo

#Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base UNet
unet = UNet2DConditionModel.from_pretrained(
    base_model,
    subfolder="unet",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# Apply LoRA config
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v"],
    lora_dropout=0.1,
    bias="none"
)
unet = get_peft_model(unet, lora_config)

# Load LoRA weights
lora_weights = load_file(f"{lora_path}/pytorch_lora_weights.safetensors")
unet.load_state_dict(lora_weights, strict=False)

# Merge and unload PEFT wrapper
unet = unet.merge_and_unload()

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    unet=unet,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

pipe.enable_attention_slicing()

# Generation function
def generate(prompt, negative_prompt, guidance_scale, steps, seed):
    generator = torch.manual_seed(seed) if seed != -1 else None
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator
    ).images[0]
    return image

# Gradio interface
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="A cute anime girl with pink hair, black eyes"),
        gr.Textbox(label="Negative Prompt", placeholder="blurry, bad anatomy, low res"),
        gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Slider(10, 50, value=30, step=1, label="Inference Steps"),
        gr.Slider(-1, 999999, step=1, value=-1, label="Seed (-1 for random)")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="🎨 Anime Face Generator with LoRA"
)

if __name__ == "__main__":
    demo.launch()
