import torch

from diffusers import StableDiffusionPipeline


pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True)  


from torch import autocast

prompt = "Two girls walking on the street" # enter the prompt you want
with autocast("cuda"):
  image = pipe(prompt).images[0]  


image

