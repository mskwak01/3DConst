import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

geometry_noise = torch.randn(1,4,128,128)

# import pdb; pdb.set_trace()

import pdb; pdb.set_trace()

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, strength=0.2, noise=geometry_noise).images[0]
make_image_grid([init_image, image], rows=1, cols=2)



# import torch
# from diffusers import AutoPipelineForImage2Image
# from diffusers.utils import make_image_grid, load_image

# pipeline = AutoPipelineForImage2Image.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )
# pipeline.enable_model_cpu_offload()
# # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

# # prepare image
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
# init_image = load_image(url)

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# # pass prompt and image to pipeline
# image = pipeline(prompt, image=init_image).images[0]
# make_image_grid([init_image, image], rows=1, cols=2)