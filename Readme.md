# Synthetic Chest X-ray (CXR) Generation

This project focuses on generating synthetic Chest X-ray (CXR) images using both **Generative Adversarial Networks (GANs)** and **Diffusion Models**, along with image editing techniques.

---

##  Models Used

### 🔹 Generative Adversarial Networks (GANs)

- **DCGAN**
- **StyleGAN**
- **Denoising Diffusion GAN (DDGAN)**
- **Conditional GAN (cGAN)**

---

### 🔹 Diffusion Models

- **Stable Diffusion**
  - SD v1.4  : Pre-trained model weights are present at [link](https://github.com/P-Rajiv/cxr_gen_models.git).

  - Below is sample infernce for single image generation.
  ``` python
  import torch
  from diffusers import StableDiffusionPipeline

  # 1. Load the model from your Hugging Face repository
  model_id = "P-RAJIV/cxr_stable_diffusion_v1_4"
  pipe = StableDiffusionPipeline.from_pretrained(
      model_id, 
      torch_dtype=torch.float16  # Use float16 to save VRAM
  )

  # 2. Move the pipeline to GPU
  device = "cuda" if torch.cuda.is_available() else "cpu"
  pipe = pipe.to(device)

  # 3. Define your prompt (e.g., specific clinical findings)
  prompt = "chest x-ray showing pleural effusion"

  # 4. Generate the image
  # num_inference_steps=50 is standard for quality/speed balance
  image = pipe(prompt, num_inference_steps=50).images[0]

  # 5. Save the result
  image.save("generated_cxr.png")
  ```
  
  
  - SD v1.5  
- **Stable Diffusion XL (SDXL)**
- **FLUX**


- For balanced multiple image generation inference script is present at `main_folder/sample_images.py`

- The text file with balanced prompts roughly 150 unique prompts per disease is present as `{disease_name}.txt` in the folder `diseasewise_prompts`. 
---

### 🔹 Image Editing

- **DDIM Inversion**
  - Enables controlled editing of generated images
  - Useful for fine-grained modifications in medical imaging

---

##  Overview

- Generate high-quality synthetic CXR images
- Compare GAN-based vs Diffusion-based approaches
- Enable controllable image editing using inversion techniques
- Applications in:
  - Data augmentation
  - Rare disease synthesis
  - Bias analysis

---

##  Project Structure (Example)
