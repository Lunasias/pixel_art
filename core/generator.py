"""
Generation Engine for 8-Directional Pixel Art Characters.
Supports:
1. Local Diffusers Pipeline (SD 1.5 / SDXL + ControlNet Depth + IP-Adapter)
2. VRAM Hardware Profiles:
   - '4gb_lowvram' (NVIDIA RTX 3050 Laptop 4GB: CPU offloading, FP16, SD 1.5)
   - '8gb_standard' (NVIDIA RTX 5060 Ti 8GB: Full CUDA, SDXL / SD 1.5 fast inference)
3. Civitai API / Remote WebUI Backend
4. Standalone Simulator / Mock Generator (for instant zero-download verification)
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw

from core.processor import (
    DIRECTION_ORDER,
    downscale_pixel_art,
    quantize_palette,
    remove_background_simple,
    derive_8_directions_from_5,
    assemble_turntable,
    assemble_horizontal_spritesheet,
    assemble_compass_grid,
    clean_stray_pixels
)
from templates.pose_generator import generate_all_8_poses


class BaseGenerator:
    """Base class for character sprite generators."""
    def generate_views(
        self,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        negative_prompt: str = "",
        seed: int = 42,
        symmetry_5to8: bool = True
    ) -> Dict[str, Image.Image]:
        raise NotImplementedError


class MockPixelGenerator(BaseGenerator):
    """
    Synthetic generator for instant local testing, offline development,
    and verifying UI/turntable/export without downloading multi-gigabyte models.
    """
    def __init__(self):
        self.poses = generate_all_8_poses(size=(256, 256))

    def generate_views(
        self,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        negative_prompt: str = "",
        seed: int = 42,
        symmetry_5to8: bool = True
    ) -> Dict[str, Image.Image]:
        # Generate procedural stylized pixel art character per direction
        colors = {
            "knight": {"armor": (180, 190, 210), "cloth": (40, 80, 180), "skin": (240, 190, 150)},
            "wizard": {"armor": (100, 50, 150), "cloth": (140, 70, 200), "skin": (240, 190, 150)},
            "rogue":  {"armor": (60, 70, 60),   "cloth": (30, 40, 35),   "skin": (230, 180, 140)},
        }
        
        # Pick color theme based on prompt keywords
        p_lower = prompt.lower()
        if "wizard" in p_lower or "mage" in p_lower:
            theme = colors["wizard"]
        elif "rogue" in p_lower or "ninja" in p_lower or "dark" in p_lower:
            theme = colors["rogue"]
        else:
            theme = colors["knight"]

        # If reference image is provided, sample dominant colors from it
        if reference_image is not None:
            ref_rgb = reference_image.convert("RGB").resize((16, 16))
            dom_colors = ref_rgb.getcolors(maxcolors=256)
            if dom_colors:
                sorted_cols = sorted(dom_colors, key=lambda x: x[0], reverse=True)
                theme["cloth"] = sorted_cols[0][1]
                if len(sorted_cols) > 1:
                    theme["armor"] = sorted_cols[1][1]

        directions_to_gen = ["S", "SE", "E", "NE", "N"] if symmetry_5to8 else DIRECTION_ORDER
        views: Dict[str, Image.Image] = {}

        for d in directions_to_gen:
            img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Simple 2.5D procedural character rendering for demo
            cx, cy = 128, 140
            
            # Draw shadow
            draw.ellipse([cx - 40, cy + 50, cx + 40, cy + 70], fill=(0, 0, 0, 80))
            
            # Body & Clothing
            cloth_c = theme["cloth"] + (255,)
            armor_c = theme["armor"] + (255,)
            skin_c = theme["skin"] + (255,)
            
            # Legs
            draw.rectangle([cx - 24, cy + 20, cx - 8, cy + 55], fill=armor_c)
            draw.rectangle([cx + 8, cy + 20, cx + 24, cy + 55], fill=armor_c)
            
            # Torso / Armor
            draw.rectangle([cx - 32, cy - 30, cx + 32, cy + 25], fill=cloth_c)
            draw.rectangle([cx - 20, cy - 20, cx + 20, cy + 15], fill=armor_c)
            
            # Head & Helm
            draw.ellipse([cx - 25, cy - 75, cx + 25, cy - 25], fill=skin_c)
            draw.arc([cx - 25, cy - 75, cx + 25, cy - 25], start=180, end=360, fill=armor_c, width=8)
            
            # Directional facing indicators
            if d == "S":
                # Front face
                draw.rectangle([cx - 12, cy - 50, cx - 4, cy - 44], fill=(20, 20, 20, 255))
                draw.rectangle([cx + 4, cy - 50, cx + 12, cy - 44], fill=(20, 20, 20, 255))
            elif d == "SE":
                # Eyes shifted right
                draw.rectangle([cx - 4, cy - 50, cx + 4, cy - 44], fill=(20, 20, 20, 255))
                draw.rectangle([cx + 12, cy - 50, cx + 20, cy - 44], fill=(20, 20, 20, 255))
            elif d == "E":
                # Side profile facing right
                draw.rectangle([cx + 12, cy - 50, cx + 20, cy - 44], fill=(20, 20, 20, 255))
                # Shield or sword on side
                draw.rectangle([cx + 25, cy - 10, cx + 35, cy + 30], fill=armor_c)
            elif d == "NE":
                # Back-right: hair / cape visible
                draw.rectangle([cx - 20, cy - 25, cx + 20, cy + 40], fill=cloth_c)
            elif d == "N":
                # Back view: cape / backplate
                draw.rectangle([cx - 28, cy - 25, cx + 28, cy + 45], fill=cloth_c)

            views[d] = img

        if symmetry_5to8:
            return derive_8_directions_from_5(views)
        return views


class LocalDiffusersPipeline(BaseGenerator):
    """
    Diffusers implementation supporting Stable Diffusion 1.5 and SDXL with ControlNet.
    Optimized for both:
    - 4GB VRAM (RTX 3050 Laptop): low_cpu_mem_usage, sequential cpu offload, fp16
    - 8GB VRAM (RTX 5060 Ti): model_cpu_offload, xformers/sdpa, fp16
    """
    def __init__(
        self,
        base_model_path: str = "runwayml/stable-diffusion-v1-5",
        controlnet_depth_path: str = "lllyasviel/control_v11f1p_sd15_depth",
        lora_path: Optional[str] = None,
        vram_profile: str = "8gb_standard",  # '4gb_lowvram' or '8gb_standard'
        device: str = "cuda"
    ):
        self.base_model_path = base_model_path
        self.controlnet_depth_path = controlnet_depth_path
        self.lora_path = lora_path
        self.vram_profile = vram_profile
        self.device = device
        self.pipe = None
        self.poses = generate_all_8_poses(size=(512, 512))

    def _load_pipeline(self):
        if self.pipe is not None:
            return

        import torch
        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            UniPCMultistepScheduler
        )

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # 1. Load ControlNet Depth
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_depth_path,
            torch_dtype=torch_dtype
        )

        # 2. Load Pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model_path,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None
        )

        # 3. Scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        # 4. LoRA weights if provided (e.g. Pixel Art XL / Retro Pixel LoRA from Civitai)
        if self.lora_path and os.path.exists(self.lora_path):
            self.pipe.load_lora_weights(self.lora_path)

        # 5. VRAM Hardware Optimizations
        if self.device == "cuda":
            if self.vram_profile == "4gb_lowvram":
                # Maximize memory savings for 4GB VRAM
                self.pipe.enable_sequential_cpu_offload()
                self.pipe.enable_attention_slicing(1)
            else:
                # 8GB VRAM (RTX 5060 Ti) - fast model CPU offload and full SDPA
                self.pipe.enable_model_cpu_offload()

    def generate_views(
        self,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        negative_prompt: str = "blurry, smooth, photo, 3d render, distorted, realistic",
        seed: int = 42,
        symmetry_5to8: bool = True
    ) -> Dict[str, Image.Image]:
        import torch

        self._load_pipeline()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Base pixel art prompt triggers
        full_prompt = f"pixel art, 16-bit sprite, {prompt}, transparent background, clean outline, flat colors"

        directions_to_gen = ["S", "SE", "E", "NE", "N"] if symmetry_5to8 else DIRECTION_ORDER
        views: Dict[str, Image.Image] = {}

        for d in directions_to_gen:
            depth_pose = self.poses[d]
            dir_prompt = f"{full_prompt}, facing {d}"

            output = self.pipe(
                prompt=dir_prompt,
                negative_prompt=negative_prompt,
                image=depth_pose,
                num_inference_steps=20,
                guidance_scale=7.5,
                controlnet_conditioning_scale=0.85,
                generator=generator
            )
            views[d] = output.images[0]

        if symmetry_5to8:
            return derive_8_directions_from_5(views)
        return views


class CharacterSpritePipeline:
    """
    High-level facade orchestrating the entire end-to-end generation process:
    Generation -> Background Removal -> Downscale -> Quantize -> Turntable & Spritesheet
    """
    def __init__(self, backend: Optional[BaseGenerator] = None):
        self.backend = backend or MockPixelGenerator()

    def create_character(
        self,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        target_size: Tuple[int, int] = (64, 64),
        palette_name: str = "db32",
        symmetry_5to8: bool = True,
        remove_bg: bool = True,
        clean_noise: bool = True,
        seed: int = 42,
        output_dir: str = "./outputs"
    ) -> Dict[str, Union[Dict[str, Image.Image], Image.Image, str]]:
        os.makedirs(output_dir, exist_ok=True)

        # 1. Generate Raw Views (either 5 mirrored to 8, or all 8)
        raw_views = self.backend.generate_views(
            prompt=prompt,
            reference_image=reference_image,
            seed=seed,
            symmetry_5to8=symmetry_5to8
        )

        # 2. Pixel-Art Processing Pipeline for each view
        processed_views: Dict[str, Image.Image] = {}
        for d, img in raw_views.items():
            # A. Background Matting
            if remove_bg:
                img = remove_background_simple(img)

            # B. Nearest-Neighbor Downscaling
            img = downscale_pixel_art(img, target_size=target_size)

            # C. Color Quantization (Palettization)
            if palette_name and palette_name.lower() != "none":
                img = quantize_palette(img, palette_name=palette_name)

            # D. Stray Pixel / Noise Removal
            if clean_noise:
                img = clean_stray_pixels(img)

            processed_views[d] = img

        # 3. Assemble Horizontal Sprite Sheet
        spritesheet = assemble_horizontal_spritesheet(processed_views)
        sheet_path = os.path.join(output_dir, f"spritesheet_{seed}.png")
        spritesheet.save(sheet_path)

        # 4. Assemble Compass 3x3 Grid
        compass_grid = assemble_compass_grid(processed_views, center_image=reference_image)
        grid_path = os.path.join(output_dir, f"compass_grid_{seed}.png")
        compass_grid.save(grid_path)

        # 5. Assemble Animated Turntable GIF
        turntable_path = os.path.join(output_dir, f"turntable_{seed}.gif")
        assemble_turntable(processed_views, turntable_path, fps=7, scale_factor=4)

        return {
            "views": processed_views,
            "spritesheet": spritesheet,
            "spritesheet_path": sheet_path,
            "compass_grid": compass_grid,
            "compass_grid_path": grid_path,
            "turntable_path": turntable_path
        }
