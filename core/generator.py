"""
Generation Engine for 8-Directional Pixel Art Characters.
Supports:
1. Stable Diffusion WebUI Forge API (Local RTX 5060 Ti / Stability Matrix on http://127.0.0.1:7860)
2. Local Diffusers Pipeline (SD 1.5 / SDXL + ControlNet Depth)
3. VRAM Hardware Profiles:
   - '4gb_lowvram' (NVIDIA RTX 3050 Laptop 4GB: CPU offloading, FP16)
   - '8gb_standard' (NVIDIA RTX 5060 Ti 8GB: Full CUDA, fast SDPA)
4. Standalone Simulator / Mock Generator (zero-download verification)
"""

import base64
import io
import os
from typing import Dict, List, Optional, Tuple, Union
import requests
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
    clean_stray_pixels,
    slice_horizontal_strip
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


def is_forge_available(url: str = "http://127.0.0.1:7860", timeout: float = 1.5) -> bool:
    """Checks whether Stable Diffusion WebUI Forge is running and responsive."""
    try:
        resp = requests.get(f"{url.rstrip('/')}/sdapi/v1/progress", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


class ForgeWebUIGenerator(BaseGenerator):
    """
    Direct connector to local Stable Diffusion WebUI Forge (Stability Matrix).
    Leverages already installed models (Animagine XL 3.1, AOM3, Pixel Art XL LoRA)
    with zero duplicate downloads and maximum VRAM acceleration.
    """
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:7860",
        checkpoint: Optional[str] = None,
        lora_name: str = "pixel-art-xl-v1.1",
        timeout: int = 180
    ):
        self.base_url = base_url.rstrip("/")
        self.checkpoint = checkpoint
        self.lora_name = lora_name
        self.timeout = timeout

    def generate_views(
        self,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        negative_prompt: str = "",
        seed: int = 42,
        symmetry_5to8: bool = True
    ) -> Dict[str, Image.Image]:
        if not is_forge_available(self.base_url):
            raise ConnectionError(
                f"Cannot connect to Stable Diffusion WebUI Forge at {self.base_url}. "
                "Please make sure Stability Matrix / Forge is running."
            )

        # Standard high-quality negative prompt
        if not negative_prompt or not negative_prompt.strip():
            negative_prompt = (
                "(worst quality, low quality:1.4), (character sheet:1.5), "
                "deformed, bad anatomy, bad hands, missing fingers, extra limbs, photo, realistic, "
                "green skin, colored skin, alien skin, scenery, complex background"
            )

        # Direction mapping for character angle prompting
        dir_prompts = {
            "S": "front view, facing camera, looking at viewer",
            "SE": "three quarter front view, facing front-right",
            "E": "side view, right profile",
            "NE": "three quarter back view, facing back-right",
            "N": "back view, facing away from camera, rear view",
            "NW": "three quarter back view, facing back-left",
            "W": "side view, left profile",
            "SW": "three quarter front view, facing front-left"
        }

        directions_to_gen = ["S", "SE", "E", "NE", "N"] if symmetry_5to8 else DIRECTION_ORDER
        views: Dict[str, Image.Image] = {}

        for d in directions_to_gen:
            facing_cue = dir_prompts.get(d, f"facing {d}")
            pos_parts = [
                prompt.strip(),
                facing_cue,
                "full body, standing, solid clean white background, simple background, white backdrop"
            ]

            # Append LoRA if not already in prompt
            if "<lora:" not in prompt and self.lora_name:
                pos_parts.append(f"<lora:{self.lora_name}:1.0>")

            pos_prompt = ", ".join(pos_parts)

            payload = {
                "prompt": pos_prompt,
                "negative_prompt": negative_prompt,
                "steps": 28,
                "cfg_scale": 7.0,
                "width": 640,
                "height": 832,
                "sampler_name": "Euler a",
                "seed": seed,
            }

            resp = requests.post(f"{self.base_url}/sdapi/v1/txt2img", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            if "images" in data and len(data["images"]) > 0:
                raw_b64 = data["images"][0]
                img_data = base64.b64decode(raw_b64)
                img = Image.open(io.BytesIO(img_data)).convert("RGBA")
                views[d] = img
            else:
                raise RuntimeError("Forge API returned no images")

        if symmetry_5to8:
            return derive_8_directions_from_5(views)
        return views


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
        colors = {
            "knight": {"armor": (180, 190, 210), "cloth": (40, 80, 180), "skin": (240, 190, 150)},
            "wizard": {"armor": (100, 50, 150), "cloth": (140, 70, 200), "skin": (240, 190, 150)},
            "rogue":  {"armor": (60, 70, 60),   "cloth": (30, 40, 35),   "skin": (230, 180, 140)},
        }

        p_lower = prompt.lower()
        if "wizard" in p_lower or "mage" in p_lower:
            theme = colors["wizard"]
        elif "rogue" in p_lower or "ninja" in p_lower or "dark" in p_lower:
            theme = colors["rogue"]
        else:
            theme = colors["knight"]

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
            # Use solid chroma green background to simulate realistic AI generation backdrop
            img = Image.new("RGBA", (256, 256), (0, 255, 0, 255))
            draw = ImageDraw.Draw(img)

            cx, cy = 128, 140

            # Draw shadow
            draw.ellipse([cx - 40, cy + 50, cx + 40, cy + 70], fill=(0, 0, 0, 80))

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
                draw.rectangle([cx - 12, cy - 50, cx - 4, cy - 44], fill=(20, 20, 20, 255))
                draw.rectangle([cx + 4, cy - 50, cx + 12, cy - 44], fill=(20, 20, 20, 255))
            elif d == "SE":
                draw.rectangle([cx - 4, cy - 50, cx + 4, cy - 44], fill=(20, 20, 20, 255))
                draw.rectangle([cx + 12, cy - 50, cx + 20, cy - 44], fill=(20, 20, 20, 255))
            elif d == "E":
                draw.rectangle([cx + 12, cy - 50, cx + 20, cy - 44], fill=(20, 20, 20, 255))
                draw.rectangle([cx + 25, cy - 10, cx + 35, cy + 30], fill=armor_c)
            elif d == "NE":
                draw.rectangle([cx - 20, cy - 25, cx + 20, cy + 40], fill=cloth_c)
            elif d == "N":
                draw.rectangle([cx - 28, cy - 25, cx + 28, cy + 45], fill=cloth_c)

            views[d] = img

        if symmetry_5to8:
            return derive_8_directions_from_5(views)
        return views


class LocalDiffusersPipeline(BaseGenerator):
    """
    Diffusers implementation supporting Stable Diffusion 1.5 and SDXL with ControlNet.
    Optimized for:
    - 4GB VRAM (RTX 3050 Laptop): sequential cpu offload, fp16
    - 8GB VRAM (RTX 5060 Ti): model_cpu_offload, xformers/sdpa, fp16
    """
    def __init__(
        self,
        base_model_path: str = "runwayml/stable-diffusion-v1-5",
        controlnet_depth_path: str = "lllyasviel/control_v11f1p_sd15_depth",
        lora_path: Optional[str] = None,
        vram_profile: str = "8gb_standard",
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

        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_depth_path,
            torch_dtype=torch_dtype
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model_path,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        if self.lora_path and os.path.exists(self.lora_path):
            self.pipe.load_lora_weights(self.lora_path)

        if self.device == "cuda":
            if self.vram_profile == "4gb_lowvram":
                self.pipe.enable_sequential_cpu_offload()
                self.pipe.enable_attention_slicing(1)
            else:
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

        # Use solid green backdrop prompt instead of "transparent background"
        full_prompt = (
            f"pixel art, 16-bit sprite, {prompt}, "
            "solid bright green background, chroma key green background, clean outline, flat colors"
        )

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

        # 1. Generate Raw Views
        raw_views = self.backend.generate_views(
            prompt=prompt,
            reference_image=reference_image,
            seed=seed,
            symmetry_5to8=symmetry_5to8
        )

        # 2. Pixel-Art Processing Pipeline for each view
        processed_views: Dict[str, Image.Image] = {}
        for d, img in raw_views.items():
            # A. Robust Border Floodfill Background Removal
            if remove_bg:
                img = remove_background_simple(img)

            # B. Edge-Preserving Downscaling
            img = downscale_pixel_art(img, target_size=target_size, preserve_details=True)

            # C. Perceptually Weighted Color Quantization
            if palette_name and palette_name.lower() != "none":
                img = quantize_palette(img, palette_name=palette_name)

            # D. Stray Pixel / Isolated Dot Removal
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
