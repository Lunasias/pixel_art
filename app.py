"""
PixelArt 8-Dir Studio - Web UI
Generates consistent 8-directional top-down/isometric pixel art characters.
Supports:
- 5-to-8 directional symmetry (zero limb drift)
- Strict nearest-neighbor downscaling & palette quantization (Pico-8, DB32, Endesga)
- Automatic background removal & stray noise cleaning
- Local RTX 3050 (4GB) and RTX 5060 Ti (8GB) hardware profiles
"""

import os
import yaml
from typing import Optional, Tuple
from PIL import Image
import gradio as gr

from core.processor import PALETTES, DIRECTION_ORDER, downscale_pixel_art, quantize_palette, remove_background_simple, clean_stray_pixels, assemble_turntable
from core.generator import CharacterSpritePipeline, MockPixelGenerator, LocalDiffusersPipeline

# Load configuration
CONFIG_PATH = "config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = {}

# Initialize cached pipelines
mock_pipeline = CharacterSpritePipeline(backend=MockPixelGenerator())
local_pipeline = None  # Loaded lazily if requested


def get_pipeline(engine_choice: str, vram_profile: str) -> CharacterSpritePipeline:
    global local_pipeline
    if engine_choice == "Mock / Demo Mode (Instant, No GPU needed)":
        return mock_pipeline

    if local_pipeline is None:
        backend = LocalDiffusersPipeline(
            vram_profile="4gb_lowvram" if "4GB" in vram_profile else "8gb_standard"
        )
        local_pipeline = CharacterSpritePipeline(backend=backend)
    return local_pipeline


def generate_character(
    prompt: str,
    ref_image: Optional[Image.Image],
    size_str: str,
    palette_choice: str,
    symmetry: bool,
    remove_bg: bool,
    clean_noise: bool,
    engine: str,
    vram_profile: str,
    seed: int
):
    if not prompt or not prompt.strip():
        prompt = "a fantasy knight in silver armor with a blue cape"

    # Parse target size
    dim = int(size_str.split("x")[0].strip())
    target_size = (dim, dim)

    palette_name = None if palette_choice == "None (Full Color)" else palette_choice.lower()

    # Get pipeline
    pipeline = get_pipeline(engine, vram_profile)

    # Run Generation
    res = pipeline.create_character(
        prompt=prompt,
        reference_image=ref_image,
        target_size=target_size,
        palette_name=palette_name,
        symmetry_5to8=symmetry,
        remove_bg=remove_bg,
        clean_noise=clean_noise,
        seed=int(seed),
        output_dir="./outputs"
    )

    views = res["views"]
    ordered_views = [views[d] for d in DIRECTION_ORDER if d in views]

    return (
        res["turntable_path"],      # Animated Turntable GIF
        res["spritesheet"],         # Spritesheet Image
        res["compass_grid"],        # 3x3 Compass Grid Image
        res["spritesheet_path"],    # Download Spritesheet
        res["turntable_path"],      # Download Turntable
        ordered_views               # Gallery of 8 directions
    )


def process_custom_image(
    input_img: Optional[Image.Image],
    size_str: str,
    palette_choice: str,
    remove_bg: bool,
    clean_noise: bool
):
    if input_img is None:
        return None, None

    dim = int(size_str.split("x")[0].strip())
    target_size = (dim, dim)
    palette_name = None if palette_choice == "None (Full Color)" else palette_choice.lower()

    img = input_img.convert("RGBA")
    if remove_bg:
        img = remove_background_simple(img)
    img = downscale_pixel_art(img, target_size=target_size)
    if palette_name:
        img = quantize_palette(img, palette_name=palette_name)
    if clean_noise:
        img = clean_stray_pixels(img)

    # Upscale for crisp display
    display_img = img.resize((dim * 4, dim * 4), Image.Resampling.NEAREST)
    return img, display_img


# Build Web Interface
with gr.Blocks(title="PixelArt 8-Dir Studio") as demo:
    gr.Markdown(
        """
        # 🎮 PixelArt 8-Dir Studio (PixelLab Alternative)
        ### AI-Powered 8-Directional Consistent Pixel Art Character Generator
        Generates top-down / isometric sprites with **locked character identity**, **5-to-8 directional symmetry**, and **crisp palette quantization**.
        """
    )

    with gr.Tabs():
        # --- TAB 1: 8-DIRECTION GENERATOR ---
        with gr.Tab("🎨 8-Direction Character Generator"):
            with gr.Row():
                # Left Column: Inputs & Controls
                with gr.Column(scale=4):
                    prompt_input = gr.Textbox(
                        label="Character Prompt",
                        placeholder="e.g. cute wizard in purple robes casting a spell, top-down pixel art",
                        value="a heroic knight in polished steel armor with a royal blue cloak",
                        lines=2
                    )

                    ref_input = gr.Image(
                        label="Reference Concept Image (Optional)",
                        type="pil"
                    )

                    with gr.Row():
                        size_dropdown = gr.Dropdown(
                            label="Target Resolution (Pixels)",
                            choices=["32x32", "48x48", "64x64", "128x128"],
                            value="64x64"
                        )
                        palette_dropdown = gr.Dropdown(
                            label="Color Palette",
                            choices=["DB32", "Pico-8", "Endesga-32", "GameBoy", "Sweetie-16", "None (Full Color)"],
                            value="DB32"
                        )

                    with gr.Row():
                        symmetry_cb = gr.Checkbox(
                            label="5-to-8 Direction Symmetry (Mirror SE/E/NE -> SW/W/NW)",
                            value=True,
                            info="Recommended: Ensures 100% limb alignment & eliminates left/right drift"
                        )
                        remove_bg_cb = gr.Checkbox(
                            label="Auto-Remove Background (Transparent Alpha)",
                            value=True
                        )
                        clean_noise_cb = gr.Checkbox(
                            label="Clean Stray Noise Pixels",
                            value=True
                        )

                    with gr.Accordion("⚙️ Hardware & Generator Settings", open=False):
                        engine_radio = gr.Radio(
                            label="Generation Engine",
                            choices=[
                                "Mock / Demo Mode (Instant, No GPU needed)",
                                "Local SD 1.5 + ControlNet (Local CUDA)"
                            ],
                            value="Mock / Demo Mode (Instant, No GPU needed)"
                        )
                        vram_radio = gr.Radio(
                            label="VRAM Hardware Profile",
                            choices=[
                                "4GB Laptop Low VRAM (RTX 3050)",
                                "8GB Standard (RTX 5060 Ti Desktop)"
                            ],
                            value="8GB Standard (RTX 5060 Ti Desktop)"
                        )
                        seed_slider = gr.Slider(
                            label="Seed",
                            minimum=1,
                            maximum=999999,
                            step=1,
                            value=42
                        )

                    generate_btn = gr.Button("⚡ Generate 8-Direction Sprite Pack", variant="primary", size="lg")

                # Right Column: Output & Previews
                with gr.Column(scale=5):
                    with gr.Row():
                        turntable_out = gr.Image(
                            label="🔄 360° Rotating Turntable Preview",
                            type="filepath",
                            interactive=False
                        )
                        compass_out = gr.Image(
                            label="🧭 3x3 Compass Grid Layout",
                            type="pil",
                            interactive=False
                        )

                    spritesheet_out = gr.Image(
                        label="📜 Game-Ready Spritesheet (1x8 Horizontal Strip)",
                        type="pil",
                        interactive=False
                    )

                    with gr.Row():
                        download_sheet = gr.File(label="Download Spritesheet (.png)")
                        download_turntable = gr.File(label="Download Turntable (.gif)")

                    gallery_out = gr.Gallery(
                        label="Individual Frames (S, SE, E, NE, N, NW, W, SW)",
                        columns=4,
                        rows=2,
                        height="auto"
                    )

            # Wiring up generation button
            generate_btn.click(
                fn=generate_character,
                inputs=[
                    prompt_input, ref_input, size_dropdown, palette_dropdown,
                    symmetry_cb, remove_bg_cb, clean_noise_cb,
                    engine_radio, vram_radio, seed_slider
                ],
                outputs=[
                    turntable_out, spritesheet_out, compass_out,
                    download_sheet, download_turntable, gallery_out
                ]
            )

        # --- TAB 2: PIXEL LABORATORY ---
        with gr.Tab("🧪 Pixel Lab (Post-Processor)"):
            gr.Markdown("Upload any image or high-res concept art to convert it into crisp, true pixel art.")
            with gr.Row():
                with gr.Column():
                    lab_input = gr.Image(label="Source Image", type="pil")
                    lab_size = gr.Dropdown(
                        label="Target Resolution",
                        choices=["16x16", "32x32", "48x48", "64x64", "128x128"],
                        value="64x64"
                    )
                    lab_palette = gr.Dropdown(
                        label="Color Palette",
                        choices=["DB32", "Pico-8", "Endesga-32", "GameBoy", "Sweetie-16", "None (Full Color)"],
                        value="DB32"
                    )
                    lab_bg = gr.Checkbox(label="Remove Background", value=True)
                    lab_clean = gr.Checkbox(label="Clean Stray Pixels", value=True)
                    lab_btn = gr.Button("Convert to Pixel Art", variant="primary")

                with gr.Column():
                    lab_display = gr.Image(label="Pixel-Perfect Preview (Upscaled x4)", type="pil")
                    lab_raw = gr.Image(label="Exact Raw File (1:1 scale)", type="pil")

            lab_btn.click(
                fn=process_custom_image,
                inputs=[lab_input, lab_size, lab_palette, lab_bg, lab_clean],
                outputs=[lab_raw, lab_display]
            )

        # --- TAB 3: SETUP & CIVITAI GUIDE ---
        with gr.Tab("📖 Setup & Civitai Guide"):
            gr.Markdown(
                """
                ### 🚀 Recommended Models from Civitai
                To achieve high-quality results matching PixelLab.ai locally, download these models:

                1. **SD 1.5 Sprite Checkpoint (Best for 4GB-8GB VRAM)**:
                   - [Pixel Art Sprite Diffusion (Civitai)](https://civitai.com/models/22/pixel-art-sprite-diffusion)
                   - Trigger Words: `PixelartFSS` (Front), `PixelartRSS` (Right), `PixelartBSS` (Back), `PixelartLSS` (Left)
                   - Size: ~3.9 GB safetensors

                2. **SDXL Checkpoint (Recommended for RTX 5060 Ti 8GB)**:
                   - [Super_PixelArt_XL_M_V1 (Civitai)](https://civitai.com/models/581162/superpixelartxlmv1)
                   - Mix of pixel models creating 128x128 pixel-perfect sprites from 1024x1024 generation.
                   - Trigger words: `pixelart, sprite, 16bit`

                3. **ControlNet Depth for Orientation Control**:
                   - [lllyasviel/control_v11f1p_sd15_depth (HuggingFace)](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth)
                   - Allows feeding the bundled 8-direction mannequin depth poses to lock angles.

                ---
                ### 💻 Hardware Profiles:
                - **RTX 3050 Laptop (4GB VRAM)**:
                  - Select *4GB Laptop Low VRAM* profile.
                  - Uses SD 1.5 + sequential CPU offload + attention slicing.
                - **RTX 5060 Ti Desktop (8GB VRAM)**:
                  - Select *8GB Standard* profile.
                  - Fast SDPA acceleration, can run full batching and SDXL.
                """
            )

if __name__ == "__main__":
    host = config.get("app", {}).get("host", "127.0.0.1")
    port = config.get("app", {}).get("port", 7860)
    print(f"Starting PixelArt 8-Dir Studio at http://{host}:{port} ...")
    demo.launch(server_name=host, server_port=port, share=False)
