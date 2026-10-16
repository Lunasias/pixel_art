"""
PixelArt 8-Dir Studio - Web UI
Generates and post-processes consistent 8-directional top-down/isometric pixel art characters.
Supports:
- Direct WebUI Forge (Stability Matrix) API integration on http://127.0.0.1:7860
- Robust OpenCV border floodfill background removal (zero Swiss cheese / internal holes)
- High-detail 16/32-bit edge-preserving downscaling (256x256, 128x128, 64x64, 48x48, 32x32)
- Perceptual palette quantization (DB32, Pico-8, Endesga-32, Sweetie-16, GameBoy)
- 5-to-8 directional symmetry (zero limb drift)
- Complete Sprite Sheet Slicer & Game-Ready ZIP Exporter
"""

import os
import yaml
from typing import Optional, Tuple, Dict
from PIL import Image
import gradio as gr

from core.processor import (
    PALETTES,
    DIRECTION_ORDER,
    downscale_pixel_art,
    quantize_palette,
    remove_background_border_floodfill,
    clean_stray_pixels,
    derive_8_directions_from_5,
    slice_horizontal_strip,
    slice_grid_3x3,
    assemble_horizontal_spritesheet,
    assemble_compass_grid,
    assemble_turntable,
    package_frames_zip
)
from core.generator import (
    CharacterSpritePipeline,
    MockPixelGenerator,
    LocalDiffusersPipeline,
    ForgeWebUIGenerator,
    is_forge_available
)

# Load configuration
CONFIG_PATH = "config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = {}

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize cached pipelines
mock_pipeline = CharacterSpritePipeline(backend=MockPixelGenerator())
local_diffusers_pipeline = None
forge_pipeline = None


def get_pipeline(engine_choice: str, vram_profile: str) -> CharacterSpritePipeline:
    global local_diffusers_pipeline, forge_pipeline

    if "Mock" in engine_choice:
        return mock_pipeline

    if "Forge" in engine_choice:
        forge_url = config.get("forge", {}).get("url", "http://127.0.0.1:7860")
        lora = config.get("forge", {}).get("default_lora", "pixel-art-xl-v1.1")
        timeout = config.get("forge", {}).get("timeout", 180)
        backend = ForgeWebUIGenerator(base_url=forge_url, lora_name=lora, timeout=timeout)
        forge_pipeline = CharacterSpritePipeline(backend=backend)
        return forge_pipeline

    if local_diffusers_pipeline is None:
        backend = LocalDiffusersPipeline(
            vram_profile="4gb_lowvram" if "4GB" in vram_profile else "8gb_standard"
        )
        local_diffusers_pipeline = CharacterSpritePipeline(backend=backend)
    return local_diffusers_pipeline


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
        prompt = "a heroic knight in polished steel armor with a royal blue cloak"

    dim = int(size_str.split("x")[0].strip())
    target_size = (dim, dim)
    palette_name = None if palette_choice == "None (Full Color)" else palette_choice.lower()

    pipeline = get_pipeline(engine, vram_profile)

    res = pipeline.create_character(
        prompt=prompt,
        reference_image=ref_image,
        target_size=target_size,
        palette_name=palette_name,
        symmetry_5to8=symmetry,
        remove_bg=remove_bg,
        clean_noise=clean_noise,
        seed=int(seed),
        output_dir=OUTPUT_DIR
    )

    views = res["views"]
    assert isinstance(views, dict)
    ordered_views = [views[d] for d in DIRECTION_ORDER if d in views]

    # Create ZIP package
    zip_path = os.path.join(OUTPUT_DIR, f"hero_8dir_{seed}.zip")
    package_frames_zip(views, zip_path, prefix=f"hero_{dim}px")

    return (
        res["turntable_path"],      # Turntable GIF
        res["spritesheet"],         # Spritesheet
        res["compass_grid"],        # Compass Grid
        res["spritesheet_path"],    # Download Spritesheet
        res["turntable_path"],      # Download Turntable
        zip_path,                   # Download ZIP
        ordered_views               # Gallery
    )


def slice_and_process_sheet(
    sheet_img: Optional[Image.Image],
    sheet_type: str,
    size_str: str,
    palette_choice: str,
    tolerance: int,
    remove_bg: bool,
    clean_noise: bool,
    name_prefix: str
):
    if sheet_img is None:
        return None, None, None, None, None, None, []

    dim = int(size_str.split("x")[0].strip())
    target_size = (dim, dim)
    palette_name = None if palette_choice == "None (Full Color)" else palette_choice.lower()
    prefix = name_prefix.strip() or "character"

    # 1. Slice frames
    if "1x8" in sheet_type:
        raw_frames = slice_horizontal_strip(sheet_img, count=8, order=DIRECTION_ORDER)
    elif "1x5" in sheet_type:
        five = slice_horizontal_strip(sheet_img, count=5, order=["S", "SE", "E", "NE", "N"])
        raw_frames = derive_8_directions_from_5(five)
    elif "3x3" in sheet_type:
        grid_frames = slice_grid_3x3(sheet_img)
        raw_frames = {d: grid_frames[d] for d in DIRECTION_ORDER if d in grid_frames}
    else:
        raw_frames = slice_horizontal_strip(sheet_img, count=8, order=DIRECTION_ORDER)

    # 2. Process each frame
    processed_views: Dict[str, Image.Image] = {}
    for d in DIRECTION_ORDER:
        if d in raw_frames:
            f = raw_frames[d]
            if remove_bg:
                f = remove_background_border_floodfill(f, tolerance=int(tolerance))
            f = downscale_pixel_art(f, target_size=target_size, preserve_details=True)
            if palette_name:
                f = quantize_palette(f, palette_name=palette_name)
            if clean_noise:
                f = clean_stray_pixels(f)
            processed_views[d] = f

    # 3. Assemble Exports
    spritesheet = assemble_horizontal_spritesheet(processed_views)
    sheet_path = os.path.join(OUTPUT_DIR, f"{prefix}_spritesheet_{dim}px.png")
    spritesheet.save(sheet_path)

    compass_grid = assemble_compass_grid(processed_views)
    compass_path = os.path.join(OUTPUT_DIR, f"{prefix}_compass_{dim}px.png")
    compass_grid.save(compass_path)

    turntable_path = os.path.join(OUTPUT_DIR, f"{prefix}_turntable_{dim}px.gif")
    assemble_turntable(processed_views, turntable_path, fps=7, scale_factor=max(1, 256 // dim))

    zip_path = os.path.join(OUTPUT_DIR, f"{prefix}_frames_{dim}px.zip")
    package_frames_zip(processed_views, zip_path, prefix=prefix)

    ordered_views = [processed_views[d] for d in DIRECTION_ORDER if d in processed_views]

    return (
        turntable_path,
        spritesheet,
        compass_grid,
        sheet_path,
        turntable_path,
        zip_path,
        ordered_views
    )


def process_custom_image(
    input_img: Optional[Image.Image],
    size_str: str,
    palette_choice: str,
    tolerance: int,
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
        img = remove_background_border_floodfill(img, tolerance=int(tolerance))
    img = downscale_pixel_art(img, target_size=target_size, preserve_details=True)
    if palette_name:
        img = quantize_palette(img, palette_name=palette_name)
    if clean_noise:
        img = clean_stray_pixels(img)

    display_scale = max(2, min(8, 256 // dim))
    display_img = img.resize((dim * display_scale, dim * display_scale), Image.Resampling.NEAREST)
    return img, display_img


def check_forge_status() -> str:
    forge_url = config.get("forge", {}).get("url", "http://127.0.0.1:7860")
    if is_forge_available(forge_url):
        return f"🟢 Forge WebUI Connected at {forge_url}"
    return f"⚪ Forge WebUI Not Running ({forge_url})"


# Build Gradio Web Interface
with gr.Blocks(title="PixelArt 8-Dir Studio") as demo:
    gr.Markdown(
        """
        # 🎮 PixelArt 8-Dir Studio
        ### High-Precision 8-Directional Consistent Pixel Art Generator & Game Exporter
        Supports top-down / isometric sprites with **locked character identity**, **border floodfill matting**, and **crisp 16/32-bit palette quantization**.
        """
    )

    with gr.Tabs():
        # --- TAB 1: 8-DIRECTION CHARACTER GENERATOR ---
        with gr.Tab("🎨 8-Direction Generator"):
            with gr.Row():
                with gr.Column(scale=4):
                    prompt_input = gr.Textbox(
                        label="Character Prompt",
                        placeholder="e.g. 1girl, paladin, gorgeous anime girl, silver armor, blue cape, full body",
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
                            choices=["32x32", "48x48", "64x64", "128x128", "256x256"],
                            value="64x64"
                        )
                        palette_dropdown = gr.Dropdown(
                            label="Color Palette",
                            choices=["DB32", "Pico-8", "Endesga-32", "Sweetie-16", "GameBoy", "None (Full Color)"],
                            value="DB32"
                        )

                    with gr.Row():
                        symmetry_cb = gr.Checkbox(
                            label="5-to-8 Direction Symmetry (Mirror SE/E/NE -> SW/W/NW)",
                            value=True,
                            info="Recommended: Guarantees 100% limb alignment and zero drift"
                        )
                        remove_bg_cb = gr.Checkbox(
                            label="Border Floodfill Cutout (Transparent RGBA)",
                            value=True
                        )
                        clean_noise_cb = gr.Checkbox(
                            label="Clean Stray Noise Pixels",
                            value=True
                        )

                    with gr.Accordion("⚙️ Engine & Hardware Profile", open=True):
                        engine_radio = gr.Radio(
                            label="Generation Engine",
                            choices=[
                                "Mock / Demo Mode (Instant, No GPU needed)",
                                "Stable Diffusion WebUI Forge (Local 7860)",
                                "Local SD 1.5 + ControlNet (Local CUDA)"
                            ],
                            value="Mock / Demo Mode (Instant, No GPU needed)"
                        )
                        forge_status_text = gr.Markdown(value=check_forge_status)

                        vram_radio = gr.Radio(
                            label="Hardware VRAM Profile",
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
                        label="📜 Spritesheet (1x8 Horizontal Strip)",
                        type="pil",
                        interactive=False
                    )

                    with gr.Row():
                        download_sheet = gr.File(label="Download Spritesheet (.png)")
                        download_turntable = gr.File(label="Download Turntable (.gif)")
                        download_zip = gr.File(label="Download 8-Dir Frames (.zip)")

                    gallery_out = gr.Gallery(
                        label="Individual Frames (S, SE, E, NE, N, NW, W, SW)",
                        columns=4,
                        rows=2,
                        height="auto"
                    )

            generate_btn.click(
                fn=generate_character,
                inputs=[
                    prompt_input, ref_input, size_dropdown, palette_dropdown,
                    symmetry_cb, remove_bg_cb, clean_noise_cb,
                    engine_radio, vram_radio, seed_slider
                ],
                outputs=[
                    turntable_out, spritesheet_out, compass_out,
                    download_sheet, download_turntable, download_zip, gallery_out
                ]
            )

        # --- TAB 2: SHEET SLICER & GAME EXPORTER ---
        with gr.Tab("✂️ Sheet Slicer & Game Exporter"):
            gr.Markdown(
                """
                ### Import Pre-Generated Character Sheets (from Forge, Midjourney, ChatGPT)
                Upload a 1x8 strip, 1x5 strip, or 3x3 grid. The studio will slice, apply **border floodfill matting**,
                downscale with **edge preservation**, **quantize colors**, and package into game-ready sprites!
                """
            )
            with gr.Row():
                with gr.Column(scale=4):
                    sheet_input = gr.Image(label="Source Sheet Image", type="pil")
                    sheet_layout = gr.Radio(
                        label="Sheet Layout Format",
                        choices=[
                            "1x8 Horizontal Strip (S, SE, E, NE, N, NW, W, SW)",
                            "1x5 Horizontal Strip (S, SE, E, NE, N -> Auto Mirror to 8)",
                            "3x3 Compass Grid"
                        ],
                        value="1x8 Horizontal Strip (S, SE, E, NE, N, NW, W, SW)"
                    )
                    with gr.Row():
                        slicer_size = gr.Dropdown(
                            label="Output Resolution",
                            choices=["32x32", "48x48", "64x64", "128x128", "256x256"],
                            value="256x256"
                        )
                        slicer_palette = gr.Dropdown(
                            label="Color Palette",
                            choices=["None (Full Color)", "DB32", "Pico-8", "Endesga-32", "Sweetie-16", "GameBoy"],
                            value="None (Full Color)"
                        )

                    with gr.Row():
                        slicer_tolerance = gr.Slider(
                            label="Background Tolerance",
                            minimum=5,
                            maximum=80,
                            step=1,
                            value=28,
                            info="Adjust for green/white screen sensitivity"
                        )
                        slicer_prefix = gr.Textbox(label="Character Class / Name", value="hero_idle")

                    with gr.Row():
                        slicer_bg = gr.Checkbox(label="Border Floodfill Cutout", value=True)
                        slicer_clean = gr.Checkbox(label="Clean Stray Noise Pixels", value=True)

                    slice_btn = gr.Button("✂️ Slice, Process & Export 8-Direction Pack", variant="primary", size="lg")

                with gr.Column(scale=5):
                    with gr.Row():
                        slicer_turntable_out = gr.Image(label="🔄 360° Turntable Preview", type="filepath")
                        slicer_compass_out = gr.Image(label="🧭 3x3 Compass Grid", type="pil")

                    slicer_sheet_out = gr.Image(label="📜 Processed 1x8 Strip", type="pil")

                    with gr.Row():
                        slicer_dl_sheet = gr.File(label="Spritesheet (.png)")
                        slicer_dl_turntable = gr.File(label="Turntable (.gif)")
                        slicer_dl_zip = gr.File(label="Game Engine ZIP (.zip)")

                    slicer_gallery = gr.Gallery(
                        label="Individual Sliced Frames",
                        columns=4,
                        rows=2,
                        height="auto"
                    )

            slice_btn.click(
                fn=slice_and_process_sheet,
                inputs=[
                    sheet_input, sheet_layout, slicer_size, slicer_palette,
                    slicer_tolerance, slicer_bg, slicer_clean, slicer_prefix
                ],
                outputs=[
                    slicer_turntable_out, slicer_sheet_out, slicer_compass_out,
                    slicer_dl_sheet, slicer_dl_turntable, slicer_dl_zip, slicer_gallery
                ]
            )

        # --- TAB 3: PIXEL LABORATORY ---
        with gr.Tab("🧪 Pixel Lab (Single Image Post-Processor)"):
            gr.Markdown("Convert any single concept art or sprite into sharp, retro 16/32-bit pixel art.")
            with gr.Row():
                with gr.Column():
                    lab_input = gr.Image(label="Source Image", type="pil")
                    lab_size = gr.Dropdown(
                        label="Target Resolution",
                        choices=["16x16", "32x32", "48x48", "64x64", "128x128", "256x256"],
                        value="64x64"
                    )
                    lab_palette = gr.Dropdown(
                        label="Color Palette",
                        choices=["DB32", "Pico-8", "Endesga-32", "Sweetie-16", "GameBoy", "None (Full Color)"],
                        value="DB32"
                    )
                    lab_tol = gr.Slider(label="Background Floodfill Tolerance", minimum=5, maximum=80, value=28, step=1)
                    lab_bg = gr.Checkbox(label="Border Floodfill Cutout (No holes in hair/clothing)", value=True)
                    lab_clean = gr.Checkbox(label="Clean Stray Pixels (Preserves weapon tips)", value=True)
                    lab_btn = gr.Button("Convert to Pixel Art", variant="primary")

                with gr.Column():
                    lab_display = gr.Image(label="Pixel-Perfect Preview (Upscaled)", type="pil")
                    lab_raw = gr.Image(label="Exact Raw File (1:1 scale)", type="pil")

            lab_btn.click(
                fn=process_custom_image,
                inputs=[lab_input, lab_size, lab_palette, lab_tol, lab_bg, lab_clean],
                outputs=[lab_raw, lab_display]
            )

        # --- TAB 4: SETUP & PROMPT RECIPES ---
        with gr.Tab("📖 Setup & Prompt Recipes"):
            gr.Markdown(
                """
                ### 🚀 Recommended WebUI Forge Workflow (Animagine XL 3.1 + Pixel Art LoRA)
                To generate gorgeous, consistent anime characters on your local RTX 5060 Ti:

                #### 1. Positive Prompt Recipe:
                ```text
                1girl, <class_name>, gorgeous anime girl, voluptuous proportions, <sexy minimalist clothing>,
                full body, solid bright green background, chroma key green background, clean background <lora:pixel-art-xl-v1.1:1.0>
                ```
                *(Note: NEVER put 'pixel art' in positive prompt when using pixel-art-xl LoRA)*

                #### 2. Negative Prompt Recipe:
                ```text
                (worst quality, low quality:1.4), (character sheet:1.5), deformed, bad anatomy, bad hands, missing fingers, extra limbs, photo, realistic
                ```

                #### 3. Recommended Generation Settings:
                - **Resolution:** 832x1216 (Single character) or 1024x1024 (Sprite Sheet)
                - **Steps:** 28
                - **CFG Scale:** 7.0
                - **Sampler:** DPM++ 2M or Euler a
                - **Background:** Always request solid chroma green background to allow 100% clean border floodfill with zero edge fringing!
                """
            )

if __name__ == "__main__":
    host = config.get("app", {}).get("host", "127.0.0.1")
    port = config.get("app", {}).get("port", 7865)
    print(f"Starting PixelArt 8-Dir Studio at http://{host}:{port} ...")
    demo.launch(server_name=host, server_port=port, share=False)
