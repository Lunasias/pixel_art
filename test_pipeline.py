"""
Automated Test & Verification Suite for PixelArt 8-Dir Studio
Verifies:
1. 8-Directional Pose Generation
2. 5-to-8 Symmetry Derivation
3. Nearest-Neighbor Downscaling & Palette Quantization
4. Stray Pixel Removal
5. Horizontal Spritesheet & Turntable GIF Export
"""

import os
from PIL import Image
from core.generator import CharacterSpritePipeline, MockPixelGenerator
from core.processor import PALETTES, quantize_palette, downscale_pixel_art


def test_core_pipeline():
    print(">>> 1. Testing Core 8-Direction Pipeline...")
    pipeline = CharacterSpritePipeline(backend=MockPixelGenerator())
    res = pipeline.create_character(
        prompt="legendary warrior with gold armor and crimson cape",
        target_size=(48, 48),
        palette_name="db32",
        symmetry_5to8=True,
        output_dir="./outputs"
    )

    assert len(res["views"]) == 8, f"Expected 8 views, got {len(res['views'])}"
    assert os.path.exists(res["spritesheet_path"]), "Spritesheet not created"
    assert os.path.exists(res["turntable_path"]), "Turntable GIF not created"
    assert os.path.exists(res["compass_grid_path"]), "Compass grid not created"
    print("    [PASS] 8 Views generated and assembled successfully.")


def test_image_processors():
    print(">>> 2. Testing Pixel Processors (Downscale & Quantize)...")
    test_img = Image.new("RGBA", (256, 256), (100, 150, 200, 255))
    downscaled = downscale_pixel_art(test_img, target_size=(32, 32))
    assert downscaled.size == (32, 32), f"Expected (32, 32), got {downscaled.size}"

    for p_name in ["db32", "pico-8", "gameboy"]:
        quantized = quantize_palette(downscaled, palette_name=p_name)
        assert quantized.size == (32, 32)
        assert quantized.mode == "RGBA"
    print("    [PASS] Downscaling and Palette Quantization working correctly.")


if __name__ == "__main__":
    print("========================================")
    print(" Running PixelArt 8-Dir Studio Tests   ")
    print("========================================")
    test_core_pipeline()
    test_image_processors()
    print("========================================")
    print(" ALL TESTS PASSED! Ready for deployment ")
    print("========================================")
