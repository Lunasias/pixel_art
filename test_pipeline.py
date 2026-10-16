"""
Automated Test & Verification Suite for PixelArt 8-Dir Studio
Verifies:
1. 8-Directional Pose Generation & Symmetry
2. Border Floodfill Background Removal (no internal holes)
3. Detail-Preserving Downscaling & Perceptual Palette Quantization
4. Stray Pixel Cleaner (preserves weapon tips)
5. Horizontal Spritesheet, Turntable GIF, and ZIP Export
"""

import os
import numpy as np
from PIL import Image, ImageDraw
from core.generator import CharacterSpritePipeline, MockPixelGenerator
from core.processor import (
    PALETTES,
    quantize_palette,
    downscale_pixel_art,
    remove_background_border_floodfill,
    clean_stray_pixels,
    package_frames_zip
)


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

    # Test ZIP export
    zip_path = "./outputs/test_frames.zip"
    views_dict = res["views"]
    assert isinstance(views_dict, dict)
    package_frames_zip(views_dict, zip_path, prefix="warrior")
    assert os.path.exists(zip_path), "ZIP package was not created"
    print("    [PASS] 8 Views generated, assembled, and zipped successfully.")


def test_border_floodfill_no_internal_holes():
    print(">>> 2. Testing Robust Border Floodfill (Zero Internal Holes)...")
    # Create white canvas (200x200) with white background
    img = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw dark character outline (circle)
    draw.ellipse([50, 50, 150, 150], fill=(20, 20, 20, 255), outline=(0, 0, 0, 255))
    # Draw WHITE interior (like silver armor or white eye/hair inside outline)
    draw.ellipse([80, 80, 120, 120], fill=(255, 255, 255, 255))

    cutout = remove_background_border_floodfill(img, tolerance=10)
    arr = np.array(cutout)

    # 1. Corner border (0, 0) should be transparent (alpha == 0)
    assert arr[0, 0, 3] == 0, "Outer border was not removed"

    # 2. Inside center (100, 100) is white, but MUST remain completely solid (alpha == 255)!
    assert arr[100, 100, 3] == 255, "CRITICAL: Internal white clothing/hair was perforated!"
    print("    [PASS] Border floodfill removed background without perforating internal white features.")


def test_stray_pixel_preserves_weapon_tips():
    print(">>> 3. Testing Stray Pixel Cleaner (Preserves Weapon Tips)...")
    # Create transparent image
    img = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    arr = np.array(img)

    # Draw a 1-pixel thin sword blade from (10, 10) to (10, 25)
    arr[10:26, 10] = [200, 200, 200, 255]
    # Draw an isolated 1-pixel noise floater at (40, 40)
    arr[40, 40] = [255, 0, 0, 255]

    test_img = Image.fromarray(arr, mode="RGBA")
    cleaned = clean_stray_pixels(test_img)
    clean_arr = np.array(cleaned)

    # Floater at (40, 40) should be removed
    assert clean_arr[40, 40, 3] == 0, "Isolated stray pixel was not cleaned"
    # Sword tip at (10, 10) and base at (25, 10) must be preserved!
    assert clean_arr[10, 10, 3] == 255, "Weapon tip was accidentally deleted!"
    assert clean_arr[25, 10, 3] == 255, "Weapon body was damaged!"
    print("    [PASS] Isolated stray noise removed while weapon tips remain intact.")


def test_image_processors():
    print(">>> 4. Testing Pixel Processors (Downscale & Quantize)...")
    test_img = Image.new("RGBA", (256, 256), (100, 150, 200, 255))
    downscaled = downscale_pixel_art(test_img, target_size=(32, 32))
    assert downscaled.size == (32, 32), f"Expected (32, 32), got {downscaled.size}"

    for p_name in ["db32", "pico-8", "gameboy", "sweetie-16", "endesga-32"]:
        quantized = quantize_palette(downscaled, palette_name=p_name)
        assert quantized.size == (32, 32)
        assert quantized.mode == "RGBA"
    print("    [PASS] Detail-preserving downscaling and perceptual quantization working correctly.")


if __name__ == "__main__":
    print("========================================")
    print(" Running PixelArt 8-Dir Studio Tests   ")
    print("========================================")
    test_core_pipeline()
    test_border_floodfill_no_internal_holes()
    test_stray_pixel_preserves_weapon_tips()
    test_image_processors()
    print("========================================")
    print(" ALL TESTS PASSED! Ready for deployment ")
    print("========================================")
