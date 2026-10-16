"""
Pixel Processing Engine
Handles border floodfill background removal, edge-preserving pixel downscaling,
perceptually weighted palette quantization, stray pixel cleaning, 5-to-8 directional mirroring,
and game-ready sprite sheet / turntable assembly.
"""

import io
import zipfile
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image

# Standard curated game palettes (RGB hex)
PALETTES: Dict[str, List[Tuple[int, int, int]]] = {
    "db32": [
        (0, 0, 0), (34, 32, 52), (69, 40, 60), (102, 57, 49), (143, 86, 59),
        (223, 113, 38), (217, 160, 102), (238, 195, 154), (251, 242, 54),
        (153, 229, 80), (106, 190, 48), (55, 148, 110), (75, 105, 47),
        (82, 75, 36), (50, 60, 57), (63, 63, 116), (48, 96, 130),
        (91, 110, 225), (99, 155, 255), (95, 205, 228), (203, 219, 252),
        (255, 255, 255), (155, 173, 183), (132, 126, 135), (105, 106, 106),
        (89, 86, 82), (118, 66, 138), (172, 50, 50), (217, 87, 99),
        (215, 123, 186), (143, 151, 74), (138, 111, 48)
    ],
    "pico-8": [
        (0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
        (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
        (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
        (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170)
    ],
    "endesga-32": [
        (190, 74, 47), (215, 118, 67), (234, 212, 170), (228, 166, 114),
        (184, 111, 80), (115, 62, 57), (62, 39, 49), (162, 38, 51),
        (228, 59, 68), (247, 118, 34), (254, 174, 52), (254, 231, 97),
        (99, 199, 77), (62, 137, 72), (38, 92, 66), (25, 60, 62),
        (18, 78, 137), (0, 153, 219), (44, 232, 244), (255, 255, 255),
        (192, 203, 220), (139, 155, 180), (90, 105, 136), (58, 68, 102),
        (38, 43, 68), (24, 20, 37), (255, 0, 68), (254, 132, 143),
        (254, 174, 178), (255, 213, 218), (175, 42, 130), (118, 29, 90)
    ],
    "gameboy": [
        (15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)
    ],
    "sweetie-16": [
        (26, 28, 44), (93, 39, 93), (177, 62, 83), (239, 125, 87),
        (255, 205, 117), (167, 240, 112), (56, 183, 100), (37, 113, 121),
        (41, 54, 111), (59, 93, 201), (65, 166, 246), (115, 239, 247),
        (244, 244, 244), (148, 176, 194), (86, 108, 134), (51, 60, 87)
    ]
}

# Standard rotation order (counter-clockwise from South / Front)
DIRECTION_ORDER = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
DIRECTION_NAMES = {
    "S": "South (Front)",
    "SE": "South-East",
    "E": "East (Right)",
    "NE": "North-East",
    "N": "North (Back)",
    "NW": "North-West",
    "W": "West (Left)",
    "SW": "South-West"
}


def remove_background_border_floodfill(
    image: Image.Image,
    tolerance: int = 12,
    min_component_size: int = 80
) -> Image.Image:
    """
    Robust background removal using OpenCV border floodfill + connected component filtering.
    Starts floodfill exclusively from borders/corners with conservative tolerance so internal
    features (light hair highlights, specular glints on skin/armor, light eyes) are never
    perforated with holes or eroded.
    """
    img_rgba = image.convert("RGBA")
    arr = np.array(img_rgba)
    h, w = arr.shape[:2]

    rgb = arr[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # OpenCV floodFill requires a mask of size (h + 2, w + 2)
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Dense seed points around all four image borders
    seeds = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2),
        (w // 4, 0), (3 * w // 4, 0), (w // 4, h - 1), (3 * w // 4, h - 1),
        (0, h // 4), (0, 3 * h // 4), (w - 1, h // 4), (w - 1, 3 * h // 4)
    ]

    diff = (int(tolerance), int(tolerance), int(tolerance))
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE

    for sx, sy in seeds:
        if flood_mask[sy + 1, sx + 1] == 0:
            cv2.floodFill(bgr, flood_mask, (sx, sy), 0, diff, diff, flags)

    # Extract inner mask (255 = flooded background)
    bg_mask = flood_mask[1:h + 1, 1:w + 1] == 255

    # Also clean ground contact shadow at bottom 10%
    bottom_y = int(h * 0.90)
    for y in range(bottom_y, h):
        for x in range(w):
            p = rgb[y, x]
            if p[0] > 185 and p[1] > 185 and p[2] > 185:
                bg_mask[y, x] = True

    # Initial foreground mask
    fg_mask = (~bg_mask).astype(np.uint8) * 255

    # Filter out tiny floating background dust
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    clean_fg = np.zeros_like(fg_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_size:
            clean_fg[labels == i] = 255

    # If the image already had transparent areas, respect them
    if np.any(arr[:, :, 3] < 255):
        clean_fg = np.minimum(clean_fg, arr[:, :, 3])

    arr[:, :, 3] = clean_fg
    return Image.fromarray(arr, mode="RGBA")


def remove_background_simple(image: Image.Image, tolerance: int = 28) -> Image.Image:
    """
    Backwards-compatible wrapper routing to the robust border floodfill.
    """
    return remove_background_border_floodfill(image, tolerance=tolerance)


def remove_background_ai(image: Image.Image) -> Image.Image:
    """
    AI background removal via rembg (falls back to border floodfill if rembg not installed).
    """
    try:
        from rembg import remove
        return remove(image)
    except Exception:
        return remove_background_border_floodfill(image)


def downscale_pixel_art(
    image: Image.Image,
    target_size: Tuple[int, int] = (64, 64),
    preserve_details: bool = True
) -> Image.Image:
    """
    Downscales images to target sprite resolution while preserving sharp retro 16/32-bit details.
    When downscaling from high-res (e.g. 512+ to 64/128), uses high-quality Lanczos resampling
    followed by binary alpha thresholding to prevent facial melting and broken weapon edges.
    """
    img = image.convert("RGBA")
    w, h = img.size
    tw, th = target_size

    if (w, h) == (tw, th):
        return img

    if preserve_details and (w > tw * 1.5 or h > th * 1.5):
        # Lanczos downscale preserves structural line density (eyes, curves, silhouettes)
        downscaled = img.resize((tw, th), Image.Resampling.LANCZOS)
        arr = np.array(downscaled)
        # Snap alpha to binary (threshold at 64 to preserve delicate hair tips and fingertips)
        arr[:, :, 3] = np.where(arr[:, :, 3] > 64, 255, 0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGBA")
    else:
        return img.resize((tw, th), Image.Resampling.NEAREST)


def quantize_palette(
    image: Image.Image,
    palette_name: Optional[str] = "db32",
    custom_palette: Optional[List[Tuple[int, int, int]]] = None,
    max_colors: int = 32
) -> Image.Image:
    """
    Quantizes image colors to a crisp retro pixel art palette using human-perceptual
    weighted Euclidean distance (2*dR^2 + 4*dG^2 + 3*dB^2) while preserving alpha transparency.
    """
    img = image.convert("RGBA")
    data = np.array(img)
    alpha = data[:, :, 3]
    rgb = data[:, :, :3]

    if custom_palette is not None and len(custom_palette) > 0:
        pal_colors = np.array(custom_palette, dtype=np.float32)
    elif palette_name and palette_name.lower() in PALETTES:
        pal_colors = np.array(PALETTES[palette_name.lower()], dtype=np.float32)
    else:
        # Dynamic K-Means / Median Cut palette if no preset specified
        pil_rgb = Image.fromarray(rgb)
        quantized_pil = pil_rgb.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT).convert("RGB")
        res = np.array(quantized_pil)
        out = np.dstack((res, alpha))
        return Image.fromarray(out, mode="RGBA")

    visible_mask = alpha > 0
    if np.any(visible_mask):
        flat_rgb = rgb[visible_mask].astype(np.float32)
        # Perceptually weighted color difference: Human eye is most sensitive to Green, then Red, then Blue
        diff = flat_rgb[:, np.newaxis, :] - pal_colors[np.newaxis, :, :]  # (N, P, 3)
        weights = np.array([2.0, 4.0, 3.0], dtype=np.float32)
        dists = np.sum((diff ** 2) * weights, axis=2)
        closest_indices = np.argmin(dists, axis=1)

        new_rgb = np.copy(rgb)
        new_rgb[visible_mask] = pal_colors[closest_indices].astype(np.uint8)
    else:
        new_rgb = rgb

    out = np.dstack((new_rgb, alpha))
    return Image.fromarray(out, mode="RGBA")


def clean_stray_pixels(image: Image.Image, max_island_size: int = 1) -> Image.Image:
    """
    Removes isolated orphan pixels (single noise specks) without clipping weapon tips,
    hair points, or antennas. Uses connected component analysis on opaque regions so that
    tips of connected lines (swords, staves) remain 100% intact.
    """
    img = image.convert("RGBA")
    data = np.array(img)
    alpha = data[:, :, 3]
    opaque = (alpha > 32).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opaque, connectivity=8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_island_size:
            # Clear only completely isolated 1-pixel floaters
            data[:, :, 3][labels == i] = 0

    return Image.fromarray(data, mode="RGBA")


def mirror_horizontal(image: Image.Image) -> Image.Image:
    """Horizontally flip an image (for SE -> SW, E -> W, NE -> NW)."""
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def derive_8_directions_from_5(five_views: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
    """
    Enforces 5-to-8 directional symmetry:
    Input must contain 'S', 'SE', 'E', 'NE', 'N'.
    Automatically generates:
      'SW' = mirror('SE')
      'W'  = mirror('E')
      'NW' = mirror('NE')
    Guarantees zero limb drift and perfectly consistent character proportions.
    """
    required = ["S", "SE", "E", "NE", "N"]
    for key in required:
        if key not in five_views:
            raise ValueError(f"Missing required direction '{key}' in five_views")

    res = dict(five_views)
    res["SW"] = mirror_horizontal(five_views["SE"])
    res["W"]  = mirror_horizontal(five_views["E"])
    res["NW"] = mirror_horizontal(five_views["NE"])

    return {k: res[k] for k in DIRECTION_ORDER}


def slice_grid_3x3(sheet: Image.Image) -> Dict[str, Image.Image]:
    """
    Slices a 3x3 compass grid sheet into 8 directions:
    Grid Layout:
      [ NW ][  N ][ NE ]
      [  W ][ C* ][  E ]   (* Center is reference concept or idle)
      [ SW ][  S ][ SE ]
    """
    img = sheet.convert("RGBA")
    w, h = img.size
    cell_w = w // 3
    cell_h = h // 3

    layout = [
        ["NW", "N", "NE"],
        ["W",  "CENTER", "E"],
        ["SW", "S", "SE"]
    ]

    frames: Dict[str, Image.Image] = {}
    for r in range(3):
        for c in range(3):
            name = layout[r][c]
            box = (c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h)
            cell = img.crop(box)
            frames[name] = cell

    return frames


def slice_horizontal_strip(
    sheet: Image.Image,
    count: int = 8,
    order: Optional[List[str]] = None
) -> Dict[str, Image.Image]:
    """
    Slices a 1xN horizontal strip into directions.
    """
    if order is None:
        order = DIRECTION_ORDER[:count]
    img = sheet.convert("RGBA")
    w, h = img.size
    cell_w = w // count

    frames: Dict[str, Image.Image] = {}
    for i, name in enumerate(order):
        box = (i * cell_w, 0, (i + 1) * cell_w, h)
        frames[name] = img.crop(box)
    return frames


def assemble_horizontal_spritesheet(
    frames_dict: Dict[str, Image.Image],
    order: Optional[List[str]] = None,
    padding: int = 0
) -> Image.Image:
    """
    Stitches individual directional frames into a game-ready horizontal sprite sheet.
    """
    if order is None:
        order = DIRECTION_ORDER

    valid_frames = [frames_dict[d] for d in order if d in frames_dict]
    if not valid_frames:
        raise ValueError("No matching frames found for spritesheet assembly")

    frame_w, frame_h = valid_frames[0].size
    n = len(valid_frames)

    sheet_w = (frame_w * n) + (padding * (n - 1))
    sheet_h = frame_h

    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
    for i, frame in enumerate(valid_frames):
        x = i * (frame_w + padding)
        sheet.paste(frame, (x, 0))

    return sheet


def assemble_compass_grid(
    frames_dict: Dict[str, Image.Image],
    center_image: Optional[Image.Image] = None
) -> Image.Image:
    """
    Packs 8 directions into a 3x3 compass layout (with optional center frame).
    """
    sample = list(frames_dict.values())[0]
    fw, fh = sample.size
    grid = Image.new("RGBA", (fw * 3, fh * 3), (0, 0, 0, 0))

    pos_map = {
        "NW": (0, 0), "N": (1, 0), "NE": (2, 0),
        "W":  (0, 1),             "E":  (2, 1),
        "SW": (0, 2), "S": (1, 2), "SE": (2, 2),
    }

    for dir_key, (col, row) in pos_map.items():
        if dir_key in frames_dict:
            grid.paste(frames_dict[dir_key], (col * fw, row * fh))

    if center_image:
        center_resized = center_image.resize((fw, fh), Image.Resampling.NEAREST)
        grid.paste(center_resized, (fw, fh))

    return grid


def assemble_turntable(
    frames_dict: Dict[str, Image.Image],
    output_path: str,
    fps: int = 6,
    scale_factor: int = 4
) -> str:
    """
    Creates an animated turntable (.gif or .webp) rotating smoothly through all 8 angles.
    """
    ordered_frames = []
    for d in DIRECTION_ORDER:
        if d in frames_dict:
            frame = frames_dict[d]
            if scale_factor > 1:
                w, h = frame.size
                frame = frame.resize((w * scale_factor, h * scale_factor), Image.Resampling.NEAREST)
            ordered_frames.append(frame)

    if not ordered_frames:
        raise ValueError("No frames available for turntable")

    duration = int(1000 / fps)

    if output_path.lower().endswith(".webp"):
        ordered_frames[0].save(
            output_path,
            save_all=True,
            append_images=ordered_frames[1:],
            duration=duration,
            loop=0,
            lossless=True
        )
    else:  # Default to GIF
        ordered_frames[0].save(
            output_path,
            save_all=True,
            append_images=ordered_frames[1:],
            duration=duration,
            loop=0,
            disposal=2
        )
    return output_path


def package_frames_zip(
    frames_dict: Dict[str, Image.Image],
    zip_path: str,
    prefix: str = "hero"
) -> str:
    """
    Packages individual directional frames into a ZIP file ready for game engine import.
    Example files inside zip:
      hero_S.png, hero_SE.png, hero_E.png, hero_NE.png,
      hero_N.png, hero_NW.png, hero_W.png, hero_SW.png
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in DIRECTION_ORDER:
            if d in frames_dict:
                img_byte_arr = io.BytesIO()
                frames_dict[d].save(img_byte_arr, format="PNG")
                zf.writestr(f"{prefix}_{d}.png", img_byte_arr.getvalue())
    return zip_path
