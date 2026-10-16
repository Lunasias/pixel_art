"""
Pixel Processing Engine
Handles background removal, nearest-neighbor downscaling, color quantization (palettes),
stray pixel cleaning, 5-to-8 directional mirroring, and sprite sheet / turntable assembly.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
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


def remove_background_simple(image: Image.Image, tolerance: int = 25) -> Image.Image:
    """
    Fast background removal using corner color floodfill/distance keyer.
    Works seamlessly on flat AI generation backgrounds without requiring GPU.
    """
    img = image.convert("RGBA")
    data = np.array(img)
    rgb = data[:, :, :3]
    
    # Sample 4 corners
    corners = [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]]
    bg_color = np.median(corners, axis=0)
    
    # Calculate Euclidean distance from bg color
    dist = np.sqrt(np.sum((rgb.astype(np.float32) - bg_color) ** 2, axis=2))
    mask = dist <= tolerance
    
    # Set matching pixels to transparent
    data[:, :, 3][mask] = 0
    return Image.fromarray(data, mode="RGBA")


def remove_background_ai(image: Image.Image) -> Image.Image:
    """
    AI-powered background removal via rembg (falls back to simple keyer if rembg is not installed).
    """
    try:
        from rembg import remove
        return remove(image)
    except Exception:
        return remove_background_simple(image)


def downscale_pixel_art(image: Image.Image, target_size: Tuple[int, int] = (64, 64)) -> Image.Image:
    """
    Strict Nearest-Neighbor downscaling to preserve hard pixel edges.
    Never uses bilinear or bicubic filtering to avoid anti-aliasing blur.
    """
    img = image.convert("RGBA")
    return img.resize(target_size, Image.Resampling.NEAREST)


def quantize_palette(
    image: Image.Image,
    palette_name: Optional[str] = "db32",
    custom_palette: Optional[List[Tuple[int, int, int]]] = None,
    max_colors: int = 32
) -> Image.Image:
    """
    Quantizes image colors to a crisp pixel art palette while preserving exact alpha transparency.
    """
    img = image.convert("RGBA")
    data = np.array(img)
    alpha = data[:, :, 3]
    rgb = data[:, :, :3]
    
    # Determine palette colors
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

    # Map each visible pixel to the nearest palette color in Euclidean RGB space
    h, w, _ = rgb.shape
    visible_mask = alpha > 0
    
    if np.any(visible_mask):
        flat_rgb = rgb[visible_mask].astype(np.float32)
        # Compute squared distances between pixels and palette colors
        # (N, 1, 3) - (1, P, 3) -> (N, P)
        dists = np.sum((flat_rgb[:, np.newaxis, :] - pal_colors[np.newaxis, :, :]) ** 2, axis=2)
        closest_indices = np.argmin(dists, axis=1)
        
        # Replace colors with palette equivalents
        new_rgb = np.copy(rgb)
        new_rgb[visible_mask] = pal_colors[closest_indices].astype(np.uint8)
    else:
        new_rgb = rgb

    out = np.dstack((new_rgb, alpha))
    return Image.fromarray(out, mode="RGBA")


def clean_stray_pixels(image: Image.Image) -> Image.Image:
    """
    Removes orphan single pixels (noise common in AI pixel art generators).
    An opaque pixel with fewer than 2 orthogonal/diagonal opaque neighbors is removed.
    """
    img = image.convert("RGBA")
    data = np.array(img)
    alpha = data[:, :, 3]
    opaque = (alpha > 32).astype(np.uint8)
    
    h, w = opaque.shape
    padded = np.pad(opaque, 1, mode="constant", constant_values=0)
    
    # Count 8-way neighbors
    neighbor_counts = (
        padded[0:h, 0:w] + padded[0:h, 1:w+1] + padded[0:h, 2:w+2] +
        padded[1:h+1, 0:w]                     + padded[1:h+1, 2:w+2] +
        padded[2:h+2, 0:w] + padded[2:h+2, 1:w+1] + padded[2:h+2, 2:w+2]
    )
    
    # If pixel is opaque but has < 2 neighbors, it's an isolated stray pixel
    stray_mask = (opaque == 1) & (neighbor_counts < 2)
    data[:, :, 3][stray_mask] = 0
    
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
    Ensures 100% geometric and limb alignment between left and right!
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


def slice_horizontal_strip(sheet: Image.Image, count: int = 8, order: Optional[List[str]] = None) -> Dict[str, Image.Image]:
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
    Scales up with nearest-neighbor so it looks sharp in previews.
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
