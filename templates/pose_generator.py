"""
Pose and Guide Template Generator for 8-Directional ControlNet.
Generates 3D-angled top-down depth / wireframe stick poses for:
S (0°), SE (45°), E (90°), NE (135°), N (180°), NW (225°), W (270°), SW (315°).
"""

from typing import Dict, Tuple
import math
import numpy as np
from PIL import Image, ImageDraw

# Standard 8 directions
DIRECTIONS = {
    "S": 0,       # Facing camera (Front)
    "SE": 45,     # Front-right
    "E": 90,      # Right profile
    "NE": 135,    # Back-right
    "N": 180,     # Back
    "NW": 225,    # Back-left
    "W": 270,     # Left profile
    "SW": 315     # Front-left
}


def create_topdown_pose_depth(
    angle_deg: float,
    size: Tuple[int, int] = (512, 512),
    tilt_angle_deg: float = 30.0
) -> Image.Image:
    """
    Renders a pseudo-3D mannequin depth map for a top-down / 2.5D RPG perspective character.
    Colors represent depth (brighter = closer to camera).
    """
    w, h = size
    img = Image.new("L", size, 0)  # Grayscale depth map (0 = deep black background)
    draw = ImageDraw.Draw(img)
    
    rad = math.radians(angle_deg)
    tilt_rad = math.radians(tilt_angle_deg)
    
    # Character center pivot (ground level at bottom center)
    cx, cy = w / 2, h * 0.72
    scale = h * 0.38
    
    # 3D points relative to character root (X: right, Y: forward, Z: up)
    # Head, Shoulders, Hands, Torso, Hips, Feet
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
    cos_t = math.cos(tilt_rad)
    sin_t = math.sin(tilt_rad)
    
    def project(x: float, y: float, z: float) -> Tuple[float, float, int]:
        # Rotate around Z axis (heading angle)
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        rz = z
        
        # Tilt perspective (Top-down view looking down by tilt_rad)
        px = cx + rx * scale
        py = cy - (rz * cos_t - ry * sin_t) * scale
        
        # Depth value (Z-distance towards camera)
        depth_val = 128 + int((rz * sin_t + ry * cos_t) * 70)
        depth_val = max(30, min(255, depth_val))
        return px, py, depth_val

    # Body joints in local coordinates
    head_z = 1.35
    neck_z = 1.15
    hip_z = 0.65
    
    # Shoulders
    l_shoulder = (-0.28, 0.0, 1.10)
    r_shoulder = (0.28, 0.0, 1.10)
    
    # Hands (slightly forward)
    l_hand = (-0.35, 0.15, 0.60)
    r_hand = (0.35, 0.15, 0.60)
    
    # Feet
    l_foot = (-0.18, 0.0, 0.05)
    r_foot = (0.18, 0.0, 0.05)
    
    # Project parts
    hx, hy, hd = project(0, 0, head_z)
    nx, ny, nd = project(0, 0, neck_z)
    hip_x, hip_y, hip_d = project(0, 0, hip_z)
    
    ls_x, ls_y, ls_d = project(*l_shoulder)
    rs_x, rs_y, rs_d = project(*r_shoulder)
    
    lh_x, lh_y, lh_d = project(*l_hand)
    rh_x, rh_y, rh_d = project(*r_hand)
    
    lf_x, lf_y, lf_d = project(*l_foot)
    rf_x, rf_y, rf_d = project(*r_foot)
    
    # Draw body parts with depth gradients (drawing back to front)
    # Legs
    leg_w = int(scale * 0.14)
    draw.line([(hip_x, hip_y), (lf_x, lf_y)], fill=int((hip_d + lf_d)/2), width=leg_w)
    draw.line([(hip_x, hip_y), (rf_x, rf_y)], fill=int((hip_d + rf_d)/2), width=leg_w)
    
    # Torso
    torso_w = int(scale * 0.32)
    draw.line([(hip_x, hip_y), (nx, ny)], fill=int((hip_d + nd)/2), width=torso_w)
    
    # Arms
    arm_w = int(scale * 0.10)
    draw.line([(ls_x, ls_y), (lh_x, lh_y)], fill=int((ls_d + lh_d)/2), width=arm_w)
    draw.line([(rs_x, rs_y), (rh_x, rh_y)], fill=int((rs_d + rh_d)/2), width=arm_w)
    
    # Head (Sphere with depth)
    head_rad = int(scale * 0.20)
    draw.ellipse(
        [hx - head_rad, hy - head_rad, hx + head_rad, hy + head_rad],
        fill=hd
    )
    
    # Draw face nose/visor direction indicator for orientation cue
    fx, fy, fd = project(0, -0.22, head_z)
    draw.line([(hx, hy), (fx, fy)], fill=min(255, hd + 35), width=int(scale * 0.05))
    
    return img


def generate_all_8_poses(size: Tuple[int, int] = (512, 512)) -> Dict[str, Image.Image]:
    """Generates a full set of 8 depth poses for ControlNet guidance."""
    return {
        direction: create_topdown_pose_depth(angle, size=size)
        for direction, angle in DIRECTIONS.items()
    }


def create_3x3_anchor_template(
    reference_image: Image.Image,
    poses_dict: Dict[str, Image.Image],
    cell_size: Tuple[int, int] = (256, 256)
) -> Image.Image:
    """
    Assembles a 3x3 anchor template sheet matching PixelLab / Sprute layout.
    Center cell = Reference Character
    Outer 8 cells = Directional guidance / poses
    """
    cw, ch = cell_size
    grid = Image.new("RGB", (cw * 3, ch * 3), (128, 128, 128))
    
    pos_map = {
        "NW": (0, 0), "N": (1, 0), "NE": (2, 0),
        "W":  (0, 1),             "E":  (2, 1),
        "SW": (0, 2), "S": (1, 2), "SE": (2, 2),
    }
    
    # Paste poses into perimeter slots
    for dir_key, (c, r) in pos_map.items():
        if dir_key in poses_dict:
            pose_resized = poses_dict[dir_key].resize(cell_size, Image.Resampling.BILINEAR)
            grid.paste(pose_resized.convert("RGB"), (c * cw, r * ch))
            
    # Paste reference into center slot
    ref_resized = reference_image.resize(cell_size, Image.Resampling.BILINEAR)
    grid.paste(ref_resized.convert("RGB"), (cw, ch))
    
    return grid
