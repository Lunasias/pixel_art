# 🎮 PixelArt 8-Dir Studio (PixelLab.ai Open-Source Alternative)

> **AI-Powered 8-Directional Consistent Pixel Art Character Generator**  
> รองรับการสร้างตัวละครพิกเซลอาร์ตมุมมองด้านบน (Top-Down / Isometric) 8 ทิศทาง โดยรักษาเอกลักษณ์ของตัวละคร (Character Consistency) ไม่ให้เพี้ยน พร้อมแปลงเป็น True Pixel Art อัตโนมัติ

---

## ✨ คุณสมบัติเด่น (Key Features)

1. **Character Consistency (รักษาภาพลักษณ์เดิมข้าม 8 ทิศทาง)**:
   - **5-to-8 Directional Symmetry**: เจน 5 ทิศทางหลัก (`S`, `SE`, `E`, `NE`, `N`) แล้วสะท้อนแนวนอน (Horizontal Mirror) เป็น `SW`, `W`, `NW` โดยอัตโนมัติ การันตีตำแหน่งข้อต่อและสัดส่วนตรงกัน 100% ป้องกันปัญหาแขนขาบิดเบี้ยว
   - **IP-Adapter + ControlNet Guidance**: ใช้ภาพ Concept อ้างอิง 1 ภาพในการล็อกหน้าตาและเสื้อผ้า พร้อมใช้ Depth Map 3D 8 มุมมองควบคุมทิศทางการหัน
   - **3x3 Compass Grid**: รองรับการเจนภาพทั้ง 8 ทิศทางบน Canvas ผืนเดียวรอบทิศทาง (Single-pass attention)
2. **True Pixel Art Pipeline (ระบบแปลงเป็นพิกเซลอาร์ตแท้)**:
   - **Strict Nearest-Neighbor Downscaling**: ย่อขนาดภาพเป็น 32x32, 48x48, 64x64 หรือ 128x128 โดยไม่เกิดอาการเบลอ (Anti-aliasing blur)
   - **Palette Quantization**: บังคับพาเลทสีมาตรฐานเกม (DB32, Pico-8, Endesga-32, GameBoy, Sweetie-16) ป้องกันสีไล่เฉดฟุ้งของ AI
   - **Stray Pixel Cleaner**: อัลกอริทึมลบพิกเซลขยะรอบตัวละครอัตโนมัติ
   - **Alpha Transparency Matting**: ไดคัทฉากหลังให้เป็นพื้นหลังโปร่งใส (RGBA)
3. **Game-Ready Export**:
   - ส่งออกเป็น Spritesheet แนวนอน (1x8 Strip)
   - ส่งออกเป็น 3x3 Compass Grid
   - ส่งออกเป็น Turntable Animation 360° (.gif / .webp)
   - พรีวิวหมุนรอบตัวแบบ Real-time บน Web UI
4. **Hardware Profiles (รองรับตั้งแต่ 4GB จนถึง 8GB+ VRAM)**:
   - **4GB Low VRAM (RTX 3050 Laptop)**: Sequential CPU Offloading + SD 1.5 + FP16
   - **8GB Standard (RTX 5060 Ti Desktop)**: Fast SDPA + Model Offloading + SDXL / SD 1.5 Batching

---

## 🧭 ลำดับทิศทางทั้ง 8 ทิศ (Canonical Direction Order)

| รหัสทิศ | ทิศทาง | องศา (Angle) | วิธีการสร้าง |
| :---: | :---: | :---: | :---: |
| **S** | South (หน้าตรง) | 0° | AI Generated |
| **SE** | South-East (หน้าเฉียงขวา) | 45° | AI Generated |
| **E** | East (หันขวา) | 90° | AI Generated |
| **NE** | North-East (หลังเฉียงขวา) | 135° | AI Generated |
| **N** | North (หลังตรง) | 180° | AI Generated |
| **NW** | North-West (หลังเฉียงซ้าย) | 225° | Mirrored from NE |
| **W** | West (หันซ้าย) | 270° | Mirrored from E |
| **SW** | South-West (หน้าเฉียงซ้าย) | 315° | Mirrored from SE |

---

## 🚀 การติดตั้งและเริ่มต้นใช้งาน (Quickstart)

แนะนำให้ใช้ [`uv`](https://github.com/astral-sh/uv) เพราะติดตั้ง Python และ Dependency ได้เร็วมาก:

### 1. Clone โปรเจกต์
```bash
git clone https://github.com/Lunasias/pixel_art.git
cd pixel_art
```

### 2. สร้าง Virtual Environment
```bash
# สร้าง environment ด้วย Python 3.12
uv venv .venv --python 3.12
source .venv/bin/activate   # บน Windows: .venv\Scripts\activate
```

### 3. ติดตั้ง Dependencies
```bash
# ติดตั้ง Core และ Web UI (รัน Mock Mode ได้ทันทีโดยไม่ต้องโหลดโมเดลหลาย GB)
uv pip install -e .

# (ตัวเลือกเสริม) ติดตั้งโมดูล AI สำหรับรัน Local GPU ด้วย PyTorch + Diffusers
uv pip install -e ".[ai]"
```

### 4. รัน Web UI
```bash
python app.py
```
เปิดบราวเซอร์ไปที่ `http://127.0.0.1:7860`

---

## 🎨 โมเดลแนะนำจาก Civitai และ Hugging Face

### สำหรับการรันบนเครื่อง RTX 5060 Ti (8GB VRAM) / RTX 3050 (4GB VRAM):

1. **SD 1.5 Checkpoint (แนะนำสำหรับ 4GB - 8GB)**:
   - **[Pixel Art Sprite Diffusion (Civitai)](https://civitai.com/models/22/pixel-art-sprite-diffusion)**  
     เทรนมาเพื่อสร้าง Sprite Sheet 4-8 ทิศโดยเฉพาะ  
     *Trigger words:* `PixelartFSS` (หน้า), `PixelartRSS` (ขวา), `PixelartBSS` (หลัง), `PixelartLSS` (ซ้าย)
   - **ControlNet Depth Model**:
     - `lllyasviel/control_v11f1p_sd15_depth` (โหลดอัตโนมัติผ่าน Diffusers)

2. **SDXL Checkpoint (เหมาะมากสำหรับ RTX 5060 Ti 8GB)**:
   - **[Super_PixelArt_XL_M_V1 (Civitai)](https://civitai.com/models/581162/superpixelartxlmv1)**  
     ผสมผสาน LoRA เพื่อสร้างภาพพิกเซลคมกริบขนาด 128x128 จากการเจนที่ 1024x1024  
     *Trigger words:* `pixelart, sprite, 16bit`
   - **[Pixel Art Sprite Sheet (space candy media)](https://civitai.com/models/1028198/pixel-art-sprite-sheet-space-candy-media)**  
     LoRA ช่วยคุมโครงร่างตัวละครหลายมุมมอง

---

## 💻 การสลับใช้งานระหว่างเครื่อง 4GB (RTX 3050) และ 8GB (RTX 5060 Ti)

ในหน้า Web UI หรือในไฟล์ `config.yaml`:
- **บนเครื่อง RTX 3050 Laptop (4GB)**:
  - เลือก VRAM Profile: `4GB Laptop Low VRAM`
  - ระบบจะเปิดใช้งาน `enable_sequential_cpu_offload()` และลด Attention Slicing เหลือ 1
  - แนะนำขนาด Sprite: `32x32` หรือ `48x48`
- **บนเครื่อง RTX 5060 Ti Desktop (8GB)**:
  - เลือก VRAM Profile: `8GB Standard`
  - ระบบจะรันด้วย Fast SDPA และ Model Offload ทำงานได้เต็มประสิทธิภาพ
  - รองรับขนาด Sprite: `64x64` และ `128x128`

---

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

```
pixel_art/
├── assets/
│   └── poses/             # Depth Map Mannequin 8 ทิศทางสำหรับ ControlNet
├── core/
│   ├── generator.py       # ระบบสร้างภาพ 8 ทิศ (Diffusers, ControlNet, Mock)
│   └── processor.py       # ระบบ Downscaling, Palettization, Alpha Matting, Slicer
├── templates/
│   └── pose_generator.py  # ตัวคำนวณและเรนเดอร์ท่าทาง 3D 8 ทิศทาง
├── outputs/               # โฟลเดอร์เก็บผลลัพธ์ Spritesheet และ Turntable GIF
├── app.py                 # Gradio Web UI หลัก
├── config.yaml            # ค่าคอนฟิกฮาร์ดแวร์และพาเลทสี
├── pyproject.toml         # การตั้งค่าแพ็กเกจ
└── requirements.txt       # รายการ Library
```

---

## 📜 การใช้งานผ่าน Python Code (Standalone Library)

คุณสามารถนำ Core Module ไปเขียนสคริปต์แบบอัตโนมัติในเกมของคุณได้:

```python
from core.generator import CharacterSpritePipeline, MockPixelGenerator

pipeline = CharacterSpritePipeline(backend=MockPixelGenerator())

# เจนตัวละคร 8 ทิศทาง
result = pipeline.create_character(
    prompt="a cyber knight in neon armor",
    target_size=(64, 64),
    palette_name="db32",
    symmetry_5to8=True,
    output_dir="./outputs"
)

print("Spritesheet saved at:", result["spritesheet_path"])
print("Turntable GIF saved at:", result["turntable_path"])
```

---

## 📄 License
MIT License - สามารถนำไปพัฒนาต่อยอดในโปรเจกต์เกมเชิงพาณิชย์และส่วนตัวได้อย่างอิสระ!
