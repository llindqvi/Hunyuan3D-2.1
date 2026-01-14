# Hunyuan3D-2.1 Installation Guide

This guide documents the installation steps for Hunyuan3D-2.1, particularly for systems with custom PyTorch installations (e.g., RTX 5090).

## Prerequisites

- Python 3.10
- Conda or Miniconda (for Python environment)
- PyTorch compatible with your GPU (already installed)
- Git
- C++ compiler (g++)
- wget or curl

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd Hunyuan3D-2.1
```

### 2. Create and Activate Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch (if not already installed)

**If you already have a compatible PyTorch installation** (e.g., for RTX 5090), skip this step.

Otherwise, install according to the README:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install Python Dependencies (excluding bpy)

The `bpy==4.0` package is not available on PyPI, so we install all other requirements:

```bash
grep -v "^bpy==" requirements.txt | pip install -r /dev/stdin
```

This installs all dependencies except Blender Python bindings, which are typically not required for core functionality.

### 5. Install Custom Rasterizer

```bash
cd hy3dpaint/custom_rasterizer
pip install -e .
cd ../..
```

### 6. Compile DifferentiableRenderer

```bash
cd hy3dpaint/DifferentiableRenderer
c++ -O3 -Wall -shared -std=c++11 -fPIC \
  $(python3-config --includes) \
  -I$(python -c "import pybind11; print(pybind11.get_include())") \
  -I$(python -c "import numpy; print(numpy.get_include())") \
  mesh_inpaint_processor.cpp \
  -o mesh_inpaint_processor$(python3-config --extension-suffix)
cd ../..
```

### 7. Download RealESRGAN Model

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
```

## Verification

Verify the installation was successful:

```bash
source venv/bin/activate

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check installed packages
pip list | grep -E "transformers|diffusers|gradio"

# Check compiled extension
ls -lh hy3dpaint/DifferentiableRenderer/*.so

# Check downloaded model
ls -lh hy3dpaint/ckpt/RealESRGAN_x4plus.pth
```

## Troubleshooting

### bpy Package Not Found

The `bpy==4.0` package is not available as a prebuilt wheel on PyPI. This is normal and typically doesn't affect core 3D generation functionality. If you specifically need Blender integration:

- Install Blender separately from https://www.blender.org/
- Or build bpy from source following Blender documentation

### Python.h Not Found During Compilation

Make sure you have Python development headers installed. If using conda:
```bash
conda install python-dev
```

Or on Ubuntu/Debian:
```bash
sudo apt-get install python3-dev
```

### CUDA Compatibility Issues

If you have a newer GPU (e.g., RTX 5090), you may need a newer PyTorch version than specified in the README. Keep your existing compatible PyTorch installation instead of downgrading to 2.5.1.

## Running the Application

### Gradio Web Interface

```bash
source venv/bin/activate
python3 gradio_app.py \
  --model_path tencent/Hunyuan3D-2.1 \
  --subfolder hunyuan3d-dit-v2-1 \
  --texgen_model_path tencent/Hunyuan3D-2.1 \
  --low_vram_mode
```

### Python API Usage

```python
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# Generate mesh
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
mesh_untextured = shape_pipeline(image='assets/demo.png')[0]

# Generate texture
paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
mesh_textured = paint_pipeline(mesh_path, image_path='assets/demo.png')
```

## System Requirements

- **VRAM Requirements:**
  - Shape generation: 10 GB
  - Texture generation: 21 GB
  - Both together: 29 GB

- **Recommended:**
  - Python 3.10
  - CUDA 12.x
  - 32GB+ system RAM
  - RTX 3090 or better

## Installation Environment

This installation was tested with:
- **OS:** Linux
- **Python:** 3.10.10
- **PyTorch:** 2.9.1
- **CUDA:** 12.x (via cupy-cuda12x)
- **GPU:** NVIDIA RTX 5090
