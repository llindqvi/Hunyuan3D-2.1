import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import os
import time
from datetime import datetime
import torch
import gc
import trimesh
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

# ========== INPUT ==========
image_path = 'assets/example_images/052.png'

# Generate timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
input_name = os.path.splitext(os.path.basename(image_path))[0]
output_dir = os.path.join('save_dir', f'{input_name}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)

# Output paths
output_glb = os.path.join(output_dir, f'{input_name}.glb')
output_textured_obj = os.path.join(output_dir, f'{input_name}_textured.obj')
output_textured_glb = os.path.join(output_dir, f'{input_name}_textured.glb')

# ========== TIMING ==========
timings = {}
total_start = time.time()

# ========== SHAPE GENERATION ==========
print(f"Starting shape generation for {image_path}...")
phase_start = time.time()
model_path = 'tencent/Hunyuan3D-2.1'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
timings['Pipeline Loading'] = time.time() - phase_start

phase_start = time.time()
image = Image.open(image_path)
if image.mode == 'RGB':
    print("Image has no alpha channel, performing background removal...")
    rembg = BackgroundRemover()
    image = rembg(image)
    print("Background removal complete")
else:
    print("Image already has alpha channel, skipping background removal")
    image = image.convert("RGBA")
timings['Image Preprocessing'] = time.time() - phase_start

phase_start = time.time()
mesh = pipeline_shapegen(image=image)[0]
timings['Shape Generation'] = time.time() - phase_start

phase_start = time.time()
mesh.export(output_glb)
timings['Shape Export'] = time.time() - phase_start
print(f"Shape generation complete, saved to {output_glb}")

# ========== FREE MEMORY BEFORE TEXTURE GENERATION ==========
print("Freeing GPU memory...")
phase_start = time.time()
del pipeline_shapegen
gc.collect()
torch.cuda.empty_cache()
timings['Memory Cleanup'] = time.time() - phase_start
print(f"GPU memory freed. Available: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

# ========== TEXTURE GENERATION ==========
print("Starting texture generation...")
phase_start = time.time()
max_num_view = 4  # Reduced from 6 for memory
resolution = 512
conf = Hunyuan3DPaintConfig(max_num_view, resolution)
conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
paint_pipeline = Hunyuan3DPaintPipeline(conf)
timings['Texture Pipeline Loading'] = time.time() - phase_start

phase_start = time.time()
output_mesh_path = paint_pipeline(
    mesh_path = output_glb, 
    image_path = image_path,
    output_mesh_path = output_textured_obj,
    save_glb = False
)
timings['Texture Generation'] = time.time() - phase_start
print(f"Texture generation complete, saved to {output_mesh_path}")

# ========== CONVERT TO GLB WITH TRIMESH ==========
print("Converting textured OBJ to GLB...")
phase_start = time.time()
try:
    mesh_textured = trimesh.load(output_mesh_path, force='mesh')
    mesh_textured.export(output_textured_glb)
    timings['GLB Conversion'] = time.time() - phase_start
    print(f"GLB export complete, saved to {output_textured_glb}")
    
    # Clean up intermediate files - keep only textured GLB
    import glob
    print("Cleaning up intermediate files...")
    cleanup_patterns = [
        output_glb,  # untextured GLB
        output_textured_obj,  # textured OBJ
        output_mesh_path.replace('.obj', '.mtl'),  # MTL file
        output_mesh_path.replace('.obj', '.jpg'),  # base texture
        output_mesh_path.replace('.obj', '_metallic.jpg'),  # metallic map
        output_mesh_path.replace('.obj', '_roughness.jpg'),  # roughness map
        os.path.join(output_dir, 'white_mesh_remesh.obj'),  # temp mesh
    ]
    for pattern in cleanup_patterns:
        for file in glob.glob(pattern):
            if os.path.exists(file):
                os.remove(file)
                print(f"  Removed: {os.path.basename(file)}")
    print(f"Final output: {output_textured_glb}")
except Exception as e:
    timings['GLB Conversion'] = time.time() - phase_start
    print(f"GLB conversion failed: {e}")

# ========== TIMING SUMMARY ==========
total_time = time.time() - total_start
print("\n" + "="*60)
print("TIMING SUMMARY")
print("="*60)
for phase, duration in timings.items():
    print(f"{phase:.<40} {duration:>8.2f}s")
print("-"*60)
print(f"{'TOTAL TIME':.<40} {total_time:>8.2f}s")
print("="*60)
