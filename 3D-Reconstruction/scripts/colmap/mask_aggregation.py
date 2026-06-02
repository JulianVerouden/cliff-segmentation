import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
import json

# -----------------------------
# CONFIG
# -----------------------------
workspace_path = Path(r"colmap_data\dense")  # your COLMAP dense workspace
fused_ply_path = workspace_path / "fused.ply"
output_ply_path = workspace_path / "fused_with_mask.ply"

# Path to mask images (dictionary: image_name -> mask file path)
mask_dir = Path("masks")
mask_images = {f.name: cv2.imread(str(mask_dir / f.name), cv2.IMREAD_UNCHANGED)
               for f in mask_dir.iterdir() if f.suffix in [".png", ".jpg"]}

# Path to COLMAP camera/pose info (assumes JSON exported from COLMAP, see note below)
# Format: cameras.json -> {camera_id: {"K": [[...]]}}
#         images.json -> {image_name: {"R": [[...]], "t": [...], "camera_id": ...}}
cameras_file = workspace_path / "cameras.json"
images_file = workspace_path / "images.json"

# -----------------------------
# LOAD COLMAP DATA
# -----------------------------
with open(cameras_file) as f:
    cameras = json.load(f)  # camera_id -> {"K": 3x3 list}

with open(images_file) as f:
    images = json.load(f)   # image_name -> {"R": 3x3 list, "t": 3 list, "camera_id": int}

# Convert to numpy
for cam in cameras.values():
    cam["K"] = np.array(cam["K"], dtype=np.float32)
for img in images.values():
    img["R"] = np.array(img["R"], dtype=np.float32)
    img["t"] = np.array(img["t"], dtype=np.float32).reshape((3,1))

# -----------------------------
# LOAD DENSE POINT CLOUD
# -----------------------------
pcd = o3d.io.read_point_cloud(str(fused_ply_path))
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)  # floats 0..1

# Initialize mask array
mask_values = np.zeros(len(points), dtype=np.float32)
counts = np.zeros(len(points), dtype=np.int32)

# -----------------------------
# REPROJECT POINTS TO MASKS
# -----------------------------
for img_name, mask in mask_images.items():
    if img_name not in images:
        print(f"Skipping {img_name}, not in COLMAP images.")
        continue

    img_data = images[img_name]
    cam_data = cameras[str(img_data["camera_id"])]
    K = cam_data["K"]
    R = img_data["R"]
    t = img_data["t"]

    # Transform points into camera coordinates
    X_cam = (R @ points.T + t).T  # Nx3

    # Perspective division
    u = (K[0,0]*X_cam[:,0]/X_cam[:,2] + K[0,2]).astype(int)
    v = (K[1,1]*X_cam[:,1]/X_cam[:,2] + K[1,2]).astype(int)

    h, w = mask.shape[:2]
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (X_cam[:,2] > 0)

    # Accumulate mask values
    mask_values[valid] += mask[v[valid], u[valid]] / 255.0
    counts[valid] += 1

# Average across views
mask_values = mask_values / np.maximum(counts, 1)

# -----------------------------
# SAVE NEW PLY WITH MASK
# -----------------------------
# Combine RGB + mask into "colors"
colors_with_mask = np.hstack([colors, mask_values[:, None]])  # Nx4

# Open3D doesn't support 4-channel colors, so save mask separately as "scalar"
pcd.colors = o3d.utility.Vector3dVector(colors)  # keep RGB for visualization
pcd.points = o3d.utility.Vector3dVector(points)

# Use Open3D PLY writer with extra scalar property
# We'll write manually for the 4th channel
with open(output_ply_path, "w") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(points)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("property float mask\n")
    f.write("end_header\n")
    for i in range(len(points)):
        x, y, z = points[i]
        r, g, b = (colors[i] * 255).astype(int)
        mask_val = float(mask_values[i])
        f.write(f"{x} {y} {z} {r} {g} {b} {mask_val}\n")

print(f"Saved fused point cloud with mask to {output_ply_path}")