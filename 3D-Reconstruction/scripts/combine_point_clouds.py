import numpy as np
import struct
from scipy.spatial import cKDTree
import sys
import os
import glob
from pathlib import Path

from scipy.spatial.transform import Rotation
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import ReconstructionConfig

# Loads COLMAP Sim(3) transform from:
# scale qx qy qz qw tx ty tz
def load_colmap_transform(transform_path):
    vals = np.loadtxt(transform_path)

    scale = vals[0]

    qx, qy, qz, qw = vals[1:5]

    tx, ty, tz = vals[5:8]

    # scipy expects [x, y, z, w]
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

    t = np.array([tx, ty, tz])

    return scale, R, t

def apply_transform(points_struct, scale, R, t):

    xyz = np.stack([
        points_struct['x'],
        points_struct['y'],
        points_struct['z']
    ], axis=1)

    xyz_transformed = scale * (xyz @ R.T) + t

    points_struct['x'] = xyz_transformed[:, 0]
    points_struct['y'] = xyz_transformed[:, 1]
    points_struct['z'] = xyz_transformed[:, 2]

    return points_struct

# Reads a binary little-endian PLY and returns header and structured array of vertices
def read_ply_binary(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == "end_header":
                break

        # Determine number of vertices and properties
        vertex_count = 0
        properties = []
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line.startswith("property"):
                properties.append(line.split()[1:])

        # Map PLY types to numpy dtypes
        dtype_map = {'float':'f4', 'double':'f8', 'uchar':'u1','uint8':'u1'}
        dtypes = [dtype_map.get(p[0], 'f4') for p in properties]
        names = [p[1] for p in properties]
        vertex_dtype = np.dtype({'names': names, 'formats': dtypes})

        # Read binary data
        vertices = np.fromfile(f, dtype=vertex_dtype, count=vertex_count)
    return header, vertices

# Writes a binary PLY with an extra float channel at the end.
def write_ply_binary_ecef(path, vertices, extra_property_name="mask"):
    N = len(vertices)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property double x",
        "property double y",
        "property double z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"property float {extra_property_name}",
        "end_header\n"
    ]
    with open(path, "wb") as f:
        f.write("\n".join(header).encode('ascii'))

        # Write each vertex: XYZ(float32), RGB(uchar), mask(float32)
        for v in vertices:
            f.write(struct.pack("<ddd", v['x'], v['y'], v['z']))
            f.write(struct.pack("BBB", v['red'], v['green'], v['blue']))
            f.write(struct.pack("<f", v[extra_property_name]))

# Writes a binary PLY with an extra float channel at the end.
def write_ply_binary(path, vertices, extra_property_name="mask"):
    N = len(vertices)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"property float {extra_property_name}",
        "end_header\n"
    ]
    with open(path, "wb") as f:
        f.write("\n".join(header).encode('ascii'))

        # Write each vertex: XYZ(float32), RGB(uchar), mask(float32)
        for v in vertices:
            f.write(struct.pack("<fff", v['x'], v['y'], v['z']))
            f.write(struct.pack("BBB", v['red'], v['green'], v['blue']))
            f.write(struct.pack("<f", v[extra_property_name]))
            
def combine_clusters(cfg: ReconstructionConfig):
    cluster_point_clouds = glob.glob(os.path.abspath(cfg.cluster_point_clouds) + f"\\*.ply")
    combined_point_cloud = []

    for cluster in cluster_point_clouds:
        _, points_cluster = read_ply_binary(cluster)
        combined_point_cloud.append(points_cluster)

    # Merge all structured arrays into one
    combined_point_cloud = np.concatenate(combined_point_cloud)

    print(combined_point_cloud.shape)
    print(combined_point_cloud.dtype)

    write_ply_binary(Path(cfg.dataset_output, f"masked_point_cloud_{cfg.dataset_name}.ply"), combined_point_cloud, "mask")

def add_mask_channel(cfg: ReconstructionConfig, file_name):
    # Paths
    fused_path = Path(cfg.dense_path, "fused.ply")
    mask_path = Path(cfg.dense_path_masks, "fused.ply")
    output_path = Path(cfg.dataset_output, "cluster_point_clouds", f"{file_name}.ply")
    
    os.makedirs(Path(cfg.dataset_output, "cluster_point_clouds"), exist_ok=True)

    # Load PLYs
    _, points_fused = read_ply_binary(fused_path)
    _, points_mask  = read_ply_binary(mask_path)

    # Build KD-tree
    xyz_fused = np.stack([points_fused['x'], points_fused['y'], points_fused['z']], axis=1)
    xyz_mask  = np.stack([points_mask['x'], points_mask['y'], points_mask['z']], axis=1)

    # Use red channel of mask as the mask value (adjust if stored differently)
    mask_values_mask = points_mask['red'].astype(np.float32) / 255.0

    tree = cKDTree(xyz_mask)
    _, idx = tree.query(xyz_fused, k=1)  # nearest neighbor
    mask_values_fused = mask_values_mask[idx]

    # Merge into new structured array
    points_with_mask = np.empty(len(points_fused), dtype=[
        ('x','f4'),('y','f4'),('z','f4'),
        ('red','u1'),('green','u1'),('blue','u1'),
        ('mask','f4')
    ])

    points_with_mask_ecef = np.empty(len(points_fused), dtype=[
        ('x','f8'),('y','f8'),('z','f8'),
        ('red','u1'),('green','u1'),('blue','u1'),
        ('mask','f4')
    ])

    points_with_mask['x'] = points_fused['x']
    points_with_mask['y'] = points_fused['y']
    points_with_mask['z'] = points_fused['z']
    points_with_mask['red'] = points_fused['red']
    points_with_mask['green'] = points_fused['green']
    points_with_mask['blue'] = points_fused['blue']
    points_with_mask['mask'] = mask_values_fused

    print(np.min(points_with_mask['x']))
    print(np.max(points_with_mask['x']))

    # ECEF Align
    # transform_path = Path(cfg.sparse_aligned_path, "0", "transform.txt")

    # scale, R, t = load_colmap_transform(transform_path)

    # points_with_mask = apply_transform(points_with_mask, scale, R, t)

    # Write output
    write_ply_binary(output_path, points_with_mask)
    print(f"Saved fused point cloud with mask channel to: {output_path}")
