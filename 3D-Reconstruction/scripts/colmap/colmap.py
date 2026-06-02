import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import ReconstructionConfig
class Colmap:
    def __init__(self, cfg: ReconstructionConfig):
        self.cfg = cfg

        if not self.cfg.colmap_bin.exists():
            raise FileNotFoundError(f"COLMAP binary not found at {self.cfg.colmap_bin}")
        if not self.cfg.colmap_image_path.exists():
            os.makedirs(self.cfg.colmap_image_path)
        if not self.cfg.colmap_mask_path.exists():
            os.makedirs(self.cfg.colmap_mask_path)

    def run_colmap(self, args):
        cmd = [str(self.cfg.colmap_bin)] + args
        print("\nRunning COLMAP command:\n", " ".join(cmd))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        if process.stdout != None:
            for line in process.stdout:
                print(line, end="")
            process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"COLMAP command failed with exit code {process.returncode}")

    def extract_features(self):
        args = [
            "feature_extractor",
            f"--database_path={self.cfg.database_path}",
            f"--image_path={self.cfg.colmap_image_path}",
            "--ImageReader.camera_model=SIMPLE_RADIAL",
            "--ImageReader.single_camera=1",
            f"--SiftExtraction.max_image_size={self.cfg.max_img_size_feature_extraction}",
            f"--SiftExtraction.max_num_features={self.cfg.max_num_features}",
            "--log_to_stderr", "1"
        ]
        self.run_colmap(args)

    def match_features_sequential(self):
        args = [
            "sequential_matcher",
            f"--database_path={self.cfg.database_path}",
            f"--SequentialMatching.overlap={self.cfg.sequence_overlap}",
            "--log_to_stderr", "1"
        ]
        self.run_colmap(args)

    def match_features_spatial(self):
        args = [
            "spatial_matcher",
            f"--database_path={self.cfg.database_path}",
            f"--SpatialMatching.max_num_neighbors={self.cfg.max_num_neighbors}",
            "--log_to_stderr", "1"
        ]
        self.run_colmap(args)

    def match_features_exhaustive(self):
        args = [
            "exhaustive_matcher",
            f"--database_path={self.cfg.database_path}",
            "--log_to_stderr", "1"
        ]
        self.run_colmap(args)

    def sparse_reconstruction(self, output_path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        args = [
            "mapper",
            f"--database_path={self.cfg.database_path}",
            f"--image_path={self.cfg.colmap_image_path}",
            f"--output_path={output_path}",
            "--Mapper.num_threads=8",
            "--Mapper.multiple_models=0",
            "--Mapper.extract_colors=0",
            "--log_to_stderr", "1",
            "--Mapper.ba_global_frames_ratio=1.4",
            "--Mapper.ba_global_points_ratio=1.4",
            "--Mapper.ba_global_max_num_iterations=20",
            "--Mapper.ba_use_gpu=1",
            "--Mapper.ba_global_frames_freq=1000",
            "--Mapper.ba_global_points_freq=500000",
            "--Mapper.ba_local_max_num_iterations=15",
        ]

        self.run_colmap(args)

    def image_undistortion(self, image_path, sparse_model_path, output_path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        args = [            
            "image_undistorter",
            f"--image_path={image_path}",
            f"--input_path={sparse_model_path}",
            f"--output_path={output_path}",
            "--output_type=COLMAP",
            f"--max_image_size={self.cfg.max_img_size_dense}",
            "--log_to_stderr", "1"
        ]
        self.run_colmap(args)

    def dense_stereo(self, output_path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        args=[            
            "patch_match_stereo",
            f"--workspace_path={output_path}",
            "--workspace_format=COLMAP",
            f"--PatchMatchStereo.geom_consistency={int(self.cfg.geom_consistency)}",
            f"--PatchMatchStereo.max_image_size={self.cfg.max_img_size_dense}",
            f"--PatchMatchStereo.num_iterations={self.cfg.num_iterations}",
            f"--PatchMatchStereo.cache_size={self.cfg.cache_size}",
            "--log_to_stderr", "1",
            "--PatchMatchStereo.filter", "1"
        ]
        self.run_colmap(args)

    def stereo_fusion(self, workspace_path, output_path):
        args=[            
            "stereo_fusion",
            f"--workspace_path={workspace_path}",
            "--workspace_format=COLMAP",
            "--input_type=photometric",
            f"--output_path={output_path}",
            f"--StereoFusion.num_threads={self.cfg.num_threads}",
            f"--StereoFusion.use_cache={self.cfg.use_cache}",
            f"--StereoFusion.cache_size={self.cfg.cache_size}",
            f"--StereoFusion.max_image_size={self.cfg.max_img_size_dense}",
            f"--StereoFusion.max_num_pixels={self.cfg.max_num_pixels}",
            "--log_to_stderr", "1"
        ]
        self.run_colmap(args)

    def model_aligner(self):
        sparse_path = Path(self.cfg.sparse_path,"0")
        sparse_aligned_path = Path(self.cfg.sparse_aligned_path,"0")

        args=[            
            "model_aligner",
            f"--input_path={sparse_path}",
            f"--output_path={sparse_aligned_path}",
            f"--database_path={self.cfg.database_path}",
            "--ref_is_gps=1",
            "--alignment_type=ecef",
            "--alignment_max_error=3.0",
            "--log_to_stderr=1",
            f"--transform_path={sparse_aligned_path}\\transform.txt"
        ]
        self.run_colmap(args)

    def bin_to_txt(self):
        input_path = Path(self.cfg.sparse_path, "0")

        output_path = Path(self.cfg.sparse_path, "sparse_txt")
        output_path.mkdir(parents=True, exist_ok=True)

        args=[            
            "model_converter",
            f"--input_path={input_path}",
            f"--output_path={output_path}",
            "--output_type=TXT"
        ]
        self.run_colmap(args)

    def txt_to_bin(self, cluster):
        input_path= Path(self.cfg.sparse_path, f"sparse_cluster_{cluster}_txt")
        output_path= Path(self.cfg.sparse_path, f"sparse_cluster_{cluster}")
        output_path.mkdir(parents=True, exist_ok=True)

        args=[            
            "model_converter",
            f"--input_path={input_path}",
            f"--output_path={output_path}",
            "--output_type=BIN"
        ]
        self.run_colmap(args)