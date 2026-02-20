
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os
import csv
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path

from config import TrainConfig

# ---------- Helper functions ----------
def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = {}
    info = image._getexif() # type: ignore
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]
                exif_data["GPSInfo"] = gps_data
            else:
                exif_data[decoded] = value
    return exif_data

def rational_to_float(x):
    """Convert an EXIF rational (tuple or IFDRational) to float."""
    try:
        return float(x[0]) / float(x[1])
    except TypeError:
        return float(x)

def get_decimal_from_dms(dms, ref):
    degrees, minutes, seconds = dms
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def get_lat_lon(exif_data):
    gps_info = exif_data.get("GPSInfo")
    if not gps_info:
        return None, None, None

    lat = gps_info.get("GPSLatitude")
    lat_ref = gps_info.get("GPSLatitudeRef")
    lon = gps_info.get("GPSLongitude")
    lon_ref = gps_info.get("GPSLongitudeRef")
    alt = gps_info.get("GPSAltitude")

    if lat and lon and lat_ref and lon_ref:
        latitude = get_decimal_from_dms([rational_to_float(x) for x in lat], lat_ref)
        longitude = get_decimal_from_dms([rational_to_float(x) for x in lon], lon_ref)
        altitude = rational_to_float(alt) if alt else None
        return latitude, longitude, altitude
    return None, None, None

def get_tile_count(file, cfg: TrainConfig):
    file_name = Path(file).stem
    tile_count = sum(file_name in f for f in os.listdir(cfg.tiles_img_dir))
    print(tile_count)

    return tile_count

# ---------- Main processing ----------
def extract_metadata(cfg: TrainConfig):
    input_folder = Path(cfg.image_dir)

    rows = []

    for file in os.listdir(input_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.tif')):
            path = os.path.join(input_folder, file)
            try:
                exif = get_exif_data(path)
                lat, lon, alt = get_lat_lon(exif)
                tile_count = get_tile_count(file, cfg)
                rows.append({
                    "filename": Path(file).stem,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                    "tile_count": tile_count,
                    "set": "train"
                })
                print(f"{file}: {lat}, {lon}, {alt}, {tile_count}")
            except Exception as e:
                print(f"Error reading {file}: {e}")

    # ---------- Write to CSV ----------
    with open(cfg.metadata_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["filename", "latitude", "longitude", "altitude", "tile_count", "set"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nDone! GPS data written to {cfg.metadata_file}")

def main(cfg: TrainConfig) -> None:
    extract_metadata(cfg)