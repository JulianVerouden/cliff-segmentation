import sys
from pathlib import Path
import csv

import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import ReconstructionConfig

# Extract XMP field for GPS and flight angles for DJI Mavic 3M
def get_xmp_fields(image_path):
    
    with open(image_path, 'rb') as f:
        data = f.read()

    start = data.find(b'<x:xmpmeta')
    end = data.find(b'</x:xmpmeta')
    if start == -1 or end == -1:
        return {}

    xmp_bytes = data[start:end+12]
    xmp_str = xmp_bytes.decode('utf-8', errors='ignore')

    ns = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'drone': 'http://www.dji.com/drone-dji/1.0/'
    }

    root = ET.fromstring(xmp_str)
    result = {}
    for desc in root.findall('.//rdf:Description', ns):
        for key in ['GpsLatitude', 'GpsLongitude', 'AbsoluteAltitude',
                    'FlightPitchDegree', 'FlightYawDegree', 'FlightRollDegree']:
            value = desc.get(f'{{{ns["drone"]}}}{key}')
            if value is not None:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    return result

def collect_exif(cfg: ReconstructionConfig):
  images = list((p.resolve() for p in Path(cfg.dataset_dir).glob("**/*") if p.suffix.lower() in {".png", ".jpg"})) # For now only checks .JPG, .jpg, .PNG and .png files
  csv_rows = []
  txt_lines = []

  for image in images:
    xmp_data = get_xmp_fields(image)

    lat = xmp_data.get('GpsLatitude', -1)
    lon = xmp_data.get('GpsLongitude', -1)
    alt = xmp_data.get('AbsoluteAltitude', -1)
    pitch = xmp_data.get('FlightPitchDegree', -1)
    yaw = xmp_data.get('FlightYawDegree', -1)
    roll = xmp_data.get('FlightRollDegree', -1)

    csv_rows.append({
        "filename": image.name,
        "latitude": lat,
        "longitude": lon,
        "altitude": alt,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll
    })
    
    txt_lines.append(f"{image.name} {lat} {lon} {alt} {yaw} {pitch} {roll}")
  
  return csv_rows, txt_lines

# Write Metadata to csv
def write_csv(output_csv, csv_rows):
  with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ["filename", "latitude", "longitude", "altitude", "pitch", "yaw", "roll"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(row)
  
  print(f"Metadata written to CSV: {output_csv}")

# Write image metadata to a COLMAP-compatible GPS .txt
def write_txt(output_txt, txt_lines):
  with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("\n".join(txt_lines))

  print(f"COLMAP-compatible GPS TXT written to: {output_txt}")

def main(cfg: ReconstructionConfig) -> None:
  csv_rows, txt_lines = collect_exif(cfg)
  write_csv(cfg.metadata_csv, csv_rows)
  write_txt(cfg.metadata_txt, txt_lines)

