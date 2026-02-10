import os

root = "data\images\cabo_espichel\tiles"

for dirpath, dirnames, filenames in os.walk(root):
    for fname in filenames:
        old_path = os.path.join(dirpath, fname)

        # Rule 1: .jpg → .tif
        if fname.endswith(".jpg"):
            new_name = fname[:-4] + ".tif"
            new_path = os.path.join(dirpath, new_name)
            print(f"Renaming: {old_path} -> {new_path}")
            os.rename(old_path, new_path)

        # Rule 2: *_mask.png → remove _mask and convert to .tif
        elif fname.endswith("_mask.png"):
            base = fname[:-9]        # remove '_mask.png'
            new_name = base + ".tif"
            new_path = os.path.join(dirpath, new_name)
            print(f"Renaming: {old_path} -> {new_path}")
            os.rename(old_path, new_path)