import torch
import sys
import os

MODEL_PATH = r"data\checkpoints\unet_resnet50_aug_pre_training_50.pth"   # <-- change this to your file

print(f"Inspecting: {MODEL_PATH}\n")

if not os.path.exists(MODEL_PATH):
    print("❌ File not found.")
    sys.exit()

try:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
except Exception as e:
    print("❌ ERROR while loading model:")
    print(type(e).__name__, str(e))
    sys.exit()

print("\n✅ File loaded successfully!\n")

print("🔍 Type of loaded object:", type(checkpoint))

# If it's a dict, show its keys
if isinstance(checkpoint, dict):
    print("\nKeys in checkpoint:")
    for k in checkpoint.keys():
        print(" •", k)

    # Check if it looks like a state_dict
    first_key = next(iter(checkpoint))
    if isinstance(checkpoint[first_key], torch.Tensor):
        print("\n🟢 This appears to be a valid state_dict.")
    else:
        print("\n🟠 This dictionary is not a standard model state_dict.")
else:
    print("\n🔴 This is NOT a state_dict.")
    print("It is likely a full pickled model object, which is causing your error.")
