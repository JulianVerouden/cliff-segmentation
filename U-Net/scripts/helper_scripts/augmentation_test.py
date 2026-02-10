import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Base normalization ---
base_norm = [
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
]

# --- Define individual augmentations ---
def get_individual_augmentations():
    return [
        ("HorizontalFlip", A.HorizontalFlip(p=1.0)),
        ("VerticalFlip", A.VerticalFlip(p=1.0)),
        ("Rotate90", A.RandomRotate90(p=1.0)),
        ("Affine", A.Affine(scale=(1.1, 1.1),
                            translate_percent=(0.1, 0.1),
                            rotate=(30, 30), p=1.0)),
        ("RandomCrop", A.RandomCrop(height=196, width=196, p=1.0)),
        ("BrightnessContrast", A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)),
        ("ColorJitter", A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0)),
        ("GaussianBlur", A.GaussianBlur(blur_limit=(7, 7), p=1.0)),
        ("GaussNoise", A.GaussNoise(var_limit=(50.0, 50.0), p=1.0)),
        ("ElasticTransform", A.ElasticTransform(alpha=500, sigma=20, alpha_affine=20, p=1.0)),
        ("GridDistortion", A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)),
    ]

def visualize_strong_augmentations_interactive(image_path, mask_path, per_page=3):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask at: {mask_path}")

    augmentations = get_individual_augmentations()
    n_pages = (len(augmentations) + per_page - 1) // per_page

    # Precompute all augmented samples
    results = []
    for name, aug in augmentations:
        transform = A.Compose([aug])
        augmented = transform(image=image, mask=mask)
        img_tensor = augmented["image"]
        mask_tensor = augmented["mask"]
        results.append((name, img_tensor, mask_tensor))

    # --- Setup figure ---
    fig, axes = plt.subplots(per_page, 2, figsize=(10, 4 * per_page))
    plt.subplots_adjust(bottom=0.15, hspace=0.25)
    axes = axes.reshape(-1, 2)

    # --- Slider setup ---
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'Page', 0, n_pages - 1, valinit=0, valstep=1)

    def update(val):
        page = int(slider.val)
        start_idx = page * per_page
        end_idx = min(start_idx + per_page, len(results))

        for i, axpair in enumerate(axes):
            for ax in axpair:
                ax.clear()

            if start_idx + i < end_idx:
                name, aug_img, aug_mask = results[start_idx + i]
                axpair[0].imshow(aug_img)
                axpair[0].set_title(f"{name} (Image)")
                axpair[0].axis("off")

                axpair[1].imshow(aug_mask, cmap="gray")
                axpair[1].set_title(f"{name} (Mask)")
                axpair[1].axis("off")
            else:
                axpair[0].axis("off")
                axpair[1].axis("off")

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)  # initial render
    plt.show()


# --- Example usage ---
if __name__ == "__main__":
    image_path = r"data\images\cabo_espichel\tiles\test\DJI_20231108134640_0088_D_point5_0_14.jpg"
    mask_path = r"data\masks\cabo_espichel\tiles\test\DJI_20231108134640_0088_D_point5_0_14_mask.png"

    visualize_strong_augmentations_interactive(image_path, mask_path, per_page=3)
