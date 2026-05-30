import cv2
import numpy as np
import time
from pathlib import Path
from scratch_cnn import _render_toy_gear


def generate_synthetic_gears(total_pairs: int = 500) -> None:
    train_dir_intact = Path("gear_data/intact")
    train_dir_defect = Path("gear_data/defective")
    backup_dir_intact = Path("backup_intact_gears")
    backup_dir_defect = Path("backup_defective_gears")

    for folder in [train_dir_intact, train_dir_defect, backup_dir_intact, backup_dir_defect]:
        folder.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()
    batch_id = int(time.time())  # لتوليد أسماء فريدة لا تمسح القديم

    print("Generating 1000 3D synthetic gears (Adding to existing data)...")
    print("This might take a minute, please wait...")

    for i in range(1, total_pairs + 1):
        # ترس سليم
        img_intact = _render_toy_gear(150, defective=False, rng=rng)
        img_intact = np.clip(
            (img_intact - img_intact.min())
            / (img_intact.max() - img_intact.min())
            * 255,
            0,
            255,
        ).astype(np.uint8)
        name_intact = f"synth_3d_{batch_id}_intact_{i}.png"

        cv2.imwrite(str(train_dir_intact / name_intact), img_intact)
        cv2.imwrite(str(backup_dir_intact / name_intact), img_intact)

        # ترس معيب
        img_defect = _render_toy_gear(150, defective=True, rng=rng)
        img_defect = np.clip(
            (img_defect - img_defect.min())
            / (img_defect.max() - img_defect.min())
            * 255,
            0,
            255,
        ).astype(np.uint8)
        name_defect = f"synth_3d_{batch_id}_defect_{i}.png"

        cv2.imwrite(str(train_dir_defect / name_defect), img_defect)
        cv2.imwrite(str(backup_dir_defect / name_defect), img_defect)

        if i % 100 == 0:
            print(f"Processed {i * 2}/1000 images...")

    print("SUCCESS: 1000 NEW 3D images appended to your dataset!")


if __name__ == "__main__":
    generate_synthetic_gears()