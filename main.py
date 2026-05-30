from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scratch_cnn import (
    DataWorkspace,
    IMAGE_EXTENSIONS,
    build_gear_cnn,
    convolve2d,
    load_csv_dataset,
    load_labeled_dataset,
    make_toy_gear_dataset,
    prepare_data_workspace,
    preprocess_image,
)
from geometric_sonification import FFTSignalModel, image_to_frequency_spectrum
from fusion_coordinator import FusionCoordinator


PROJECT_NAME = "DecodeLabs Robotics & Automation Internship , PROJECT 2"
DEVELOPER_NAME = "Alfarouq Ibrahim"
DEVELOPER_ROLE = "Robotics & Automation Intern"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train and run {PROJECT_NAME}."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Labeled dataset folder with class subfolders, for example dataset/intact and dataset/defective.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("gear_data"),
        help="Managed data workspace used when --dataset or --labels is not supplied.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="CSV file for flat-folder training. Format: filename,label.",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Create/refresh the managed data workspace and labels CSV, then exit.",
    )
    parser.add_argument(
        "--import-current",
        action="store_true",
        help="Copy image files from the project folder into gear_data/unlabeled if they are not already there.",
    )
    parser.add_argument(
        "--init-labels",
        type=Path,
        help="Write a label template CSV for the images in the current folder, then exit.",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Train on generated toy gears instead of real labeled images.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--synthetic-samples", type=int, default=20)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--load-model", type=Path)
    parser.add_argument(
        "--infer",
        nargs="*",
        type=Path,
        help="Images to classify after training. Defaults to test_gear*.png.",
    )
    parser.add_argument(
        "--infer-dir",
        type=Path,
        help="Classify every supported image in a folder after training/loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    image_shape = (1, args.image_size, args.image_size)

    if args.init_labels:
        write_label_template(args.init_labels)
        return

    if args.dataset and args.labels:
        raise ValueError("Use either --dataset or --labels, not both.")

    if args.prepare_data:
        workspace = prepare_data_workspace(
            args.data_dir,
            import_from=Path(".") if args.import_current else None,
        )
        print_workspace_summary(workspace)
        return

    # Memory Management: Skip heavy loading during inference mode
    if args.synthetic_demo:
        x, y, class_names = make_toy_gear_dataset(
            samples_per_class=args.synthetic_samples,
            image_size=args.image_size,
            seed=args.seed,
        )
        print("Running a synthetic gear sanity demo.")
    elif args.labels:
        if args.epochs > 0:
            x, y, class_names = load_csv_dataset(args.labels, (args.image_size, args.image_size))
            print(f"Loaded {len(y)} labeled images from {args.labels}.")
        else:
            class_names = ["intact", "defective"]
            x, y = None, None
    elif args.dataset:
        if args.epochs > 0:
            x, y, class_names = load_labeled_dataset(args.dataset, (args.image_size, args.image_size))
            print(f"Loaded {len(y)} labeled images from {args.dataset}.")
        else:
            class_names = ["intact", "defective"]
            x, y = None, None
    else:
        workspace = prepare_data_workspace(
            args.data_dir,
            import_from=Path(".") if args.import_current else None,
        )
        print_workspace_summary(workspace)
        if args.epochs > 0:
            ready_classes = [name for name, count in workspace.class_counts.items() if count > 0]
            if len(ready_classes) < 2:
                print("\nNot enough labeled training data yet.")
                print(f"Move images into {workspace.data_dir / 'intact'} and {workspace.data_dir / 'defective'}.")
                print("Then run: python main.py")
                print("For a math-only smoke test instead, run: python main.py --synthetic-demo")
                return

            x, y, class_names = load_csv_dataset(workspace.labels_path, (args.image_size, args.image_size))
            print(f"Loaded {len(y)} labeled images from {workspace.labels_path}.")
        else:
            print("\n⚡ Inference Mode: Skipping heavy dataset load...")
            class_names = ["intact", "defective"]
            x, y = None, None

    model = build_gear_cnn(image_shape, class_names=class_names, seed=args.seed)

    if args.load_model:
        model.load(args.load_model)
        print(f"Loaded weights from {args.load_model}.")

    if args.epochs > 0:
        history = model.fit(
            x,
            y,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            rng=rng,
        )
        for epoch, (loss, accuracy) in enumerate(
            zip(history.losses, history.accuracies), start=1
        ):
            print(f"epoch {epoch:02d} | loss={loss:.4f} | accuracy={accuracy:.3f}")

    if args.save_model:
        model.save(args.save_model)
        print(f"Saved weights to {args.save_model}.")

    image_paths = collect_inference_images(args.infer, args.infer_dir)
    if image_paths:
        print(f"\n{PROJECT_NAME} Pipeline Active")
        print(f"Developer: {DEVELOPER_NAME} | {DEVELOPER_ROLE}")
        import cv2
        
        signal_model = FFTSignalModel()
        coordinator = FusionCoordinator(model, signal_model, class_names=("intact", "defective"))
        
        # Calibration for tolerance gate
        THRESHOLD_MAX = 45.0 
        
        for image_path in image_paths:
            # --- 1. EXTRACT INDEPENDENT AI DECISIONS FOR XAI ---
            sample = preprocess_image(image_path, (args.image_size, args.image_size))[None, :, :, :]
            spectrum = image_to_frequency_spectrum(str(image_path), num_samples=2048, spectrum_bins=512)
            
            # Visual AI (CNN) decision
            v_probs = model.predict_proba(sample)[0]
            v_conf = max(v_probs) * 100
            v_label = "intact" if v_probs[0] > v_probs[1] else "defective"
            
            # FFT Signal AI decision
            s_probs = signal_model.predict_proba(spectrum)[0]
            s_conf = max(s_probs) * 100
            s_label = "intact" if s_probs[0] > s_probs[1] else "defective"
            
            # Final Fused decision
            result = coordinator.predict_fusion(sample, spectrum)
            ai_label = result.label
            ai_confidence = max(result.fused_probabilities) * 100
            
            # --- 2. DECODELABS DETERMINISTIC PIPELINE ---
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None: continue
            
            raw_resized = cv2.resize(img_bgr, (400, 400))
            gray = cv2.cvtColor(raw_resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive thresholding and morphological closing
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            plc_signal = "PASS"
            plc_color = (0, 255, 0)
            
            if contours:
                main_c = max(contours, key=cv2.contourArea)
                cv2.drawContours(raw_resized, [main_c], -1, (255, 255, 0), 1)
                
                hull_indices = cv2.convexHull(main_c, returnPoints=False) 
                
                try:
                    defects = cv2.convexityDefects(main_c, hull_indices)
                    if defects is not None:
                        max_defect_dist = 0
                        best_defect = None
                        
                        for i in range(defects.shape[0]):
                            s, e, f, d_raw = defects[i, 0]
                            actual_distance = d_raw / 256.0 
                            
                            if actual_distance > THRESHOLD_MAX and actual_distance > max_defect_dist:
                                max_defect_dist = actual_distance
                                best_defect = (s, e, f)
                        
                        if best_defect is not None:
                            s, e, f = best_defect
                            
                            pt_start = main_c[s][0]
                            pt_end = main_c[e][0]
                            pt_farthest = main_c[f][0]
                            
                            defect_pts = np.array([pt_start, pt_end, pt_farthest])
                            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(defect_pts)
                            
                            pad = 12
                            x_box = max(0, x_rect - pad)
                            y_box = max(0, y_rect - pad)
                            w_box = w_rect + (pad * 2)
                            h_box = h_rect + (pad * 2)
                            
                            cv2.rectangle(raw_resized, (x_box, y_box), (x_box+w_box, y_box+h_box), (0, 0, 255), 2)
                            cv2.putText(raw_resized, "FAIL: TOOTH DEFECT", (x_box, max(15, y_box-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                            
                            plc_signal = "FAIL (DEFECT)"
                            plc_color = (0, 0, 255)
                except Exception as e:
                    pass

            # --- 3. DASHBOARD UI BUILD ---
            dashboard = np.zeros((450, 1250, 3), dtype=np.uint8)
            
            sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
            filtered = convolve2d(cv2.resize(gray, (args.image_size, args.image_size)), sobel)
            filtered = np.clip(np.abs(filtered), 0, 255).astype(np.uint8)
            filtered_bgr = cv2.cvtColor(cv2.resize(filtered, (400, 400)), cv2.COLOR_GRAY2BGR)
            filtered_bgr[:, :, 0] = 0 
            filtered_bgr[:, :, 2] = 0 
            
            # XAI Dashboard Panel
            final_panel = np.zeros((400, 400, 3), dtype=np.uint8)
            
            cv2.putText(final_panel, "DecodeLabs Internship", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.putText(final_panel, f"{DEVELOPER_NAME} | Intern", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(final_panel, "1. CLASSICAL PLC GATE:", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(final_panel, plc_signal, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, plc_color, 2)
            
            v_color = (0, 255, 0) if v_label == "intact" else (0, 0, 255)
            cv2.putText(final_panel, "2. VISUAL AI (CNN):", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(final_panel, f"{v_label.upper()} ({v_conf:.1f}%)", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, v_color, 2)
            
            s_color = (0, 255, 0) if s_label == "intact" else (0, 0, 255)
            cv2.putText(final_panel, "3. FFT SIGNAL AI:", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(final_panel, f"{s_label.upper()} ({s_conf:.1f}%)", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, s_color, 2)
            
            cv2.putText(final_panel, "-"*30, (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(final_panel, "FINAL FUSED DECISION:", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            ai_color = (0, 255, 0) if ai_label == "intact" else (0, 0, 255)
            cv2.putText(final_panel, f"{ai_label.upper()} ({ai_confidence:.1f}%)", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ai_color, 2)
            
            dashboard[40:440, 10:410] = raw_resized
            dashboard[40:440, 430:830] = filtered_bgr
            dashboard[40:440, 850:1250] = final_panel
            
            cv2.arrowedLine(dashboard, (415, 240), (425, 240), (255, 255, 255), 3, tipLength=0.5)
            cv2.arrowedLine(dashboard, (835, 240), (845, 240), (255, 255, 255), 3, tipLength=0.5)
            
            cv2.putText(dashboard, "1. DECODELABS TOLERANCE GATE", (45, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(dashboard, "2. EDGE-AI FEATURE MAP", (510, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(dashboard, "3. EXPLAINABLE AI VERDICT", (900, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            
            window_name = PROJECT_NAME
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1500, 500)
            cv2.imshow(window_name, dashboard)
            
            cv2.waitKey(0) 
            
        cv2.destroyAllWindows()

    show_convolution_probe()

def show_convolution_probe() -> None:
    image_path = Path("test_gear.png")
    if not image_path.exists():
        return

    sample = preprocess_image(image_path, (64, 64))[0]
    sobel_horizontal = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=np.float32,
    )
    feature_map = convolve2d(sample, sobel_horizontal)
    print(
        "\nConvolution probe: "
        f"test_gear.png -> feature map {feature_map.shape}, "
        f"min={feature_map.min():.3f}, max={feature_map.max():.3f}"
    )


def collect_inference_images(
    image_args: list[Path] | None,
    image_dir: Path | None,
) -> list[Path]:
    image_paths: list[Path] = []

    if image_args is None and image_dir is None:
        image_paths.extend(sorted(Path(".").glob("test_gear*.png")))
    elif image_args is not None:
        for image_arg in image_args:
            if image_arg.is_dir():
                image_paths.extend(iter_image_files(image_arg))
            elif image_arg.is_file():
                image_paths.append(image_arg)

    if image_dir is not None:
        image_paths.extend(iter_image_files(image_dir))

    return sorted(dict.fromkeys(image_paths))


def iter_image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_label_template(labels_path: Path) -> None:
    image_paths = sorted(
        path
        for path in Path(".").iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    lines = ["filename,label"]
    lines.extend(f"{path.name}," for path in image_paths)
    labels_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {labels_path} with {len(image_paths)} image rows.")
    print("Fill the label column with intact or defective, then run with --labels.")


def print_workspace_summary(workspace: DataWorkspace) -> None:
    print(PROJECT_NAME)
    print(f"Developer: {DEVELOPER_NAME} | {DEVELOPER_ROLE}")
    print(f"Data workspace: {workspace.data_dir}")
    print(f"Labels CSV: {workspace.labels_path}")
    if workspace.imported_count:
        print(f"Imported {workspace.imported_count} image(s) into {workspace.data_dir / 'unlabeled'}.")
    for class_name, count in workspace.class_counts.items():
        print(f"{class_name}: {count} labeled image(s)")
    print(f"unlabeled: {workspace.unlabeled_count} image(s)")


if __name__ == "__main__":
    main()