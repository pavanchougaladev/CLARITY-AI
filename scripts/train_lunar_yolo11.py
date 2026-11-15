"""
ISRO LunarFusion — Lunar Crater YOLO11 Training Pipeline
=========================================================
Trains and certifies fine-tuned Ultralytics YOLO11 models on multi-mission
lunar imagery (Chandrayaan-1/2, LROC NAC, Apollo, TMC).
Supports Apple Silicon GPU (MPS), CUDA, or CPU with automatic export to
models/yolo11_lunar_crater.pt and AGENTS.md Gate compliance metrics.
"""

import os
import argparse
import shutil
import torch
from ultralytics import YOLO

def train_lunar_yolo11(
    data_yaml: str = "data/lunar_crater_yolo/lunar_craters.yaml",
    base_model: str = "yolo11n.pt",
    epochs: int = 15,
    batch_size: int = 8,
    imgsz: int = 640,
    output_model_path: str = "models/yolo11_lunar_crater.pt",
    project_dir: str = "runs/train_lunar",
    device: str = None
):
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"

    print("=" * 70)
    print("🚀 STARTING MULTI-MISSION LUNAR CRATER YOLO11 TRAINING")
    print(f"  • Base Model:       {base_model}")
    print(f"  • Dataset YAML:     {data_yaml}")
    print(f"  • Target Hardware:  {device.upper()}")
    print(f"  • Image Dimensions: {imgsz}x{imgsz}")
    print(f"  • Batch Size:       {batch_size}")
    print(f"  • Epochs:           {epochs}")
    print(f"  • Export Target:    {output_model_path}")
    print("=" * 70)

    # Initialize model (can be yolo11n.pt, yolo11s.pt, or prior checkpoint)
    model = YOLO(base_model)

    try:
        results = model.train(
            data=os.path.abspath(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            project=project_dir,
            name="lunar_crater_yolo11",
            exist_ok=True,
            plots=True,
            save=True,
            val=True,
            workers=2,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            cos_lr=True,
            warmup_epochs=2,
            augment=True,
            hsv_h=0.015,
            hsv_s=0.2,
            hsv_v=0.3,
            degrees=180.0,    # Invariant to lunar orbital attitude rotation
            flipud=0.5,
            fliplr=0.5
        )
    except Exception as e:
        if device == "mps":
            print(f"\n⚠ MPS training encountered an issue ({e}). Falling back to CPU...")
            device = "cpu"
            results = model.train(
                data=os.path.abspath(data_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                device="cpu",
                project=project_dir,
                name="lunar_crater_yolo11",
                exist_ok=True,
                plots=True,
                save=True,
                val=True,
                workers=2,
                optimizer="AdamW",
                lr0=0.001,
                lrf=0.01,
                cos_lr=True,
                warmup_epochs=2,
                augment=True,
                degrees=180.0,
                flipud=0.5,
                fliplr=0.5
            )
        else:
            raise e

    # Locate best trained weights
    save_dir = getattr(results, "save_dir", None)
    best_weights_path = None
    candidate_paths = [
        os.path.join(save_dir, "weights", "best.pt") if save_dir else None,
        os.path.join("runs", "detect", project_dir, "lunar_crater_yolo11", "weights", "best.pt"),
        os.path.join(project_dir, "lunar_crater_yolo11", "weights", "best.pt"),
        os.path.join("runs", "train_lunar", "lunar_crater_yolo11", "weights", "best.pt")
    ]
    for cp in candidate_paths:
        if cp and os.path.exists(cp):
            best_weights_path = cp
            break

    if best_weights_path and os.path.exists(best_weights_path):
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        shutil.copy(best_weights_path, output_model_path)
        print(f"\n✅ Certified best model weights exported to: {output_model_path}")
    else:
        print(f"\n⚠ Best weights not found in standard directories. Using {output_model_path}")

    # Certification and evaluation pass
    print("\n🔍 Certifying final trained model on validation split...")
    eval_model_path = output_model_path if os.path.exists(output_model_path) else base_model
    trained_model = YOLO(eval_model_path)
    val_results = trained_model.val(data=os.path.abspath(data_yaml), device=device, verbose=False)

    precision = float(val_results.box.mp)
    recall = float(val_results.box.mr)
    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)
    speed_ms = val_results.speed.get("inference", 0.0)

    print("\n" + "=" * 70)
    print("📊 CERTIFIED LUNAR CRATER YOLO11 QUALITY METRICS")
    print(f"  • Precision:       {precision:.4f} ({precision*100:.1f}%)")
    print(f"  • Recall:          {recall:.4f} ({recall*100:.1f}%)")
    print(f"  • mAP@50:          {map50:.4f} ({map50*100:.1f}%)")
    print(f"  • mAP@50-95:       {map50_95:.4f} ({map50_95*100:.1f}%)")
    print(f"  • Inference Speed: {speed_ms:.2f} ms per tile")
    print(f"  • Model File:      {output_model_path} ({os.path.getsize(output_model_path)/(1024*1024):.2f} MB)")
    print("=" * 70)

    return output_model_path, val_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Mission Lunar YOLO11 Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base YOLO11 model")
    parser.add_argument("--device", type=str, default=None, help="Target device (mps, cpu, cuda)")
    args = parser.parse_args()

    train_lunar_yolo11(epochs=args.epochs, batch_size=args.batch, base_model=args.model, device=args.device)
