#!/usr/bin/env python3
"""
Compile YOLO Segmentation/Detection Models (.pt) to Hailo Executable Format (.hef)
Targeted for Raspberry Pi AI HAT+ (Hailo-8 / Hailo-8L)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Ensure USER environment variable is always defined for Hailo compiler tools
if "USER" not in os.environ:
    os.environ["USER"] = os.environ.get("USERNAME", "acer")
from ultralytics import YOLO

# Available categories
AVAILABLE_CATEGORIES = [
    "steel_defects", "bottle", "grid", "tile"
]


def compile_single_model(category, arch="hailo8", model_path=None, data_path=None, imgsz=640, output_dir=None, fast=False):
    base_dir = Path(__file__).resolve().parent

    # Determine model path
    if model_path:
        pt_file = Path(model_path).resolve()
    else:
        candidate_pt = (base_dir / "full_training" / f"{category}_production_best.pt").resolve()
        if not candidate_pt.exists():
            candidate_pt = (base_dir.parent / "full_training" / f"{category}_production_best.pt").resolve()
        pt_file = candidate_pt

    if not pt_file.exists():
        print(f"❌ Error: Model weights not found at {pt_file}")
        return False

    # Determine data.yaml path
    if data_path:
        data_yaml = Path(data_path).resolve()
    else:
        candidate_data = (base_dir / "hailo_calibration_data" / category / "data.yaml").resolve()
        if not candidate_data.exists():
            candidate_data = (Path.home() / "hailo_workspace" / "hailo_calibration_data" / category / "data.yaml").resolve()
        data_yaml = candidate_data

    if not data_yaml.exists():
        print(f"❌ Error: Calibration dataset not found at {data_yaml}")
        return False

    print("=" * 60)
    print(f"🚀 Compiling HEF for Category: '{category}'")
    print(f"   Model Weights : {pt_file.name}")
    print(f"   Calibration   : {data_yaml}")
    print(f"   Target HW     : {arch.upper()} ({'26 TOPS AI HAT+' if arch == 'hailo8' else '13 TOPS AI HAT+'})")
    print(f"   Image Size    : {imgsz}x{imgsz}")
    print(f"   Mode          : {'Fast PTQ (Level 1, ~1.5m)' if fast else 'Full QAT Fine-Tuning (Level 2, ~20m)'}")
    print("=" * 60)

    try:
        if fast:
            import hailo_sdk_client
            orig_load_model_script = hailo_sdk_client.ClientRunner.load_model_script
            def patched_load_model_script(self, script):
                lines = []
                for line in script.splitlines():
                    if "model_optimization_flavor(optimization_level=2)" in line:
                        lines.append("model_optimization_flavor(optimization_level=1)")
                    elif not line.strip().startswith("post_quantization_optimization(finetune"):
                        lines.append(line)
                return orig_load_model_script(self, "\n".join(lines))
            hailo_sdk_client.ClientRunner.load_model_script = patched_load_model_script

        # Load the trained YOLO model
        model = YOLO(str(pt_file))

        # Export natively to Hailo format using Ultralytics + Hailo Dataflow Compiler
        exported_path = model.export(
            format="hailo",
            name=arch,
            data=str(data_yaml),
            imgsz=imgsz
        )

        print(f"\n✅ Export finished successfully: {exported_path}")

        # If an output directory is specified, copy the HEF there
        if output_dir:
            out_path = Path(output_dir).resolve()
            out_path.mkdir(parents=True, exist_ok=True)

            # Locate .hef in the export directory or alongside model
            hef_file = None
            if exported_path and Path(exported_path).suffix == ".hef":
                hef_file = Path(exported_path)
            else:
                # Search in adjacent folders
                search_dir = pt_file.parent / f"{pt_file.stem}_hailo_model"
                candidates = list(search_dir.glob("*.hef")) + list(pt_file.parent.glob(f"{pt_file.stem}*.hef"))
                if candidates:
                    hef_file = candidates[0]

            if hef_file and hef_file.exists():
                dest_file = out_path / f"{category}_{arch}.hef"
                shutil.copy2(hef_file, dest_file)
                size_mb = dest_file.stat().st_size / (1024 * 1024)
                print(f"📦 HEF artifact saved to: {dest_file} ({size_mb:.2f} MB)")

        return True

    except Exception as e:
        print(f"\n❌ Compilation failed for '{category}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compile trained YOLO .pt models to Hailo .hef for Raspberry Pi AI HAT+",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c", "--category",
        type=str,
        default="tile",
        help=f"Category name to compile (choices: {', '.join(AVAILABLE_CATEGORIES)})"
    )
    parser.add_argument(
        "-a", "--arch",
        type=str,
        default="hailo8",
        choices=["hailo8l", "hailo8"],
        help="Target hardware: 'hailo8' (26 TOPS AI HAT+) or 'hailo8l' (13 TOPS AI HAT+)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Custom path to .pt model weights"
    )
    parser.add_argument(
        "-d", "--data",
        type=str,
        default=None,
        help="Custom path to data.yaml calibration file"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Model input image resolution (square)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast Post-Training Quantization (~1.5 min) instead of 4-epoch QAT (~20 min)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compile all 15 categories sequentially"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="hef_output",
        help="Directory to store compiled .hef files"
    )

    args = parser.parse_args()

    if args.all:
        print(f"\nStarting batch compilation for all {len(AVAILABLE_CATEGORIES)} categories targeting {args.arch}...")
        results = {}
        for cat in AVAILABLE_CATEGORIES:
            success = compile_single_model(
                category=cat,
                arch=args.arch,
                imgsz=args.imgsz,
                output_dir=args.output_dir,
                fast=args.fast
            )
            results[cat] = "SUCCESS" if success else "FAILED"

        print("\n" + "=" * 40)
        print("Batch Compilation Summary:")
        for cat, status in results.items():
            print(f"  {cat:<15}: {status}")
        print("=" * 40)
    else:
        compile_single_model(
            category=args.category,
            arch=args.arch,
            model_path=args.model,
            data_path=args.data,
            imgsz=args.imgsz,
            output_dir=args.output_dir,
            fast=args.fast
        )


if __name__ == "__main__":
    main()
