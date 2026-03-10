from __future__ import annotations

import argparse

try:
    from ml.four_model_manifest import validate_four_model_manifest
except ModuleNotFoundError:
    from four_model_manifest import validate_four_model_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    manifest, errors = validate_four_model_manifest(args.manifest)
    print(f"Manifest: {manifest.path}")
    print(f"Data:     {manifest.data_path}")
    print(f"Device:   {manifest.device}")
    print(f"Workers:  {manifest.workers}")
    print(f"Outputs:")
    print(f"  - bidding: {manifest.outputs.bidding}")
    print(f"  - passing: {manifest.outputs.passing}")
    print(f"  - playing: {manifest.outputs.playing}")
    print(f"  - belief:  {manifest.outputs.belief}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Manifest looks valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
