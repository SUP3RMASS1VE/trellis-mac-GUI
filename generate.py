"""
Generate a 3D mesh from a single image using TRELLIS.2 on Apple Silicon.
"""

# runner sets up MPS/backend env vars and sys.path; it must be imported first.
import runner

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from an image using TRELLIS.2")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", default="output_3d", help="Output filename without extension (default: output_3d)")
    parser.add_argument(
        "--pipeline-type", default="512",
        choices=runner.PIPELINE_TYPES,
        help="Pipeline resolution (default: 512)",
    )
    parser.add_argument(
        "--texture-size", type=int, default=1024,
        choices=runner.TEXTURE_SIZES,
        help="Texture resolution for PBR baking (default: 1024)",
    )
    parser.add_argument(
        "--no-texture", action="store_true",
        help="Skip texture baking, export geometry only",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override sampler steps for all three flow phases (default: pipeline JSON, usually 12)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")
        sys.exit(1)

    print("=" * 60)
    print("TRELLIS.2 on Apple Silicon")
    print("=" * 60)
    print(f"Input: {args.image}")

    try:
        result = runner.generate(
            args.image,
            seed=args.seed,
            pipeline_type=args.pipeline_type,
            texture_size=args.texture_size,
            texture=not args.no_texture,
            steps=args.steps,
            output=args.output,
        )
    except runner.WatchdogError as e:
        print(f"\nERROR: {e}\n")
        sys.exit(2)

    print(f"\nTotal time: {result['gen_time']:.1f}s generation + {result['bake_time']:.0f}s baking")


if __name__ == "__main__":
    main()
