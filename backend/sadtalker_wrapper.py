#!/usr/bin/env python3
"""
SadTalker Wrapper Script
Provides a compatible interface for the backend API.

Usage:
    python3 run_inference.py --audio /path/to/audio.mp3 --image /path/to/face.png --output /path/to/output.mp4

This wrapper translates to SadTalker's native interface.
"""

import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="SadTalker wrapper for backend API")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--image", required=True, help="Path to source face image")
    parser.add_argument("--output", required=True, help="Path for output video file")
    parser.add_argument("--size", type=int, default=512, choices=[256, 512], help="Output resolution")
    parser.add_argument("--enhancer", default="gfpgan", help="Face enhancer (gfpgan or RestoreFormer)")
    parser.add_argument("--version", default="v15", help="Ignored (for MuseTalk compat)")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    # Get the directory of this script (SadTalker root)
    script_dir = Path(__file__).parent.absolute()

    # Create temp results directory
    result_dir = script_dir / "results_temp"
    result_dir.mkdir(exist_ok=True)

    # Build SadTalker command
    cmd = [
        sys.executable,
        str(script_dir / "inference.py"),
        "--source_image", args.image,
        "--driven_audio", args.audio,
        "--result_dir", str(result_dir),
        "--size", str(args.size),
        "--preprocess", "crop",  # Good for portrait images
        "--enhancer", args.enhancer,
    ]

    print(f"Running SadTalker: {' '.join(cmd)}")

    # Run SadTalker
    result = subprocess.run(cmd, cwd=str(script_dir))

    if result.returncode != 0:
        print(f"Error: SadTalker failed with code {result.returncode}")
        sys.exit(result.returncode)

    # Find the output video (SadTalker creates timestamped directories)
    output_videos = list(result_dir.glob("*.mp4"))

    if not output_videos:
        print("Error: No output video found")
        sys.exit(1)

    # Get the most recent video
    latest_video = max(output_videos, key=lambda p: p.stat().st_mtime)

    # Move to the requested output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(latest_video), str(output_path))

    print(f"Output video saved to: {output_path}")

    # Clean up temp results
    for f in result_dir.iterdir():
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()

if __name__ == "__main__":
    main()
