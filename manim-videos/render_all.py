#!/usr/bin/env python3
"""
VoxFormer STT Manim Video Renderer
===================================

Main entry point for rendering all VoxFormer STT explanation videos.

Usage:
    # Render all videos (production quality)
    python render_all.py --all

    # Render specific video
    python render_all.py --video 1

    # Preview mode (low quality, fast)
    python render_all.py --video 1 --preview

    # Render specific scene
    python render_all.py --scene AudioWaveformScene

    # List all scenes
    python render_all.py --list
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Video modules and their scenes
VIDEOS = {
    1: {
        "name": "Audio Pipeline",
        "file": "scenes/01_audio_pipeline.py",
        "scenes": [
            "IntroScene",
            "AudioWaveformScene",
            "STFTScene",
            "MelFilterBankScene",
            "LogCompressionScene",
            "AudioPipelineSummary",
        ]
    },
    2: {
        "name": "Transformer Foundations",
        "file": "scenes/02_transformer_foundations.py",
        "scenes": [
            "TransformerIntroScene",
            "ConvSubsamplingScene",
            "PositionEncodingIntroScene",
            "RoPEScene",
            "AttentionIntroScene",
            "ScaledDotProductScene",
            "MultiHeadAttentionScene",
            "TransformerFoundationsSummary",
        ]
    },
    3: {
        "name": "Conformer Block",
        "file": "scenes/03_conformer_block.py",
        "scenes": [
            "ConformerIntroScene",
            "ConformerOverviewScene",
            "FeedForwardScene",
            "ConvModuleScene",
            "ConformerStackScene",
            "ConformerSummaryScene",
        ]
    },
    4: {
        "name": "CTC Loss",
        "file": "scenes/04_ctc_loss.py",
        "scenes": [
            "CTCIntroScene",
            "CTCProblemScene",
            "CTCBlankTokenScene",
            "CTCAlgorithmScene",
            "CTCGradientScene",
            "CTCDecodingScene",
            "CTCSummaryScene",
        ]
    },
    5: {
        "name": "Full Pipeline",
        "file": "scenes/05_full_pipeline.py",
        "scenes": [
            "FinalIntroScene",
            "EndToEndScene",
            "ModelConfigScene",
            "TrainingScene",
            "InferenceScene",
            "FinalSummaryScene",
        ]
    },
}

# Quality presets
QUALITY_PRESETS = {
    "preview": "-ql",           # 480p, 15fps
    "low": "-ql",               # 480p, 15fps
    "medium": "-qm",            # 720p, 30fps
    "high": "-qh",              # 1080p, 60fps
    "production": "-qp",        # 1440p, 60fps
    "4k": "-qk",                # 4K, 60fps
}


def list_all_scenes():
    """List all available scenes"""
    print("\n" + "=" * 60)
    print("VoxFormer STT Manim Videos - Scene List")
    print("=" * 60)

    total_scenes = 0
    for video_num, video_info in VIDEOS.items():
        print(f"\nVideo {video_num}: {video_info['name']}")
        print(f"  File: {video_info['file']}")
        print("  Scenes:")
        for i, scene in enumerate(video_info['scenes'], 1):
            print(f"    {i}. {scene}")
            total_scenes += 1

    print(f"\n{'=' * 60}")
    print(f"Total: {len(VIDEOS)} videos, {total_scenes} scenes")
    print("=" * 60 + "\n")


def render_scene(file_path: str, scene_name: str, quality: str = "medium", preview: bool = False):
    """Render a single scene"""
    quality_flag = QUALITY_PRESETS.get(quality, "-qm")
    if preview:
        quality_flag = "-pql"  # Preview with low quality

    cmd = ["manim", quality_flag, file_path, scene_name]

    print(f"\n{'=' * 60}")
    print(f"Rendering: {scene_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60 + "\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Successfully rendered: {scene_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to render: {scene_name}")
        print(f"  Error: {e}")
        return False
    except FileNotFoundError:
        print("\n✗ Error: 'manim' command not found.")
        print("  Please install manim: pip install manimgl")
        return False


def render_video(video_num: int, quality: str = "medium", preview: bool = False):
    """Render all scenes for a video"""
    if video_num not in VIDEOS:
        print(f"Error: Video {video_num} not found. Valid: 1-5")
        return False

    video_info = VIDEOS[video_num]
    print(f"\n{'#' * 60}")
    print(f"# Rendering Video {video_num}: {video_info['name']}")
    print(f"# Scenes: {len(video_info['scenes'])}")
    print("#" * 60)

    success_count = 0
    for scene in video_info['scenes']:
        if render_scene(video_info['file'], scene, quality, preview):
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"Video {video_num} Complete: {success_count}/{len(video_info['scenes'])} scenes rendered")
    print("=" * 60)

    return success_count == len(video_info['scenes'])


def render_all(quality: str = "medium"):
    """Render all videos"""
    print("\n" + "#" * 60)
    print("# VoxFormer STT - Rendering All Videos")
    print("# This may take several hours for production quality")
    print("#" * 60)

    results = {}
    for video_num in VIDEOS:
        results[video_num] = render_video(video_num, quality)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for video_num, success in results.items():
        status = "✓ Complete" if success else "✗ Failed"
        print(f"  Video {video_num}: {status}")
    print("=" * 60 + "\n")


def find_scene(scene_name: str):
    """Find which file contains a scene"""
    for video_num, video_info in VIDEOS.items():
        if scene_name in video_info['scenes']:
            return video_info['file'], video_num
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="VoxFormer STT Manim Video Renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python render_all.py --list                    # List all scenes
  python render_all.py --video 1                 # Render Video 1
  python render_all.py --video 1 --preview       # Preview Video 1
  python render_all.py --scene AudioWaveformScene  # Render specific scene
  python render_all.py --all                     # Render all videos
  python render_all.py --all --quality 4k        # Render all in 4K
        """
    )

    parser.add_argument("--list", "-l", action="store_true",
                        help="List all available scenes")
    parser.add_argument("--video", "-v", type=int, choices=[1, 2, 3, 4, 5],
                        help="Render specific video (1-5)")
    parser.add_argument("--scene", "-s", type=str,
                        help="Render specific scene by name")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Render all videos")
    parser.add_argument("--preview", "-p", action="store_true",
                        help="Preview mode (fast, low quality)")
    parser.add_argument("--quality", "-q", type=str, default="medium",
                        choices=["preview", "low", "medium", "high", "production", "4k"],
                        help="Render quality (default: medium)")

    args = parser.parse_args()

    # Change to script directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)

    if args.list:
        list_all_scenes()
    elif args.scene:
        file_path, video_num = find_scene(args.scene)
        if file_path:
            render_scene(file_path, args.scene, args.quality, args.preview)
        else:
            print(f"Error: Scene '{args.scene}' not found.")
            print("Use --list to see available scenes.")
    elif args.video:
        render_video(args.video, args.quality, args.preview)
    elif args.all:
        render_all(args.quality)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
