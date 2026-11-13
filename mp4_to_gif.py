"""Convert MP4 videos to looping GIFs.

Setup (once):
    pip install imageio-ffmpeg

Examples:
    python mp4_to_gif.py eval.mp4                      # -> eval.gif next to the video
    python mp4_to_gif.py videos/                       # every .mp4 in the folder and its subfolders
    python mp4_to_gif.py eval.mp4 --fps 15 --width 300
    python mp4_to_gif.py eval.mp4 --start 2 --duration 4
    python mp4_to_gif.py videos/ --out gifs/
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_ffmpeg() -> str:
    """Use the ffmpeg binary bundled with imageio-ffmpeg, else one on PATH."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        import shutil

        exe = shutil.which("ffmpeg")
        if exe is None:
            sys.exit("ffmpeg not found. Run: pip install imageio-ffmpeg")
        return exe


def convert(ffmpeg, src, dst, fps, width, start, duration, loop):
    # Scale to the requested width (height follows), then build one colour
    # palette for the whole clip. This gives far better colours than the
    # default GIF palette.
    scale = f"scale={width}:-1:flags=lanczos," if width else ""
    filters = (
        f"fps={fps},{scale}split[a][b];"
        "[a]palettegen=stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=5"
    )

    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(src), "-filter_complex", filters, "-loop", str(loop), str(dst)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED  {src.name}: {result.stderr.strip()}")
        return False

    size_mb = dst.stat().st_size / 1e6
    print(f"OK      {src.name} -> {dst}  ({size_mb:.1f} MB)")
    return True


def main():
    p = argparse.ArgumentParser(description="Convert MP4 videos to GIFs.")
    p.add_argument("input", type=Path, help="an .mp4 file or a folder of .mp4 files")
    p.add_argument("--out", type=Path, help="output folder (default: next to each video)")
    p.add_argument("--fps", type=int, default=15, help="frames per second (default: 15)")
    p.add_argument("--width", type=int, default=0,
                   help="output width in pixels; 0 keeps the original size (default: 0)")
    p.add_argument("--start", type=float, default=0, help="start time in seconds")
    p.add_argument("--duration", type=float, default=0,
                   help="length in seconds; 0 converts to the end (default: 0)")
    p.add_argument("--loop", type=int, default=0,
                   help="0 loops forever, -1 plays once (default: 0)")
    args = p.parse_args()

    if args.input.is_dir():
        # rglob also searches every subfolder
        videos = sorted(args.input.rglob("*.mp4"))
    elif args.input.is_file():
        videos = [args.input]
    else:
        sys.exit(f"Not found: {args.input}")
    if not videos:
        sys.exit(f"No .mp4 files in {args.input}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()
    done = 0
    for src in videos:
        if args.out and args.input.is_dir():
            # keep the subfolder structure so equal file names do not collide
            dst = args.out / src.relative_to(args.input).with_suffix(".gif")
            dst.parent.mkdir(parents=True, exist_ok=True)
        else:
            dst = (args.out or src.parent) / (src.stem + ".gif")
        done += convert(ffmpeg, src, dst, args.fps, args.width,
                        args.start, args.duration, args.loop)
    print(f"{done} of {len(videos)} converted.")


if __name__ == "__main__":
    main()