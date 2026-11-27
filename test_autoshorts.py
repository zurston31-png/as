"""Offline checks: ranking, captions, and a real ffmpeg render of synthetic clips."""
import subprocess
import tempfile
from pathlib import Path

from autoshorts import captions, duration, rank, render


def test_rank_by_outlier_score():
    vids = rank([
        {"id": "a", "title": "big channel", "views": 1_000_000, "subs": 2_000_000},
        {"id": "b", "title": "small channel", "views": 200_000, "subs": 5_000},
        {"id": "c", "title": "tiny channel", "views": 5_000, "subs": 10},
    ])
    assert [v["id"] for v in vids] == ["b", "c", "a"]
    assert vids[1]["score"] == 5.0  # subs floored at 1000


def test_captions_cover_full_audio():
    srt = captions(["one two three four five", "six"], [2.0, 1.0])
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,600\nONE TWO THREE FOUR\n")
    assert "00:00:02,000 --> 00:00:03,000\nSIX" in srt


def test_render_vertical_with_audio():
    work = Path(tempfile.mkdtemp())
    segments = []
    for i, (size, secs) in enumerate([("1920x1080", 1.2), ("720x1280", 0.8)]):
        clip, audio = work / f"c{i}.mp4", work / f"a{i}.mp3"
        # clip shorter than its audio, so the loop path is exercised
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"testsrc=size={size}:rate=25:duration=0.5", str(clip)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"sine=frequency=440:duration={secs}", str(audio)], check=True)
        segments.append((clip, audio, f"line number {i}"))
    out = work / "out.mp4"
    render(segments, out, work)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert probe == "1080,1920"
    assert abs(duration(out) - 2.0) < 0.25
