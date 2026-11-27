"""Daily trending YouTube Shorts: research -> script -> voice -> footage -> render -> upload.

Free parts: YouTube Data API, Pixabay, edge-tts, ffmpeg. Paid part: one Claude call per video.
Env: ANTHROPIC_API_KEY, PIXABAY_API_KEY, YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN.
"""
import argparse
import asyncio
import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path

import anthropic
import edge_tts
import requests
from pydantic import BaseModel

STATE = Path(__file__).with_name("posted.json")
VOICE = "en-US-AndrewMultilingualNeural"  # warmer, more expressive than GuyNeural
YT = "https://www.googleapis.com/youtube/v3"


# ---------- YouTube ----------

def yt_token():
    r = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": os.environ["YT_CLIENT_ID"].strip(),
        "client_secret": os.environ["YT_CLIENT_SECRET"].strip(),
        # tolerate pasting the whole "YT_REFRESH_TOKEN = 1//..." line
        "refresh_token": os.environ["YT_REFRESH_TOKEN"].split("=")[-1].strip(),
        "grant_type": "refresh_token",
    }, timeout=30)
    if not r.ok:  # Google's body says which value is wrong (invalid_client vs invalid_grant)
        raise SystemExit(f"YouTube login failed: {r.status_code} {r.text}")
    return r.json()["access_token"]


def trending(token, niche, days=14):
    """Recent Shorts in the niche, ranked by views relative to channel size (outlier score)."""
    h = {"Authorization": f"Bearer {token}"}
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    found = requests.get(f"{YT}/search", headers=h, timeout=30, params={
        "part": "snippet", "q": niche, "type": "video", "videoDuration": "short",
        "order": "viewCount", "publishedAfter": since, "maxResults": 25,
    })
    found.raise_for_status()
    items = found.json()["items"]
    if not items:
        return []
    ids = ",".join(i["id"]["videoId"] for i in items)
    stats = requests.get(f"{YT}/videos", headers=h, timeout=30,
                         params={"part": "statistics", "id": ids}).json()["items"]
    views = {v["id"]: int(v["statistics"].get("viewCount", 0)) for v in stats}
    chan_ids = ",".join({i["snippet"]["channelId"] for i in items})
    chans = requests.get(f"{YT}/channels", headers=h, timeout=30,
                         params={"part": "statistics", "id": chan_ids}).json()["items"]
    subs = {c["id"]: int(c["statistics"].get("subscriberCount", 0)) for c in chans}
    return rank([{
        "id": i["id"]["videoId"],
        "title": i["snippet"]["title"],
        "views": views.get(i["id"]["videoId"], 0),
        "subs": subs.get(i["snippet"]["channelId"], 0),
    } for i in items])


def rank(videos):
    for v in videos:
        v["score"] = round(v["views"] / max(v["subs"], 1000), 1)  # floor so tiny channels don't explode the score
    return sorted(videos, key=lambda v: v["score"], reverse=True)


def upload(token, path, script, privacy):
    meta = {
        "snippet": {"title": script.title[:100], "description": script.description,
                    "tags": script.tags[:15], "categoryId": "28"},
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False,
                   "containsSyntheticMedia": True},
    }
    start = requests.post(
        "https://www.googleapis.com/upload/youtube/v3/videos",
        params={"uploadType": "resumable", "part": "snippet,status"},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                 "X-Upload-Content-Type": "video/mp4"},
        data=json.dumps(meta), timeout=60)
    start.raise_for_status()
    with open(path, "rb") as f:
        done = requests.put(start.headers["Location"], data=f, timeout=600,
                            headers={"Content-Type": "video/mp4"})
    done.raise_for_status()
    return done.json()["id"]


# ---------- Script (Claude) ----------

class Line(BaseModel):
    text: str
    footage_query: str


class Script(BaseModel):
    source_video_id: str
    title: str
    description: str
    tags: list[str]
    lines: list[Line]


def write_script(niche, videos, used_ids):
    pool = [v for v in videos if v["id"] not in used_ids][:12]
    if not pool:
        return None
    prompt = f"""You write ~45-second YouTube Shorts for a channel about {niche}.
These Shorts are breaking out right now (score = views divided by channel subscribers):
{json.dumps(pool, indent=1)}

Pick ONE and write an original Short on the same topic or angle, not a copy of its title.
- 100-120 spoken words in 8-12 lines. The first line is the hook: a concrete, surprising detail.
- Only state facts that are well established. Use no numbers you are unsure of.
- footage_query: 2-4 plain words that find matching vertical stock footage on Pixabay (e.g. "galaxy stars", "sun surface").
- title under 70 characters; description of 1-2 sentences ending with 3-4 hashtags including #shorts; 5-10 tags.
- source_video_id: the id of the video you picked."""
    resp = anthropic.Anthropic().messages.parse(
        model="claude-opus-5-5",
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
        output_format=Script,
    )
    if resp.stop_reason == "refusal":
        return None
    return resp.parsed_output


# ---------- Voice, footage, render ----------

def speak(lines, out_dir):
    async def run():
        paths = []
        for i, line in enumerate(lines):
            p = out_dir / f"line{i}.mp3"
            await edge_tts.Communicate(line, VOICE, rate="+12%").save(str(p))
            paths.append(p)
        return paths
    return asyncio.run(run())


def find_clip(query, out_path, skip):
    r = requests.get("https://pixabay.com/api/videos/", timeout=30, params={
        "key": os.environ["PIXABAY_API_KEY"], "q": query, "per_page": 20, "safesearch": "true"})
    r.raise_for_status()
    hits = [h for h in r.json()["hits"] if h["id"] not in skip and h["videos"]["large"]["url"]]
    # Pixabay has no orientation filter: prefer vertical clips, render() crops the rest
    hits.sort(key=lambda h: h["videos"]["large"]["height"] <= h["videos"]["large"]["width"])
    if not hits:
        return None
    hit = hits[0]
    out_path.write_bytes(requests.get(hit["videos"]["large"]["url"], timeout=120).content)
    skip.add(hit["id"])
    return hit["user"]


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout)


def srt_time(t):
    ms = round(t * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def captions(lines, durations, words_per_caption=4):
    """SRT with short chunks; each line's time is split by word count."""
    out, t, n = [], 0.0, 1
    for text, d in zip(lines, durations):
        words = text.split()
        chunks = [words[i:i + words_per_caption] for i in range(0, len(words), words_per_caption)]
        for chunk in chunks:
            span = d * len(chunk) / len(words)
            out.append(f"{n}\n{srt_time(t)} --> {srt_time(t + span)}\n{' '.join(chunk).upper()}\n")
            t, n = t + span, n + 1
    return "\n".join(out)


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def render(segments, out_path, work):
    """segments: [(clip_path, audio_path, text)] -> vertical 1080x1920 mp4 with burned captions."""
    durs = [duration(a) for _, a, _ in segments]
    parts = []
    for i, ((clip, _, _), d) in enumerate(zip(segments, durs)):
        part = work / f"part{i}.mp4"
        ff("-stream_loop", "-1", "-i", str(clip), "-t", f"{d:.3f}", "-an",
           "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,setsar=1",
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(part))
        parts.append(part)
    (work / "v.txt").write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    (work / "a.txt").write_text("".join(f"file '{Path(a).resolve()}'\n" for _, a, _ in segments))
    ff("-f", "concat", "-safe", "0", "-i", str(work / "v.txt"), "-c", "copy", str(work / "video.mp4"))
    ff("-f", "concat", "-safe", "0", "-i", str(work / "a.txt"), "-c:a", "aac", str(work / "audio.m4a"))
    (work / "subs.srt").write_text(captions([t for _, _, t in segments], durs))
    style = "FontName=DejaVu Sans,FontSize=16,Bold=1,Outline=3,Shadow=0,Alignment=2,MarginV=40"
    ff("-i", str(work / "video.mp4"), "-i", str(work / "audio.m4a"),
       "-vf", f"subtitles={work / 'subs.srt'}:force_style='{style}'",
       "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac",
       "-shortest", str(out_path))


# ---------- Main ----------

def make_one(token, niche, videos, state, privacy, out_dir):
    script = write_script(niche, videos, {p["source"] for p in state})
    if not script:
        return None
    work = Path(tempfile.mkdtemp())
    texts = [l.text for l in script.lines]
    audios = speak(texts, work)
    used, credits, segments = set(), set(), []
    for i, (line, audio) in enumerate(zip(script.lines, audios)):
        clip = work / f"clip{i}.mp4"
        who = find_clip(line.footage_query, clip, used) or find_clip(niche, clip, used)
        if not who:
            raise RuntimeError(f"no footage for {line.footage_query!r}")
        credits.add(who)
        segments.append((clip, audio, line.text))
    script.description += f"\n\nFootage: Pixabay ({', '.join(sorted(credits))}). Voice: AI."
    out = out_dir / f"{dt.date.today()}-{script.source_video_id}.mp4"
    render(segments, out, work)
    video_id = upload(token, out, script, privacy) if privacy != "none" else None
    state.append({"date": str(dt.date.today()), "source": script.source_video_id,
                  "title": script.title, "youtube_id": video_id})
    STATE.write_text(json.dumps(state, indent=1))
    print(f"made {out}" + (f" -> https://youtu.be/{video_id}" if video_id else " (not uploaded)"))
    return video_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niche", default="space universe facts")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--privacy", default="public", choices=["public", "unlisted", "private", "none"],
                    help="'none' renders without uploading")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else []
    token = yt_token()
    videos = trending(token, a.niche)
    print("trending:", *[f"  {v['score']:>6}x  {v['title']}" for v in videos[:8]], sep="\n")
    for _ in range(a.count):
        try:
            make_one(token, a.niche, videos, state, a.privacy, out_dir)
        except Exception as e:  # one bad video shouldn't stop the rest
            print("failed:", e)


if __name__ == "__main__":
    main()
