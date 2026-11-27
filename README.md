# autoshorts

Makes trending YouTube Shorts every day, on GitHub's free servers:

1. **Research:** finds Shorts in your niche from the last 14 days that got far more views than their channel's size (YouTube Data API, free).
2. **Script:** Claude writes an original ~45-second script on one of those topics (about 3-4 cents per video).
3. **Voice:** free Microsoft neural voice (edge-tts).
4. **Footage:** free vertical stock clips from Pexels, credited in the description.
5. **Render:** ffmpeg builds a 1080×1920 video with captions.
6. **Upload:** posts to your channel and records the topic in `posted.json` so it isn't repeated.

## One-time setup (about 20 minutes)

### 1. Claude API key
Create a key at https://console.anthropic.com → API Keys. Add a few dollars of credit.

### 2. Pexels API key (free)
Sign up at https://www.pexels.com/api/ and copy your key.

### 3. YouTube access
1. Go to https://console.cloud.google.com, create a project, and enable **YouTube Data API v3**.
2. **OAuth consent screen:** choose External, add your Gmail as a test user, then click **Publish app**. Otherwise Google expires your sign-in after 7 days.
3. **Credentials → Create credentials → OAuth client ID → Desktop app.** Copy the client ID and secret.
4. On your PC:
   ```
   pip install google-auth-oauthlib
   python get_youtube_token.py YOUR_CLIENT_ID YOUR_CLIENT_SECRET
   ```
   Sign in with the account that owns your channel and allow access. It prints `YT_REFRESH_TOKEN`.

### 4. Add the secrets to GitHub
Repo → **Settings → Secrets and variables → Actions → New repository secret**. Add:
`ANTHROPIC_API_KEY`, `PEXELS_API_KEY`, `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN`.

Optional: under **Variables**, set `SHORTS_PER_DAY` (default 5, about the most the default YouTube quota allows).

### 5. Test it
Repo → **Actions → Daily Shorts → Run workflow**, with privacy `none`. The finished video appears under the run's **Artifacts** without being uploaded. When you like it, run with `public`. After that it runs by itself every day.

## Limits you can't remove
- **Google keeps uploads private at first.** Videos uploaded through a new, unaudited Google Cloud project are locked to private. Apply for the free audit at https://support.google.com/youtube/contact/yt_api_form. Until it's approved, switch each video to public in YouTube Studio.
- **Daily quota:** 10,000 units a day by default. One run uses about 100 for research plus about 1,600 per upload, so roughly 5 uploads a day.
- **YouTube spam rules:** channels that post lots of near-identical AI videos can be demonetized or removed. Start with 1-3 a day and check the results.

## Run locally
```
pip install -r requirements.txt
python autoshorts.py --niche "space universe facts" --count 1 --privacy none
python -m pytest -q
```
