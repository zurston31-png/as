"""Run once on your own computer to get YT_REFRESH_TOKEN.

pip install google-auth-oauthlib
python get_youtube_token.py CLIENT_ID CLIENT_SECRET
"""
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

client_id, client_secret = sys.argv[1], sys.argv[2]
flow = InstalledAppFlow.from_client_config(
    {"installed": {"client_id": client_id, "client_secret": client_secret,
                   "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                   "token_uri": "https://oauth2.googleapis.com/token"}},
    scopes=["https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube.readonly"],
)
creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")
print("\nYT_REFRESH_TOKEN =", creds.refresh_token)
