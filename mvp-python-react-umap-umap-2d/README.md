# Music Taste Embedding Map

Python + React MVP for mapping music taste with text embeddings and UMAP.

The app embeds song metadata, descriptions, reviews, and optionally YouTube top comments. It then reduces the vectors to 2D with UMAP and displays them in a React SVG scatter plot.

Audio extraction from YouTube is intentionally not included. For YouTube links, the app uses the official YouTube Data API for metadata and comments. Future audio support should use user-uploaded or otherwise authorized audio files.

## Project Structure

```text
music-embedding-umap/
  backend/
    app.py
    requirements.txt
    data/
      sample_songs.json
    outputs/
      songs_with_embeddings.json
      songs_with_umap.json
  frontend/
    package.json
    src/
      App.tsx
      main.tsx
      styles.css
  README.md
```

## Backend Setup

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8787
```

Or:

```powershell
cd backend
python app.py
```

Backend URL: http://127.0.0.1:8787

The first `Build Index` request can take a while because `dragonkue/multilingual-e5-small-ko-v2` must be downloaded.

## YouTube Import

Set a YouTube Data API key before running the backend:

```powershell
$env:YOUTUBE_API_KEY="your_api_key_here"
uvicorn app:app --reload --port 8787
```

The app accepts normal YouTube and YouTube Music links, extracts the video id, fetches video metadata, imports top-level comments using the official `commentThreads.list` API, appends or updates the song in `backend/data/sample_songs.json`, and rebuilds the embedding index.

No YouTube audio is downloaded or scraped.

## Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL: http://localhost:5173

## GitHub Pages Website

This repository includes a GitHub Actions workflow at `.github/workflows/deploy-pages.yml`.

GitHub Pages can only host the static React frontend. It cannot run the Python FastAPI backend, sentence-transformers, UMAP, or YouTube Data API import. For that reason, the deployed website automatically uses `frontend/public/sample_songs_umap.json` as static demo data.

The public GitHub Pages version supports:

- viewing the music taste map
- clicking songs for details
- simple in-browser text search
- simple in-browser similar song lookup
- simple in-browser taste recommendations

The local full version supports:

- real sentence-transformers embeddings
- real UMAP rebuilds
- YouTube metadata/comment import with `YOUTUBE_API_KEY`
- backend-powered similarity search

To publish:

```powershell
git init
git add .
git commit -m "Build music embedding UMAP MVP"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Then in GitHub:

1. Open the repository settings.
2. Go to `Pages`.
3. Set `Source` to `GitHub Actions`.
4. Push to `main` again if the workflow has not run yet.

Your site will be available at:

```text
https://YOUR_USERNAME.github.io/YOUR_REPO/
```

## Features

- Text embedding with `sentence-transformers`
- E5-compatible prefixes: `passage:` for songs and `query:` for searches
- YouTube metadata and top comments as extra text signal
- UMAP 2D scatter plot
- Similar song search by text query
- Similar song lookup from a selected point
- Taste profile recommendations from liked songs

## API

- `GET /api/songs`: returns the current song list and UMAP coordinates.
- `POST /api/build-index`: rebuilds embeddings and UMAP coordinates.
- `POST /api/import-youtube`: imports YouTube metadata/comments and rebuilds the index.
- `POST /api/search`: returns songs near a text query.
- `GET /api/similar/{song_id}`: returns songs similar to one song.
- `POST /api/user-profile`: recommends songs from liked song ids.

YouTube import example:

```json
{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "genre": "YouTube",
  "tags": ["youtube", "comments", "imported"],
  "max_comments": 20,
  "rebuild_index": true
}
```

Search example:

```json
{
  "query": "dreamy late-night K-pop with soft comments",
  "top_k": 5
}
```

## Test Flow

1. Start the backend.
2. Call `POST /api/build-index`.
3. Call `POST /api/search`.
4. Set `YOUTUBE_API_KEY`, paste a YouTube link in the frontend, and click `Import`.
5. Confirm the new point appears on the map.
6. Click a point to see details and imported top comments.
7. Try `Find Similar Songs` and taste recommendations.
