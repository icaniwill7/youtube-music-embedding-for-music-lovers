from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover
    cosine_similarity = None

try:
    import umap
except Exception:  # pragma: no cover
    umap = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
SAMPLE_PATH = DATA_DIR / "sample_songs.json"
EMBEDDINGS_PATH = OUTPUT_DIR / "songs_with_embeddings.json"
UMAP_PATH = OUTPUT_DIR / "songs_with_umap.json"
MODEL_NAME = "dragonkue/multilingual-e5-small-ko-v2"
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

app = FastAPI(title="Music Taste Embedding Map API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Any | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class UserProfileRequest(BaseModel):
    liked_song_ids: list[str] = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class YouTubeImportRequest(BaseModel):
    youtube_url: str = Field(..., min_length=5)
    title: str | None = None
    artist: str | None = None
    genre: str = "YouTube"
    tags: list[str] = Field(default_factory=lambda: ["youtube", "comments"])
    max_comments: int = Field(default=20, ge=0, le=100)
    rebuild_index: bool = True


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sample_songs() -> list[dict[str, Any]]:
    seeds = [
        ("Blue Dawn Signal", "Luna Field", "K-pop", ["fresh", "band pop", "dawn"], "A bright K-pop band track with airy guitar."),
        ("Neon Rain", "City Echo", "Synth Pop", ["dreamy", "city", "synth"], "A glossy synth-pop song for rainy nights."),
        ("Paper Airplane", "Mellow Apricot", "Indie Pop", ["warm", "acoustic", "daily"], "A gentle indie-pop song with acoustic guitar."),
        ("Chrome Heartbeat", "Nova Unit", "Electronic", ["fast", "digital", "drive"], "A high-energy electronic track with metallic synths."),
        ("Foggy Window", "Haerin Park", "R&B", ["smooth", "night", "groove"], "A soft R&B song with warm chords."),
        ("Summer Arcade", "Pixel Soda", "Dance Pop", ["bright", "summer", "dance"], "A playful dance-pop track with bright synths."),
        ("Quiet Orbit", "Jin Sol", "Ballad", ["piano", "calm", "emotional"], "A restrained piano ballad with a quiet mood."),
        ("Dusty Vinyl", "Brown Street", "Jazz", ["vintage", "cafe", "relaxed"], "A relaxed jazz-pop song with soft piano."),
        ("Highway Bloom", "Road Mint", "Rock", ["guitar", "drive", "open"], "A breezy rock song for long drives."),
        ("Moon Jelly", "Sea Glass", "Dream Pop", ["floating", "ocean", "dreamy"], "A reverb-heavy dream-pop song with floating vocals."),
        ("Metro Pulse", "Line 7", "Hip-hop", ["beat", "city", "cool"], "A crisp hip-hop track with low bass."),
        ("Peach Sunset", "Orenji", "K-pop", ["sweet", "youth", "sunset"], "A sweet K-pop song with a sunset glow."),
        ("Snow Lantern", "Yuri Han", "Folk", ["winter", "acoustic", "quiet"], "A minimal folk song with gentle warmth."),
        ("Velvet Bassline", "Mood Circuit", "R&B", ["bass", "sleek", "groove"], "A sleek R&B track led by a soft bassline."),
        ("Spark Run", "Vivid Track", "Dance Pop", ["energy", "workout", "fast"], "An uptempo dance-pop song with strong kicks."),
        ("Afterimage", "Mono Lake", "Alternative", ["lonely", "guitar", "memory"], "A spacious alternative track with wistful guitars."),
        ("Tiny Festival", "Cherry Parade", "Indie Pop", ["cute", "festival", "light"], "A light indie-pop song with handclap rhythm."),
        ("Deep Current", "Abyss Tone", "Ambient", ["deep", "meditative", "slow"], "A slow ambient track with wide space."),
        ("Glass Runner", "Prism Beat", "Electronic", ["cold", "tense", "motion"], "A tense electronic song with sharp arpeggios."),
        ("Diary Page", "Min Seo", "Ballad", ["confession", "piano", "delicate"], "A delicate piano ballad with intimate vocals."),
    ]
    return [
        {
            "id": f"song-{index + 1:03d}",
            "title": title,
            "artist": artist,
            "genre": genre,
            "tags": tags,
            "review": f"A {', '.join(tags)} mood stands out in this sample review.",
            "description": description,
        }
        for index, (title, artist, genre, tags, description) in enumerate(seeds)
    ]


def read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    ensure_dirs()
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def ensure_sample_data() -> None:
    ensure_dirs()
    if not SAMPLE_PATH.exists():
        write_json(SAMPLE_PATH, sample_songs())


def load_songs() -> list[dict[str, Any]]:
    ensure_sample_data()
    songs = read_json(SAMPLE_PATH, [])
    if not isinstance(songs, list):
        raise ValueError("sample_songs.json must contain a list of songs.")
    return songs


def save_songs(songs: list[dict[str, Any]]) -> None:
    write_json(SAMPLE_PATH, songs)


def public_song(song: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": song.get("id"),
        "title": song.get("title"),
        "artist": song.get("artist"),
        "genre": song.get("genre"),
        "tags": song.get("tags", []),
        "review": song.get("review", ""),
        "description": song.get("description", ""),
        "youtube_url": song.get("youtube_url"),
        "youtube_video_id": song.get("youtube_video_id"),
        "youtube_comments": song.get("youtube_comments", []),
        "x": song.get("x"),
        "y": song.get("y"),
        "similarity": song.get("similarity"),
    }


def embedding_text(song: dict[str, Any]) -> str:
    tags = ", ".join(song.get("tags", []))
    comments = " ".join(song.get("youtube_comments", [])[:30])
    youtube_title = song.get("youtube_title", "")
    return (
        f"{song.get('title', '')} - {song.get('artist', '')}. "
        f"Genre: {song.get('genre', '')}. "
        f"Tags: {tags}. "
        f"YouTube title: {youtube_title}. "
        f"Description: {song.get('description', '')}. "
        f"Review: {song.get('review', '')}. "
        f"Top comments: {comments}"
    )


def get_model() -> Any:
    global _model
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")
    if _model is None:
        print(
            f"Loading sentence-transformers model: {MODEL_NAME}. "
            "First run may take several minutes while the model downloads."
        )
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    return np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=True),
        dtype=np.float32,
    )


def circular_coordinates(count: int) -> np.ndarray:
    if count == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32)
    coords = []
    for index in range(count):
        angle = 2 * math.pi * index / count
        coords.append([math.cos(angle), math.sin(angle)])
    return np.asarray(coords, dtype=np.float32)


def build_umap(embeddings: np.ndarray) -> np.ndarray:
    count = len(embeddings)
    if count < 3 or umap is None:
        return circular_coordinates(count)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(10, count - 1)),
        random_state=42,
        metric="cosine",
    )
    return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)


def build_index() -> list[dict[str, Any]]:
    songs = load_songs()
    if not songs:
        write_json(EMBEDDINGS_PATH, [])
        write_json(UMAP_PATH, [])
        return []

    texts = [f"passage: {embedding_text(song)}" for song in songs]
    embeddings = encode_texts(texts)
    coords = build_umap(embeddings)

    songs_with_embeddings: list[dict[str, Any]] = []
    songs_with_umap: list[dict[str, Any]] = []

    for song, vector, coord in zip(songs, embeddings, coords, strict=True):
        base_song = {**song, "embedding_text": embedding_text(song)}
        songs_with_embeddings.append({**base_song, "embedding": vector.tolist()})
        songs_with_umap.append({**base_song, "x": float(coord[0]), "y": float(coord[1])})

    write_json(EMBEDDINGS_PATH, songs_with_embeddings)
    write_json(UMAP_PATH, songs_with_umap)
    return songs_with_umap


def load_index() -> list[dict[str, Any]]:
    data = read_json(EMBEDDINGS_PATH, [])
    if not data:
        build_index()
        return read_json(EMBEDDINGS_PATH, [])
    return data


def rank_by_vector(vector: np.ndarray, top_k: int, exclude_id: str | None = None) -> list[dict[str, Any]]:
    if cosine_similarity is None:
        raise RuntimeError("scikit-learn is not installed.")
    songs = load_index()
    candidates = [song for song in songs if song.get("id") != exclude_id]
    if not candidates:
        return []

    matrix = np.asarray([song["embedding"] for song in candidates], dtype=np.float32)
    scores = cosine_similarity(vector.reshape(1, -1), matrix)[0]
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    umap_by_id = {song.get("id"): song for song in read_json(UMAP_PATH, [])}
    results = []
    for index in ranked_indices:
        song = {**candidates[int(index)]}
        song.update(umap_by_id.get(song.get("id"), {}))
        song["similarity"] = float(scores[int(index)])
        results.append(public_song(song))
    return results


def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().replace("www.", "")

    if host in {"youtu.be", "music.youtube.com"} and parsed.path:
        if host == "youtu.be":
            return parsed.path.strip("/")
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id:
            return query_id

    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id:
            return query_id
        match = re.search(r"/(?:shorts|embed)/([^/?#]+)", parsed.path)
        if match:
            return match.group(1)

    if re.fullmatch(r"[\w-]{11}", url.strip()):
        return url.strip()

    raise ValueError("Could not extract a YouTube video id from the URL.")


def youtube_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("Set YOUTUBE_API_KEY before importing YouTube metadata.")

    url = f"{YOUTUBE_API_BASE}/{path}?{urlencode({**params, 'key': api_key})}"
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "music-embedding-umap/0.1"})
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_youtube_metadata(video_id: str, max_comments: int) -> dict[str, Any]:
    videos = youtube_get(
        "videos",
        {"part": "snippet", "id": video_id, "maxResults": 1},
    )
    items = videos.get("items", [])
    if not items:
        raise ValueError(f"YouTube video not found: {video_id}")

    snippet = items[0].get("snippet", {})
    comments: list[str] = []

    if max_comments > 0:
        comment_data = youtube_get(
            "commentThreads",
            {
                "part": "snippet",
                "videoId": video_id,
                "order": "relevance",
                "textFormat": "plainText",
                "maxResults": max_comments,
            },
        )
        for item in comment_data.get("items", []):
            comment = (
                item.get("snippet", {})
                .get("topLevelComment", {})
                .get("snippet", {})
                .get("textDisplay", "")
            )
            if comment:
                comments.append(strip_html(comment))

    return {
        "youtube_title": snippet.get("title", ""),
        "youtube_channel": snippet.get("channelTitle", ""),
        "youtube_description": snippet.get("description", ""),
        "youtube_tags": snippet.get("tags", []),
        "youtube_comments": comments,
    }


def next_imported_song_id(songs: list[dict[str, Any]]) -> str:
    existing = {song.get("id") for song in songs}
    index = 1
    while True:
        song_id = f"yt-{index:03d}"
        if song_id not in existing:
            return song_id
        index += 1


def import_youtube_song(request: YouTubeImportRequest) -> dict[str, Any]:
    video_id = extract_youtube_video_id(request.youtube_url)
    songs = load_songs()

    existing = next((song for song in songs if song.get("youtube_video_id") == video_id), None)
    metadata = fetch_youtube_metadata(video_id, request.max_comments)

    title = request.title or metadata["youtube_title"] or f"YouTube video {video_id}"
    artist = request.artist or metadata["youtube_channel"] or "YouTube"
    description_parts = [
        metadata.get("youtube_description", "")[:1200],
        "Imported from YouTube metadata and top comments.",
    ]
    review = " ".join(metadata.get("youtube_comments", [])[:10])
    if not review:
        review = "No public top comments were imported."

    song = {
        "id": existing.get("id") if existing else next_imported_song_id(songs),
        "title": title,
        "artist": artist,
        "genre": request.genre,
        "tags": list(dict.fromkeys([*request.tags, *metadata.get("youtube_tags", [])[:8]])),
        "review": review,
        "description": " ".join(part for part in description_parts if part).strip(),
        "youtube_url": request.youtube_url,
        "youtube_video_id": video_id,
        **metadata,
    }

    if existing:
        songs = [song if item.get("id") == existing.get("id") else item for item in songs]
    else:
        songs.append(song)
    save_songs(songs)
    return song


def error_response(error: Exception) -> dict[str, str]:
    return {"error": str(error)}


@app.get("/api/songs")
def api_songs() -> dict[str, Any]:
    try:
        ensure_sample_data()
        if UMAP_PATH.exists():
            songs = read_json(UMAP_PATH, [])
        else:
            songs = [{**song, "x": None, "y": None} for song in load_songs()]
        return {"songs": [public_song(song) for song in songs]}
    except Exception as exc:
        return error_response(exc)


@app.post("/api/build-index")
def api_build_index() -> dict[str, Any]:
    try:
        songs = build_index()
        return {"songs": [public_song(song) for song in songs]}
    except Exception as exc:
        return error_response(exc)


@app.post("/api/import-youtube")
def api_import_youtube(request: YouTubeImportRequest) -> dict[str, Any]:
    try:
        song = import_youtube_song(request)
        songs = build_index() if request.rebuild_index else load_songs()
        indexed = next((item for item in songs if item.get("id") == song.get("id")), song)
        return {"song": public_song(indexed), "songs": [public_song(item) for item in songs]}
    except Exception as exc:
        return error_response(exc)


@app.post("/api/search")
def api_search(request: SearchRequest) -> dict[str, Any]:
    try:
        query_embedding = encode_texts([f"query: {request.query}"])[0]
        return {"results": rank_by_vector(query_embedding, request.top_k)}
    except Exception as exc:
        return error_response(exc)


@app.get("/api/similar/{song_id}")
def api_similar(song_id: str) -> dict[str, Any]:
    try:
        songs = load_index()
        selected = next((song for song in songs if song.get("id") == song_id), None)
        if selected is None:
            return {"error": f"Song not found: {song_id}"}
        vector = np.asarray(selected["embedding"], dtype=np.float32)
        return {"results": rank_by_vector(vector, 5, exclude_id=song_id)}
    except Exception as exc:
        return error_response(exc)


@app.post("/api/user-profile")
def api_user_profile(request: UserProfileRequest) -> dict[str, Any]:
    try:
        if cosine_similarity is None:
            raise RuntimeError("scikit-learn is not installed.")
        songs = load_index()
        liked = [song for song in songs if song.get("id") in set(request.liked_song_ids)]
        if not liked:
            return {"error": "No liked songs matched the current index."}
        vectors = np.asarray([song["embedding"] for song in liked], dtype=np.float32)
        profile = vectors.mean(axis=0)
        exclude = {song.get("id") for song in liked}
        candidates = [song for song in songs if song.get("id") not in exclude]
        if not candidates:
            return {"results": []}
        matrix = np.asarray([song["embedding"] for song in candidates], dtype=np.float32)
        scores = cosine_similarity(profile.reshape(1, -1), matrix)[0]
        ranked_indices = np.argsort(scores)[::-1][: request.top_k]
        umap_by_id = {song.get("id"): song for song in read_json(UMAP_PATH, [])}
        results = []
        for index in ranked_indices:
            song = {**candidates[int(index)]}
            song.update(umap_by_id.get(song.get("id"), {}))
            song["similarity"] = float(scores[int(index)])
            results.append(public_song(song))
        return {"results": results}
    except Exception as exc:
        return error_response(exc)


if __name__ == "__main__":
    import uvicorn

    ensure_sample_data()
    print("Backend running at http://127.0.0.1:8787")
    print("The first Build Index request may take a while because the model must download.")
    print("Set YOUTUBE_API_KEY to enable /api/import-youtube.")
    uvicorn.run("app:app", host="127.0.0.1", port=8787, reload=True)
