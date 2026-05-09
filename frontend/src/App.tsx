import { MouseEvent, useEffect, useMemo, useState } from "react";
import { Loader2, Map, Music2, Search, Sparkles } from "lucide-react";

const API_BASE = "http://127.0.0.1:8787";
const isStaticPage =
  window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1";

type Song = {
  id: string;
  title: string;
  artist: string;
  genre: string;
  tags: string[];
  review: string;
  description: string;
  youtube_url?: string;
  youtube_video_id?: string;
  youtube_comments?: string[];
  x: number | null;
  y: number | null;
  similarity?: number;
};

type Point = Song & {
  sx: number;
  sy: number;
};

const palette = [
  "#2563eb",
  "#dc2626",
  "#16a34a",
  "#9333ea",
  "#ea580c",
  "#0891b2",
  "#be123c",
  "#4f46e5",
  "#65a30d",
  "#c026d3",
  "#0f766e",
  "#b45309"
];

function colorForGenre(genre: string, genres: string[]) {
  const index = Math.max(0, genres.indexOf(genre));
  return palette[index % palette.length];
}

function formatScore(score?: number) {
  return typeof score === "number" ? `${(score * 100).toFixed(1)}%` : "";
}

async function readApi<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  const data = await response.json();
  if (data.error) {
    throw new Error(data.error);
  }
  return data;
}

async function readStaticSongs(): Promise<Song[]> {
  const response = await fetch(`${import.meta.env.BASE_URL}sample_songs_umap.json`);
  const data = await response.json();
  return data.songs;
}

function songText(song: Song) {
  return [
    song.title,
    song.artist,
    song.genre,
    song.tags.join(" "),
    song.description,
    song.review,
    ...(song.youtube_comments ?? [])
  ]
    .join(" ")
    .toLowerCase();
}

function tokens(text: string) {
  return text
    .toLowerCase()
    .split(/[^a-z0-9가-힣]+/i)
    .filter((token) => token.length > 1);
}

function localRank(queryText: string, candidates: Song[], topK = 5, excludeId?: string) {
  const queryTokens = tokens(queryText);
  const querySet = new Set(queryTokens);
  return candidates
    .filter((song) => song.id !== excludeId)
    .map((song) => {
      const textTokens = tokens(songText(song));
      const textSet = new Set(textTokens);
      const overlap = [...querySet].filter((token) => textSet.has(token)).length;
      const tagBoost = song.tags.filter((tag) => querySet.has(tag.toLowerCase())).length * 0.25;
      const score = querySet.size ? (overlap + tagBoost) / Math.sqrt(querySet.size * Math.max(1, textSet.size)) : 0;
      return { ...song, similarity: Math.min(0.99, score) };
    })
    .sort((a, b) => (b.similarity ?? 0) - (a.similarity ?? 0))
    .slice(0, topK);
}

export default function App() {
  const [songs, setSongs] = useState<Song[]>([]);
  const [selectedSong, setSelectedSong] = useState<Song | null>(null);
  const [hoveredSong, setHoveredSong] = useState<Song | null>(null);
  const [query, setQuery] = useState("");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [searchResults, setSearchResults] = useState<Song[]>([]);
  const [similarSongs, setSimilarSongs] = useState<Song[]>([]);
  const [profileResults, setProfileResults] = useState<Song[]>([]);
  const [likedSongIds, setLikedSongIds] = useState<string[]>([]);
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");
  const [demoMode, setDemoMode] = useState(isStaticPage);

  useEffect(() => {
    loadSongs();
  }, []);

  const genres = useMemo(
    () => Array.from(new Set(songs.map((song) => song.genre))).sort(),
    [songs]
  );

  const highlightedIds = useMemo(
    () => new Set(searchResults.map((song) => song.id)),
    [searchResults]
  );

  const points = useMemo<Point[]>(() => {
    const drawable = songs.filter(
      (song): song is Song & { x: number; y: number } =>
        typeof song.x === "number" && typeof song.y === "number"
    );
    if (!drawable.length) {
      return [];
    }

    const xs = drawable.map((song) => song.x);
    const ys = drawable.map((song) => song.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const width = 760;
    const height = 520;
    const pad = 48;

    return drawable.map((song) => {
      const xRatio = maxX === minX ? 0.5 : (song.x - minX) / (maxX - minX);
      const yRatio = maxY === minY ? 0.5 : (song.y - minY) / (maxY - minY);
      return {
        ...song,
        sx: pad + xRatio * (width - pad * 2),
        sy: height - pad - yRatio * (height - pad * 2)
      };
    });
  }, [songs]);

  async function loadSongs() {
    try {
      setError("");
      if (isStaticPage) {
        const staticSongs = await readStaticSongs();
        setSongs(staticSongs);
        setSelectedSong(staticSongs[0] ?? null);
        return;
      }
      const data = await readApi<{ songs: Song[] }>("/api/songs");
      setSongs(data.songs);
      setSelectedSong((current) => current ?? data.songs[0] ?? null);
    } catch (err) {
      const staticSongs = await readStaticSongs();
      setDemoMode(true);
      setSongs(staticSongs);
      setSelectedSong(staticSongs[0] ?? null);
      setError("Backend is not running, so the static GitHub Pages demo data is loaded.");
    }
  }

  async function buildIndex() {
    try {
      setLoading("build");
      setError("");
      setSearchResults([]);
      setSimilarSongs([]);
      setProfileResults([]);
      if (demoMode) {
        const staticSongs = await readStaticSongs();
        setSongs(staticSongs);
        setSelectedSong(staticSongs[0] ?? null);
        return;
      }
      const data = await readApi<{ songs: Song[] }>("/api/build-index", { method: "POST" });
      setSongs(data.songs);
      setSelectedSong(data.songs[0] ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to build index.");
    } finally {
      setLoading("");
    }
  }

  async function importYoutube() {
    if (!youtubeUrl.trim()) {
      return;
    }
    try {
      setLoading("youtube");
      setError("");
      if (demoMode) {
        setError("YouTube import needs the local FastAPI backend and YOUTUBE_API_KEY. GitHub Pages shows a static demo.");
        return;
      }
      const data = await readApi<{ song: Song; songs: Song[] }>("/api/import-youtube", {
        method: "POST",
        body: JSON.stringify({
          youtube_url: youtubeUrl,
          genre: "YouTube",
          tags: ["youtube", "comments", "imported"],
          max_comments: 20,
          rebuild_index: true
        })
      });
      setSongs(data.songs);
      setSelectedSong(data.song);
      setYoutubeUrl("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to import YouTube data.");
    } finally {
      setLoading("");
    }
  }

  async function searchSongs() {
    if (!query.trim()) {
      return;
    }
    try {
      setLoading("search");
      setError("");
      if (demoMode) {
        setSearchResults(localRank(query, songs, 5));
        return;
      }
      const data = await readApi<{ results: Song[] }>("/api/search", {
        method: "POST",
        body: JSON.stringify({ query, top_k: 5 })
      });
      setSearchResults(data.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed.");
    } finally {
      setLoading("");
    }
  }

  async function findSimilar(songId: string) {
    try {
      setLoading("similar");
      setError("");
      if (demoMode) {
        const selected = songs.find((song) => song.id === songId);
        if (selected) {
          setSimilarSongs(localRank(songText(selected), songs, 5, songId));
        }
        return;
      }
      const data = await readApi<{ results: Song[] }>(`/api/similar/${songId}`);
      setSimilarSongs(data.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to find similar songs.");
    } finally {
      setLoading("");
    }
  }

  async function buildUserProfile() {
    if (!likedSongIds.length) {
      setError("Choose at least one liked song.");
      return;
    }
    try {
      setLoading("profile");
      setError("");
      if (demoMode) {
        const likedSongs = songs.filter((song) => likedSongIds.includes(song.id));
        const profileText = likedSongs.map(songText).join(" ");
        setProfileResults(localRank(profileText, songs.filter((song) => !likedSongIds.includes(song.id)), 5));
        return;
      }
      const data = await readApi<{ results: Song[] }>("/api/user-profile", {
        method: "POST",
        body: JSON.stringify({ liked_song_ids: likedSongIds, top_k: 5 })
      });
      setProfileResults(data.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Taste recommendation failed.");
    } finally {
      setLoading("");
    }
  }

  function toggleLiked(songId: string) {
    setLikedSongIds((current) => {
      if (current.includes(songId)) {
        return current.filter((id) => id !== songId);
      }
      if (current.length >= 5) {
        return current;
      }
      return [...current, songId];
    });
  }

  function selectPoint(song: Song, event?: MouseEvent<SVGCircleElement>) {
    event?.stopPropagation();
    setSelectedSong(song);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">
            <Map size={16} />
            Text + Comments Embedding
          </div>
          <h1>Music Taste Embedding Map</h1>
          <p>Embed song descriptions, reviews, and YouTube top comments into a 2D taste map.</p>
          {demoMode && <p className="demo-note">Static GitHub Pages demo mode. Local backend unlocks real embeddings and YouTube import.</p>}
        </div>
        <button className="primary-button" onClick={buildIndex} disabled={loading === "build"}>
          {loading === "build" ? <Loader2 className="spin" size={18} /> : <Sparkles size={18} />}
          Build Index
        </button>
      </header>

      {error && <div className="error-card">{error}</div>}

      <section className="layout">
        <div className="map-column">
          <div className="toolbar-card stacked-tools">
            <div className="search-row">
              <Search size={18} />
              <input
                value={query}
                placeholder="Example: dreamy K-pop for late-night listening"
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") searchSongs();
                }}
              />
              <button onClick={searchSongs} disabled={loading === "search"}>
                {loading === "search" ? "Searching" : "Search"}
              </button>
            </div>
            <div className="search-row">
              <Music2 size={18} />
              <input
                value={youtubeUrl}
                placeholder="Paste a YouTube or YouTube Music link"
                onChange={(event) => setYoutubeUrl(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") importYoutube();
                }}
              />
              <button onClick={importYoutube} disabled={loading === "youtube"}>
                {loading === "youtube" ? "Importing" : "Import"}
              </button>
            </div>
          </div>

          <div className="map-card">
            <div className="map-header">
              <div>
                <h2>UMAP Scatter Plot</h2>
                <span>{points.length ? `${points.length} songs indexed` : "Build Index to create coordinates"}</span>
              </div>
              <div className="legend">
                {genres.map((genre) => (
                  <span key={genre}>
                    <i style={{ backgroundColor: colorForGenre(genre, genres) }} />
                    {genre}
                  </span>
                ))}
              </div>
            </div>

            <div className="svg-wrap">
              <svg viewBox="0 0 760 520" role="img" aria-label="Music embedding scatter plot">
                <rect x="0" y="0" width="760" height="520" rx="8" />
                {points.map((point) => {
                  const isSelected = selectedSong?.id === point.id;
                  const isHighlighted = highlightedIds.has(point.id);
                  return (
                    <circle
                      key={point.id}
                      cx={point.sx}
                      cy={point.sy}
                      r={isSelected ? 9 : isHighlighted ? 8 : 6}
                      fill={colorForGenre(point.genre, genres)}
                      stroke={isSelected ? "#111827" : isHighlighted ? "#facc15" : "#ffffff"}
                      strokeWidth={isSelected || isHighlighted ? 4 : 2}
                      onClick={(event) => selectPoint(point, event)}
                      onMouseEnter={() => setHoveredSong(point)}
                      onMouseLeave={() => setHoveredSong(null)}
                    />
                  );
                })}
                {!points.length && (
                  <text x="380" y="260" textAnchor="middle">
                    Build Index to display the map.
                  </text>
                )}
              </svg>
              {hoveredSong && (
                <div className="tooltip">
                  <strong>{hoveredSong.title}</strong>
                  <span>{hoveredSong.artist}</span>
                </div>
              )}
            </div>
          </div>

          <div className="profile-card">
            <div className="section-title">
              <Music2 size={18} />
              <h2>Taste Profile Test</h2>
            </div>
            <div className="check-grid">
              {songs.map((song) => (
                <label key={song.id} className={likedSongIds.includes(song.id) ? "checked" : ""}>
                  <input
                    type="checkbox"
                    checked={likedSongIds.includes(song.id)}
                    onChange={() => toggleLiked(song.id)}
                  />
                  <span>{song.title}</span>
                </label>
              ))}
            </div>
            <button className="secondary-button" onClick={buildUserProfile} disabled={loading === "profile"}>
              {loading === "profile" ? "Recommending" : "Find Songs Near My Taste"}
            </button>
          </div>
        </div>

        <aside className="side-column">
          <DetailPanel song={selectedSong} loading={loading === "similar"} onSimilar={findSimilar} />
          <ResultList title="Search Results" songs={searchResults} empty="Search results will appear here." onPick={setSelectedSong} />
          <ResultList title="Similar Songs" songs={similarSongs} empty="Pick a song, then find similar tracks." onPick={setSelectedSong} />
          <ResultList title="Taste Recommendations" songs={profileResults} empty="Choose liked songs to get recommendations." onPick={setSelectedSong} />
        </aside>
      </section>
    </main>
  );
}

function DetailPanel({
  song,
  loading,
  onSimilar
}: {
  song: Song | null;
  loading: boolean;
  onSimilar: (songId: string) => void;
}) {
  if (!song) {
    return (
      <section className="card detail-panel">
        <h2>Details</h2>
        <p className="muted">Select a point on the map.</p>
      </section>
    );
  }

  return (
    <section className="card detail-panel">
      <div className="detail-heading">
        <div>
          <h2>{song.title}</h2>
          <p>{song.artist}</p>
        </div>
        <span>{song.genre}</span>
      </div>
      <div className="tag-row">
        {song.tags.map((tag) => (
          <span key={tag}>{tag}</span>
        ))}
      </div>
      {song.youtube_url && (
        <a className="youtube-link" href={song.youtube_url} target="_blank" rel="noreferrer">
          Open YouTube source
        </a>
      )}
      <h3>Description</h3>
      <p>{song.description}</p>
      <h3>Review / Comment Summary</h3>
      <p>{song.review}</p>
      {!!song.youtube_comments?.length && (
        <>
          <h3>Imported Top Comments</h3>
          <ul className="comment-list">
            {song.youtube_comments.slice(0, 5).map((comment, index) => (
              <li key={`${song.id}-comment-${index}`}>{comment}</li>
            ))}
          </ul>
        </>
      )}
      <button className="secondary-button" onClick={() => onSimilar(song.id)} disabled={loading}>
        {loading ? "Finding" : "Find Similar Songs"}
      </button>
    </section>
  );
}

function ResultList({
  title,
  songs,
  empty,
  onPick
}: {
  title: string;
  songs: Song[];
  empty: string;
  onPick: (song: Song) => void;
}) {
  return (
    <section className="card result-list">
      <h2>{title}</h2>
      {!songs.length && <p className="muted">{empty}</p>}
      {songs.map((song) => (
        <button key={song.id} className="result-item" onClick={() => onPick(song)}>
          <strong>{song.title}</strong>
          <span>
            {song.artist} - {song.genre}
          </span>
          {typeof song.similarity === "number" && <em>{formatScore(song.similarity)}</em>}
        </button>
      ))}
    </section>
  );
}
