# SatQuery AI — Backend

FastAPI backend for the SatQuery AI satellite and remote-sensing image analysis platform.

---

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Foundation: FastAPI, MongoDB, health check |
| 2 | ✅ Complete | Image Management System: upload, validation, storage, metadata |
| 3 | ✅ Complete | Session System: sessions, image linking, questions, answers, history |
| 4 | ✅ Complete | Real AI Analysis: Gemini Vision API, analyses collection, session integration |
| 5+ | 🔜 Planned | Comparison analysis, multi-intent, spectral models, authentication |

---

## Phase 2 — Image Management System

Phase 2 adds the complete image upload and metadata pipeline:

```
Frontend
  ↓
Upload Image (multipart/form-data)
  ↓
FastAPI /api/images/upload
  ↓
File Validation (extension → MIME → size → content)
  ↓
Image Metadata Extraction
  ↓
Optional GeoTIFF Metadata (CRS, bands, bounds, resolution)
  ↓
Generate image_id (ULID-based, collision-safe)
  ↓
Store Original (uploads/original/)
  ↓
Create Processed Image (uploads/processed/)
  ↓
Create Thumbnail (uploads/thumbnails/, max 512×512)
  ↓
Save Metadata in MongoDB (satquery.images)
  ↓
Return Structured JSON Response
```

---

## Supported Image Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| JPEG | `.jpg`, `.jpeg` | Width, height, format, MIME, size |
| PNG | `.png` | Width, height, format, MIME, size |
| TIFF | `.tif`, `.tiff` | Width, height, format, MIME, size |
| GeoTIFF | `.tif`, `.tiff` | All TIFF fields + CRS, projection, bands, bounds, resolution, NoData |

> **Important:** Not every TIFF is a GeoTIFF. The backend checks for actual geospatial tags.
> No satellite metadata is ever fabricated.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Runtime environment |
| `API_PREFIX` | `/api` | URL prefix for all API routes |
| `MONGODB_URI` | _(required)_ | MongoDB connection string |
| `DATABASE_NAME` | `satquery` | MongoDB database name |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated allowed origins |
| `UPLOAD_DIR` | `uploads` | Root directory for stored images |
| `MAX_FILE_SIZE_MB` | `20` | Maximum upload size in megabytes |
| `THUMBNAIL_SIZE` | `512` | Max thumbnail dimension (px) |

---

## Installation

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Run

```bash
uvicorn app.main:app --reload
```

The API starts at: **http://127.0.0.1:8000**

Interactive docs: **http://127.0.0.1:8000/docs**

---

## API Endpoints

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Service health + MongoDB status |

### Images (Phase 2)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/images/upload` | Upload a satellite/remote-sensing image |
| GET | `/api/images/{image_id}` | Get image metadata by ID |
| DELETE | `/api/images/{image_id}` | Delete image and all stored files |

---

## Upload Example

```bash
curl -X POST http://127.0.0.1:8000/api/images/upload \
  -F "file=@satellite_image.jpg"
```

**Example response:**

```json
{
  "image_id": "img_01JABC123DEFGHIJKLMNO",
  "filename": "satellite_image.jpg",
  "format": "JPEG",
  "mime_type": "image/jpeg",
  "width": 1080,
  "height": 939,
  "size": 361532,
  "status": "ready",
  "is_geospatial": false,
  "geospatial": null,
  "created_at": "2026-08-28T05:30:00Z"
}
```

**GeoTIFF response example:**

```json
{
  "image_id": "img_01JABC...",
  "filename": "sentinel2.tif",
  "format": "TIFF",
  "is_geospatial": true,
  "geospatial": {
    "is_geospatial": true,
    "crs": "EPSG:32643",
    "width": 4096,
    "height": 4096,
    "bands": 4,
    "resolution": [10.0, 10.0],
    "bounds": { "left": 72.1, "bottom": 23.1, "right": 72.2, "top": 23.2 },
    "nodata": null
  }
}
```

---

## File Size Limit

Default: **20 MB** (configurable via `MAX_FILE_SIZE_MB`).

Files larger than the limit return HTTP 413.

---

## Storage Structure

```
backend/
└── uploads/
    ├── original/           ← Original uploaded file (never modified)
    │   └── img_<ULID>.jpg
    ├── processed/          ← Orientation-corrected / format-safe copy
    │   └── img_<ULID>.jpg
    └── thumbnails/         ← JPEG thumbnail, max 512×512, aspect-ratio preserved
        └── img_<ULID>.jpg
```

> `uploads/` is in `.gitignore` — uploaded images are not tracked in git.

---

## MongoDB — Image Collection

Collection: `satquery.images`

Unique index on: `image_id`

**Document structure:**

```json
{
  "_id": "...",
  "image_id": "img_01JABC...",
  "filename": "satellite.jpg",
  "stored_filename": "img_01JABC.jpg",
  "mime_type": "image/jpeg",
  "format": "JPEG",
  "size": 361532,
  "width": 1080,
  "height": 939,
  "status": "ready",
  "is_geospatial": false,
  "geospatial": null,
  "storage": {
    "original": "...",
    "processed": "...",
    "thumbnail": "..."
  },
  "created_at": "2026-08-28T05:30:00Z"
}
```

---

## Tests

```bash
# Install pytest if not already installed
pip install pytest httpx

# Run all tests
pytest tests/ -v
```

---

## Phase 3 — Session System

Phase 3 adds the session system linking images to conversation history:

```
Session
  ├── image_ids[]          ← uploaded images linked to this session
  └── questions[]          ← each question + answer pair
        └── analysis_id    ← Phase 4: references analyses collection
```

### Session Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/sessions` | Create a new session |
| `GET` | `/api/sessions` | List sessions (paginated) |
| `GET` | `/api/sessions/{id}` | Get a session |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `POST` | `/api/sessions/{id}/images/{img_id}` | Link image to session |
| `GET` | `/api/sessions/{id}/images` | List session images |
| `POST` | `/api/sessions/{id}/questions` | Add a question |
| `GET` | `/api/sessions/{id}/questions` | List questions |
| `PATCH` | `/api/sessions/{id}/questions/{q_id}` | Update question/answer |

---

## Phase 4 — Real AI Analysis

Phase 4 adds a full vision AI analysis pipeline:

```
Frontend
   ↓  POST /api/analysis { image_id, query, session_id? }
FastAPI
   ↓
analysis_service.py
   ↓ validates image + session
   ↓ creates analyses document (status=processing)
   ↓ resolves image path
   ↓ builds prompt (ai/prompts.py)
   ↓
GeminiProvider  ←  google-genai SDK
   ↓ sends image bytes + prompt
   ↓ receives text response
   ↓
response_parser.py  →  ParsedAnalysisResult
   ↓ updates analyses document (status=completed/failed)
   ↓ updates session question
   ↓
MongoDB  →  Frontend
```

### Analysis Endpoint

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analysis` | Run AI analysis on an uploaded image |

**Request body:**
```json
{
  "image_id": "img_01ABC...",
  "query": "Identify buildings in this image.",
  "session_id": "ses_01XYZ..."  // optional
}
```

**Response:**
```json
{
  "analysis_id": "analysis_01DEF...",
  "image_id": "img_01ABC...",
  "session_id": "ses_01XYZ...",
  "query": "Identify buildings in this image.",
  "analysis_type": "general",
  "status": "completed",
  "answer": "The image contains several visible building structures...",
  "findings": [],
  "confidence": null,
  "confidence_available": false,
  "provider": "gemini",
  "model": "gemini-1.5-flash",
  "created_at": "2026-08-28T17:00:00Z",
  "completed_at": "2026-08-28T17:00:03Z"
}
```

> **Note:** `confidence` is always `null` and `confidence_available` is always `false`.
> Gemini does not return calibrated numerical confidence values.

### AI Provider Configuration

Add these variables to `Backend/.env`:

```env
# Phase 4 — AI Provider
AI_PROVIDER=gemini
AI_API_KEY=          ← add your Gemini API key here
AI_MODEL=gemini-1.5-flash
AI_TIMEOUT_SECONDS=60
```

Get a Gemini API key at: https://aistudio.google.com/app/apikey

The application **starts without a key** but returns HTTP 503 on analysis requests:
```json
{ "detail": "AI provider is not configured. Please set AI_API_KEY in your .env file." }
```

### Adding a New AI Provider

1. Create `app/ai/myprovider_provider.py` implementing `BaseAIProvider`.
2. Register it in `app/ai/provider.py` `_PROVIDER_REGISTRY`.
3. Set `AI_PROVIDER=myprovider` in `.env`.

### MongoDB — Analyses Collection

Each analysis is stored as:
```json
{
  "analysis_id": "analysis_01...",
  "session_id": "ses_01...",
  "image_id": "img_01...",
  "query": "...",
  "analysis_type": "general",
  "status": "completed",
  "answer": "...",
  "findings": [],
  "confidence": null,
  "confidence_available": false,
  "provider": "gemini",
  "model": "gemini-1.5-flash",
  "created_at": "...",
  "completed_at": "..."
}
```

Indexes: `analysis_id` (unique), `session_id`, `image_id`, `created_at DESC`.

---

## Security Notes

- Extensions and MIME types are validated server-side
- File content is verified with Pillow (rejects fake/corrupt images)
- Decompression-bomb protection active (max 256 megapixels)
- Path traversal prevented — uploaded filenames never used as filesystem paths
- Server-generated filenames based on image_id
- `.env` is gitignored — never committed
- Python stack traces are never returned to the client
- **`AI_API_KEY` is server-side only — never sent to the frontend, never logged, never stored in MongoDB**
"# Set_Qeuey_Backend" 
