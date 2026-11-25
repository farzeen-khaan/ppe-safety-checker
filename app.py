"""F6 — The web app: FastAPI and one HTML page. Vercel's entry point (`app` in app.py) and the local server.

Run locally:
    uvicorn app:app --reload          # then open http://127.0.0.1:8000 (API docs at /docs)

Two calls per photo, because a serverless function can't stream a second result and may be
frozen after it responds:
    POST /api/analyze                  photo -> verdicts, annotated JPEG, saved row, alert sent
    POST /api/analyses/{id}/summary    the written summary, saved over the standard one
Every request is independent: nothing is kept between requests except the loaded model.
"""

import base64
import hashlib
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src import storage
from src.discord_webhook import summary_note
from src.pipeline import PENDING, analyze, write_summary
from src.storage import StorageUnavailable
from src.verdicts import count_verdicts

MAX_UPLOAD_BYTES = 3 * 1024 * 1024  # well under Vercel's 4.5 MB request limit; the page sends about 0.3 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIDE = 1280  # the annotated JPEG returned; the page resizes uploads to this too
ANALYZE_LIMIT, SUMMARY_LIMIT = 10, 30  # per client per hour: a public demo, the model API and alerts cost
RATE_WINDOW = timedelta(hours=1)
RETRY_AFTER_DB_S = 5  # a Neon branch waking up usually answers within seconds
PAGE = Path(__file__).resolve().parent / "web" / "index.html"

app = FastAPI(
    title="PPE Safety Checker API",
    version="1.0.0",
    description="Checks every worker in a construction-site photo for a helmet and a high-visibility vest. "
    "The page at / calls these endpoints; summaries are automated and should be checked against the figures.",
)


# --- Response models ---

class Person(BaseModel):
    index: int = Field(description="1, 2, 3 … within the photo")
    status: str = Field(description="compliant, missing helmet, missing vest, missing helmet and vest, or too far to judge")
    confidence: float
    box: list[float] = Field(description="x1, y1, x2, y2 in the uploaded image's pixels")


class Counts(BaseModel):
    people: int
    compliant: int
    in_violation: int
    too_far: int


class AnalyzeResponse(BaseModel):
    id: int
    severity: str
    counts: Counts
    people: list[Person]
    annotated_jpeg: str = Field(description="data:image/jpeg;base64,… (never stored)")
    alert_status: str = Field(description="sent, failed, disabled (alerts off) or not_needed")
    processing_ms: int


class Summary(BaseModel):
    id: int
    summary: str
    recommended_action: str
    automated: bool = Field(description="True when written by the language model; False for the standard summary")
    note: str


class HistoryItem(BaseModel):
    id: int
    created_at: str = Field(description="UTC, e.g. 2026-11-25T13:00:00Z")
    image_name: str
    severity: str
    counts: Counts
    summary: str = Field(description="automated, standard or pending")
    alert_status: str


class AnalysisDetail(HistoryItem):
    summary_text: str
    recommended_action: str
    note: str
    people: list[Person]
    conf_threshold: float
    processing_ms: int


class Health(BaseModel):
    status: str
    database: str = Field(description="ok, or unreachable (Neon may be waking up)")


# --- Errors: one JSON shape, the right status code ---

class RateLimited(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after


@app.exception_handler(StorageUnavailable)
async def database_waking(request: Request, exc: StorageUnavailable) -> JSONResponse:
    return JSONResponse(
        {"detail": "The incident log is waking up or unreachable. Try again in a few seconds."},
        status_code=503, headers={"Retry-After": str(RETRY_AFTER_DB_S)},
    )


@app.exception_handler(RateLimited)
async def rate_limited(request: Request, exc: RateLimited) -> JSONResponse:
    minutes = max(1, round(exc.retry_after / 60))
    return JSONResponse(
        {"detail": f"Too many requests from your connection. Try again in about {minutes} min."},
        status_code=429, headers={"Retry-After": str(exc.retry_after)},
    )


# --- Helpers ---

def client_key(request: Request) -> str:
    """A salted hash of the client's IP: enough to rate-limit, never stored as the IP itself.

    Vercel puts the client's public IP in x-real-ip and x-forwarded-for. The salt is derived from
    DATABASE_URL (already a secret), so no extra setting is needed and the hashes can't be
    reversed by trying every IPv4 address without it.
    """
    ip = (request.headers.get("x-real-ip") or request.headers.get("x-forwarded-for", "").split(",")[0]
          or (request.client.host if request.client else "unknown")).strip()
    salt = hashlib.sha256(storage.database_url().encode()).hexdigest()
    return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()


def check_rate(bucket: str, limit: int, request: Request) -> None:
    retry_after = storage.rate_limit(bucket, client_key(request), limit, RATE_WINDOW)
    if retry_after is not None:
        raise RateLimited(retry_after)


def jpeg_data_url(image_bgr: np.ndarray) -> str:
    height, width = image_bgr.shape[:2]
    scale = MAX_IMAGE_SIDE / max(height, width)
    if scale < 1:
        image_bgr = cv2.resize(image_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise HTTPException(500, "Could not encode the annotated image.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


def people_out(rows) -> list[Person]:
    return [Person(index=i, status=v.status, confidence=v.person.confidence,
                   box=[v.person.x1, v.person.y1, v.person.x2, v.person.y2]) for i, v in enumerate(rows, 1)]


def summary_kind(row: dict) -> str:
    if row["report_source"] == "llm":
        return "automated"
    return "pending" if row["report_error"] == PENDING else "standard"


def counts_of(row: dict) -> Counts:
    return Counts(people=row["people_count"], compliant=row["compliant_count"],
                  in_violation=row["violation_count"], too_far=row["too_far_count"])


def history_item(row: dict) -> dict:
    return dict(id=row["id"], created_at=row["created_at"], image_name=row["image_name"], severity=row["severity"],
                counts=counts_of(row), summary=summary_kind(row), alert_status=row["alert_status"])


def safe_name(filename: str | None) -> str:
    name = Path(filename or "photo.jpg").name.strip() or "photo.jpg"
    return name[:200]


# --- Endpoints ---

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


@app.post("/api/analyze", response_model=AnalyzeResponse, status_code=201,
          responses={400: {}, 413: {}, 415: {}, 429: {}, 503: {}})
async def analyze_photo(request: Request, photo: UploadFile = File(description="JPG, PNG or WEBP, at most 3 MB")):
    """Detect workers, apply the rules, save the analysis and send the alert. Returns before the summary exists."""
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD_BYTES + 64 * 1024:  # multipart overhead
        raise HTTPException(413, f"The photo is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
                                 "Use the page, which resizes photos before uploading, or a smaller photo.")
    if photo.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Upload a JPG, PNG or WEBP photo.")
    data = await photo.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"The photo is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "That file is not an image this app can read. Upload a JPG, PNG or WEBP photo.")

    check_rate("analyze", ANALYZE_LIMIT, request)
    result = analyze(image, safe_name(photo.filename))
    if result.save_error is not None:  # the row is what the summary call needs
        raise StorageUnavailable(result.save_error)
    c = count_verdicts(result.verdicts)
    return AnalyzeResponse(
        id=result.analysis_id, severity=result.report.severity,
        counts=Counts(people=c["people"], compliant=c["compliant"], in_violation=c["violations"], too_far=c["too_far"]),
        people=people_out(result.verdicts), annotated_jpeg=jpeg_data_url(result.annotated),
        alert_status=result.alert_status, processing_ms=result.processing_ms,
    )


@app.post("/api/analyses/{analysis_id}/summary", response_model=Summary, responses={404: {}, 429: {}, 503: {}})
def summary(analysis_id: int, request: Request):
    """Write the summary for a saved analysis (automated, or standard if the model is unavailable) and save it."""
    check_rate("summary", SUMMARY_LIMIT, request)
    try:
        report = write_summary(analysis_id)
    except KeyError:
        raise HTTPException(404, f"No analysis with id {analysis_id}.") from None
    return Summary(id=analysis_id, summary=report.summary, recommended_action=report.recommended_action,
                   automated=report.automated, note=summary_note(report.automated))


@app.get("/api/history", response_model=list[HistoryItem], responses={503: {}})
def history(limit: int = Query(50, ge=1, le=100)):
    """Recent analyses, newest first. Text only: the log stores no photos."""
    return [history_item(r) for r in storage.list_analyses(limit=limit)]


@app.get("/api/analyses/{analysis_id}", response_model=AnalysisDetail, responses={404: {}, 503: {}})
def analysis(analysis_id: int):
    """One analysis with its per-person verdicts."""
    row = storage.get_analysis(analysis_id)
    if row is None:
        raise HTTPException(404, f"No analysis with id {analysis_id}.")
    people = [Person(index=p["person_index"], status=p["status"], confidence=p["confidence"],
                     box=[p["x1"], p["y1"], p["x2"], p["y2"]]) for p in row["people"]]
    return AnalysisDetail(
        **history_item(row), summary_text=row["report_summary"], recommended_action=row["report_action"],
        note=summary_note(row["report_source"] == "llm"), people=people,
        conf_threshold=row["conf_threshold"], processing_ms=row["processing_ms"],
    )


@app.get("/api/health", response_model=Health)
def health():
    """Liveness, and whether the database answers (it may take a few seconds to wake)."""
    try:
        storage.ping()
        database = "ok"
    except StorageUnavailable:
        database = "unreachable"
    return Health(status="ok", database=database)
