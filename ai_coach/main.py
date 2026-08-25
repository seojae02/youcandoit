"""AI 면접 코치 서버.

[사후 배치 레이어]  POST /api/analyze-video   녹화 영상 → 4가지 지표
                    POST /api/final-feedback  답변 + 지표 → 코칭 리포트
[실시간 레이어]     GET  /video_feed          MJPEG 스트림 (분석 동시 수행)
                    GET  /api/analysis        현재까지의 실시간 지표
"""

import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import state_manager
from gemini_processor import (
    GeminiUnavailable,
    generate_questions_from_resume,
    get_comprehensive_feedback,
)
from video_analyzer import (
    analyze_video_file,
    cleanup_camera,
    generate_frames,
    reset_realtime,
)

# 음성 분석은 pyaudio·whisper 설치가 까다로워 선택 기능으로 둔다.
# 없어도 서버는 정상적으로 뜨고, 녹화 분석은 그대로 동작한다.
try:
    from audio_analyzer import AudioUnavailable, get_analyzer
    AUDIO_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    get_analyzer = None
    AudioUnavailable = RuntimeError
    AUDIO_IMPORT_ERROR = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    cleanup_camera()
    if get_analyzer is not None:
        try:
            get_analyzer().stop()
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(title="YouCanDoIt — AI 면접 코치", lifespan=lifespan)

# 와일드카드 출처와 allow_credentials=True 는 브라우저가 거부하는 조합이다.
# 프론트 주소가 정해지면 ALLOWED_ORIGINS 에 넣고 쿠키 인증을 켤 수 있다.
_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()] or ["*"],
    allow_credentials=bool(_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 요청/응답 모델
# ============================================================

class ResumeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    resume_content: str = Field(..., alias="자소서")


class QAPair(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    question: str = Field(..., alias="예상질문")
    answer: str = Field(..., alias="대답")


class VideoAnalysisResult(BaseModel):
    shoulder_tilt_count: int
    gaze_off_center_count: int
    average_smile_score: float
    average_blink_count: float


class FinalFeedbackRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    qa_list: List[QAPair] = Field(..., alias="질문&대답 리스트")
    video_result: VideoAnalysisResult = Field(..., alias="비디오_분석_결과")


# ============================================================
# [사후 배치] 녹화 영상 분석
# ============================================================

@app.post("/api/analyze-video")
def api_analyze_video(file: UploadFile = File(...)):
    """영상 파일 → 4가지 행동 지표.

    async 를 쓰지 않는 것이 의도적이다. 프레임마다 두 개의 모델을 돌리는
    CPU 작업이라, async 핸들러 안에서 그냥 호출하면 이벤트 루프가 통째로
    막혀 그동안 다른 요청이 전부 대기한다. 동기 함수로 두면 FastAPI 가
    스레드풀에서 실행해준다.
    """
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    temp_path = Path(tempfile.gettempdir()) / f"youcandoit_{uuid.uuid4()}{suffix}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        stride = int(os.getenv("BATCH_FRAME_STRIDE", "1"))
        return analyze_video_file(temp_path, frame_stride=stride)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


# ============================================================
# [LLM] 질문 생성 / 종합 피드백
# ============================================================

@app.post("/api/generate-questions")
def api_generate_questions(request: ResumeRequest):
    try:
        return {"questions": generate_questions_from_resume(request.resume_content)}
    except GeminiUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/final-feedback")
def api_final_feedback(request: FinalFeedbackRequest):
    qa_list = [qa.model_dump(by_alias=True) for qa in request.qa_list]
    try:
        feedback = get_comprehensive_feedback(qa_list, request.video_result.model_dump())
    except GeminiUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"final_feedback": feedback}


# ============================================================
# [실시간] 스트리밍 및 지표 조회
# ============================================================

@app.get("/video_feed")
async def video_feed():
    """MJPEG 스트림. 프레임을 내보내면서 동시에 지표를 계산한다."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/analysis")
def get_analysis_data():
    """실시간 레이어가 지금까지 계산한 값. 프론트가 주기적으로 폴링한다."""
    return state_manager.get_all_data()


@app.post("/api/session/reset")
def reset_session():
    """새 면접을 시작할 때 누적 지표를 비운다."""
    state_manager.reset()
    reset_realtime()
    return {"status": "reset"}


# ============================================================
# [실시간] 음성 분석 제어
# ============================================================

@app.post("/api/audio/start")
def start_audio():
    if get_analyzer is None:
        raise HTTPException(
            status_code=503,
            detail=f"음성 분석을 사용할 수 없습니다: {AUDIO_IMPORT_ERROR}",
        )
    try:
        return get_analyzer().start()
    except AudioUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/audio/stop")
def stop_audio():
    if get_analyzer is None:
        raise HTTPException(status_code=503, detail="음성 분석을 사용할 수 없습니다.")
    return get_analyzer().stop()


# ============================================================
# 상태 확인
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "audio_available": get_analyzer is not None,
        "audio_error": AUDIO_IMPORT_ERROR,
        "video_source": os.getenv("VIDEO_SOURCE", "0"),
        "gemini_key_set": bool(os.getenv("GEMINI_API_KEY")),
    }


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
def read_root():
    """데모 콘솔. 스트림과 실시간 지표를 눈으로 확인하기 위한 페이지다.

    실제 서비스 화면이 아니라, 프론트엔드가 붙기 전까지 서버 동작을
    검증하고 API 사용법을 보여주는 용도다.
    """
    demo = STATIC_DIR / "demo.html"
    if demo.exists():
        return demo.read_text(encoding="utf-8")
    return (
        "<h1>YouCanDoIt — AI 면접 코치</h1>"
        "<p>API 문서: <a href='/docs'>/docs</a></p>"
        "<p>상태 확인: <a href='/api/health'>/api/health</a></p>"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
