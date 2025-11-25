from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List
import uvicorn
import shutil
import os
import uuid

# 모듈 임포트
from video_analyzer import generate_frames, cleanup_camera, analyze_video_file
from state_manager import get_all_data
from gemini_processor import generate_questions_from_resume, get_comprehensive_feedback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 데이터 모델 정의 (보내주신 JSON과 100% 일치) ---

class ResumeRequest(BaseModel):
    resume_content: str = Field(..., alias="자소서")

class QAPair(BaseModel):
    # "예상질문": "..."
    question: str = Field(..., alias="예상질문") 
    # "대답": "..."
    answer: str = Field(..., alias="대답") 

class VideoAnalysisResult(BaseModel):
    # "비디오_분석_결과" 안의 키들과 일치
    shoulder_tilt_count: int
    gaze_off_center_count: int
    average_smile_score: float
    average_blink_count: float

class FinalFeedbackRequest(BaseModel):
    # 전체 JSON 구조 정의
    qa_list: List[QAPair] = Field(..., alias="질문&대답 리스트")
    video_result: VideoAnalysisResult = Field(..., alias="비디오_분석_결과")

# =========================================================
# [API 3] 종합 피드백 (JSON Body로 받음) - ★여기가 안되셨던 부분★
# =========================================================
@app.post("/api/final-feedback")
async def api_final_feedback(request: FinalFeedbackRequest):
    """
    Body (Raw JSON): 
    {
      "질문&대답 리스트": [...],
      "비디오_분석_결과": {...}
    }
    """
    # 1. Pydantic 모델을 딕셔너리로 변환 (한글 키 유지를 위해 by_alias=True)
    # 리스트 내부의 QAPair도 딕셔너리로 변환해야 함
    qa_list_dict = [qa.dict(by_alias=True) for qa in request.qa_list]
    video_data_dict = request.video_result.dict()

    # 2. Gemini에게 전달
    feedback = get_comprehensive_feedback(qa_list_dict, video_data_dict)
    
    return {"final_feedback": feedback}

# =========================================================
# [API 1] 비디오 분석 (파일 업로드)
# =========================================================
@app.post("/api/analyze-video")
async def api_analyze_video(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4()}.mp4"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = analyze_video_file(temp_filename)
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

# =========================================================
# [API 2] 자소서 질문 생성
# =========================================================
@app.post("/api/generate-questions")
async def api_generate_questions(request: ResumeRequest):
    questions = generate_questions_from_resume(request.resume_content)
    return {"questions": questions}

# =========================================================
# [기타] 실시간 스트리밍 & 메인 페이지
# =========================================================
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/analysis")
async def get_analysis_data():
    return get_all_data()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return "<h1>✅ AI 면접 코치 서버 (3-Step API 모드)</h1>"

@app.on_event("shutdown")
def on_shutdown():
    cleanup_camera()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)