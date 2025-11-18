from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
import os

# 💡 모듈 임포트
# 주의: 같은 폴더 내에 video_analyzer.py, state_manager.py, gemini_processor.py가 있어야 합니다.
from video_analyzer import generate_frames, cleanup_camera, analyze_video_file
from state_manager import get_all_data
from gemini_processor import generate_questions_from_resume, get_feedback_on_answer

app = FastAPI()

# --- CORS 설정 (프론트엔드 연동을 위해 필수) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 프론트엔드 주소(예: http://localhost:3000)로 변경 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 모델 (데이터 검증용) ---
class ResumeInput(BaseModel):
    resume_text: str

class FeedbackInput(BaseModel):
    question: str
    answer: str

# ==========================================
# 1. [핵심 기능] 녹화 파일 분석 API
# ==========================================
@app.post("/api/analyze-recording")
async def analyze_recording(file: UploadFile = File(...)):
    """
    프론트엔드에서 녹화된 영상 파일(blob/mp4 등)을 받아 분석 결과를 반환합니다.
    """
    # 1. 받은 파일을 서버에 임시 저장
    temp_filename = f"temp_{file.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. video_analyzer 모듈의 파일 분석 함수 호출
        # (전체 영상을 처음부터 끝까지 분석하여 통계를 냄)
        result = analyze_video_file(temp_filename)
        
        # 3. 분석 결과 JSON 반환
        return {
            "status": "success",
            "data": result 
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
        
    finally:
        # 4. 분석이 끝나면 임시 파일 삭제 (서버 용량 확보)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# ==========================================
# 2. [핵심 기능] Gemini AI API
# ==========================================
@app.post("/api/generate-questions")
async def api_generate_questions(input_data: ResumeInput):
    """
    자소서를 받아서 면접 질문 리스트를 생성합니다.
    """
    questions = generate_questions_from_resume(input_data.resume_text)
    return {"questions": questions}

@app.post("/api/get-feedback")
async def api_get_feedback(input_data: FeedbackInput):
    """
    면접 질문과 답변을 받아서 피드백을 생성합니다.
    """
    feedback = get_feedback_on_answer(input_data.question, input_data.answer)
    return {"feedback": feedback}

# ==========================================
# 3. [시연용] 실시간 스트리밍 API (선택 사항)
# ==========================================
@app.get("/video_feed")
async def video_feed():
    # 백엔드 컴퓨터의 웹캠을 켜서 실시간으로 보여줌 (시연용)
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/analysis")
async def get_analysis_data():
    # 실시간 분석 데이터를 반환 (시연용)
    return get_all_data()

# ==========================================
# 4. 메인 페이지 (서버 상태 확인용)
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>AI 면접 코치 서버</title>
            <style>
                body { font-family: sans-serif; text-align: center; padding-top: 50px; }
                h1 { color: #007BFF; }
                .box { border: 1px solid #ddd; padding: 20px; display: inline-block; border-radius: 10px; }
            </style>
        </head>
        <body>
            <div class="box">
                <h1>✅ AI 면접 코치 서버 가동 중</h1>
                <p>프론트엔드 요청을 기다리고 있습니다.</p>
                <br>
                <p>📹 <a href="/video_feed">실시간 웹캠 테스트 (백엔드 캠)</a></p>
                <p>📄 <a href="/docs">API 문서 보기 (Swagger)</a></p>
            </div>
        </body>
    </html>
    """

# --- 서버 종료 이벤트 ---
@app.on_event("shutdown")
def on_shutdown():
    print("서버 종료: 카메라 자원을 해제합니다.")
    cleanup_camera()

if __name__ == "__main__":
    # 로컬 실행: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)