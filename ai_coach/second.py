# --- 1. 라이브러리 임포트 ---
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from PIL import ImageFont, ImageDraw, Image
from mediapipe.framework.formats import landmark_pb2
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn
import asyncio

# --- FastAPI 앱 인스턴스 생성 ---
app = FastAPI()

# --- 💡 1. 카메라를 전역 변수로 한번만 초기화 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 찾을 수 없거나 열 수 없습니다. 카메라 연결 상태를 확인해주세요.")
# ---------------------------------------------

# --- 2. 초기 설정 (MediaPipe 모델 로드) ---
pose_model_path = './models/pose_landmarker_lite.task'
face_model_path = './models/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.IMAGE)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.IMAGE)

pose_landmarker = PoseLandmarker.create_from_options(pose_options)
face_landmarker = FaceLandmarker.create_from_options(face_options)

# --- 3. 한글 폰트 설정 및 보조 함수 ---
FONT_PATH = 'C:/Windows/Fonts/malgun.ttf' 
try:
    font = ImageFont.truetype(FONT_PATH, 20)
except IOError:
    font = ImageFont.load_default()
# (draw_text_korean, calculate_tilt_angle, get_gaze_ratio 함수는 이전과 동일하게 유지)
def draw_text_korean(image, text, pos, font, color):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=(0,0,0), stroke_width=2)
    draw.text(pos, text, font=font, fill=color)
    return cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)

def calculate_tilt_angle(a, b):
    a = np.array(a); b = np.array(b)
    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.degrees(radians)
    return angle

def get_gaze_ratio(face_landmarks):
    try:
        r_eye_outer = face_landmarks[33]; r_eye_inner = face_landmarks[133]; r_iris = face_landmarks[468]
        l_eye_outer = face_landmarks[263]; l_eye_inner = face_landmarks[362]; l_iris = face_landmarks[473]
        if r_eye_inner is None or l_eye_inner is None: return 0.0

        r_eye_width = math.hypot(r_eye_inner.x - r_eye_outer.x, r_eye_inner.y - r_eye_outer.y)
        r_iris_to_center = math.hypot(r_iris.x - (r_eye_inner.x + r_eye_outer.x)/2, r_iris.y - (r_eye_inner.y + r_eye_outer.y)/2)
        l_eye_width = math.hypot(l_eye_inner.x - l_eye_outer.x, l_eye_inner.y - l_eye_outer.y)
        l_iris_to_center = math.hypot(l_iris.x - (l_eye_inner.x + l_eye_outer.x)/2, l_iris.y - (l_eye_inner.y + l_eye_outer.y)/2)
        if r_eye_width == 0 or l_eye_width == 0: return 0.0

        r_ratio = r_iris_to_center / r_eye_width; l_ratio = l_iris_to_center / l_eye_width
        gaze_score = 0
        if r_iris.x > (r_eye_inner.x + r_eye_outer.x)/2 : gaze_score += r_ratio * 2
        else: gaze_score -= r_ratio * 2
        if l_iris.x > (l_eye_inner.x + l_eye_outer.x)/2 : gaze_score += l_ratio * 2
        else: gaze_score -= l_ratio * 2
        return gaze_score
    except (ZeroDivisionError, IndexError, TypeError):
        return 0.0

# --- 4. 실시간 데이터 저장을 위한 글로벌 변수 ---
analysis_data = {
    "shoulder_angle": 0.0, "head_angle": 0.0,
    "gaze_status": "N/A", "gaze_score": 0.0,
    "blinks_per_minute": 0.0, "smile_score": 0.0,
    "shoulder_tilt_count": 0, "head_tilt_count": 0,
    "gaze_off_center_count": 0,
}
show_landmarks = True

# --- 5. 비디오 스트리밍 및 AI 분석 로직 ---
async def generate_frames():
    start_time = time.time()
    blink_count, shoulder_tilt_count, head_tilt_count, gaze_off_center_count = 0, 0, 0, 0
    is_blinking, is_shoulder_tilted, is_head_tilted, is_gaze_off_center = False, False, False, False
    
    global analysis_data, show_landmarks

    while True:
        try: # --- 💡 2. 루프 내에 강력한 에러 처리 추가 ---
            ret, frame = cap.read()
            if not ret:
                print("카메라에서 프레임을 읽을 수 없습니다.")
                break
            
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            pose_detection_result = pose_landmarker.detect(mp_image)
            face_detection_result = face_landmarker.detect(mp_image)

            shoulder_angle, head_angle = 180, 180
            gaze_score, smile_score = 0.0, 0.0
            
            if pose_detection_result.pose_landmarks:
                p_landmarks = pose_detection_result.pose_landmarks[0]
                shoulder_l = [p_landmarks[11].x, p_landmarks[11].y]; shoulder_r = [p_landmarks[12].x, p_landmarks[12].y]
                shoulder_angle = calculate_tilt_angle(shoulder_l, shoulder_r)
                ear_l = [p_landmarks[7].x, p_landmarks[7].y]; ear_r = [p_landmarks[8].x, p_landmarks[8].y]
                head_angle = calculate_tilt_angle(ear_l, ear_r)
            
            if face_detection_result.face_landmarks:
                face_landmarks = face_detection_result.face_landmarks[0]
                gaze_score = get_gaze_ratio(face_landmarks)
                
                blendshapes = face_detection_result.face_blendshapes[0]
                eye_blink_score = (next((c.score for c in blendshapes if c.category_name == 'eyeBlinkLeft'), 0) + next((c.score for c in blendshapes if c.category_name == 'eyeBlinkRight'), 0)) / 2
                if eye_blink_score > 0.7 and not is_blinking:
                    blink_count += 1
                    is_blinking = True
                elif eye_blink_score < 0.4:
                    is_blinking = False

                mouth_smile = (next((c.score for c in blendshapes if c.category_name == 'mouthSmileLeft'), 0) + next((c.score for c in blendshapes if c.category_name == 'mouthSmileRight'), 0)) / 2
                smile_score = mouth_smile * 100

            elapsed_time = time.time() - start_time
            blinks_per_minute = (blink_count / elapsed_time) * 60 if elapsed_time > 1 else 0

            # 이벤트 카운팅
            if abs(shoulder_angle) < 175:
                if not is_shoulder_tilted: shoulder_tilt_count += 1; is_shoulder_tilted = True
            else: is_shoulder_tilted = False

            if abs(head_angle) < 175:
                if not is_head_tilted: head_tilt_count += 1; is_head_tilted = True
            else: is_head_tilted = False

            if abs(gaze_score) > 0.5:
                if not is_gaze_off_center: gaze_off_center_count += 1; is_gaze_off_center = True
            else: is_gaze_off_center = False
            
            # 랜드마크 그리기
            if show_landmarks:
                if pose_detection_result.pose_landmarks:
                    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_detection_result.pose_landmarks[0]])
                    mp.solutions.drawing_utils.draw_landmarks(frame, pose_landmarks_proto, mp.solutions.pose.POSE_CONNECTIONS)
                if face_detection_result.face_landmarks:
                    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in face_detection_result.face_landmarks[0]])
                    mp.solutions.drawing_utils.draw_landmarks(image=frame, landmark_list=face_landmarks_proto, connections=mp.solutions.face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(200,200,200), thickness=1))
                    mp.solutions.drawing_utils.draw_landmarks(image=frame, landmark_list=face_landmarks_proto, connections=mp.solutions.face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,255), thickness=1))

            # 글로벌 변수 업데이트
            if abs(gaze_score) < 0.2: gaze_status = "정면"
            elif gaze_score <= -0.2: gaze_status = "왼쪽"
            else: gaze_status = "오른쪽"
            analysis_data = {
                "shoulder_angle": round(shoulder_angle, 2), "head_angle": round(head_angle, 2),
                "gaze_status": gaze_status, "gaze_score": round(gaze_score, 2),
                "blinks_per_minute": round(blinks_per_minute, 1), "smile_score": round(smile_score, 1),
                "shoulder_tilt_count": shoulder_tilt_count, "head_tilt_count": head_tilt_count,
                "gaze_off_center_count": gaze_off_center_count,
            }
            
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag: continue
            
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            await asyncio.sleep(0.01)
        
        except Exception as e:
            print(f"스트리밍 루프에서 에러 발생: {e}")
            # 에러가 발생해도 루프를 계속 진행하여 스트림이 끊기지 않도록 함
            continue

# --- 6. FastAPI 엔드포인트 설정 ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # 웹 페이지 UI (이전과 동일)
    html_content = """
    <html>
        <head><title>AI Interview Coach</title>
            <style>
                body { font-family: sans-serif; display: flex; padding: 20px; }
                #video-container { margin-right: 30px; }
                #data-container { border: 1px solid #ddd; padding: 20px; border-radius: 8px; background-color: #f9f9f9; width: 300px; }
                h1, h2 { color: #333; border-bottom: 2px solid #007BFF; padding-bottom: 10px; }
                p { font-size: 1.1em; margin: 12px 0; }
                span { font-weight: bold; color: #0056b3; padding: 2px 5px; background-color: #e9ecef; border-radius: 3px;}
            </style>
        </head>
        <body>
            <div id="video-container">
                <h1>AI Interview Coach - Live</h1>
                <img src="/video_feed" width="640" height="480" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            </div>
            <div id="data-container">
                <h2>실시간 분석 데이터</h2>
                <p>어깨 기울기: <span id="shoulder_angle">0.0</span></p>
                <p>고개 기울기: <span id="head_angle">0.0</span></p>
                <p>시선 방향: <span id="gaze_status">N/A</span> (<span id="gaze_score">0.0</span>)</p>
                <p>분당 눈깜빡임: <span id="blinks_per_minute">0.0</span></p>
                <p>스마일 지수: <span id="smile_score">0.0</span></p>
                <hr>
                <h2>이탈 횟수</h2>
                <p>자세 이탈 (어깨): <span id="shoulder_tilt_count">0</span></p>
                <p>자세 이탈 (고개): <span id="head_tilt_count">0</span></p>
                <p>시선 이탈: <span id="gaze_off_center_count">0</span></p>
            </div>
            <script>
                setInterval(function() {
                    fetch('/api/analysis')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('shoulder_angle').innerText = data.shoulder_angle;
                            document.getElementById('head_angle').innerText = data.head_angle;
                            document.getElementById('gaze_status').innerText = data.gaze_status;
                            document.getElementById('gaze_score').innerText = data.gaze_score;
                            document.getElementById('blinks_per_minute').innerText = data.blinks_per_minute;
                            document.getElementById('smile_score').innerText = data.smile_score;
                            document.getElementById('shoulder_tilt_count').innerText = data.shoulder_tilt_count;
                            document.getElementById('head_tilt_count').innerText = data.head_tilt_count;
                            document.getElementById('gaze_off_center_count').innerText = data.gaze_off_center_count;
                        });
                }, 500);
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/analysis")
async def get_analysis_data():
    return analysis_data

# --- 7. 서버 종료 시 카메라 해제 ---
@app.on_event("shutdown")
def shutdown_event():
    print("서버를 종료하며 카메라를 해제합니다.")
    cap.release()