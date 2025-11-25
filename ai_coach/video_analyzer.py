import cv2
import mediapipe as mp
import numpy as np
import time
import math
from mediapipe.framework.formats import landmark_pb2
import asyncio
from state_manager import update_video_data

# --- 1. MediaPipe 모델 초기화 ---
# (주의: 실행 위치에 ./models 폴더와 task 파일이 있어야 합니다)
pose_model_path = './models/pose_landmarker_lite.task'
face_model_path = './models/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 실시간/파일 분석 모두 IMAGE 모드(프레임 단위 처리) 사용
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.IMAGE)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.IMAGE)

pose_landmarker = PoseLandmarker.create_from_options(pose_options)
face_landmarker = FaceLandmarker.create_from_options(face_options)

# --- 2. 보조 계산 함수들 ---
def calculate_tilt_angle(a, b):
    """두 점(어깨/귀) 사이의 기울기 각도를 계산합니다."""
    a = np.array(a); b = np.array(b)
    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.degrees(radians)
    return angle

def get_gaze_ratio(face_landmarks):
    """눈 랜드마크를 기반으로 시선 비율(좌/우 치우침)을 계산합니다."""
    try:
        # 오른쪽 눈
        r_eye_outer = face_landmarks[33]; r_eye_inner = face_landmarks[133]; r_iris = face_landmarks[468]
        # 왼쪽 눈
        l_eye_outer = face_landmarks[263]; l_eye_inner = face_landmarks[362]; l_iris = face_landmarks[473]
        
        # 눈 길이 및 홍채 위치 계산
        r_eye_width = math.hypot(r_eye_inner.x - r_eye_outer.x, r_eye_inner.y - r_eye_outer.y)
        l_eye_width = math.hypot(l_eye_inner.x - l_eye_outer.x, l_eye_inner.y - l_eye_outer.y)
        
        # 홍채가 중앙에서 얼마나 벗어났는지 확인
        r_iris_to_center = math.hypot(r_iris.x - (r_eye_inner.x + r_eye_outer.x)/2, r_iris.y - (r_eye_inner.y + r_eye_outer.y)/2)
        l_iris_to_center = math.hypot(l_iris.x - (l_eye_inner.x + l_eye_outer.x)/2, l_iris.y - (l_eye_inner.y + l_eye_outer.y)/2)
        
        if r_eye_width == 0 or l_eye_width == 0: return 0.0
        
        r_ratio = r_iris_to_center / r_eye_width
        l_ratio = l_iris_to_center / l_eye_width
        
        gaze_score = 0
        # 방향 판별 (양수: 오른쪽, 음수: 왼쪽)
        if r_iris.x > (r_eye_inner.x + r_eye_outer.x)/2 : gaze_score += r_ratio * 2
        else: gaze_score -= r_ratio * 2
        
        if l_iris.x > (l_eye_inner.x + l_eye_outer.x)/2 : gaze_score += l_ratio * 2
        else: gaze_score -= l_ratio * 2
        
        return gaze_score
    except: return 0.0

# =========================================================
# 💡 [기능 1] 녹화 파일 분석 함수 (API 업로드용)
# =========================================================
def analyze_video_file(file_path: str) -> dict:
    """
    영상 파일을 프레임 단위로 읽어 전체 통계를 분석합니다.
    """
    video = cv2.VideoCapture(file_path)
    if not video.isOpened(): return {"error": "파일 열기 실패"}

    # --- 통계 변수 초기화 ---
    shoulder_tilt_count = 0
    gaze_off_center_count = 0
    total_smile_score = 0
    blink_count = 0
    
    # --- 상태 추적 플래그 (이탈 중복 카운트 방지) ---
    is_shoulder_tilted = False
    is_gaze_off_center = False
    is_blinking = False
    
    processed_frames = 0

    # 프레임 루프 시작
    while True:
        ret, frame = video.read()
        if not ret: break
        processed_frames += 1
        
        # 이미지 변환 (BGR -> RGB -> MediaPipe Image)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # AI 감지 실행
        pose_res = pose_landmarker.detect(mp_image)
        face_res = face_landmarker.detect(mp_image)
        
        # 1. 자세 분석 (어깨 기울기)
        s_angle = 180
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks[0]
            try:
                s_angle = calculate_tilt_angle([lm[11].x, lm[11].y], [lm[12].x, lm[12].y])
            except: pass
            
        # 어깨 이탈 카운트 (기준: 175도 미만)
        if abs(s_angle) < 175:
            if not is_shoulder_tilted:
                shoulder_tilt_count += 1
                is_shoulder_tilted = True
        else: is_shoulder_tilted = False

        # 2. 얼굴 분석 (시선, 미소, 눈깜빡임)
        if face_res.face_landmarks:
            # 시선 이탈 카운트 (기준: 점수 0.5 초과)
            g_score = get_gaze_ratio(face_res.face_landmarks[0])
            if abs(g_score) > 0.5:
                if not is_gaze_off_center:
                    gaze_off_center_count += 1
                    is_gaze_off_center = True
            else: is_gaze_off_center = False
            
            if face_res.face_blendshapes:
                shapes = face_res.face_blendshapes[0]
                
                # 눈 깜빡임 카운트 (Threshold: 0.6 닫힘 / 0.3 열림)
                blink = (next((s.score for s in shapes if s.category_name=='eyeBlinkLeft'),0) + 
                         next((s.score for s in shapes if s.category_name=='eyeBlinkRight'),0))/2
                if blink > 0.6 and not is_blinking:
                    blink_count += 1
                    is_blinking = True
                elif blink < 0.3: is_blinking = False
                
                # 미소 점수 누적 (0~100점)
                smile = (next((s.score for s in shapes if s.category_name=='mouthSmileLeft'),0) + 
                         next((s.score for s in shapes if s.category_name=='mouthSmileRight'),0))/2 * 100
                total_smile_score += smile

    # --- 최종 결과 계산 ---
    fps = video.get(cv2.CAP_PROP_FPS) or 30
    duration_min = (processed_frames / fps) / 60
    if duration_min == 0: duration_min = 0.01 # 0 나누기 방지용

    avg_smile = total_smile_score / processed_frames if processed_frames > 0 else 0
    avg_blink = blink_count / duration_min

    video.release()
    
    # 요청하신 4가지 지표 반환
    return {
        "shoulder_tilt_count": shoulder_tilt_count,     # 어깨 이탈 횟수
        "gaze_off_center_count": gaze_off_center_count, # 시선 이탈 횟수
        "average_smile_score": round(avg_smile, 1),     # 평균 미소 점수
        "average_blink_count": round(avg_blink, 1)      # 분당 평균 눈깜빡임
    }

# =========================================================
# 💡 [기능 2] 실시간 스트리밍 (시연/로컬 테스트용)
# =========================================================
cap = None # 전역 변수 (Lazy Loading)

async def generate_frames():
    global cap
    # 요청이 들어왔을 때 카메라를 켭니다.
    if cap is None or not cap.isOpened():
        print("📷 웹캠 스트리밍 시작...")
        cap = cv2.VideoCapture(0)

    start_time = time.time()
    blink_count = 0
    is_blinking = False

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            
            # --- 실시간 데이터 업데이트 로직 (간소화) ---
            # (여기서 계산된 값은 state_manager를 통해 /api/analysis로 나감)
            # 실제 데모 때는 위 analyze_video_file과 동일한 로직으로 계산하여
            # update_video_data({...})를 호출해주면 됩니다.
            
            # 이미지 인코딩 및 전송
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            await asyncio.sleep(0.05)

        except Exception as e:
            print(f"스트리밍 에러: {e}")
            await asyncio.sleep(0.1)

def cleanup_camera():
    """서버 종료 시 카메라 자원 해제"""
    global cap
    if cap and cap.isOpened():
        cap.release()
        print("📷 카메라 자원 해제됨")