import cv2
import mediapipe as mp
import numpy as np
import time
import math
from mediapipe.framework.formats import landmark_pb2
import asyncio
from state_manager import update_video_data

# --- 모델 초기화 (전역 사용) ---
pose_model_path = './models/pose_landmarker_lite.task'
face_model_path = './models/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 1. 실시간용 옵션 (LIVE_STREAM 모드 대신 IMAGE 모드로 프레임 단위 처리)
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.IMAGE)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.IMAGE)

pose_landmarker = PoseLandmarker.create_from_options(pose_options)
face_landmarker = FaceLandmarker.create_from_options(face_options)

# --- 보조 함수들 ---
def calculate_tilt_angle(a, b):
    a = np.array(a); b = np.array(b)
    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.degrees(radians)
    return angle

def get_gaze_ratio(face_landmarks):
    try:
        r_eye_outer = face_landmarks[33]; r_eye_inner = face_landmarks[133]; r_iris = face_landmarks[468]
        l_eye_outer = face_landmarks[263]; l_eye_inner = face_landmarks[362]; l_iris = face_landmarks[473]
        
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
    except: return 0.0

# ==========================================
# 💡 [기능 1] 실시간 스트리밍용 함수 (시연용)
# ==========================================
cap = cv2.VideoCapture(0) # 전역 카메라 (시연용)

async def generate_frames():
    # ... (기존 코드와 동일하게 유지 - 시연을 위해 필요) ...
    # (코드 길이를 줄이기 위해 생략하지만, 기존 generate_frames 로직을 그대로 두세요)
    # 실시간 분석 결과를 state_manager에 업데이트하는 부분 포함
    pass 

def cleanup_camera():
    if cap.isOpened(): cap.release()


# ==========================================
# 💡 [기능 2] (NEW) 녹화 파일 분석용 함수
# ==========================================
def analyze_video_file(file_path: str) -> dict:
    """
    업로드된 비디오 파일을 분석하여 요약 통계를 반환합니다.
    """
    video = cv2.VideoCapture(file_path)
    if not video.isOpened():
        return {"error": "파일을 열 수 없습니다."}

    # 1. 통계 변수 초기화
    shoulder_tilt_count = 0
    head_tilt_count = 0
    gaze_off_center_count = 0
    
    total_smile_score = 0
    blink_count = 0
    
    # 상태 추적 변수
    is_shoulder_tilted = False
    is_head_tilted = False
    is_gaze_off_center = False
    is_blinking = False
    
    processed_frames = 0

    # 2. 프레임 루프 (처음부터 끝까지)
    while True:
        ret, frame = video.read()
        if not ret:
            break # 영상 끝

        processed_frames += 1
        
        # 이미지 변환 (BGR -> RGB -> MP Image)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # AI 감지
        pose_result = pose_landmarker.detect(mp_image)
        face_result = face_landmarker.detect(mp_image)
        
        # --- A. 자세 분석 ---
        shoulder_angle = 180
        head_angle = 180
        
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]
            try:
                shoulder_angle = calculate_tilt_angle([lm[11].x, lm[11].y], [lm[12].x, lm[12].y])
                head_angle = calculate_tilt_angle([lm[7].x, lm[7].y], [lm[8].x, lm[8].y])
            except: pass

        # 어깨 이탈 카운트
        if abs(shoulder_angle) < 175:
            if not is_shoulder_tilted:
                shoulder_tilt_count += 1
                is_shoulder_tilted = True
        else: is_shoulder_tilted = False

        # 고개 이탈 카운트
        if abs(head_angle) < 175:
            if not is_head_tilted:
                head_tilt_count += 1
                is_head_tilted = True
        else: is_head_tilted = False

        # --- B. 얼굴/시선/표정 분석 ---
        gaze_score = 0.0
        smile_score = 0.0
        
        if face_result.face_landmarks:
            # 시선
            gaze_score = get_gaze_ratio(face_result.face_landmarks[0])
            
            # 시선 이탈 카운트
            if abs(gaze_score) > 0.5:
                if not is_gaze_off_center:
                    gaze_off_center_count += 1
                    is_gaze_off_center = True
            else: is_gaze_off_center = False

            # 블렌드쉐이프 (눈깜빡임, 미소)
            if face_result.face_blendshapes:
                shapes = face_result.face_blendshapes[0]
                
                # 눈 깜빡임
                blink_left = next((s.score for s in shapes if s.category_name=='eyeBlinkLeft'), 0)
                blink_right = next((s.score for s in shapes if s.category_name=='eyeBlinkRight'), 0)
                avg_blink = (blink_left + blink_right) / 2
                
                if avg_blink > 0.6 and not is_blinking:
                    blink_count += 1
                    is_blinking = True
                elif avg_blink < 0.3:
                    is_blinking = False
                
                # 미소
                smile_left = next((s.score for s in shapes if s.category_name=='mouthSmileLeft'), 0)
                smile_right = next((s.score for s in shapes if s.category_name=='mouthSmileRight'), 0)
                smile_score = (smile_left + smile_right) / 2 * 100
                
                # 미소 점수 누적 (나중에 평균 내기 위함)
                total_smile_score += smile_score

    # 3. 최종 통계 계산
    video_fps = video.get(cv2.CAP_PROP_FPS)
    if video_fps == 0: video_fps = 30 # 기본값 방어
    
    duration_min = (processed_frames / video_fps) / 60
    if duration_min == 0: duration_min = 0.01 # 0 나누기 방지

    avg_smile = total_smile_score / processed_frames if processed_frames > 0 else 0
    avg_blink_per_min = blink_count / duration_min

    video.release()
    
    # 4. 결과 반환 (요청하신 4가지 핵심 지표)
    return {
        "shoulder_tilt_count": shoulder_tilt_count,     # 어깨 이탈 횟수
        "gaze_off_center_count": gaze_off_center_count, # 시선 이탈 횟수
        "average_smile_score": round(avg_smile, 1),     # 평균 미소 점수
        "average_blink_count": round(avg_blink_per_min, 1) # 분당 평균 눈깜빡임
    }