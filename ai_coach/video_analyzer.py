"""영상 분석 모듈.

녹화 파일 분석(사후 배치)과 웹캠 실시간 분석이 FrameAnalyzer 하나를
공유한다. 두 경로가 각자 집계 로직을 갖고 있으면 같은 영상인데도
실시간 화면과 최종 리포트의 숫자가 어긋나기 때문이다.
"""

import asyncio
import math
import os
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from state_manager import update_video_data

BASE_DIR = Path(__file__).resolve().parent
POSE_MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"
FACE_MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

# 실시간 소스. 0 이면 웹캠, 파일 경로면 그 영상을 재생한다.
# 프론트엔드 개발자가 웹캠 없이도 실시간 경로를 붙여볼 수 있게 하기 위한 것.
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")

# 실시간에서 몇 프레임마다 추론할지. 매 프레임 추론하면 FPS 가 떨어진다.
REALTIME_DETECT_EVERY = int(os.getenv("REALTIME_DETECT_EVERY", "3"))


# ============================================================
# MediaPipe 초기화 (지연 로딩 + 락)
# ============================================================

# MediaPipe Tasks 객체는 스레드 안전을 보장하지 않는다.
# 요청이 동시에 들어오면 같은 인스턴스의 detect() 가 겹쳐 호출되므로
# 추론 구간을 락으로 감싼다.
_mp_lock = threading.Lock()
_landmarkers = None


def _get_landmarkers():
    """모델을 처음 필요할 때 한 번만 로드한다."""
    global _landmarkers
    if _landmarkers is not None:
        return _landmarkers

    with _mp_lock:
        if _landmarkers is not None:
            return _landmarkers

        for path in (POSE_MODEL_PATH, FACE_MODEL_PATH):
            if not path.exists():
                raise FileNotFoundError(f"MediaPipe 모델 파일이 없습니다: {path}")

        base = mp.tasks.BaseOptions
        vision = mp.tasks.vision

        pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base(model_asset_path=str(POSE_MODEL_PATH)),
                running_mode=vision.RunningMode.IMAGE,
            )
        )
        face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=str(FACE_MODEL_PATH)),
                output_face_blendshapes=True,
                running_mode=vision.RunningMode.IMAGE,
            )
        )
        _landmarkers = (pose, face)
    return _landmarkers


# ============================================================
# 보조 계산
# ============================================================

def calculate_tilt_angle(a, b) -> float:
    """두 점을 잇는 선의 기울기(도).

    정면을 본 사람의 좌우 어깨는 이미지 좌표에서 거의 수평이라
    이 값이 ±180 도 부근에 모인다. 그래서 "기울지 않음"의 기준이
    0 이 아니라 180 이다.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))


def get_gaze_ratio(face_landmarks) -> float:
    """홍채가 눈 중앙에서 얼마나 벗어났는지. 양수는 오른쪽, 음수는 왼쪽."""
    try:
        r_out, r_in, r_iris = face_landmarks[33], face_landmarks[133], face_landmarks[468]
        l_out, l_in, l_iris = face_landmarks[263], face_landmarks[362], face_landmarks[473]
    except (IndexError, TypeError):
        return 0.0

    def side_score(outer, inner, iris):
        width = math.hypot(inner.x - outer.x, inner.y - outer.y)
        if width == 0:
            return 0.0
        cx = (inner.x + outer.x) / 2
        cy = (inner.y + outer.y) / 2
        offset = math.hypot(iris.x - cx, iris.y - cy) / width
        return offset * 2 if iris.x > cx else -offset * 2

    return side_score(r_out, r_in, r_iris) + side_score(l_out, l_in, l_iris)


def _blendshape(shapes, name: str) -> float:
    return next((s.score for s in shapes if s.category_name == name), 0.0)


# ============================================================
# 프레임 분석기 — 실시간과 배치가 함께 쓴다
# ============================================================

class FrameAnalyzer:
    """프레임을 하나씩 받아 지표를 누적한다."""

    # 어깨가 이 각도보다 더 기울면 '이탈'로 본다 (180 이 수평).
    SHOULDER_LEVEL_DEG = 175.0
    # 홍채 이탈 점수가 이 값을 넘으면 '시선 이탈'로 본다.
    GAZE_OFF_THRESHOLD = 0.5
    # 깜빡임 판정 임계값을 둘로 나눈다(히스테리시스).
    # 하나만 두면 눈이 반쯤 감긴 구간에서 값이 경계를 오가며
    # 한 번의 깜빡임이 수십 번으로 집계된다.
    BLINK_CLOSE = 0.6   # 이 위로 올라가야 '감음'
    BLINK_OPEN = 0.3    # 이 아래로 내려와야 다시 '뜸'

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        # 누적 지표
        self.shoulder_tilt_count = 0
        self.gaze_off_center_count = 0
        self.blink_count = 0
        self.analyzed_frames = 0
        self._smile_sum = 0.0
        self._smile_frames = 0
        # 상태 플래그 — 상태가 바뀌는 순간에만 집계하기 위한 것.
        # 없으면 30fps 에서 1초만 기울어도 30 회로 잡힌다.
        self._shoulder_tilted = False
        self._gaze_off = False
        self._blinking = False
        # 최신 순간값 (실시간 표시용)
        self.shoulder_angle = 180.0
        self.gaze_score = 0.0
        self.smile_score = 0.0

    # --- 상태 기계: MediaPipe 없이 단위 테스트할 수 있게 분리했다 ---

    def update_shoulder(self, angle: float) -> None:
        self.shoulder_angle = angle
        if abs(angle) < self.SHOULDER_LEVEL_DEG:
            if not self._shoulder_tilted:
                self.shoulder_tilt_count += 1
                self._shoulder_tilted = True
        else:
            self._shoulder_tilted = False

    def update_gaze(self, score: float) -> None:
        self.gaze_score = score
        if abs(score) > self.GAZE_OFF_THRESHOLD:
            if not self._gaze_off:
                self.gaze_off_center_count += 1
                self._gaze_off = True
        else:
            self._gaze_off = False

    def update_blink(self, score: float) -> None:
        if score > self.BLINK_CLOSE:
            if not self._blinking:
                self.blink_count += 1
                self._blinking = True
        elif score < self.BLINK_OPEN:
            self._blinking = False

    def update_smile(self, score: float) -> None:
        self.smile_score = score
        self._smile_sum += score
        self._smile_frames += 1

    # --- 프레임 처리 ---

    def process(self, frame_bgr) -> dict:
        """프레임 한 장을 분석하고 현재 스냅샷을 돌려준다."""
        self.analyzed_frames += 1

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_landmarker, face_landmarker = _get_landmarkers()
        with _mp_lock:
            pose_res = pose_landmarker.detect(mp_image)
            face_res = face_landmarker.detect(mp_image)

        # 자세
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks[0]
            if len(lm) > 12:
                self.update_shoulder(
                    calculate_tilt_angle([lm[11].x, lm[11].y], [lm[12].x, lm[12].y])
                )

        # 얼굴
        if face_res.face_landmarks:
            self.update_gaze(get_gaze_ratio(face_res.face_landmarks[0]))

            if face_res.face_blendshapes:
                shapes = face_res.face_blendshapes[0]
                blink = (
                    _blendshape(shapes, "eyeBlinkLeft")
                    + _blendshape(shapes, "eyeBlinkRight")
                ) / 2
                self.update_blink(blink)

                smile = (
                    _blendshape(shapes, "mouthSmileLeft")
                    + _blendshape(shapes, "mouthSmileRight")
                ) / 2 * 100
                self.update_smile(smile)

        return self.snapshot()

    # --- 결과 ---

    @property
    def average_smile_score(self) -> float:
        # 얼굴이 잡힌 프레임만으로 평균을 낸다. 전체 프레임으로 나누면
        # 얼굴이 화면 밖에 있던 구간 때문에 점수가 부당하게 낮아진다.
        if self._smile_frames == 0:
            return 0.0
        return self._smile_sum / self._smile_frames

    def snapshot(self) -> dict:
        """실시간 조회용 — 순간값과 누적값을 함께 담는다."""
        return {
            "shoulder_angle": round(self.shoulder_angle, 1),
            "gaze_score": round(self.gaze_score, 3),
            "smile_score": round(self.smile_score, 1),
            "shoulder_tilt_count": self.shoulder_tilt_count,
            "gaze_off_center_count": self.gaze_off_center_count,
            "blink_count": self.blink_count,
            "analyzed_frames": self.analyzed_frames,
        }

    def summary(self, duration_min: float) -> dict:
        """최종 리포트용 4가지 지표."""
        safe_minutes = duration_min if duration_min > 0 else 0.01
        return {
            "shoulder_tilt_count": self.shoulder_tilt_count,
            "gaze_off_center_count": self.gaze_off_center_count,
            "average_smile_score": round(self.average_smile_score, 1),
            # 총 횟수는 영상이 길수록 커져서 비교가 안 된다. 분당으로 정규화한다.
            "average_blink_count": round(self.blink_count / safe_minutes, 1),
        }


# ============================================================
# [배치] 녹화 파일 분석
# ============================================================

def analyze_video_file(file_path, frame_stride: int = 1) -> dict:
    """영상 파일 전체를 분석해 4가지 지표를 반환한다.

    frame_stride 를 올리면 그만큼 프레임을 건너뛰어 빨라진다.
    다만 깜빡임은 0.1~0.4 초짜리 짧은 이벤트라 건너뛰면 놓치므로
    기본값은 1(모든 프레임)로 둔다.
    """
    video = cv2.VideoCapture(str(file_path))
    if not video.isOpened():
        raise ValueError(f"영상 파일을 열 수 없습니다: {file_path}")

    analyzer = FrameAnalyzer()
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    read_frames = 0

    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            read_frames += 1
            if (read_frames - 1) % frame_stride:
                continue
            analyzer.process(frame)
    finally:
        video.release()

    duration_min = (read_frames / fps) / 60 if fps else 0.0
    return analyzer.summary(duration_min)


# ============================================================
# [실시간] 웹캠 스트리밍 + 지표 산출
# ============================================================

_capture = None
_capture_lock = threading.Lock()
realtime_analyzer = FrameAnalyzer()


def _resolve_source():
    return int(VIDEO_SOURCE) if str(VIDEO_SOURCE).isdigit() else VIDEO_SOURCE


def _open_capture():
    global _capture
    with _capture_lock:
        if _capture is None or not _capture.isOpened():
            source = _resolve_source()
            _capture = cv2.VideoCapture(source)
            if not _capture.isOpened():
                raise RuntimeError(f"영상 소스를 열 수 없습니다: {source!r}")
        return _capture


def reset_realtime() -> None:
    """새 면접 세션을 시작할 때 누적값을 비운다."""
    realtime_analyzer.reset()
    update_video_data(realtime_analyzer.snapshot())


async def generate_frames():
    """MJPEG 스트림을 내보내면서 동시에 지표를 계산해 state_manager 로 보낸다."""
    capture = _open_capture()
    is_webcam = isinstance(_resolve_source(), int)
    frame_index = 0

    update_video_data({"is_running": True})
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if is_webcam:
                    await asyncio.sleep(0.1)
                    continue
                # 파일 소스는 끝나면 처음으로 되감는다 (프론트 연동 테스트용)
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if is_webcam:
                frame = cv2.flip(frame, 1)  # 거울처럼 보이도록

            frame_index += 1
            if frame_index % REALTIME_DETECT_EVERY == 0:
                try:
                    update_video_data(realtime_analyzer.process(frame))
                except Exception as exc:  # 추론이 실패해도 스트림은 유지한다
                    print(f"[realtime] 분석 실패: {exc}")

            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )
            await asyncio.sleep(0.03)
    finally:
        # 클라이언트가 끊기면 여기로 온다
        update_video_data({"is_running": False})


def cleanup_camera() -> None:
    global _capture
    with _capture_lock:
        if _capture is not None and _capture.isOpened():
            _capture.release()
            print("카메라 자원을 해제했습니다.")
        _capture = None
