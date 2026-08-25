"""모듈 간 실시간 분석 데이터를 공유하는 중앙 저장소.

영상 분석 스레드와 음성 분석 스레드가 동시에 값을 쓰고,
API 핸들러가 그 값을 읽는다. 그래서 Lock 으로 보호한다.
"""

import copy
import threading

_lock = threading.Lock()

# 실시간 레이어가 채우는 값. 초기값은 "아직 분석 전" 상태를 뜻한다.
_INITIAL_STATE = {
    "video": {
        "is_running": False,
        "shoulder_angle": 0.0,
        "gaze_score": 0.0,
        "smile_score": 0.0,
        "shoulder_tilt_count": 0,
        "gaze_off_center_count": 0,
        "blink_count": 0,
        "analyzed_frames": 0,
    },
    "audio": {
        "is_running": False,
        "transcription": "",
        "speaking_speed_epm": 0.0,
        "filler_word_count": 0,
        "average_volume": 0.0,
        "average_pitch_hz": 0.0,
    },
}

live_analysis_data = copy.deepcopy(_INITIAL_STATE)


def update_video_data(data: dict) -> None:
    """영상 분석 스레드가 호출한다."""
    with _lock:
        live_analysis_data["video"].update(data)


def update_audio_data(data: dict) -> None:
    """음성 분석 스레드가 호출한다."""
    with _lock:
        live_analysis_data["audio"].update(data)


def get_all_data() -> dict:
    """API 핸들러가 호출한다.

    깊은 복사로 반환한다. 얕은 복사(dict.copy())를 쓰면 중첩된
    "video"/"audio" 딕셔너리가 원본을 그대로 가리키기 때문에,
    호출자가 값을 읽는 동안 분석 스레드가 그 값을 바꿀 수 있다.
    Lock 을 걸어놓고도 보호가 새는 셈이라 deepcopy 를 쓴다.
    """
    with _lock:
        return copy.deepcopy(live_analysis_data)


def reset(section: str | None = None) -> None:
    """새 면접 세션을 시작할 때 누적값을 초기화한다."""
    with _lock:
        if section is None:
            live_analysis_data.update(copy.deepcopy(_INITIAL_STATE))
        else:
            live_analysis_data[section] = copy.deepcopy(_INITIAL_STATE[section])
