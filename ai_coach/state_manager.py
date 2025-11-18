# state_manager.py
import threading

# 💡 1. 'threading.Lock'
# 여러 스레드(음성, 영상)가 동시에 데이터를 쓰려고 할 때
# 데이터가 깨지는 것을 막아주는 "자물쇠"입니다.
_lock = threading.Lock()

# 💡 2. 중앙 데이터 저장소 (딕셔너리)
# 모든 모듈이 이곳을 참조합니다.
live_analysis_data = {
    "video": {
        "shoulder_angle": 0.0,
        "head_angle": 0.0,
        "gaze_status": "N/A",
        "shoulder_tilt_count": 0,
        "head_tilt_count": 0,
        "gaze_off_center_count": 0,
    },
    "audio": {
        "transcription": "...",
        "speaking_speed_wpm": 0.0,
        "filler_word_count": 0,
        "average_volume": 0.0,
        "average_pitch_hz": 0.0
    }
}

# 💡 3. 데이터 업데이트 함수 (video_analyzer.py가 호출)
def update_video_data(data):
    with _lock: # 데이터를 쓰기 전에 자물쇠를 잠급니다.
        live_analysis_data["video"].update(data)
    # (자물쇠는 자동으로 풀립니다)

# 💡 4. 데이터 업데이트 함수 (audio_analyzer.py가 호출)
def update_audio_data(data):
    with _lock: # 데이터를 쓰기 전에 자물쇠를 잠급니다.
        live_analysis_data["audio"].update(data)

# 💡 5. 데이터 읽기 함수 (main.py가 호출)
def get_all_data():
    with _lock: # 데이터를 읽는 중에도 다른 스레드가 쓰지 못하게 잠급니다.
        # 데이터의 복사본을 반환하여 원본을 보호합니다.
        return live_analysis_data.copy()