"""실시간 음성 분석 모듈.

3초 단위로 마이크 입력을 잘라 STT 와 운율(볼륨/피치)을 분석하고
state_manager 로 넘긴다.

무거운 자원(Whisper 모델, 마이크 스트림)은 start() 시점에 준비한다.
import 만으로 모델을 내려받거나 마이크를 열면 서버가 뜨지 않는다.
"""

import os
import threading

import numpy as np

from state_manager import update_audio_data

CHUNK_DURATION_S = 3          # 몇 초 단위로 분석할지
RATE = 16000                  # Whisper 는 16kHz 를 기대한다
BUFFER_SIZE = CHUNK_DURATION_S * RATE
READ_CHUNK = 1024

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")

# 한국어 면접에서 자주 나오는 군말
FILLER_WORDS = {"음", "어", "그", "저", "이제", "뭐", "같은", "그러니까", "아", "휴"}


class AudioUnavailable(RuntimeError):
    """pyaudio / whisper / librosa 가 없거나 마이크를 열 수 없을 때."""


class AudioAnalyzer:
    def __init__(self):
        self._model = None
        self._pyaudio = None
        self._stream = None
        self._thread = None
        self._buffer = np.array([], dtype=np.int16)
        self.is_running = False

    # --- 지연 로딩 ---

    def _load(self):
        try:
            import librosa            # noqa: F401
            import pyaudio
            import whisper
        except ImportError as exc:
            raise AudioUnavailable(
                f"음성 분석에 필요한 패키지가 없습니다: {exc}. "
                "pip install openai-whisper librosa pyaudio"
            ) from exc

        if self._model is None:
            print(f"Whisper 모델을 불러옵니다 ({WHISPER_MODEL}) ...")
            self._model = whisper.load_model(WHISPER_MODEL)
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()
        return pyaudio

    # --- 제어 ---

    def start(self):
        if self.is_running:
            return {"status": "already_running"}

        pyaudio = self._load()
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=READ_CHUNK,
        )
        self._buffer = np.array([], dtype=np.int16)
        self.is_running = True
        update_audio_data({"is_running": True})

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return {"status": "started"}

    def stop(self):
        if not self.is_running:
            return {"status": "not_running"}

        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        update_audio_data({"is_running": False})
        return {"status": "stopped"}

    # --- 분석 루프 ---

    def _loop(self):
        while self.is_running:
            try:
                raw = self._stream.read(READ_CHUNK, exception_on_overflow=False)
                self._buffer = np.concatenate(
                    (self._buffer, np.frombuffer(raw, dtype=np.int16))
                )

                if len(self._buffer) >= BUFFER_SIZE:
                    chunk = self._buffer[:BUFFER_SIZE]
                    self._buffer = self._buffer[BUFFER_SIZE:]
                    update_audio_data(self.analyze_chunk(chunk))
            except Exception as exc:
                print(f"[audio] 분석 루프 오류: {exc}")
                self.is_running = False
                update_audio_data({"is_running": False})
                break

    def analyze_chunk(self, chunk_int16) -> dict:
        """3초 분량 오디오를 분석한다. 테스트에서 직접 호출할 수 있다."""
        import librosa

        audio = chunk_int16.astype(np.float32) / 32768.0

        text = self._model.transcribe(audio, language="ko")["text"].strip()
        words = text.split()

        rms = float(np.mean(librosa.feature.rms(y=audio)))
        pitches, _ = librosa.core.piptrack(y=audio, sr=RATE)
        voiced = pitches[pitches > 0]
        pitch = float(np.mean(voiced)) if voiced.size else 0.0

        return {
            "transcription": text,
            # 한국어는 split() 단위가 '어절'이라 WPM 이 아니라 EPM 으로 부른다
            "speaking_speed_epm": round(len(words) / CHUNK_DURATION_S * 60, 2),
            "filler_word_count": sum(
                1 for w in words if w.lower().strip(".,?!") in FILLER_WORDS
            ),
            "average_volume": round(rms, 4),
            "average_pitch_hz": round(pitch, 2),
        }


_instance = None
_instance_lock = threading.Lock()


def get_analyzer() -> AudioAnalyzer:
    """실제로 필요할 때 한 번만 만든다."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = AudioAnalyzer()
        return _instance
