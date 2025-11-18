# audio_analyzer.py

import pyaudio
import numpy as np
import whisper  # 또는 from faster_whisper import WhisperModel
import librosa
import threading
import time
from state_manager import update_audio_data

# --- 1. 상수 정의 ---
CHUNK_DURATION_S = 3  # 몇 초마다 분석할 것인가 (3초 ~ 5초가 적당)
RATE = 16000          # 샘플링 레이트 (Whisper는 16kHz가 필요)
BUFFER_SIZE = CHUNK_DURATION_S * RATE # 버퍼 크기 (샘플 개수)
FORMAT = pyaudio.paInt16
CHANNELS = 1

# 분석할 필러(filler) 단어 목록
FILLER_WORDS = ["음", "어", "그", "저", "이제", "뭐", "같은", "그러니까", "아", "휴"]

class AudioAnalyzer:
    def __init__(self):
        # 💡 성능 팁: CPU에서는 'tiny' 또는 'base' 모델을 사용하세요.
        # 💡 'faster-whisper' 라이브러리를 사용하면 훨씬 더 빠릅니다.
        self.model = whisper.load_model("tiny")
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.audio_buffer = np.array([], dtype=np.int16)
        self.is_running = False
        self.thread = None

    def start_analysis(self):
        """백그라운드에서 오디오 분석 스레드를 시작합니다."""
        if self.is_running:
            print("오디오 분석이 이미 실행 중입니다.")
            return

        print("오디오 분석 스레드를 시작합니다...")
        self.is_running = True
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                  input=True, frames_per_buffer=1024)
        
        # 'daemon=True'로 메인 프로그램 종료 시 스레드도 함께 종료
        self.thread = threading.Thread(target=self._run_analysis_loop, daemon=True)
        self.thread.start()

    def stop_analysis(self):
        """오디오 분석 스레드를 중지합니다."""
        self.is_running = False
        if self.thread:
            self.thread.join()  # 스레드가 끝날 때까지 기다림
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        print("오디오 분석을 중지했습니다.")

    def _run_analysis_loop(self):
        """실제 분석이 일어나는 메인 루프 (스레드에서 실행됨)"""
        while self.is_running:
            try:
                # 1. 마이크에서 데이터 읽기
                data = self.stream.read(1024, exception_on_overflow=False)
                new_data = np.frombuffer(data, dtype=np.int16)
                
                # 2. 버퍼에 데이터 추가
                self.audio_buffer = np.concatenate((self.audio_buffer, new_data))

                # 3. 버퍼가 꽉 찼는지(3초) 확인
                if len(self.audio_buffer) >= BUFFER_SIZE:
                    # 4. 3초 분량의 "청크"를 잘라내어 분석
                    chunk_to_analyze = self.audio_buffer[:BUFFER_SIZE]
                    self.audio_buffer = self.audio_buffer[BUFFER_SIZE:] # 나머지는 다음 루프를 위해 남겨둠

                    # 5. 실제 분석 실행
                    analysis_result = self._analyze_chunk(chunk_to_analyze)
                    
                    # 6. state_manager로 결과 전송
                    update_audio_data(analysis_result)

            except Exception as e:
                print(f"오디오 분석 루프 에러: {e}")
                time.sleep(0.1)

    def _analyze_chunk(self, audio_chunk_int16):
        """3초 분량의 오디오 청크를 분석하는 핵심 함수"""
        
        # 1. 오디오 데이터를 Whisper/Librosa가 사용 가능한 float32로 변환
        audio_float = audio_chunk_int16.astype(np.float32) / 32768.0
        
        # --- 2. STT (Whisper) ---
        result = self.model.transcribe(audio_float, language="ko")
        transcription = result["text"]
        
        # --- 3. 텍스트 분석 (속도, 필러 단어) ---
        words = transcription.split()
        word_count = len(words)
        speaking_speed_wpm = (word_count / CHUNK_DURATION_S) * 60 # 분당 단어 수 (WPM)
        filler_count = sum(1 for word in words if word.lower().strip(".,?!") in FILLER_WORDS)
        
        # --- 4. 운율 분석 (Librosa) ---
        # 볼륨 (RMS Energy)
        rms = librosa.feature.rms(y=audio_float)
        average_volume = np.mean(rms)
        
        # 목소리 톤/높낮이 (Pitch)
        pitches, _ = librosa.core.piptrack(y=audio_float, sr=RATE)
        valid_pitches = pitches[pitches > 0]
        average_pitch_hz = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
        
        # 5. 결과 취합
        return {
            "transcription": transcription,
            "speaking_speed_wpm": round(speaking_speed_wpm, 2),
            "filler_word_count": filler_count,
            "average_volume": round(float(average_volume), 4),
            "average_pitch_hz": round(float(average_pitch_hz), 2)
        }

# --- main.py에서 사용할 수 있도록 인스턴스 생성 ---
audio_analyzer_instance = AudioAnalyzer()