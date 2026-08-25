"""의존성 없이 핵심 집계 로직을 검증한다.

    python selftest.py

MediaPipe 추론 없이 상태 기계만 따로 돌려본다. 임계값 판정은
'모델이 정답을 주지 않고 직접 정해야 했던' 부분이라 회귀 테스트가 필요하다.
"""

import sys

from video_analyzer import FrameAnalyzer

PASS, FAIL = [], []


def check(name, actual, expected):
    if actual == expected:
        PASS.append(name)
        print(f"  ✅ {name}: {actual}")
    else:
        FAIL.append(name)
        print(f"  ❌ {name}: 기대 {expected} / 실제 {actual}")


print("=" * 64)
print("1. 깜빡임 히스테리시스 — 경계에서 떨려도 중복 집계되지 않아야 한다")
print("=" * 64)

fa = FrameAnalyzer()
# 0.5 를 단일 임계값으로 썼다면 아래 시퀀스는 4회로 집계된다.
sequence = [0.0, 0.65, 0.55, 0.45, 0.58, 0.62, 0.20, 0.70]
for score in sequence:
    fa.update_blink(score)
naive = sum(
    1 for prev, cur in zip([0.0] + sequence, sequence) if prev <= 0.5 < cur
)
print(f"  입력: {sequence}")
print(f"  단일 임계값(0.5)이었다면: {naive}회")
check("히스테리시스 깜빡임 횟수", fa.blink_count, 2)

print()
print("=" * 64)
print("2. 상태 전이 집계 — 기울어져 있는 동안 계속 세지 않아야 한다")
print("=" * 64)

fa = FrameAnalyzer()
angles = [180, 180, 170, 170, 170, 179, 180, 168, 168]
for a in angles:
    fa.update_shoulder(a)
print(f"  입력 각도: {angles}")
print("  (175도 미만이 '기울어짐'. 프레임마다 셌다면 6회)")
check("어깨 이탈 횟수", fa.shoulder_tilt_count, 2)

fa = FrameAnalyzer()
gazes = [0.1, 0.9, 0.8, 0.6, 0.1, -0.9, -0.7, 0.0]
for g in gazes:
    fa.update_gaze(g)
print(f"  입력 시선점수: {gazes}")
check("시선 이탈 횟수", fa.gaze_off_center_count, 2)

print()
print("=" * 64)
print("3. 정규화 — 영상 길이가 달라도 비교 가능해야 한다")
print("=" * 64)

fa = FrameAnalyzer()
for _ in range(30):
    fa.update_blink(0.9)
    fa.update_blink(0.1)
check("깜빡임 30회 / 2분 = 분당", fa.summary(2.0)["average_blink_count"], 15.0)
check("깜빡임 30회 / 5분 = 분당", fa.summary(5.0)["average_blink_count"], 6.0)
check("길이 0 방어 (0으로 나누지 않음)", fa.summary(0.0)["average_blink_count"], 3000.0)

print()
print("=" * 64)
print("4. 미소 평균 — 얼굴이 잡힌 프레임만으로 평균을 낸다")
print("=" * 64)

fa = FrameAnalyzer()
fa.analyzed_frames = 10          # 10프레임 중
fa.update_smile(80.0)            # 얼굴이 잡힌 건 2프레임
fa.update_smile(60.0)
print("  10프레임 중 2프레임에서만 얼굴 검출, 점수 80 / 60")
print("  전체 프레임으로 나눴다면: 14.0 (얼굴이 없던 구간 때문에 부당하게 낮음)")
check("미소 평균", fa.summary(1.0)["average_smile_score"], 70.0)

print()
print("=" * 64)
print("5. state_manager — 스냅샷이 이후 변경에 오염되지 않아야 한다")
print("=" * 64)

import state_manager as sm

sm.reset()
sm.update_video_data({"blink_count": 7})
snapshot = sm.get_all_data()
sm.update_video_data({"blink_count": 99})
print("  스냅샷을 뜬 뒤 분석 스레드가 값을 99로 바꿈")
print("  얕은 복사(dict.copy())였다면 스냅샷도 99가 된다")
check("스냅샷 격리", snapshot["video"]["blink_count"], 7)
check("현재값은 갱신됨", sm.get_all_data()["video"]["blink_count"], 99)
sm.reset()
check("reset 동작", sm.get_all_data()["video"]["blink_count"], 0)

print()
print("=" * 64)
print(f"결과: {len(PASS)} PASS / {len(FAIL)} FAIL")
print("=" * 64)
sys.exit(1 if FAIL else 0)
