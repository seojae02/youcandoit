"""Gemini 로 예상 질문과 종합 피드백을 생성한다."""

import json
import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# 실행 위치와 무관하게 ai_coach/key.env 를 읽는다
load_dotenv(dotenv_path=BASE_DIR / "key.env")

API_KEY = os.getenv("GEMINI_API_KEY", "")

# 앞에서부터 시도하고 실패하면 다음으로 넘어간다.
# 주의: GenerativeModel(...) 은 객체만 만들 뿐 네트워크를 타지 않는다.
# 그래서 모델 가용성은 실제 호출 시점에만 판별할 수 있고,
# 폴백도 호출부에서 처리해야 한다.
MODEL_CANDIDATES = [
    m.strip()
    for m in os.getenv(
        "GEMINI_MODELS", "gemini-3.6-flash,gemini-flash-latest,gemini-2.5-flash"
    ).split(",")
    if m.strip()
]

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("경고: GEMINI_API_KEY 가 설정되지 않았습니다. ai_coach/key.env 를 확인하세요.")


class GeminiUnavailable(RuntimeError):
    pass


def _generate(prompt: str, *, as_json: bool = False) -> str:
    """모델 목록을 순서대로 시도한다. 전부 실패하면 예외를 던진다."""
    if not API_KEY:
        raise GeminiUnavailable("GEMINI_API_KEY 가 없습니다.")

    config = {"response_mime_type": "application/json"} if as_json else None
    errors = []

    for name in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(name)
            response = model.generate_content(prompt, generation_config=config)
            text = (response.text or "").strip()
            if text:
                return text
            errors.append(f"{name}: 빈 응답")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    raise GeminiUnavailable("모든 모델 호출이 실패했습니다 — " + " | ".join(errors))


# ============================================================
# [API 1] 자기소개서 → 예상 질문
# ============================================================

def generate_questions_from_resume(resume_text: str) -> list[str]:
    prompt = f"""당신은 면접관입니다. 아래 자기소개서를 읽고 지원자의 역량과 경험을
검증할 수 있는 핵심 면접 질문 3가지를 뽑아주세요.

반드시 아래 형식의 JSON 으로만 답하세요.
{{"questions": ["질문1", "질문2", "질문3"]}}

--- 자기소개서 ---
{resume_text}
---"""

    # 형식을 프롬프트로 부탁하는 대신 JSON 모드로 강제한다.
    # 예전에는 응답 문자열을 ast.literal_eval 로 파싱했는데,
    # 모델이 형식을 조금만 벗어나도 전부 실패했다.
    raw = _generate(prompt, as_json=True)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GeminiUnavailable(f"응답을 JSON 으로 해석하지 못했습니다: {raw[:200]}") from exc

    questions = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(questions, list) or not questions:
        raise GeminiUnavailable(f"질문 목록을 찾지 못했습니다: {raw[:200]}")

    return [str(q) for q in questions]


# ============================================================
# [API 3] 답변 + 행동 데이터 → 종합 피드백
# ============================================================

def _format_qa(qa_list) -> str:
    lines = []
    for idx, item in enumerate(qa_list, 1):
        if hasattr(item, "question"):
            q, a = item.question, item.answer
        elif isinstance(item, dict):
            q = item.get("예상질문") or item.get("question") or item.get("질문") or ""
            a = item.get("대답") or item.get("answer") or item.get("답변") or ""
        else:
            continue
        lines.append(f"Q{idx}. {q}\nA{idx}. {a}")
    return "\n\n".join(lines)


def _format_video(video_data) -> str:
    if not isinstance(video_data, dict) or "error" in video_data:
        return "영상 분석 데이터가 없습니다."
    return f"""- 어깨 이탈 횟수: {video_data.get('shoulder_tilt_count', 0)}회 (자세 불안정 지표)
- 시선 이탈 횟수: {video_data.get('gaze_off_center_count', 0)}회 (시선 불안정 지표)
- 평균 미소 점수: {video_data.get('average_smile_score', 0)}점 (0~100, 높을수록 밝음)
- 분당 눈 깜빡임: {video_data.get('average_blink_count', 0)}회 (높으면 긴장 신호)"""


def get_comprehensive_feedback(qa_list, video_data: dict | None = None) -> str:
    if isinstance(qa_list, dict):
        qa_list = [qa_list]
    if not isinstance(qa_list, list):
        raise ValueError("질문/답변 데이터 형식이 올바르지 않습니다.")

    prompt = f"""당신은 전문적인 AI 면접 코치입니다.
지원자의 [답변 내용]과 AI 가 분석한 [행동 데이터]를 모두 고려해 종합 피드백을 주세요.

=== [1. 행동 분석 결과] ===
{_format_video(video_data)}

=== [2. 면접 질의응답] ===
{_format_qa(qa_list)}

=== [작성 가이드] ===
1. **논리성 및 적합성** — 답변이 질문 의도를 파악했는지, 구조가 잡혀 있는지
2. **비언어적 태도** — 위 행동 데이터를 근거로 자세·시선·표정에 대해 구체적으로
3. **발견된 강점** — 가장 돋보이는 점 한 가지
4. **개선 제안** — 다음 면접에서 바로 적용할 수 있는 팁 1~2가지

말투는 지원자에게 직접 말하듯 부드러운 '해요체'를 사용하세요."""

    return _generate(prompt)
