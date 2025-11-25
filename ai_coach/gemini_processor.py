import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# 1. 환경 변수 로드 (key.env 파일 필요)
load_dotenv(dotenv_path="key.env")

# 2. API 키 설정
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ 경고: GEMINI_API_KEY가 설정되지 않았습니다.")
    else:
        genai.configure(api_key=api_key)
except Exception as e:
    print(f"오류: API 설정 중 문제 발생 {e}")

# 3. 모델 로드 (최신 모델 우선 시도)
try:
    model = genai.GenerativeModel('gemini-2.5-pro')
except:
    try:
        model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        model = None

def generate_questions_from_resume(resume_text: str) -> list:
    """
    [API 1] 자소서를 받아 핵심 면접 질문 3개를 리스트로 반환합니다.
    """
    if not model: return ["AI 모델이 로드되지 않았습니다."]

    prompt = f"""
    당신은 면접관입니다. 아래 자기소개서를 읽고, 지원자의 역량과 경험을 검증할 수 있는 
    '핵심 면접 질문' 3가지를 뽑아주세요.
    
    [조건]
    1. 질문은 한국어로 작성하세요.
    2. 서론이나 번호 매기기 없이, 오직 Python 리스트 문자열 형식으로만 답하세요.
    3. 예시: ["질문 내용 1", "질문 내용 2", "질문 내용 3"]

    --- 자기소개서 ---
    {resume_text}
    ---
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # 마크다운 코드 블록 제거
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```python", "").replace("```", "").strip()
        
        # 문자열 -> 리스트 변환
        import ast
        questions = ast.literal_eval(text)
        
        if isinstance(questions, list):
            return questions
        return ["형식 오류: 질문 리스트를 생성하지 못했습니다."]
            
    except Exception as e:
        print(f"❌ Gemini 질문 생성 오류: {e}")
        return ["질문 생성 중 오류가 발생했습니다."]

def get_comprehensive_feedback(qa_list, video_data: dict = None) -> str:
    """
    [API 3] Q&A 리스트와 비디오 데이터를 받아 종합 피드백을 생성합니다.
    """
    if not model: return "AI 모델이 로드되지 않았습니다."

    # 🔍 [데이터 전처리] 리스트인지 딕셔너리인지 확인하여 리스트로 통일
    if isinstance(qa_list, dict):
        qa_list = [qa_list]
    
    if not isinstance(qa_list, list):
        return "오류: 질문/답변 데이터 형식이 올바르지 않습니다."

    # 1. Q&A 텍스트 변환
    qa_text_buffer = []
    for idx, item in enumerate(qa_list, 1):
        # item이 딕셔너리가 아닌 경우(문자열 등) 방어
        if not isinstance(item, dict) and not hasattr(item, 'question'):
            continue

        # Pydantic 객체(.attr) 또는 딕셔너리['key'] 모두 처리
        if hasattr(item, 'question'):
            q = item.question
            a = item.answer
        else:
            # 딕셔너리일 경우 다양한 키 이름 대응 (예상질문, 질문, question 등)
            q = item.get('예상질문') or item.get('question') or item.get('질문') or ''
            a = item.get('대답') or item.get('answer') or item.get('답변') or ''
            
        qa_text_buffer.append(f"Q{idx}. {q}\nA{idx}. {a}")
    
    qa_text = "\n\n".join(qa_text_buffer)

    # 2. 비디오 데이터 텍스트 변환
    video_text = "영상 분석 데이터가 없습니다."
    if video_data and isinstance(video_data, dict) and "error" not in video_data:
        video_text = f"""
        - 어깨 이탈 횟수: {video_data.get('shoulder_tilt_count', 0)}회 (자세 불안정 지표)
        - 시선 이탈 횟수: {video_data.get('gaze_off_center_count', 0)}회 (시선 불안정 지표)
        - 평균 미소 점수: {video_data.get('average_smile_score', 0)}점 (0~100점, 높을수록 긍정적)
        - 분당 눈 깜빡임: {video_data.get('average_blink_count', 0)}회 (높으면 긴장됨)
        """

    print(f"🔍 [DEBUG] Gemini 요청 데이터:\n{video_text}\n{qa_text[:100]}...")

    # 3. 프롬프트 조합
    prompt = f"""
    당신은 전문적인 AI 면접 코치입니다.
    지원자의 [답변 내용]과 AI가 분석한 [비디오 행동 데이터]를 모두 고려하여 종합 피드백을 주세요.

    === [1. 비디오 행동 분석 결과] ===
    {video_text}

    === [2. 면접 질의응답 내용] ===
    {qa_text}

    === [피드백 작성 가이드] ===
    다음 4가지 항목에 맞춰 구체적으로 작성해주세요:
    
    1. **논리성 및 적합성**: 
       - 답변들이 질문의 의도를 잘 파악하고 있는지, 논리적인 구조를 갖췄는지 평가해주세요.
    
    2. **비언어적 태도 피드백 (중요)**: 
       - 위 [비디오 태도 분석 데이터]를 참고하여 자세, 시선, 표정에 대해 구체적으로 조언해주세요.
       - 예: "시선 이탈이 5회로 다소 높습니다. 카메라를 좀 더 응시하세요."
       - 예: "미소 점수가 높아서 인상이 좋습니다."
    
    3. **발견된 강점**: 
       - 답변 전체에서 드러나는 지원자의 직무 역량이나 태도 중 가장 돋보이는 점 1가지를 칭찬해주세요.
    
    4. **종합 개선 제안 (Action Item)**: 
       - 전체적으로 보완이 필요한 부분을 짚어주고, 다음 면접에서 바로 적용할 수 있는 구체적인 팁을 1~2가지 제안해주세요.
    
    *말투는 지원자에게 직접 말하듯이 부드럽고 격려하는 '해요체'를 사용하세요.
    """
    
    try:
        response = model.generate_content(prompt)
        print("✅ 피드백 생성 완료")
        return response.text
    except Exception as e:
        print(f"❌ Gemini 피드백 오류: {e}")
        return "피드백 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."