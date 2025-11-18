import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# 1. 환경 변수 로드
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

# 3. 모델 로드 (Pro 모델 권장)
try:
    model = genai.GenerativeModel('gemini-2.5-pro')
except:
    model = genai.GenerativeModel('gemini-pro')

def get_feedback_on_qa_list(qa_list: list) -> str:
    """
    [API 2] Q&A 리스트 전체를 받아 종합 피드백을 생성합니다.
    qa_list: Pydantic 모델(QAPair)의 리스트 또는 딕셔너리 리스트
    """
    
    # 리스트를 보기 좋은 텍스트 포맷으로 변환
    qa_text_buffer = []
    for idx, item in enumerate(qa_list, 1):
        # Pydantic 객체일 경우 .question, 아닐 경우(dict) ['question'] 접근
        # getattr(obj, name, default)를 사용하여 두 경우 모두 처리
        if hasattr(item, 'question'):
            q = item.question
            a = item.answer
        else:
            q = item.get('question', '') or item.get('예상질문', '')
            a = item.get('answer', '') or item.get('대답', '')
            
        qa_text_buffer.append(f"Q{idx}. {q}\nA{idx}. {a}")
    
    qa_text = "\n\n".join(qa_text_buffer)

    # 종합 피드백을 위한 프롬프트
    prompt = f"""
    당신은 전문적인 면접 코치입니다. 
    지원자가 진행한 아래의 [면접 질의응답 리스트]를 분석하여 종합적인 피드백을 제공해주세요.

    [면접 질의응답 리스트]
    {qa_text}

    [피드백 작성 가이드]
    다음 3가지 항목에 맞춰 구체적으로 작성해주세요:
    
    1. **논리성 및 적합성**: 
       - 답변들이 질문의 의도를 잘 파악하고 있는지, 논리적인 구조를 갖췄는지 평가해주세요.
    
    2. **발견된 강점**: 
       - 답변 전체에서 드러나는 지원자의 직무 역량이나 태도 중 가장 돋보이는 점 1가지를 칭찬해주세요.
    
    3. **개선 제안 (Action Item)**: 
       - 전체적으로 보완이 필요한 부분을 짚어주고, 다음 면접에서 바로 적용할 수 있는 구체적인 팁을 1~2가지 제안해주세요.
    
    *말투는 지원자에게 직접 말하듯이 부드럽고 격려하는 '해요체'를 사용하세요.
    """
    
    print("--- Gemini에게 종합 피드백 요청 중 ---")
    try:
        response = model.generate_content(prompt)
        print("✅ 피드백 생성 완료")
        return response.text
    except Exception as e:
        print(f"❌ Gemini 피드백 오류: {e}")
        return "피드백 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

def generate_questions_from_resume(resume_text: str) -> list:
    """
    [API 1] 자소서 기반 질문 생성
    """
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