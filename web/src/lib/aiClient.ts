/**
 * AI 서버(FastAPI)와 통신하는 단일 창구.
 *
 * 서버는 일부 요청 본문을 한글 alias 로 받는다(main.py 의 Field alias).
 * 그 매핑을 여기 가둬서, 페이지 코드는 평범한 영문 키만 다루게 한다.
 */

const BASE =
  process.env.NEXT_PUBLIC_AI_BASE_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

/** 서버가 누적해서 돌려주는 실시간 지표. state_manager 의 형태와 같다. */
export type LiveMetrics = {
  video: {
    is_running: boolean;
    shoulder_angle: number;
    gaze_score: number;
    smile_score: number;
    shoulder_tilt_count: number;
    gaze_off_center_count: number;
    blink_count: number;
    analyzed_frames: number;
  };
  audio: {
    is_running: boolean;
    transcription: string;
    speaking_speed_epm: number;
    filler_word_count: number;
    average_volume: number;
    average_pitch_hz: number;
  };
};

export type QAPair = { question: string; answer: string };

/** /api/final-feedback 이 요구하는 4개 지표. */
export type VideoSummary = {
  shoulder_tilt_count: number;
  gaze_off_center_count: number;
  average_smile_score: number;
  average_blink_count: number;
};

export class AiServerError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    // fetch 자체가 실패하면 서버가 꺼져 있거나 CORS 가 막힌 것이다.
    throw new AiServerError(
      "AI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요.",
    );
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new AiServerError(
      `AI 서버 오류 (${res.status})${detail ? `: ${detail.slice(0, 200)}` : ""}`,
    );
  }
  return res.json() as Promise<T>;
}

/** 자소서 → 예상 질문 목록. */
export async function generateQuestions(resume: string): Promise<string[]> {
  const data = await request<{ questions: string[] }>(
    "/api/generate-questions",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 자소서: resume }),
    },
  );
  return data.questions;
}

/**
 * 브라우저가 캡처한 프레임 1장을 보내고 갱신된 누적 지표를 받는다.
 * 서버가 카메라를 직접 열지 않으므로 배포 환경에서도 그대로 동작한다.
 */
export async function analyzeFrame(blob: Blob): Promise<LiveMetrics> {
  const form = new FormData();
  form.append("file", blob, "frame.jpg");
  return request<LiveMetrics>("/api/analyze-frame", {
    method: "POST",
    body: form,
  });
}

/** 새 면접을 시작할 때 서버의 누적 지표를 비운다. */
export async function resetSession(): Promise<void> {
  await request<{ status: string }>("/api/session/reset", { method: "POST" });
}

/** 답변 + 행동 지표 → Gemini 종합 코칭 리포트. */
export async function getFinalFeedback(
  qaList: QAPair[],
  video: VideoSummary,
): Promise<string> {
  const data = await request<{ final_feedback: string }>(
    "/api/final-feedback",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        "질문&대답 리스트": qaList.map((qa) => ({
          예상질문: qa.question,
          대답: qa.answer,
        })),
        비디오_분석_결과: video,
      }),
    },
  );
  return data.final_feedback;
}

/**
 * 누적 지표를 리포트 요청용 4개 값으로 줄인다.
 *
 * 서버의 average_smile_score / average_blink_count 는 FrameAnalyzer.summary() 가
 * 계산하지만, 프레임 단위 응답에는 순간값만 담긴다. 그래서 프론트가 관측한
 * 마지막 값을 그대로 쓴다.
 */
export function toVideoSummary(m: LiveMetrics): VideoSummary {
  return {
    shoulder_tilt_count: m.video.shoulder_tilt_count,
    gaze_off_center_count: m.video.gaze_off_center_count,
    average_smile_score: m.video.smile_score,
    average_blink_count: m.video.blink_count,
  };
}
