/**
 * 페이지 간에 면접 데이터를 넘기는 통로.
 *
 * 서버에 세션을 저장하지 않는 범위이므로 sessionStorage 를 쓴다.
 * 탭을 닫으면 사라지는 것이 의도된 동작이다.
 */

import type { QAPair, VideoSummary } from "./aiClient";

const QUESTIONS = "ycd.questions";
const RESULT = "ycd.result";

export type InterviewResult = {
  /** 종합 리포트는 결과 화면에서 사용자가 눌러야 생성된다(토큰 소비). */
  feedback: string | null;
  video: VideoSummary;
  qaList: QAPair[];
  finishedAt: string;
};

/** LLM 을 쓰지 않고 바로 시작할 때 쓰는 질문. */
export const DEFAULT_QUESTIONS = [
  "자기소개 부탁드립니다.",
  "본인의 장점과 보완하고 싶은 점을 말씀해 주세요.",
  "지원한 직무에 왜 적합하다고 생각하나요?",
  "협업 중 갈등이 생겼을 때 어떻게 대처했나요?",
  "앞으로 어떤 개발자가 되고 싶은지 말씀해 주세요.",
];

function read<T>(key: string): T | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function write(key: string, value: unknown): void {
  if (typeof window === "undefined") return;
  try {
    sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    // 사파리 프라이빗 모드 등에서 쓰기가 막힐 수 있다. 조용히 넘긴다.
  }
}

export const saveQuestions = (q: string[]) => write(QUESTIONS, q);
export const loadQuestions = () => read<string[]>(QUESTIONS);

export const saveResult = (r: InterviewResult) => write(RESULT, r);
export const loadResult = () => read<InterviewResult>(RESULT);
