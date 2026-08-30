"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { AiServerError, generateQuestions } from "@/lib/aiClient";
import { DEFAULT_QUESTIONS, saveQuestions } from "@/lib/session";

export default function Home() {
  const router = useRouter();
  const [resume, setResume] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /** 자소서 없이 기본 질문으로 시작. LLM 을 호출하지 않는다. */
  function startWithDefaults() {
    saveQuestions(DEFAULT_QUESTIONS);
    router.push("/session");
  }

  async function startWithResume() {
    setLoading(true);
    setError(null);
    try {
      const questions = await generateQuestions(resume.trim());
      saveQuestions(questions.length ? questions : DEFAULT_QUESTIONS);
      router.push("/session");
    } catch (e) {
      setError(
        e instanceof AiServerError
          ? e.message
          : "질문 생성에 실패했습니다. 기본 질문으로 시작할 수 있습니다.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>YouCanDoIt — AI 면접 코치</h1>
      <p>
        답변하는 동안 <strong>시선·자세·표정</strong>을 분석해 면접 태도를 수치로
        보여줍니다. 카메라 분석은 브라우저와 분석 서버 안에서만 이뤄집니다.
      </p>

      <div className="card">
        <h2>바로 시작</h2>
        <p style={{ marginBottom: 20 }}>
          기본 질문 5개로 시작합니다. 웹캠 평가 기능만 확인하려면 이쪽이
          빠릅니다.
        </p>
        <button onClick={startWithDefaults} disabled={loading}>
          기본 질문으로 시작
        </button>
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h2>
          자기소개서로 맞춤 질문 만들기{" "}
          <span className="badge">AI 사용</span>
        </h2>
        <p style={{ marginBottom: 16 }}>
          내용을 붙여넣으면 Gemini 가 예상 질문을 생성합니다. 이 단계에서만 AI
          호출이 일어납니다.
        </p>
        <textarea
          value={resume}
          onChange={(e) => setResume(e.target.value)}
          placeholder="지원 직무와 경험이 드러나도록 붙여넣어 주세요."
          disabled={loading}
        />
        <div
          style={{
            marginTop: 12,
            display: "flex",
            gap: 10,
            alignItems: "center",
          }}
        >
          <button
            className="secondary"
            onClick={startWithResume}
            disabled={resume.trim().length < 30 || loading}
          >
            {loading ? "질문을 만드는 중…" : "맞춤 질문 생성 후 시작"}
          </button>
          <span className="muted">
            {resume.trim().length < 30
              ? `${resume.trim().length}/30자`
              : `${resume.trim().length}자`}
          </span>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      <p className="muted" style={{ marginTop: 32 }}>
        카메라 권한이 필요합니다. 영상은 분석용 프레임만 서버로 전송하며 녹화본을
        저장하지 않습니다.
      </p>
    </main>
  );
}
