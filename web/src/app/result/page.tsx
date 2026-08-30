"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { AiServerError, getFinalFeedback } from "@/lib/aiClient";
import { loadResult, saveResult, type InterviewResult } from "@/lib/session";

export default function ResultPage() {
  const [result, setResult] = useState<InterviewResult | null>(null);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setResult(loadResult());
    setReady(true);
  }, []);

  async function makeReport() {
    if (!result) return;
    setLoading(true);
    setError(null);
    try {
      const feedback = await getFinalFeedback(result.qaList, result.video);
      const next = { ...result, feedback };
      setResult(next);
      saveResult(next);
    } catch (e) {
      setError(
        e instanceof AiServerError
          ? e.message
          : "리포트 생성에 실패했습니다. 잠시 후 다시 시도해 주세요.",
      );
    } finally {
      setLoading(false);
    }
  }

  if (!ready) return <main />;

  if (!result) {
    return (
      <main>
        <h1>결과가 없습니다</h1>
        <p>면접을 먼저 진행해 주세요.</p>
        <Link href="/">
          <button>처음으로</button>
        </Link>
      </main>
    );
  }

  const v = result.video;

  return (
    <main>
      <h1>면접 결과</h1>
      <p>{new Date(result.finishedAt).toLocaleString("ko-KR")}</p>

      <div className="card">
        <h2>행동 지표</h2>
        <div className="summary-grid">
          <Stat label="시선 이탈" value={`${v.gaze_off_center_count}회`} />
          <Stat label="자세 기울어짐" value={`${v.shoulder_tilt_count}회`} />
          <Stat label="미소 점수" value={v.average_smile_score.toFixed(1)} />
          <Stat label="눈 깜빡임" value={`${v.average_blink_count}회`} />
        </div>
        <p className="muted" style={{ margin: "16px 0 0" }}>
          이탈·기울어짐은 상태가 이어지는 동안 계속 세지 않고, 벗어나기 시작한
          시점에만 한 번 집계합니다.
        </p>
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h2>
          AI 코칭 리포트 <span className="badge">AI 사용</span>
        </h2>

        {result.feedback ? (
          <pre className="report">{result.feedback}</pre>
        ) : (
          <>
            <p>
              위 지표와 답변 내용을 함께 분석해 논리성·태도·강점·개선점을
              정리합니다. 이 버튼을 누를 때만 AI 를 호출합니다.
            </p>
            <button onClick={makeReport} disabled={loading}>
              {loading ? "리포트를 만드는 중…" : "리포트 생성"}
            </button>
          </>
        )}
        {error && <div className="error">{error}</div>}
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h2>질문과 답변</h2>
        {result.qaList.map((qa, i) => (
          <div key={i} className="qa">
            <strong>Q{i + 1}. {qa.question}</strong>
            <p style={{ margin: "6px 0 0" }}>{qa.answer}</p>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 24 }}>
        <Link href="/">
          <button className="secondary">다시 면접 보기</button>
        </Link>
      </div>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span className="muted">{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
