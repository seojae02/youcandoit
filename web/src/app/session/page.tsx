"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  analyzeFrame,
  resetSession,
  toVideoSummary,
  type LiveMetrics,
  type QAPair,
} from "@/lib/aiClient";
import { loadQuestions, saveResult, DEFAULT_QUESTIONS } from "@/lib/session";
import { isSpeechSupported, startDictation } from "@/lib/speech";

/** 프레임 전송 주기. 분석이 CPU 를 쓰므로 1초면 충분하다. */
const FRAME_INTERVAL_MS = 1000;
const FRAME_WIDTH = 640;

export default function SessionPage() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  /** 전송이 밀렸을 때 요청이 겹치지 않도록 하는 빗장. */
  const inFlight = useRef(false);
  const answersRef = useRef<string[]>([]);

  const [questions, setQuestions] = useState<string[]>([]);
  const [index, setIndex] = useState(0);
  const [metrics, setMetrics] = useState<LiveMetrics | null>(null);
  const [camError, setCamError] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [interim, setInterim] = useState("");
  const [finishing, setFinishing] = useState(false);

  // ── 질문 로드 ────────────────────────────────────────────────
  useEffect(() => {
    setQuestions(loadQuestions() ?? DEFAULT_QUESTIONS);
  }, []);

  // ── 카메라 시작 ──────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        await resetSession().catch(() => {
          // 서버가 꺼져 있어도 카메라는 띄운다. 분석만 실패하게 둔다.
        });
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
          audio: false, // 마이크는 받아쓰기 쪽에서 따로 연다
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
      } catch {
        if (!cancelled) {
          setCamError(
            "카메라를 사용할 수 없습니다. 브라우저 주소창의 권한 설정을 확인해 주세요.",
          );
        }
      }
    })();

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []);

  // ── 프레임 캡처 → 서버 분석 ─────────────────────────────────
  useEffect(() => {
    if (camError) return;

    const timer = setInterval(async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2 || inFlight.current) return;

      const ratio = video.videoHeight / video.videoWidth || 0.5625;
      canvas.width = FRAME_WIDTH;
      canvas.height = Math.round(FRAME_WIDTH * ratio);
      canvas
        .getContext("2d")
        ?.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.7),
      );
      if (!blob) return;

      inFlight.current = true;
      try {
        setMetrics(await analyzeFrame(blob));
        setAnalysisError(null);
      } catch {
        setAnalysisError("분석 서버에 연결할 수 없습니다.");
      } finally {
        inFlight.current = false;
      }
    }, FRAME_INTERVAL_MS);

    return () => clearInterval(timer);
  }, [camError]);

  // ── 경과 시간 ───────────────────────────────────────────────
  useEffect(() => {
    const timer = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  // ── 받아쓰기 (선택 기능) ────────────────────────────────────
  useEffect(() => {
    if (!isSpeechSupported()) return;
    return startDictation(
      (text) => {
        answersRef.current[index] = `${answersRef.current[index] ?? ""} ${text}`.trim();
        setInterim("");
      },
      (text) => setInterim(text),
    );
  }, [index]);

  const finish = useCallback(async () => {
    setFinishing(true);
    streamRef.current?.getTracks().forEach((t) => t.stop());

    const qaList: QAPair[] = questions.map((q, i) => ({
      question: q,
      answer: answersRef.current[i]?.trim() || "(답변이 기록되지 않았습니다)",
    }));

    saveResult({
      feedback: null, // 리포트는 결과 화면에서 직접 생성한다
      video: metrics
        ? toVideoSummary(metrics)
        : {
            shoulder_tilt_count: 0,
            gaze_off_center_count: 0,
            average_smile_score: 0,
            average_blink_count: 0,
          },
      qaList,
      finishedAt: new Date().toISOString(),
    });
    router.push("/result");
  }, [questions, metrics, router]);

  const v = metrics?.video;
  const mmss = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(
    elapsed % 60,
  ).padStart(2, "0")}`;

  return (
    <main>
      <div className="session-head">
        <div>
          <h1 style={{ fontSize: 20, marginBottom: 4 }}>
            질문 {index + 1} / {questions.length}
          </h1>
          <span className="muted">{mmss}</span>
        </div>
        <button className="secondary" onClick={finish} disabled={finishing}>
          {finishing ? "정리 중…" : "면접 종료"}
        </button>
      </div>

      <div className="question-box">
        {questions[index] ?? "질문을 불러오는 중…"}
      </div>

      {camError && <div className="error">{camError}</div>}
      {!camError && analysisError && (
        <div className="error">{analysisError}</div>
      )}

      <div className="session-grid">
        <div className="video-wrap">
          <video ref={videoRef} muted playsInline />
          <canvas ref={canvasRef} style={{ display: "none" }} />
        </div>

        <div className="card">
          <h2>실시간 지표</h2>
          <Metric
            label="시선 이탈"
            value={v ? `${v.gaze_off_center_count}회` : "—"}
            tone={!v ? "idle" : v.gaze_off_center_count > 8 ? "bad" : "good"}
          />
          <Metric
            label="자세 기울어짐"
            value={v ? `${v.shoulder_tilt_count}회` : "—"}
            tone={!v ? "idle" : v.shoulder_tilt_count > 8 ? "bad" : "good"}
          />
          <Metric
            label="미소 점수"
            value={v ? v.smile_score.toFixed(1) : "—"}
            tone={!v ? "idle" : v.smile_score < 10 ? "warn" : "good"}
          />
          <Metric
            label="눈 깜빡임"
            value={v ? `${v.blink_count}회` : "—"}
            tone="idle"
          />
          <Metric
            label="분석 프레임"
            value={v ? `${v.analyzed_frames}` : "—"}
            tone="idle"
          />
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h2>답변 받아쓰기</h2>
        {isSpeechSupported() ? (
          <p style={{ margin: 0 }}>
            {answersRef.current[index] || interim ? (
              <>
                {answersRef.current[index]}{" "}
                <span style={{ opacity: 0.5 }}>{interim}</span>
              </>
            ) : (
              <span className="muted">말씀하시면 여기에 기록됩니다.</span>
            )}
          </p>
        ) : (
          <p className="muted" style={{ margin: 0 }}>
            이 브라우저는 음성 인식을 지원하지 않습니다. Chrome 이나 Edge 에서
            사용해 주세요. 웹캠 평가는 그대로 동작합니다.
          </p>
        )}
      </div>

      <div style={{ marginTop: 20, display: "flex", gap: 10 }}>
        <button
          onClick={() => setIndex((i) => i + 1)}
          disabled={index >= questions.length - 1 || finishing}
        >
          다음 질문
        </button>
      </div>
    </main>
  );
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "good" | "warn" | "bad" | "idle";
}) {
  return (
    <div className="metric">
      <span className="muted">{label}</span>
      <strong data-tone={tone}>{value}</strong>
    </div>
  );
}
