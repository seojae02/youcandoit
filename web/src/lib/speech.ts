/**
 * Web Speech API 로 답변을 받아쓴다.
 *
 * 표준 라이브러리 타입에 없는 API 라 필요한 만큼만 직접 선언한다.
 * Chrome·Edge 는 webkit 접두사를 쓰고, Firefox 는 아직 지원하지 않는다.
 */

type SpeechRecognitionAlternative = { transcript: string };
type SpeechRecognitionResult = {
  isFinal: boolean;
  0: SpeechRecognitionAlternative;
};
type SpeechRecognitionEventLike = {
  resultIndex: number;
  results: { length: number; [i: number]: SpeechRecognitionResult };
};

interface RecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((e: SpeechRecognitionEventLike) => void) | null;
  onerror: (() => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
}

type RecognitionCtor = new () => RecognitionLike;

function getCtor(): RecognitionCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: RecognitionCtor;
    webkitSpeechRecognition?: RecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export const isSpeechSupported = () => getCtor() !== null;

/**
 * 받아쓰기를 시작한다. 확정된 문장이 나올 때마다 onFinal 이 호출된다.
 * 반환된 함수를 부르면 정지한다.
 */
export function startDictation(
  onFinal: (text: string) => void,
  onInterim?: (text: string) => void,
): () => void {
  const Ctor = getCtor();
  if (!Ctor) return () => {};

  const recognition = new Ctor();
  recognition.lang = "ko-KR";
  recognition.continuous = true;
  recognition.interimResults = true;

  let stopped = false;

  recognition.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i += 1) {
      const result = e.results[i];
      const text = result[0].transcript;
      if (result.isFinal) onFinal(text.trim());
      else interim += text;
    }
    onInterim?.(interim);
  };

  // continuous 를 켜도 브라우저가 침묵 구간에서 임의로 끊는다. 의도적으로 멈춘
  // 것이 아니라면 다시 살려 면접 내내 받아쓰기가 이어지게 한다.
  recognition.onend = () => {
    if (!stopped) {
      try {
        recognition.start();
      } catch {
        // 이미 시작된 상태면 무시
      }
    }
  };
  recognition.onerror = () => {};

  try {
    recognition.start();
  } catch {
    return () => {};
  }

  return () => {
    stopped = true;
    try {
      recognition.stop();
    } catch {
      // 이미 멈춘 상태면 무시
    }
  };
}
