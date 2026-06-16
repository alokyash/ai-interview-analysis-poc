"""
POC entry point: audio file in -> transcript + structured analysis report out.

Usage:
    python main.py path/to/answer.mp3
    python main.py path/to/answer.mp3 --question "Tell me about a time you debugged a tricky production issue" \
        --expected "A strong answer identifies the problem, the diagnostic steps taken, the fix, and the outcome/impact." \
        --out report.json
"""

import argparse
import json

from transcribe import transcribe_audio
from analyze import analyze_transcript


def main():
    parser = argparse.ArgumentParser(description="AI Interview Platform - Transcription + Analysis POC")
    parser.add_argument("audio_path", help="Path to interview audio/video file (mp3, wav, m4a, mp4, etc.)")
    parser.add_argument("--question", default=None, help="The interview question asked (optional, improves scoring)")
    parser.add_argument("--expected", default=None, help="Expected/reference answer text (optional, improves scoring)")
    parser.add_argument("--model-size", default="small", help="Faster-Whisper model size (tiny/base/small/medium/large-v3)")
    parser.add_argument("--out", default="report.json", help="Output report path")
    args = parser.parse_args()

    print(f"[1/3] Transcribing {args.audio_path} ...")
    transcript = transcribe_audio(args.audio_path, model_size=args.model_size)
    preview = transcript["text"][:400] + ("..." if len(transcript["text"]) > 400 else "")
    print(f"      Detected language: {transcript['language']}")
    print(f"      Transcript preview: {preview}\n")

    print("[2/3] Running LLM analysis (summary, STAR, scoring) via Llama 3.3 ...")
    analysis = analyze_transcript(
        transcript_text=transcript["text"],
        question=args.question,
        expected_answer=args.expected,
    )

    report = {
        "audio_file": args.audio_path,
        "question": args.question,
        "expected_answer": args.expected,
        "transcript": transcript["text"],
        "segments": transcript["segments"],
        "analysis": analysis,
    }

    print(f"[3/3] Writing full report to {args.out}\n")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("=== Analysis result ===")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
