"""
POC entry point: audio file + job role in -> transcript + structured
evaluation report out.

Usage:
    python main.py path/to/answer.mp3 --job-role "Python Developer"
"""

import argparse
import json

from transcribe import transcribe_audio
from analyze import analyze_transcript


def main():
    parser = argparse.ArgumentParser(description="AI Interview Platform - Transcription + Analysis POC")
    parser.add_argument("audio_path", help="Path to interview audio/video file (mp3, wav, m4a, mp4, etc.)")
    parser.add_argument("--job-role", required=True, help='Job role being interviewed for, e.g. "DevOps Engineer"')
    parser.add_argument("--model-size", default="small", help="Faster-Whisper model size (tiny/base/small/medium/large-v3)")
    parser.add_argument("--out", default="report.json", help="Output report path")
    args = parser.parse_args()

    print(f"[1/3] Transcribing {args.audio_path} ...")
    transcript = transcribe_audio(args.audio_path, model_size=args.model_size)
    preview = transcript["text"][:400000] + ("..." if len(transcript["text"]) > 400000 else "")
    print(f"      Detected language: {transcript['language']}")
    print(f"      Transcript preview: {preview}\n")

    print(f"[2/3] Inferring question, generating ideal answer, and scoring for role '{args.job_role}' ...")
    analysis = analyze_transcript(
        transcript_text=transcript["text"],
        job_role=args.job_role,
    )

    report = {
        "audio_file": args.audio_path,
        "job_role": args.job_role,
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
