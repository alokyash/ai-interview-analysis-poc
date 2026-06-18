# AI interview platform - transcription + analysis POC

Free stack, no payment required:
- **Transcription**: Faster-Whisper (runs locally, open source, no API key)
- **Analysis (summary, STAR, scoring)**: Llama 3.3 70B, hosted for free via [Groq](https://console.groq.com)
  (running a 70B model locally needs serious GPU memory, so Groq's free-tier API is the
  practical way to use Llama 3.3 for a POC)

## 1. Install system dependency

Faster-Whisper needs `ffmpeg` to read most audio/video formats.

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows: download from https://ffmpeg.org/download.html and add to PATH
```

## 2. Set up the Python environment

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Get a free Groq API key

1. Sign up at https://console.groq.com (free)
2. Create a key at https://console.groq.com/keys
3. Set it as an environment variable:

```bash
export GROQ_API_KEY=your_key_here       # on Windows (PowerShell): $env:GROQ_API_KEY="your_key_here"
```

## 4. Get a test audio file (since we don't have real interview data yet)

Record yourself answering a sample interview question on your phone (30-90 seconds is enough),
and export/airdrop it as an mp3 or m4a file. This is exactly the "prepare our own data" step
from today's conversation - this self-recorded clip is your first test sample.

## 5. Run it

```bash
python main.py path/to/your_answer.mp3 \
  --job-role "DevOps Engineer" \
  --model-size tiny
```

`The system automatically:

- Infers the interview question from the candidate's answer
- Generates an ideal/reference answer based on the selected job role
- Evaluates the candidate response
- Produces detailed feedback and scoring

## What you get back

A `report.json` like:

```json
{
  "analysis": {
    "job_role": "DevOps Engineer",
    "inferred_question": "...",
    "question_inference_confidence": "Medium",
    "ideal_answer": "...",
    "candidate_summary": "...",
    "comparison_summary": "...",
    "detected_topics": [...],
    "evaluation_categories": {
      "technical_knowledge": 60,
      "conceptual_understanding": 50,
      "communication_skills": 70,
      "problem_solving_ability": 40,
      "confidence_level": "Medium",
      "completeness_of_answer": 50
    },
    "overall_score": 56,
    "hiring_recommendation": "Borderline",
    "interview_readiness": "Needs Practice",
    "strengths": [...],
    "weaknesses": [...],
    "improvement_suggestions": [...],
    "suggested_learning_topics": [...],
    "final_interviewer_summary": "...",
    "detailed_feedback": "..."
  }
}
```

## Notes / next steps

- `--model-size small` is the default for Faster-Whisper (good speed/accuracy balance on CPU).
  Use `tiny` for faster iteration while testing, `medium` or `large-v3` for better accuracy once
  the pipeline works end-to-end.
- This is intentionally a single-file, single-answer POC to prove the pipeline. Once it works,
  the natural next steps (matching the roadmap) are: looping it over multiple Q&A pairs per
  session, adding speaker diarization, and wrapping it behind the FastAPI service.
- No data was sent anywhere paid - Faster-Whisper runs entirely on your machine, and Groq's
  free tier has no cost for this volume of usage.
## Evaluation Pipeline

Audio / Video
    ↓
Faster-Whisper
    ↓
Transcript
    ↓
Llama 3.3 70B
    ↓
Infer Interview Question
    ↓
Generate Ideal Answer
    ↓
Compare Candidate Answer
    ↓
Generate Scores and Feedback
    ↓
report.json

## Model Selection

### Faster-Whisper
Used for speech-to-text transcription.

Reasons:
- Open source
- Runs locally
- No API cost
- Good accuracy and speed

### Llama 3.3 70B (Groq)
Used for:
- Question inference
- Ideal answer generation
- Candidate evaluation
- Feedback generation

Reasons:
- Strong reasoning capabilities
- Fast inference through Groq
- Structured JSON output
- Suitable for role-based interview evaluation

The architecture is model-agnostic and can be extended to GPT-4o, Claude, Gemini, DeepSeek, or other LLMs.