 🎤 Real-Time Speech-to-Text Transcription with Whisper

This project implements a **real-time speech-to-text system** using OpenAI's Whisper model. It records audio from your microphone, transcribes it live, and prints the text to your terminal.

## 📌 Features

-  Microphone-based real-time audio capture
-  Transcription using OpenAI Whisper (`tiny`, `base`, etc.)
-  Volume detection for filtering silence
-  Multithreaded architecture for parallel recording and transcribing
-  Microphone testing before starting transcription

```bash
- pip install torch --index-url https://download.pytorch.org/whl/cpu
- pip install openai-whisper pyaudio numpy
- pip install scipy
- whisper-env-311\Scripts\activate
- python realtime_whisper.py
