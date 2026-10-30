#!/usr/bin/env python3
"""AI 410 marketing video: adds the synthetic voiceover and writes matching captions.

Speaks each line in narration.json with the Kokoro neural voice (runs locally, Apache 2.0),
places the lines inside their scenes, mixes them onto the silent video, and writes
captions.vtt from the real timings.

Setup, once:  python3 -m venv ~/.cache/ai410-kokoro-venv
              ~/.cache/ai410-kokoro-venv/bin/pip install kokoro soundfile 'misaki[en]'
              Python 3.12. Keep the venv on a short path: espeak-ng fails on long data paths.
              The first run fetches the model from hexgrad/Kokoro-82M, about 330 MB.
Usage:        ~/.cache/ai410-kokoro-venv/bin/python add-voiceover.py <silent.mp4> <out.mp4> [--voice af_heart] [--speed 1.0]
"""
import argparse, json, re, subprocess, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

def spoken(text):
    # Fixes for how the voice reads these names; captions keep the real text. [word](/.../) pins
    # the phonemes: spelling AI as "A I" isn't enough, because the voice can read "A" as the article.
    text = text.replace("AI 410", "AI four ten").replace("BS in AI", "B S in AI")
    text = text.replace("Codespaces", "Code spaces")
    text = re.sub(r"\bAI's\b", "[AI's](/ˈAˌIz/)", text)
    return re.sub(r"\bAI\b(?!')", "[AI](/ˈAˌI/)", text)

def duration(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "default=nw=1:nk=1", str(path)], text=True)
    return float(out)

def stamp(t):
    h, m = int(t // 3600), int(t % 3600 // 60)
    return f"{h:02d}:{m:02d}:{t % 60:06.3f}"

ap = argparse.ArgumentParser()
ap.add_argument("silent"); ap.add_argument("out")
ap.add_argument("--voice"); ap.add_argument("--speed", type=float)
args = ap.parse_args()
cfg = json.loads((HERE / "narration.json").read_text(encoding="utf-8"))
voice, speed = args.voice or cfg["voice"], args.speed or cfg["speed"]

import numpy as np, soundfile as sf
from kokoro import KPipeline
pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")   # "a" = American English

def speak(text, path):
    audio = [np.asarray(chunk) for _, _, chunk in pipeline(text, voice=voice, speed=speed)]
    sf.write(path, np.concatenate(audio), 24000)

tmp = Path(tempfile.mkdtemp())
inputs, filters, cues, over = [], [], [], False
for s_no, scene in enumerate(cfg["scenes"], 1):
    t = scene["start"] + cfg["lead_in"]
    for l_no, line in enumerate(scene["lines"], 1):
        wav = tmp / f"{s_no}-{l_no}.wav"
        speak(spoken(line), wav)
        d = duration(wav)
        k = len(cues)
        inputs += ["-i", str(wav)]
        filters.append(f"[{k + 1}:a]aresample=48000,adelay={int(t * 1000)}:all=1[a{k}]")
        cues.append((t, t + d, line))
        t += d + cfg["gap"]
    spare = scene["start"] + scene["length"] - (t - cfg["gap"])
    print(f"scene {s_no}: {spare:+.1f}s spare")
    over = over or spare < 0.3

n = len(cues)
mix = "".join(f"[a{k}]" for k in range(n)) + f"amix=inputs={n}:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,apad[aud]"
subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-i", args.silent, *inputs,
                       "-filter_complex", ";".join(filters) + ";" + mix,
                       "-map", "0:v", "-map", "[aud]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                       "-shortest", "-movflags", "+faststart", args.out])

vtt = ["WEBVTT", ""]
for i, (a, b, line) in enumerate(cues, 1):
    vtt += [str(i), f"{stamp(a)} --> {stamp(b)}", line, ""]
(HERE / "captions.vtt").write_text("\n".join(vtt), encoding="utf-8")
print(f"voice: {voice}, {n} lines, final length {duration(args.out):.1f}s")
if over:
    print("WARNING: a scene has under 0.3s spare; shorten its lines or raise the rate.")
