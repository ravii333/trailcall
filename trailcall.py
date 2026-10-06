"""
TrailCall — offline bird-call identification for the trail.

Records or analyzes bird audio entirely on-device using BirdNET (open-weight
acoustic species classifier, no API keys, no network calls at inference time).
Optionally turns the raw detections into a one-line trail note using a local
Gemma model via Ollama (also no API key, also fully local).

Usage:
    python trailcall.py record --seconds 15 --out recordings/clip.wav
    python trailcall.py identify recordings/clip.wav --lat 35.42 --lon -120.74
    python trailcall.py identify recordings/clip.wav --trail-note
"""

import argparse
import sys
from datetime import datetime


def record(seconds: float, out_path: str, samplerate: int = 48000) -> None:
    import sounddevice as sd
    import soundfile as sf

    print(f"Recording {seconds}s of audio... point the mic outside.")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(out_path, audio, samplerate)
    print(f"Saved to {out_path}")


def trail_note(detections: list, model: str, host: str) -> str:
    """Turn raw BirdNET detections into a one-line trail note using a local
    Gemma model served by Ollama. Runs fully on-device, no API key needed."""
    import requests

    species = sorted({d["common_name"] for d in detections})
    prompt = (
        "You are a birder's trail companion. In one short, friendly sentence, "
        "summarize what was just heard. Species detected: "
        f"{', '.join(species)}. Do not add species that weren't listed."
    )

    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return (
            "[trail note unavailable: Ollama isn't running. Install it from "
            "https://ollama.com, then run `ollama pull "
            f"{model}` and try again.]"
        )
    except requests.exceptions.RequestException as e:
        return f"[trail note failed: {e}]"

    return resp.json().get("response", "").strip()


def identify(
    path: str,
    lat: float,
    lon: float,
    min_conf: float,
    use_trail_note: bool = False,
    model: str = "gemma4:e2b",
    ollama_host: str = "http://localhost:11434",
) -> None:
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer

    analyzer = Analyzer()
    recording = Recording(
        analyzer,
        path,
        lat=lat,
        lon=lon,
        date=datetime.now(),
        min_conf=min_conf,
    )
    recording.analyze()

    if not recording.detections:
        print("No species detected above the confidence threshold.")
        return

    print(f"Detected {len(recording.detections)} call(s):\n")
    for d in recording.detections:
        print(
            f"  {d['common_name']} ({d['scientific_name']}) "
            f"- confidence {d['confidence']:.2f}, "
            f"{d['start_time']:.1f}s-{d['end_time']:.1f}s"
        )

    if use_trail_note:
        print("\nTrail note:")
        print(" ", trail_note(recording.detections, model, ollama_host))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline bird-call identifier")
    sub = parser.add_subparsers(dest="command", required=True)

    rec_p = sub.add_parser("record", help="Record audio from the microphone")
    rec_p.add_argument("--seconds", type=float, default=15.0)
    rec_p.add_argument("--out", default="recordings/clip.wav")

    id_p = sub.add_parser("identify", help="Identify bird calls in an audio file")
    id_p.add_argument("path")
    id_p.add_argument("--lat", type=float, default=0.0, help="Latitude (improves accuracy)")
    id_p.add_argument("--lon", type=float, default=0.0, help="Longitude (improves accuracy)")
    id_p.add_argument("--min-conf", type=float, default=0.25, dest="min_conf")
    id_p.add_argument(
        "--trail-note",
        action="store_true",
        dest="trail_note",
        help="Summarize detections into one spoken-style sentence using a local Gemma model (requires Ollama)",
    )
    id_p.add_argument(
        "--model",
        default="gemma4:e2b",
        help="Ollama model to use for the trail note (default: gemma4:e2b)",
    )
    id_p.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        dest="ollama_host",
        help="Ollama server URL (default: http://localhost:11434)",
    )

    args = parser.parse_args()

    if args.command == "record":
        record(args.seconds, args.out)
    elif args.command == "identify":
        identify(
            args.path,
            args.lat,
            args.lon,
            args.min_conf,
            use_trail_note=args.trail_note,
            model=args.model,
            ollama_host=args.ollama_host,
        )


if __name__ == "__main__":
    sys.exit(main())
