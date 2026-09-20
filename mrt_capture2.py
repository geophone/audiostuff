#!/usr/bin/env python3

from __future__ import annotations

import argparse
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import jax
import numpy as np

from magenta_rt import MagentaRT2Jax
from magenta_rt.audio import Waveform
from magenta_rt.config import MUSICCOCA


# ============================================================
# Recommended defaults
# ============================================================

#
# For music-like conditioning:
#
#   30-40 sec = more focused / specific style
#   50-60 sec = broader / more stable style
#
# 60 seconds is a good general default.
#
DEFAULT_CAPTURE_SECONDS = 60.0

# MusicCoCa uses 10-second audio windows.
MUSICCOCA_WINDOW_SECONDS = 10.0

# Change to mrt2_base if your ROCm GPU is fast enough.
DEFAULT_MODEL = "mrt2_small"

# Generation chunk size.
#
# 25 MRT frames = 1 second.
#
DEFAULT_CHUNK_SECONDS = 8.0

# Start playback after this many generated chunks.
#
# 3 x 8 sec = 24 seconds initial reserve.
#
DEFAULT_PREBUFFER_CHUNKS = 3

# Maximum waiting generated chunks.
DEFAULT_QUEUE_CHUNKS = 12

DEFAULT_TEMPERATURE = 1.3
DEFAULT_TOP_K = 40

DEFAULT_CFG_MUSICCOCA = 3.0
DEFAULT_CFG_NOTES = 1.0
DEFAULT_CFG_DRUMS = 1.0


# ============================================================
# External commands
# ============================================================

def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required command '{name}' not found.\n\n"
            f"Install system audio dependencies with:\n"
            f"    sudo apt install ffmpeg pulseaudio-utils"
        )


def get_default_sink() -> str:
    require_program("pactl")

    sink = subprocess.check_output(
        ["pactl", "get-default-sink"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()

    if not sink:
        raise RuntimeError(
            "Could not determine the default PulseAudio/PipeWire sink."
        )

    return sink


# ============================================================
# Capture default Ubuntu output
# ============================================================

def capture_default_sink(
    filename: Path,
    seconds: float,
) -> None:

    require_program("ffmpeg")

    sink = get_default_sink()
    monitor = f"{sink}.monitor"

    print()
    print("==================================================")
    print("CAPTURE")
    print("==================================================")
    print(f"Default sink : {sink}")
    print(f"Monitor      : {monitor}")
    print(f"Duration     : {seconds:.1f} seconds")
    print(f"Output       : {filename}")
    print()

    if 30 <= seconds <= 60:
        print(
            "Capture duration is in the recommended "
            "30-60 second music-conditioning range."
        )
    elif seconds < 30:
        print(
            "NOTE: For music-like synthesis, 30-60 seconds "
            "usually gives a more stable style embedding."
        )
    else:
        print(
            "NOTE: More than 60 seconds is allowed, but MusicCoCa "
            "will average more sections together, which can blur "
            "distinct musical styles."
        )

    print()
    print("Recording whatever is playing through the default sink...")
    print()

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subprocess.run(
        [
            "ffmpeg",

            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",

            "-f",
            "pulse",

            "-i",
            monitor,

            "-t",
            str(seconds),

            "-ac",
            "2",

            "-ar",
            "48000",

            "-c:a",
            "pcm_s16le",

            str(filename),
        ],
        check=True,
    )

    if not filename.exists():
        raise RuntimeError(
            f"Capture file wasn't created: {filename}"
        )

    if filename.stat().st_size < 1000:
        raise RuntimeError(
            "Captured WAV appears empty."
        )

    print()
    print(
        f"Capture complete: "
        f"{filename.stat().st_size / 1024 / 1024:.1f} MiB"
    )


# ============================================================
# Trim for MusicCoCa
# ============================================================

def trim_for_musiccoca(
    wav: Waveform,
    window_seconds: float = MUSICCOCA_WINDOW_SECONDS,
) -> Waveform:
    """
    Trim an audio prompt to an exact number of MusicCoCa windows.

    MusicCoCa's current configuration uses 10-second windows.

    For example:

        60.02 sec -> 60.00 sec
        57.3 sec  -> 50.00 sec
        38.4 sec  -> 30.00 sec

    This avoids creating a final mostly-zero-padded MusicCoCa window.
    """

    samples_per_window = round(
        wav.sample_rate * window_seconds
    )

    full_windows = (
        wav.num_samples
        // samples_per_window
    )

    if full_windows < 1:
        raise RuntimeError(
            f"Audio prompt must contain at least "
            f"{window_seconds:.0f} seconds."
        )

    usable_samples = (
        full_windows
        * samples_per_window
    )

    original_seconds = wav.seconds

    trimmed = wav[:usable_samples]

    removed_seconds = (
        original_seconds
        - trimmed.seconds
    )

    print()
    print("==================================================")
    print("MUSICCOCA WINDOW TRIMMING")
    print("==================================================")

    print(
        f"Original duration : "
        f"{original_seconds:.3f}s"
    )

    print(
        f"Window size       : "
        f"{window_seconds:.1f}s"
    )

    print(
        f"Full windows      : "
        f"{full_windows}"
    )

    print(
        f"Usable duration   : "
        f"{trimmed.seconds:.3f}s"
    )

    print(
        f"Removed tail      : "
        f"{removed_seconds:.3f}s"
    )

    return trimmed


# ============================================================
# MRT Waveform -> raw PulseAudio PCM
# ============================================================

def waveform_to_pcm_s16le(
    wav: Waveform,
) -> bytes:

    samples = np.asarray(
        wav.samples,
        dtype=np.float32,
    )

    if wav.sample_rate != 48000:
        raise RuntimeError(
            f"Expected MRT output at 48000 Hz, "
            f"got {wav.sample_rate} Hz"
        )

    if samples.ndim == 1:
        samples = samples[:, None]

    if samples.ndim != 2:
        raise RuntimeError(
            f"Unexpected MRT waveform shape: "
            f"{samples.shape}"
        )

    # Handle [channels, samples] if necessary.
    if (
        samples.shape[0] in (1, 2)
        and samples.shape[1] > 2
    ):
        samples = samples.T

    # Mono -> stereo.
    if samples.shape[1] == 1:
        samples = np.repeat(
            samples,
            2,
            axis=1,
        )

    if samples.shape[1] != 2:
        raise RuntimeError(
            f"Expected stereo audio, "
            f"got {samples.shape}"
        )

    samples = np.nan_to_num(
        samples,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )

    samples = np.clip(
        samples,
        -1.0,
        1.0,
    )

    pcm = (
        samples * 32767.0
    ).astype("<i2")

    return pcm.tobytes()


# ============================================================
# Playback
# ============================================================

def start_playback() -> subprocess.Popen:

    require_program("pacat")

    sink = get_default_sink()

    print()
    print("==================================================")
    print("PLAYBACK")
    print("==================================================")
    print(f"Output sink: {sink}")
    print()

    proc = subprocess.Popen(
        [
            "pacat",

            "--playback",
            "--raw",

            f"--device={sink}",

            "--format=s16le",
            "--rate=48000",
            "--channels=2",

            "--latency-msec=200",
        ],
        stdin=subprocess.PIPE,
        bufsize=0,
    )

    if proc.stdin is None:
        proc.terminate()

        raise RuntimeError(
            "Could not open pacat playback stream."
        )

    return proc


# ============================================================
# Main
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Capture system audio, use it as an MRT2 MusicCoCa "
            "style prompt, and continuously generate music."
        )
    )

    parser.add_argument(
        "--capture-seconds",
        type=float,
        default=DEFAULT_CAPTURE_SECONDS,
        help=(
            "Seconds to capture. "
            "30-60 is recommended; default: 60."
        ),
    )

    parser.add_argument(
        "--capture-file",
        default="mrt_style_capture.wav",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=[
            "mrt2_small",
            "mrt2_base",
        ],
    )

    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
    )

    parser.add_argument(
        "--prebuffer-chunks",
        type=int,
        default=DEFAULT_PREBUFFER_CHUNKS,
    )

    parser.add_argument(
        "--queue-chunks",
        type=int,
        default=DEFAULT_QUEUE_CHUNKS,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    args = parser.parse_args()

    if args.capture_seconds < 10:
        parser.error(
            "--capture-seconds must be at least 10 seconds."
        )

    if args.chunk_seconds <= 0:
        parser.error(
            "--chunk-seconds must be greater than zero."
        )

    if args.prebuffer_chunks < 1:
        parser.error(
            "--prebuffer-chunks must be >= 1."
        )

    if args.queue_chunks < args.prebuffer_chunks:
        parser.error(
            "--queue-chunks must be >= --prebuffer-chunks."
        )

    capture_file = Path(
        args.capture_file
    ).expanduser().resolve()

    # ========================================================
    # 1. Capture reference music
    # ========================================================

    capture_default_sink(
        capture_file,
        args.capture_seconds,
    )

    # ========================================================
    # 2. JAX / GPU information
    # ========================================================

    print()
    print("==================================================")
    print("JAX")
    print("==================================================")

    print(
        f"JAX version : "
        f"{jax.__version__}"
    )

    print(
        f"Backend     : "
        f"{jax.default_backend()}"
    )

    print(
        f"Devices     : "
        f"{jax.devices()}"
    )

    # ========================================================
    # 3. Load MRT
    # ========================================================

    print()
    print("==================================================")
    print("LOADING MRT")
    print("==================================================")

    print(
        f"Model: {args.model}"
    )

    mrt = MagentaRT2Jax(
        size=args.model,

        temperature=args.temperature,
        top_k=args.top_k,

        cfg_scales={
            "musiccoca": DEFAULT_CFG_MUSICCOCA,
            "notes": DEFAULT_CFG_NOTES,
            "drums": DEFAULT_CFG_DRUMS,
        },
    )

    print()
    print("MRT loaded.")

    # ========================================================
    # 4. Load captured WAV
    # ========================================================

    print()
    print("==================================================")
    print("AUDIO PROMPT")
    print("==================================================")

    print(
        f"Loading: {capture_file}"
    )

    prompt_audio = Waveform.from_file(
        str(capture_file)
    )

    print(
        f"Recorded length : "
        f"{prompt_audio.seconds:.3f}s"
    )

    print(
        f"Format          : "
        f"{prompt_audio.sample_rate} Hz, "
        f"{prompt_audio.num_channels} channels"
    )

    # ========================================================
    # 5. IMPORTANT:
    # Trim to complete 10-second MusicCoCa windows.
    # ========================================================

    prompt_audio = trim_for_musiccoca(
        prompt_audio
    )

    # ========================================================
    # 6. Audio -> MusicCoCa embedding
    # ========================================================

    print()
    print("==================================================")
    print("MUSICCOCA")
    print("==================================================")

    print(
        f"Embedding "
        f"{prompt_audio.seconds:.1f}s "
        f"of audio..."
    )

    style_embedding = mrt.embed_style(
        prompt_audio,

        # Average the complete set of 10-second style embeddings.
        pool_across_time=True,

        # Audio prompt: don't use the text->audio mapper.
        use_mapper=False,
    )

    print(
        "Style embedding:",
        np.asarray(
            style_embedding
        ).shape,
    )

    style_tokens = mrt.tokenize_style(
        style_embedding
    )

    print(
        "Style tokens:",
        np.asarray(
            style_tokens
        ).shape,
    )

    conditioning = {
        MUSICCOCA.key: style_tokens
    }

    # ========================================================
    # 7. Generation configuration
    # ========================================================

    frames_per_chunk = max(
        1,
        round(
            args.chunk_seconds * 25
        ),
    )

    actual_chunk_seconds = (
        frames_per_chunk / 25.0
    )

    prebuffer_seconds = (
        actual_chunk_seconds
        * args.prebuffer_chunks
    )

    print()
    print("==================================================")
    print("CONTINUOUS GENERATION")
    print("==================================================")

    print(
        f"Frames/chunk : "
        f"{frames_per_chunk}"
    )

    print(
        f"Audio/chunk  : "
        f"{actual_chunk_seconds:.2f}s"
    )

    print(
        f"Prebuffer    : "
        f"{args.prebuffer_chunks} chunks "
        f"({prebuffer_seconds:.1f}s)"
    )

    print(
        f"Queue        : "
        f"{args.queue_chunks} chunks"
    )

    # ========================================================
    # 8. Producer queue
    # ========================================================

    audio_queue: queue.Queue[bytes] = queue.Queue(
        maxsize=args.queue_chunks
    )

    stop_event = threading.Event()

    producer_errors: list[BaseException] = []

    generation_speeds: list[float] = []

    # ========================================================
    # 9. Continuous MRT generator
    # ========================================================

    def generate_forever() -> None:

        state = None
        chunk_number = 1

        try:

            while not stop_event.is_set():

                start = time.perf_counter()

                wav, state = mrt.generate(
                    conditioning=conditioning,

                    frames=frames_per_chunk,

                    # Maintain continuation state between chunks.
                    state=state,
                )

                # Force asynchronous JAX computation to complete.
                _ = np.asarray(
                    wav.samples
                )

                elapsed = (
                    time.perf_counter()
                    - start
                )

                speed = (
                    actual_chunk_seconds
                    / elapsed
                )

                generation_speeds.append(
                    speed
                )

                # Recent rolling average.
                if len(generation_speeds) > 20:
                    generation_speeds.pop(0)

                average_speed = (
                    sum(generation_speeds)
                    / len(generation_speeds)
                )

                pcm = waveform_to_pcm_s16le(
                    wav
                )

                while not stop_event.is_set():

                    try:

                        audio_queue.put(
                            pcm,
                            timeout=0.25,
                        )

                        break

                    except queue.Full:
                        continue

                buffer_seconds = (
                    audio_queue.qsize()
                    * actual_chunk_seconds
                )

                print(
                    f"Generated chunk {chunk_number:05d}: "
                    f"{actual_chunk_seconds:.1f}s audio "
                    f"in {elapsed:.2f}s | "
                    f"{speed:.2f}x RT | "
                    f"avg {average_speed:.2f}x | "
                    f"buffer {buffer_seconds:.0f}s",
                    flush=True,
                )

                chunk_number += 1

        except BaseException as exc:

            producer_errors.append(
                exc
            )

            stop_event.set()

    generator_thread = threading.Thread(
        target=generate_forever,
        name="mrt-generator",
        daemon=True,
    )

    generator_thread.start()

    # ========================================================
    # 10. Prebuffer
    # ========================================================

    print()
    print(
        f"Pre-generating "
        f"{prebuffer_seconds:.0f}s "
        f"before playback..."
    )

    prebuffer: list[bytes] = []

    while (
        len(prebuffer)
        < args.prebuffer_chunks
    ):

        if producer_errors:
            raise producer_errors[0]

        try:

            pcm = audio_queue.get(
                timeout=0.5,
            )

        except queue.Empty:

            continue

        prebuffer.append(
            pcm
        )

        print(
            f"Prebuffer: "
            f"{len(prebuffer)}/"
            f"{args.prebuffer_chunks} "
            f"({len(prebuffer) * actual_chunk_seconds:.0f}s)"
        )

    # ========================================================
    # 11. Start playback
    # ========================================================

    player = start_playback()

    assert player.stdin is not None

    print(
        f"Starting with "
        f"{prebuffer_seconds:.0f}s "
        f"of generated audio."
    )

    print()
    print("Press Ctrl+C to stop.")
    print()

    for pcm in prebuffer:

        player.stdin.write(
            pcm
        )

    player.stdin.flush()

    # ========================================================
    # 12. Continuous playback
    # ========================================================

    last_warning = 0.0

    try:

        while not stop_event.is_set():

            if producer_errors:
                raise producer_errors[0]

            if player.poll() is not None:
                raise RuntimeError(
                    f"pacat exited with "
                    f"status {player.returncode}"
                )

            try:

                pcm = audio_queue.get(
                    timeout=0.5
                )

            except queue.Empty:

                now = time.monotonic()

                if now - last_warning >= 5.0:

                    print(
                        "WARNING: playback buffer empty; "
                        "MRT generation is not keeping up.",
                        file=sys.stderr,
                        flush=True,
                    )

                    last_warning = now

                continue

            try:

                player.stdin.write(
                    pcm
                )

            except BrokenPipeError as exc:

                raise RuntimeError(
                    "pacat playback pipe closed."
                ) from exc

    except KeyboardInterrupt:

        print()
        print("Stopping...")

    finally:

        stop_event.set()

        try:
            player.stdin.flush()
        except Exception:
            pass

        try:
            player.stdin.close()
        except Exception:
            pass

        try:
            player.terminate()
        except Exception:
            pass

        try:

            player.wait(
                timeout=2
            )

        except subprocess.TimeoutExpired:

            player.kill()

        generator_thread.join(
            timeout=2
        )

    if producer_errors:
        raise producer_errors[0]

    return 0


if __name__ == "__main__":

    try:

        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:

        raise SystemExit(130)

    except Exception as exc:

        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )

        raise SystemExit(1)
