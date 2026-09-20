#!/usr/bin/env python3

from __future__ import annotations

import os

# Must be set before importing JAX.
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.expanduser("~/.cache/jax-mrt"),
)
os.environ.setdefault(
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "0",
)
os.environ.setdefault(
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "-1",
)

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
# Configuration
# ============================================================

CAPTURE_SECONDS = 60.0

MUSICCOCA_WINDOW_SECONDS = 10.0

MODEL = "mrt2_small"

# Playback/output chunk size.
CHUNK_SECONDS = 2.0

# IMPORTANT:
# Do NOT submit all 50 frames to MRT in one call.
#
# 5 frames = 0.2 seconds.
#
# mrt.generate() synchronizes at the end of each call, so this
# limits how much ROCm work gets queued at once.
GPU_BURST_FRAMES = 5

# Tiny throttle after each synchronized burst.
#
# The overhead is insignificant:
# 10 bursts/chunk * 0.005 = only ~50 ms per 2 sec audio.
GPU_BURST_PAUSE_SECONDS = 0.005

# Start with 6 seconds buffered.
PREBUFFER_CHUNKS = 3

# Up to 24 seconds waiting.
QUEUE_CHUNKS = 12

TEMPERATURE = 1.3
TOP_K = 40

CFG_MUSICCOCA = 3.0
CFG_NOTES = 1.0
CFG_DRUMS = 1.0


# ============================================================
# Utility
# ============================================================

def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required command '{name}' not found.\n"
            f"Install dependencies with:\n"
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
        raise RuntimeError("Could not determine default audio sink.")

    return sink


# ============================================================
# Capture
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
            "-loglevel", "warning",
            "-y",

            "-f", "pulse",
            "-i", monitor,

            "-t", str(seconds),

            "-ac", "2",
            "-ar", "48000",
            "-c:a", "pcm_s16le",

            str(filename),
        ],
        check=True,
    )

    if not filename.exists():
        raise RuntimeError("Capture file was not created.")

    print()
    print(
        f"Capture complete: "
        f"{filename.stat().st_size / 1024 / 1024:.1f} MiB"
    )


# ============================================================
# MusicCoCa trimming
# ============================================================

def trim_for_musiccoca(
    wav: Waveform,
) -> Waveform:

    samples_per_window = int(
        wav.sample_rate * MUSICCOCA_WINDOW_SECONDS
    )

    total_samples = len(wav.samples)

    full_windows = (
        total_samples // samples_per_window
    )

    if full_windows < 1:
        raise RuntimeError(
            "Audio prompt must contain at least 10 seconds."
        )

    usable_samples = (
        full_windows * samples_per_window
    )

    original_seconds = (
        total_samples / wav.sample_rate
    )

    usable_seconds = (
        usable_samples / wav.sample_rate
    )

    trimmed = wav[:usable_samples]

    print()
    print("==================================================")
    print("MUSICCOCA WINDOW TRIMMING")
    print("==================================================")
    print(f"Original duration : {original_seconds:.3f}s")
    print(f"Full windows      : {full_windows}")
    print(f"Usable duration   : {usable_seconds:.3f}s")
    print(
        f"Removed tail      : "
        f"{original_seconds - usable_seconds:.3f}s"
    )

    return trimmed


# ============================================================
# Waveform -> PCM
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
            f"Expected 48000 Hz MRT output, "
            f"got {wav.sample_rate}"
        )

    if samples.ndim == 1:
        samples = samples[:, None]

    if samples.ndim != 2:
        raise RuntimeError(
            f"Unexpected waveform shape: {samples.shape}"
        )

    # Handle channel-first output if encountered.
    if (
        samples.shape[0] in (1, 2)
        and samples.shape[1] > 2
    ):
        samples = samples.T

    if samples.shape[1] == 1:
        samples = np.repeat(
            samples,
            2,
            axis=1,
        )

    if samples.shape[1] != 2:
        raise RuntimeError(
            f"Expected stereo output, got {samples.shape}"
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
# SAFER ROCm generation
# ============================================================

def generate_chunk_safely(
    mrt,
    conditioning,
    total_frames: int,
    state,
    chunk_number: int,
):
    """
    Generate one logical audio chunk using several small MRT calls.

    This is deliberately different from:

        mrt.generate(..., frames=50)

    because MRT internally submits every streaming step before doing its
    final device_get().

    Instead we do:

        5 frames -> synchronize
        5 frames -> synchronize
        ...
        5 frames -> synchronize

    This reduces sustained AQL/MES queue pressure on gfx1151.
    """

    total_seconds = (
        total_frames / 25.0
    )

    print(
        f"[MRT] START chunk {chunk_number:05d}: "
        f"{total_frames} frames / "
        f"{total_seconds:.2f}s audio",
        flush=True,
    )

    start = time.perf_counter()

    pcm_parts: list[bytes] = []

    frames_remaining = total_frames

    burst_number = 0

    while frames_remaining > 0:

        burst_number += 1

        burst_frames = min(
            GPU_BURST_FRAMES,
            frames_remaining,
        )

        burst_start = time.perf_counter()

        #
        # IMPORTANT:
        #
        # mrt.generate() performs jax.device_get() before returning.
        # Therefore each small call becomes a GPU synchronization
        # point instead of letting dozens of streaming steps build up.
        #
        wav, state = mrt.generate(
            conditioning=conditioning,
            frames=burst_frames,
            state=state,
        )

        burst_elapsed = (
            time.perf_counter()
            - burst_start
        )

        pcm_parts.append(
            waveform_to_pcm_s16le(wav)
        )

        frames_remaining -= burst_frames

        generated_frames = (
            total_frames - frames_remaining
        )

        print(
            f"      burst {burst_number:02d}: "
            f"{burst_frames} frames in "
            f"{burst_elapsed:.3f}s "
            f"[{generated_frames}/{total_frames}]",
            flush=True,
        )

        #
        # Tiny submission throttle.
        #
        # This is intentionally here even though device_get()
        # already synchronizes.
        #
        if (
            frames_remaining > 0
            and GPU_BURST_PAUSE_SECONDS > 0
        ):
            time.sleep(
                GPU_BURST_PAUSE_SECONDS
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    realtime = (
        total_seconds / elapsed
    )

    pcm = b"".join(
        pcm_parts
    )

    print(
        f"[MRT] DONE  chunk {chunk_number:05d}: "
        f"{total_seconds:.2f}s audio in "
        f"{elapsed:.2f}s "
        f"({realtime:.2f}x RT)",
        flush=True,
    )

    return (
        pcm,
        state,
        elapsed,
    )


# ============================================================
# Playback
# ============================================================

def start_playback():

    require_program("pacat")

    sink = get_default_sink()

    print()
    print("==================================================")
    print("PLAYBACK")
    print("==================================================")
    print(f"Output sink: {sink}")

    process = subprocess.Popen(
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

    if process.stdin is None:
        raise RuntimeError(
            "Could not open pacat playback stream."
        )

    return process


# ============================================================
# Main
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--capture-seconds",
        type=float,
        default=CAPTURE_SECONDS,
    )

    parser.add_argument(
        "--capture-file",
        default="mrt_style_capture.wav",
    )

    parser.add_argument(
        "--model",
        default=MODEL,
        choices=[
            "mrt2_small",
            "mrt2_base",
        ],
    )

    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=CHUNK_SECONDS,
    )

    parser.add_argument(
        "--prebuffer-chunks",
        type=int,
        default=PREBUFFER_CHUNKS,
    )

    parser.add_argument(
        "--queue-chunks",
        type=int,
        default=QUEUE_CHUNKS,
    )

    args = parser.parse_args()

    capture_file = Path(
        args.capture_file
    ).expanduser().resolve()

    frames_per_chunk = max(
        1,
        round(
            args.chunk_seconds * 25
        ),
    )

    chunk_seconds = (
        frames_per_chunk / 25.0
    )

    # ========================================================
    # Capture
    # ========================================================

    capture_default_sink(
        capture_file,
        args.capture_seconds,
    )

    # ========================================================
    # JAX
    # ========================================================

    print()
    print("==================================================")
    print("JAX")
    print("==================================================")
    print(f"JAX version : {jax.__version__}")
    print(f"Backend     : {jax.default_backend()}")
    print(f"Devices     : {jax.devices()}")
    print(
        f"Cache       : "
        f"{os.environ['JAX_COMPILATION_CACHE_DIR']}"
    )

    # ========================================================
    # Load MRT
    # ========================================================

    print()
    print("==================================================")
    print("LOADING MRT")
    print("==================================================")
    print(f"Model: {args.model}")

    mrt = MagentaRT2Jax(
        size=args.model,

        temperature=TEMPERATURE,
        top_k=TOP_K,

        cfg_scales={
            "musiccoca": CFG_MUSICCOCA,
            "notes": CFG_NOTES,
            "drums": CFG_DRUMS,
        },
    )

    print("MRT loaded.")

    # ========================================================
    # Prompt
    # ========================================================

    print()
    print("==================================================")
    print("AUDIO PROMPT")
    print("==================================================")
    print(f"Loading: {capture_file}")

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

    prompt_audio = trim_for_musiccoca(
        prompt_audio
    )

    # ========================================================
    # MusicCoCa
    # ========================================================

    print()
    print("==================================================")
    print("MUSICCOCA")
    print("==================================================")

    print(
        f"Embedding "
        f"{prompt_audio.seconds:.1f}s of audio..."
    )

    style_embedding = mrt.embed_style(
        prompt_audio,
        pool_across_time=True,
        use_mapper=False,
    )

    print(
        "Style embedding:",
        np.asarray(style_embedding).shape,
    )

    style_tokens = mrt.tokenize_style(
        style_embedding
    )

    print(
        "Style tokens:",
        np.asarray(style_tokens).shape,
    )

    conditioning = {
        MUSICCOCA.key: style_tokens
    }

    # ========================================================
    # Warmup
    # ========================================================

    print()
    print("==================================================")
    print("GPU WARMUP")
    print("==================================================")

    t0 = time.perf_counter()

    warmup_wav, _ = mrt.generate(
        conditioning=conditioning,
        frames=1,
        state=None,
    )

    _ = waveform_to_pcm_s16le(
        warmup_wav
    )

    print(
        f"GPU warmup finished in "
        f"{time.perf_counter() - t0:.2f}s"
    )

    # Fresh stream after warmup.
    state = None

    # ========================================================
    # Generation setup
    # ========================================================

    print()
    print("==================================================")
    print("GENERATION")
    print("==================================================")
    print(f"Chunk size      : {chunk_seconds:.2f}s")
    print(f"Frames/chunk    : {frames_per_chunk}")
    print(f"GPU burst       : {GPU_BURST_FRAMES} frames")
    print(
        f"GPU burst audio : "
        f"{GPU_BURST_FRAMES / 25:.2f}s"
    )
    print(
        f"Burst pause     : "
        f"{GPU_BURST_PAUSE_SECONDS * 1000:.1f} ms"
    )
    print(
        f"Initial buffer  : "
        f"{args.prebuffer_chunks * chunk_seconds:.1f}s"
    )
    print(
        f"Maximum queue   : "
        f"{args.queue_chunks * chunk_seconds:.1f}s"
    )

    audio_queue = queue.Queue(
        maxsize=args.queue_chunks
    )

    stop_event = threading.Event()

    producer_errors: list[BaseException] = []

    # ========================================================
    # First chunk synchronously
    # ========================================================

    first_pcm, state, first_elapsed = (
        generate_chunk_safely(
            mrt=mrt,
            conditioning=conditioning,
            total_frames=frames_per_chunk,
            state=state,
            chunk_number=1,
        )
    )

    prebuffer = [
        first_pcm
    ]

    generation_state = state

    # ========================================================
    # Producer
    # ========================================================

    def generator():

        nonlocal generation_state

        chunk_number = 2

        speeds: list[float] = []

        try:

            while not stop_event.is_set():

                pcm, generation_state, elapsed = (
                    generate_chunk_safely(
                        mrt=mrt,
                        conditioning=conditioning,
                        total_frames=frames_per_chunk,
                        state=generation_state,
                        chunk_number=chunk_number,
                    )
                )

                speed = (
                    chunk_seconds / elapsed
                )

                speeds.append(speed)

                if len(speeds) > 20:
                    speeds.pop(0)

                average = (
                    sum(speeds)
                    / len(speeds)
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

                print(
                    f"[MRT] rolling average "
                    f"{average:.2f}x RT | "
                    f"queue "
                    f"{audio_queue.qsize() * chunk_seconds:.1f}s",
                    flush=True,
                )

                chunk_number += 1

        except BaseException as exc:

            producer_errors.append(
                exc
            )

            stop_event.set()

    generator_thread = threading.Thread(
        target=generator,
        name="mrt-generator",
        daemon=True,
    )

    generator_thread.start()

    # ========================================================
    # Initial prebuffer
    # ========================================================

    print()
    print(
        f"Building initial "
        f"{args.prebuffer_chunks * chunk_seconds:.1f}s buffer..."
    )

    while len(prebuffer) < args.prebuffer_chunks:

        if producer_errors:
            raise producer_errors[0]

        try:

            pcm = audio_queue.get(
                timeout=0.5
            )

        except queue.Empty:
            continue

        prebuffer.append(
            pcm
        )

        print(
            f"Buffered: "
            f"{len(prebuffer) * chunk_seconds:.1f}s / "
            f"{args.prebuffer_chunks * chunk_seconds:.1f}s",
            flush=True,
        )

    # ========================================================
    # Playback
    # ========================================================

    player = start_playback()

    assert player.stdin is not None

    print()
    print(
        f"Starting playback with "
        f"{len(prebuffer) * chunk_seconds:.1f}s ready."
    )

    print("Press Ctrl+C to stop.")
    print()

    for pcm in prebuffer:

        player.stdin.write(
            pcm
        )

    player.stdin.flush()

    # ========================================================
    # Continuous playback
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

                if now - last_warning >= 3.0:

                    print(
                        "WARNING: playback queue empty; "
                        "waiting for generation.",
                        file=sys.stderr,
                        flush=True,
                    )

                    last_warning = now

                continue

            player.stdin.write(
                pcm
            )

            player.stdin.flush()

    except KeyboardInterrupt:

        print()
        print("Stopping...")

    finally:

        stop_event.set()

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
