#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import jax

from magenta_rt import MagentaRT2Jax
from magenta_rt.audio import Waveform
from magenta_rt.config import MUSICCOCA


# ============================================================
# Defaults
# ============================================================

CAPTURE_SECONDS = 60.0

# Compare this directly with your 8-second base-model measurements.
CHUNK_SECONDS = 8.0

# First generated chunk often includes compilation/warmup.
BENCHMARK_CHUNKS = 3

# If generation is slower than realtime, size the initial reserve
# to aim for this much uninterrupted playback.
TARGET_PLAYBACK_MINUTES = 30.0

# Additional safety reserve.
SAFETY_BUFFER_SECONDS = 60.0

# Don't start with less than this much generated audio.
MIN_PREBUFFER_SECONDS = 32.0

# Prevent the automatic prebuffer from becoming absurdly large.
# If small is dramatically below realtime, the program will warn you.
MAX_PREBUFFER_SECONDS = 10 * 60.0

MODEL = "mrt2_small"

TEMPERATURE = 1.3
TOP_K = 40

CFG_MUSICCOCA = 3.0
CFG_NOTES = 1.0
CFG_DRUMS = 1.0


# ============================================================
# Utilities
# ============================================================

def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required program '{name}' was not found.\n"
            f"Install system dependencies with:\n\n"
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
# Record current system output
# ============================================================

def capture_default_sink(
    output_file: Path,
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
    print(f"Output       : {output_file}")
    print()
    print("Recording audio currently playing...")
    print()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
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

        str(output_file),
    ]

    subprocess.run(
        cmd,
        check=True,
    )

    if not output_file.exists():
        raise RuntimeError(
            f"Capture wasn't created: {output_file}"
        )

    if output_file.stat().st_size < 1000:
        raise RuntimeError(
            "Capture file appears empty."
        )

    mb = output_file.stat().st_size / 1024 / 1024

    print()
    print(
        f"Capture complete: {mb:.1f} MiB"
    )


# ============================================================
# MRT waveform -> raw stereo s16le
# ============================================================

def waveform_to_pcm_s16le(wav: Waveform) -> bytes:

    samples = np.asarray(
        wav.samples,
        dtype=np.float32,
    )

    if wav.sample_rate != 48000:
        raise RuntimeError(
            f"Expected 48000 Hz MRT output; "
            f"got {wav.sample_rate}"
        )

    if samples.ndim == 1:
        samples = samples[:, None]

    if samples.ndim != 2:
        raise RuntimeError(
            f"Unexpected MRT waveform shape: {samples.shape}"
        )

    # Handle [channels, samples].
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
            f"Expected stereo MRT audio, got {samples.shape}"
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
    print(f"Sink: {sink}")
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
            "Could not open pacat stdin."
        )

    return proc


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
        "--chunk-seconds",
        type=float,
        default=CHUNK_SECONDS,
    )

    parser.add_argument(
        "--target-minutes",
        type=float,
        default=TARGET_PLAYBACK_MINUTES,
        help=(
            "If generation is slower than realtime, size the "
            "starting buffer for approximately this many minutes "
            "of uninterrupted playback."
        ),
    )

    args = parser.parse_args()

    capture_file = Path(
        args.capture_file
    ).expanduser().resolve()

    chunk_seconds = args.chunk_seconds

    frames_per_chunk = max(
        1,
        round(chunk_seconds * 25)
    )

    # Actual duration after conversion to integral MRT frames.
    chunk_seconds = frames_per_chunk / 25.0

    # ========================================================
    # 1. Capture
    # ========================================================

    capture_default_sink(
        capture_file,
        args.capture_seconds,
    )

    # ========================================================
    # 2. JAX information
    # ========================================================

    print()
    print("==================================================")
    print("JAX")
    print("==================================================")
    print(f"Version : {jax.__version__}")
    print(f"Backend : {jax.default_backend()}")
    print(f"Devices : {jax.devices()}")

    # ========================================================
    # 3. Load MRT2 SMALL
    # ========================================================

    print()
    print("==================================================")
    print("LOADING MRT")
    print("==================================================")
    print(f"Model: {MODEL}")
    print()

    mrt = MagentaRT2Jax(
        size=MODEL,

        temperature=TEMPERATURE,
        top_k=TOP_K,

        cfg_scales={
            "musiccoca": CFG_MUSICCOCA,
            "notes": CFG_NOTES,
            "drums": CFG_DRUMS,
        },
    )

    print()
    print("MRT loaded.")

    # ========================================================
    # 4. Audio prompt
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
        f"Prompt length : "
        f"{prompt_audio.seconds:.2f} seconds"
    )

    print(
        f"Prompt format : "
        f"{prompt_audio.sample_rate} Hz, "
        f"{prompt_audio.num_channels} channel(s)"
    )

    print()
    print(
        "Embedding captured WAV with MusicCoCa..."
    )

    style_embedding = mrt.embed_style(
        prompt_audio,
        pool_across_time=True,

        # Input is audio, not text.
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
    # 5. Warmup
    # ========================================================

    print()
    print("==================================================")
    print("MRT2_SMALL WARMUP")
    print("==================================================")
    print(
        f"Generating {chunk_seconds:.1f}s "
        f"({frames_per_chunk} frames)."
    )

    print(
        "The first chunk may include JAX compilation, "
        "so it will NOT be used to estimate sustained speed."
    )

    state = None

    buffered_pcm: list[bytes] = []

    start = time.perf_counter()

    wav, state = mrt.generate(
        conditioning=conditioning,
        frames=frames_per_chunk,
        state=state,
    )

    # Force any lazy work to finish.
    _ = np.asarray(
        wav.samples
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    speed = (
        chunk_seconds / elapsed
    )

    buffered_pcm.append(
        waveform_to_pcm_s16le(wav)
    )

    print()
    print(
        f"WARMUP: {chunk_seconds:.1f}s audio "
        f"in {elapsed:.2f}s"
    )

    print(
        f"WARMUP SPEED: {speed:.3f}x realtime"
    )

    # ========================================================
    # 6. Real benchmark
    # ========================================================

    print()
    print("==================================================")
    print("STEADY-STATE BENCHMARK")
    print("==================================================")

    benchmark_times = []

    for i in range(BENCHMARK_CHUNKS):

        print(
            f"Generating benchmark chunk "
            f"{i + 1}/{BENCHMARK_CHUNKS}...",
            flush=True,
        )

        start = time.perf_counter()

        wav, state = mrt.generate(
            conditioning=conditioning,
            frames=frames_per_chunk,
            state=state,
        )

        _ = np.asarray(
            wav.samples
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        benchmark_times.append(
            elapsed
        )

        buffered_pcm.append(
            waveform_to_pcm_s16le(wav)
        )

        speed = (
            chunk_seconds / elapsed
        )

        print(
            f"  {chunk_seconds:.1f}s audio "
            f"in {elapsed:.2f}s "
            f"= {speed:.3f}x realtime"
        )

    total_benchmark_audio = (
        BENCHMARK_CHUNKS
        * chunk_seconds
    )

    total_benchmark_time = sum(
        benchmark_times
    )

    generation_rate = (
        total_benchmark_audio
        / total_benchmark_time
    )

    seconds_compute_per_audio_second = (
        1.0 / generation_rate
    )

    print()
    print("==================================================")
    print("BENCHMARK RESULT")
    print("==================================================")

    print(
        f"Sustained speed       : "
        f"{generation_rate:.3f}x realtime"
    )

    print(
        f"Compute/audio ratio   : "
        f"{seconds_compute_per_audio_second:.2f}s compute "
        f"per 1s audio"
    )

    print(
        f"Average chunk time    : "
        f"{total_benchmark_time / BENCHMARK_CHUNKS:.2f}s"
    )

    # ========================================================
    # 7. Calculate prebuffer automatically
    # ========================================================

    target_playback_seconds = (
        args.target_minutes * 60.0
    )

    if generation_rate >= 1.10:

        # Comfortable headroom.
        required_prebuffer_seconds = (
            MIN_PREBUFFER_SECONDS
        )

        print()
        print(
            "Small model is comfortably faster "
            "than realtime."
        )

    elif generation_rate >= 1.0:

        # Technically sustainable, but leave more room for jitter.
        required_prebuffer_seconds = max(
            MIN_PREBUFFER_SECONDS,
            60.0,
        )

        print()
        print(
            "Small model is slightly faster than realtime."
        )

    else:

        # During every second of playback, generation produces
        # generation_rate seconds of new audio.
        #
        # Buffer therefore loses:
        #
        #     1 - generation_rate
        #
        # seconds per playback second.

        deficit = (
            1.0 - generation_rate
        )

        required_prebuffer_seconds = (
            target_playback_seconds
            * deficit
            + SAFETY_BUFFER_SECONDS
        )

        required_prebuffer_seconds = max(
            required_prebuffer_seconds,
            MIN_PREBUFFER_SECONDS,
        )

        if (
            required_prebuffer_seconds
            > MAX_PREBUFFER_SECONDS
        ):

            print()
            print(
                "WARNING:"
            )

            print(
                f"At {generation_rate:.3f}x realtime, "
                f"approximately "
                f"{required_prebuffer_seconds / 60:.1f} minutes "
                f"would be needed for the requested "
                f"{args.target_minutes:.0f}-minute playback target."
            )

            print(
                f"Capping startup buffer at "
                f"{MAX_PREBUFFER_SECONDS / 60:.1f} minutes."
            )

            required_prebuffer_seconds = (
                MAX_PREBUFFER_SECONDS
            )

    required_chunks = math.ceil(
        required_prebuffer_seconds
        / chunk_seconds
    )

    # We already have warmup + benchmark chunks.
    required_chunks = max(
        required_chunks,
        len(buffered_pcm),
    )

    actual_prebuffer_seconds = (
        required_chunks
        * chunk_seconds
    )

    print()
    print("==================================================")
    print("AUTOMATIC BUFFER TUNING")
    print("==================================================")

    print(
        f"Chunk size          : "
        f"{chunk_seconds:.1f}s"
    )

    print(
        f"Generation speed    : "
        f"{generation_rate:.3f}x RT"
    )

    print(
        f"Starting buffer     : "
        f"{actual_prebuffer_seconds:.0f}s "
        f"({actual_prebuffer_seconds / 60:.1f} min)"
    )

    print(
        f"Chunks required     : "
        f"{required_chunks}"
    )

    # ========================================================
    # 8. Finish pre-generating initial buffer
    # ========================================================

    print()
    print("==================================================")
    print("PRE-GENERATING")
    print("==================================================")

    while len(buffered_pcm) < required_chunks:

        chunk_number = (
            len(buffered_pcm) + 1
        )

        start = time.perf_counter()

        wav, state = mrt.generate(
            conditioning=conditioning,
            frames=frames_per_chunk,
            state=state,
        )

        _ = np.asarray(
            wav.samples
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        pcm = waveform_to_pcm_s16le(
            wav
        )

        buffered_pcm.append(
            pcm
        )

        speed = (
            chunk_seconds / elapsed
        )

        buffered_seconds = (
            len(buffered_pcm)
            * chunk_seconds
        )

        print(
            f"Chunk {chunk_number:04d}: "
            f"{chunk_seconds:.1f}s in {elapsed:.2f}s | "
            f"{speed:.3f}x RT | "
            f"buffer {buffered_seconds:.0f}/"
            f"{actual_prebuffer_seconds:.0f}s",
            flush=True,
        )

    # ========================================================
    # 9. Queue
    # ========================================================

    #
    # Queue enough capacity for at least the entire initial
    # buffer plus additional future generation.
    #
    queue_chunks = max(
        required_chunks * 2,
        32,
    )

    audio_queue: queue.Queue[bytes] = queue.Queue(
        maxsize=queue_chunks
    )

    stop_event = threading.Event()

    producer_errors: list[BaseException] = []

    generation_state = state

    # Put pre-generated audio into queue.
    for pcm in buffered_pcm:
        audio_queue.put(
            pcm
        )

    # Free duplicate list container.
    buffered_pcm.clear()

    # ========================================================
    # 10. Background generation
    # ========================================================

    def generate_forever():

        nonlocal generation_state

        chunk_number = (
            required_chunks + 1
        )

        recent_times = []

        try:

            while not stop_event.is_set():

                start = time.perf_counter()

                wav, generation_state = mrt.generate(
                    conditioning=conditioning,
                    frames=frames_per_chunk,

                    # Maintain continuous MRT state.
                    state=generation_state,
                )

                _ = np.asarray(
                    wav.samples
                )

                elapsed = (
                    time.perf_counter()
                    - start
                )

                pcm = waveform_to_pcm_s16le(
                    wav
                )

                recent_times.append(
                    elapsed
                )

                if len(recent_times) > 10:
                    recent_times.pop(0)

                recent_audio = (
                    len(recent_times)
                    * chunk_seconds
                )

                recent_compute = sum(
                    recent_times
                )

                recent_rate = (
                    recent_audio
                    / recent_compute
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
                    * chunk_seconds
                )

                print(
                    f"Generated chunk {chunk_number:05d}: "
                    f"{chunk_seconds:.1f}s audio "
                    f"in {elapsed:.2f}s | "
                    f"{chunk_seconds / elapsed:.3f}x RT | "
                    f"avg {recent_rate:.3f}x | "
                    f"buffer ~{buffer_seconds:.0f}s",
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
    # 11. Start playback
    # ========================================================

    player = start_playback()

    assert player.stdin is not None

    print()
    print("==================================================")
    print("CONTINUOUS MRT2_SMALL PLAYBACK")
    print("==================================================")

    print(
        f"Starting with ~"
        f"{audio_queue.qsize() * chunk_seconds:.0f}s "
        f"of generated audio."
    )

    print("Press Ctrl+C to stop.")
    print()

    last_warning = 0.0

    # ========================================================
    # 12. Playback loop
    # ========================================================

    try:

        while not stop_event.is_set():

            if producer_errors:
                raise producer_errors[0]

            if player.poll() is not None:

                raise RuntimeError(
                    f"pacat exited unexpectedly "
                    f"with code {player.returncode}"
                )

            try:

                pcm = audio_queue.get(
                    timeout=0.5
                )

            except queue.Empty:

                now = time.monotonic()

                if now - last_warning >= 5.0:

                    print(
                        "WARNING: generated-audio buffer empty; "
                        "MRT is not keeping up.",
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
