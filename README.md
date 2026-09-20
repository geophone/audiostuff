# Clone your soundscape into realtime audio
Goal:  Take audio sample, use audio sample to generate continuous stream of AI music/sound

```bash
sudo apt install ffmpeg pulseaudio-utils
git clone git@github.com:magenta/magenta-realtime.git
cd magenta-realtime
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate

uv pip install "magenta-rt[jax]"; # or mlx for apple support

mrt models init
mrt models download
mrt checkpoints download

```
My gpu wasn't powerful enough to use mrt_base instead of mrt_small although I found the soundscapes of similar quality.

Would consider testing audio generation on a more powerful gpu.
