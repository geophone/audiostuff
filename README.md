```bash
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
