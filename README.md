# Local LLM Inference API (transformers + FastAPI + Cloudflare Tunnel)

An OpenAI-compatible chat API (`/v1/chat/completions`, streaming supported) backed by
plain Hugging Face `transformers` (no vLLM), running on your Linux GPU server and
exposed publicly through a Cloudflare Tunnel — no port forwarding or static IP needed.

We deliberately skip vLLM here: its PyPI wheels don't yet pin `torch` tightly
enough for very new GPUs, so `pip install vllm` was resolving a `torch` build
newer than the installed driver supports. Plain `transformers` has a much
simpler, more predictable dependency chain.

**Important tradeoff:** plain `transformers.generate()` has no continuous
batching like vLLM does. This server serializes requests with a lock — one
generation runs on the GPU at a time. Fine for a single user or light use;
concurrent throughput will be much lower than a vLLM-based setup.

**Target hardware assumed below: a single NVIDIA B200 (Blackwell) with ~45GB of
usable VRAM** (a partition/slice of the full card, which has 180-192GB total).
Everything is configurable via `.env` if your actual box differs.

## Important notes

- **There is no model called `gemma4:26b`.** That `name:size` format is Ollama's
  tagging convention; `transformers`/HF load models by their Hugging Face repo
  ID instead (e.g. `Qwen/Qwen3-14B`). Set whichever real model you want in `MODEL_NAME`.
- **B200 is very new hardware (Blackwell, compute capability `sm_100`).** It
  needs a recent NVIDIA driver (CUDA 12.6+/12.8 support) and a `torch` build
  compiled for a CUDA version your driver actually supports. Don't rely on
  plain `pip install torch` resolving the right one automatically — pin it
  explicitly (see install steps below) and run the sanity check first.
- **A 32B model at full bf16 needs ~64GB just for weights** — it won't fit in
  45GB. The default model here (`Qwen/Qwen3-14B`) fits comfortably at plain
  bf16 with no quantization. To run something 32B-class instead, set
  `LOAD_IN_4BIT=true` (uses `bitsandbytes`) — see the table below.

## Choosing a model for ~45GB VRAM

| Setup | `MODEL_NAME` | `LOAD_IN_4BIT` | Notes |
|---|---|---|---|
| **Default (recommended first try)** | `Qwen/Qwen3-14B` | `false` | ~28GB weights at bf16, comfortable headroom, zero quantization risk on brand-new hardware. |
| More reasoning power, if you want to push it | `Qwen/Qwen3-32B` | `true` | 4-bit via `bitsandbytes`, weights ~18-20GB. `bitsandbytes` kernels can also lag on very new GPUs — if it errors, fall back to the row above. |
| Alternative reasoning-specialist | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `false` | Distilled directly from DeepSeek-R1's reasoning traces, similar size class to the default. |

## Sanity-check your GPU stack first

Before loading a model and waiting several minutes just to hit a CUDA error,
confirm the basics:

```bash
nvidia-smi                              # driver version + "CUDA Version" it supports, confirms ~45GB visible
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`torch.version.cuda` must be **less than or equal to** the "CUDA Version" shown
by `nvidia-smi` (e.g. driver reporting `12.8` cannot run a `cu130`-built torch —
that's the "driver is too old" error, even though it's really a torch-too-new
problem). If it errors, fix the install (see below) before going further.

## Project layout

```
app/
  config.py    # settings from .env
  schemas.py   # OpenAI-compatible request/response models
  engine.py    # transformers model/tokenizer wrapper, streaming via TextIteratorStreamer
  main.py      # FastAPI app: /health, /v1/models, /v1/chat/completions
test_client.py # quick manual smoke test
Dockerfile, docker-compose.yml   # Docker path
requirements.txt                 # native pip path
cloudflared/config.yml.example    # named-tunnel template
```

## 1. Configure

```bash
cp .env.example .env
```

Edit `.env` if you want to deviate from the defaults above, and set `API_KEY`
to a long random string (required once you expose this publicly — clients
must send `Authorization: Bearer <API_KEY>`).

## 2. Run the server

### Option A — Docker (isolates the install from the host's Python)

Prereqs: Docker + the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), and root/sudo to install it — if you don't have that on this box, use Option B instead.

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# verify GPU passthrough works before building anything:
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Then:

```bash
docker compose up --build
```

First run downloads the model from Hugging Face (tens of GB for a 32B model)
into the `hf-cache` volume, then starts the API on `http://localhost:8000`.

### Option B — native Python (no venv)

`requirements.txt` already pins the `torch` install to the CUDA 12.8 wheel
index (`--extra-index-url https://download.pytorch.org/whl/cu128`) — adjust
that line first if your `nvidia-smi` reports a different CUDA version.

```bash
pip install -r requirements.txt

cp .env.example .env   # if not already done
python -m app.main
```

If `pip install` still resolves a `torch` build that doesn't match your
driver (check with the sanity-check command above), the more robust fix is
`uv`, which auto-detects the right CUDA build from your driver instead of a
hardcoded index URL:

```bash
pip uninstall -y torch torchvision torchaudio
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv pip install --system torch --torch-backend=auto
pip install -r requirements.txt   # the rest (transformers, fastapi, etc.)
```

## 3. Test locally

```bash
python test_client.py "What is 17 * 24? Show your reasoning."
```

or with curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <API_KEY>" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

With `ENABLE_THINKING=true` (Qwen3's default), responses may include a
`<think>...</think>` reasoning trace before the final answer — that's
expected, not a bug; strip it client-side if you only want the final answer.

## 4. Expose it publicly with Cloudflare Tunnel

Cloudflare Tunnel creates an outbound-only connection from your server to
Cloudflare's edge, so your server's public IP is never exposed and no
firewall/port-forwarding changes are needed.

### Install `cloudflared` (Linux, no sudo/root needed)

Install it to your home directory, **not** into the repo — the repo already
has a `cloudflared/` folder (holding `config.yml.example`), and a file and a
directory can't share that name.

```bash
curl -L --output ~/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ~/cloudflared
~/cloudflared --version
```

(If you have sudo and prefer a system package: `curl -L --output cloudflared.deb .../cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared.deb` — use the `.rpm` on RHEL/CentOS/Fedora. Then just drop the `~/` prefix from the commands below.)

### Fastest path — quick tunnel (no domain required, good for testing)

```bash
~/cloudflared tunnel --url http://localhost:8000
```

This prints a temporary `https://<random>.trycloudflare.com` URL that proxies
straight to your local server. It changes every time you restart the command —
fine for testing, not for a stable public endpoint.

### Stable path — named tunnel with your own domain

Requires a domain added to your Cloudflare account (Cloudflare DNS must be
authoritative for it). Replace `yourdomain.com` and the `llm` subdomain below
with your real values.

```bash
cd ~/LLM_infereceAPI

# 1. Authenticate - prints a URL, open it in a LOCAL browser (server is headless),
#    log in, and pick the domain to authorize. Writes ~/.cloudflared/cert.pem.
~/cloudflared tunnel login

# 2. Create the tunnel - prints a TUNNEL_ID and writes
#    ~/.cloudflared/<TUNNEL_ID>.json (credentials for this tunnel)
~/cloudflared tunnel create llm-api

# 3. Write the config, filling in the TUNNEL_ID from step 2
TUNNEL_ID=<paste-the-id-from-step-2>
DOMAIN=yourdomain.com
SUBDOMAIN=llm

mkdir -p cloudflared
cat > cloudflared/config.yml <<EOF
tunnel: ${TUNNEL_ID}
credentials-file: ${HOME}/.cloudflared/${TUNNEL_ID}.json
ingress:
  - hostname: ${SUBDOMAIN}.${DOMAIN}
    service: http://localhost:8000
  - service: http_status:404
EOF

# 4. Point the DNS record at the tunnel (creates a CNAME in your Cloudflare zone)
~/cloudflared tunnel route dns llm-api ${SUBDOMAIN}.${DOMAIN}

# 5. Run it (foreground, to confirm it connects cleanly)
~/cloudflared tunnel --config cloudflared/config.yml run llm-api
```

Once step 5 shows `Registered tunnel connection`, `Ctrl+C` it and run it in the
background instead so it survives your SSH session ending:

```bash
nohup ~/cloudflared tunnel --config ~/LLM_infereceAPI/cloudflared/config.yml run llm-api \
  > ~/cloudflared.log 2>&1 &
disown
sleep 3 && tail -n 20 ~/cloudflared.log
```

Test it:

```bash
curl https://llm.yourdomain.com/health
curl https://llm.yourdomain.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

If your server has `systemd` and you have sudo, you can install it as a proper
service instead of `nohup`:

```bash
sudo ~/cloudflared service install --config /full/path/to/cloudflared/config.yml
sudo systemctl enable --now cloudflared
```

### Harden it before leaving it running

- Keep `API_KEY` set in `.env` — the app rejects requests without the correct
  `Authorization: Bearer` header once it's set.
- Consider adding a **Cloudflare Access** application (Zero Trust dashboard) in
  front of the tunnel hostname, requiring login (Google/GitHub/OTP) before
  traffic even reaches your API — a second layer beyond the API key.
- Consider a Cloudflare rate-limiting rule on the hostname to cap requests/min
  — GPU inference is expensive to let strangers hammer, and remember this
  server processes one request at a time (no batching), so concurrent
  requests just queue up behind each other.
