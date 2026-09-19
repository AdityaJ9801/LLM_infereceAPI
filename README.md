# Local LLM Inference API (vLLM + FastAPI + Cloudflare Tunnel)

An OpenAI-compatible chat API (`/v1/chat/completions`, streaming supported) backed by
vLLM, running on your Linux GPU server and exposed publicly through a Cloudflare
Tunnel — no port forwarding or static IP needed.

**Target hardware assumed below: a single NVIDIA B200 (Blackwell) with ~45GB of
usable VRAM** (a partition/slice of the full card, which has 180-192GB total).
Everything is configurable via `.env` if your actual box differs.

## Important notes

- **There is no model called `gemma4:26b`.** That `name:size` format is Ollama's
  tagging convention; vLLM loads models by their Hugging Face repo ID instead
  (e.g. `Qwen/Qwen3-32B`). Set whichever real model you want in `MODEL_NAME`.
- **B200 is very new hardware (Blackwell, compute capability `sm_100`).** It
  needs a recent NVIDIA driver (CUDA 12.6+/12.8 support) and a vLLM/PyTorch
  build with Blackwell kernels. Don't assume an old pinned vLLM version works —
  install the latest release and run the sanity check below first.
- **A 32B model at full bf16 needs ~64GB just for weights** — it won't fit in
  45GB alongside a KV cache. The default here uses **FP8**, which vLLM can
  apply on load to an ordinary bf16 checkpoint (no need for a separately
  quantized repo), and which Blackwell's tensor cores handle natively.

## Choosing a model for ~45GB VRAM

| Setup | `MODEL_NAME` | `QUANTIZATION` | Notes |
|---|---|---|---|
| **Default (recommended first try)** | `Qwen/Qwen3-32B` | `fp8` | Strong reasoning, native "thinking" mode. Weights ~33-35GB, leaves modest KV cache headroom - keep `MAX_MODEL_LEN` around 8192 and concurrency low. |
| Safer fallback if FP8/quantized kernels misbehave on your stack | `Qwen/Qwen3-14B` | *(blank, full bf16)* | ~28GB weights, ~15GB+ free for KV cache → longer context, more concurrent requests, zero quantization risk. |
| Alternative reasoning-specialist | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | `fp8` or `awq` | Distilled directly from DeepSeek-R1's reasoning traces. |
| If you want to run larger/more concurrent later | *(any of the above)* + more GPUs | — | Set `TENSOR_PARALLEL_SIZE` to the GPU count to split one model across multiple B200s. |

If the FP8 default fails to load or throws a "no kernel image" / unsupported
architecture error, that's a sign your installed vLLM/torch build predates
Blackwell FP8 kernel support — either upgrade vLLM (`pip install -U vllm`) or
switch to the bf16 `Qwen3-14B` fallback row above while you sort that out.

## Sanity-check your GPU stack first

Before loading a 30B+ model and waiting several minutes just to hit a CUDA
error, confirm the basics:

```bash
nvidia-smi                              # driver version, confirms the GPU and ~45GB are visible
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` is `False` or it errors on the device name,
fix the driver/CUDA/torch install before going further (this is far more
common on brand-new hardware than a vLLM-specific issue).

## Project layout

```
app/
  config.py    # settings from .env
  schemas.py   # OpenAI-compatible request/response models
  engine.py    # vLLM AsyncLLMEngine wrapper + chat templating
  main.py      # FastAPI app: /health, /v1/models, /v1/chat/completions
test_client.py # quick manual smoke test
Dockerfile, docker-compose.yml, requirements-app.txt   # Docker path
requirements.txt                                       # native pip path
cloudflared/config.yml.example                          # named-tunnel template
```

## 1. Configure

```bash
cp .env.example .env
```

Edit `.env` if you want to deviate from the defaults above, and set `API_KEY`
to a long random string (required once you expose this publicly — clients
must send `Authorization: Bearer <API_KEY>`).

## 2. Run the server

### Option A — Docker (recommended: avoids hand-matching driver/CUDA/torch versions)

Prereqs: Docker + the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# verify GPU passthrough works before building anything:
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

Then:

```bash
docker compose up --build
```

First run downloads the model from Hugging Face (tens of GB for a 32B model)
into the `hf-cache` volume, then starts the API on `http://localhost:8000`.

### Option B — native Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # if not already done
python -m app.main
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

### Install `cloudflared` (Linux)

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

(Use the `.rpm` package instead on RHEL/CentOS/Fedora-based systems.)

### Fastest path — quick tunnel (no domain required, good for testing)

```bash
cloudflared tunnel --url http://localhost:8000
```

This prints a temporary `https://<random>.trycloudflare.com` URL that proxies
straight to your local server. It changes every time you restart the command —
fine for testing, not for a stable public endpoint.

### Stable path — named tunnel with your own domain

Requires a domain added to your Cloudflare account (Cloudflare DNS must be
authoritative for it).

```bash
cloudflared tunnel login                       # opens a browser link, pick your domain
cloudflared tunnel create llm-api              # prints a TUNNEL_ID, writes credentials json to ~/.cloudflared/

# copy cloudflared/config.yml.example -> cloudflared/config.yml
# fill in TUNNEL_ID, the credentials-file path, and your hostname

cloudflared tunnel route dns llm-api llm.yourdomain.com
cloudflared tunnel --config cloudflared/config.yml run llm-api
```

To keep it running persistently as a systemd service:

```bash
sudo cloudflared service install --config /full/path/to/cloudflared/config.yml
sudo systemctl enable --now cloudflared
```

### Harden it before leaving it running

- Keep `API_KEY` set in `.env` — the app rejects requests without the correct
  `Authorization: Bearer` header once it's set.
- Consider adding a **Cloudflare Access** application (Zero Trust dashboard) in
  front of the tunnel hostname, requiring login (Google/GitHub/OTP) before
  traffic even reaches your API — a second layer beyond the API key.
- Consider a Cloudflare rate-limiting rule on the hostname to cap requests/min,
  since GPU inference is expensive to let strangers hammer, and a 45GB card
  serving one 32B model has limited concurrency headroom.
