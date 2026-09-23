# Local LLM Inference API (transformers + FastAPI + Cloudflare Tunnel)

An OpenAI-compatible chat API (`/v1/chat/completions`, streaming supported) backed by
plain Hugging Face `transformers` (no vLLM), running on your Linux GPU server and
exposed publicly through a Cloudflare Tunnel — no port forwarding or static IP needed.

We deliberately skip vLLM here: its PyPI wheels don't yet pin `torch` tightly
enough for very new GPUs, so `pip install vllm` was resolving a `torch` build
newer than the installed driver supports. Plain `transformers` has a much
simpler, more predictable dependency chain.

**Important tradeoff:** plain `transformers.generate()` has no continuous
batching like vLLM does. This server instead runs up to `MAX_CONCURRENT_REQUESTS`
generations at once (each holding its own KV cache in VRAM), queues anything
beyond that (up to `MAX_QUEUE_SIZE`, then rejects with HTTP 503), and can
unload the model from VRAM entirely after `IDLE_UNLOAD_SECONDS` of no
activity, reloading it transparently on the next request. See "Concurrency,
queueing, and idle VRAM release" below. Even with concurrency > 1, per-request
throughput is well below a vLLM-based setup — there's no batched matmul
across requests, just independent GPU calls interleaved.

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

## Concurrency, queueing, and idle VRAM release

Three `.env` settings control how the server shares the GPU:

- **`MAX_CONCURRENT_REQUESTS`** (default `2`) — how many `generate()` calls run
  on the GPU at once. Each one holds its own KV cache in VRAM for the
  duration of that request, on top of the model weights, so raise this only
  as far as your free VRAM allows. With the default `Qwen/Qwen3-14B` at bf16
  (~28GB weights, ~17GB free on a 45GB card), 2 is a conservative starting
  point — watch `nvidia-smi` under load before raising it.
- **`MAX_QUEUE_SIZE`** (default `20`) — requests beyond the concurrency limit
  wait here. Once the queue itself is full, new requests get an immediate
  HTTP 503 instead of queueing indefinitely.
- **`IDLE_UNLOAD_SECONDS`** (default `600`) — if no request has run for this
  long, a background task frees the model from VRAM (`del` + `torch.cuda.empty_cache()`),
  so it stops holding memory other processes on a shared server might need.
  The next request after that transparently reloads it first — that one
  request pays the full model-load latency again. Set to `0` to disable.

`GET /health` reports live state: `model_loaded`, `active_requests`,
`queued_requests`, so you can watch this behavior in practice:

```bash
curl http://localhost:8001/health
```

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
cloudflared/config.yml.example    # template for the alternative CLI-managed tunnel (see below) - not needed for the dashboard/token flow this README uses by default
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

### Quickest path — quick tunnel (no domain required, good for a one-off test)

```bash
~/cloudflared tunnel --url http://localhost:8000   # match your .env's PORT
```

This prints a temporary `https://<random>.trycloudflare.com` URL that proxies
straight to your local server. It changes every time you restart the command —
fine for a quick test, not for a stable public endpoint.

### Stable path — dashboard-managed tunnel with a token (what this deployment actually uses)

This is a "remotely-managed" tunnel: the hostname/routing config lives in
Cloudflare's dashboard, not a local file, and the token is what authorizes
`cloudflared` to connect as that tunnel's connector.

**1. Create the tunnel** at [one.dash.cloudflare.com](https://one.dash.cloudflare.com) →
**Networks → Tunnels → Create a tunnel** → connector type **Cloudflared** →
name it (e.g. `llm-api`) → **Save tunnel**.

**2. Get the token.** The next screen shows OS-specific install commands, each
containing a long token. Pick the **Debian** tab (closest match for a Linux
GPU container) and copy just the token value — the long string after
`service install` in `sudo cloudflared service install <TOKEN>`. You don't
need sudo/systemd to use it; ignore that exact command.

**3. Run the connector with the token:**

```bash
nohup ~/cloudflared tunnel run --token <TOKEN> > ~/cloudflared.log 2>&1 &
disown
sleep 5 && tail -n 20 ~/cloudflared.log
```

Look for `Registered tunnel connection` lines — that confirms it's actually
connected to Cloudflare's edge (the dashboard's tunnel status also flips to
"Healthy"). If instead you see repeated QUIC connection errors/retries with
no successful registration, outbound UDP is likely blocked/unreliable on
this network — restart with HTTP/2 (plain TCP/443) instead:

```bash
pkill -f "cloudflared tunnel run"
nohup ~/cloudflared tunnel run --protocol http2 --token <TOKEN> > ~/cloudflared.log 2>&1 &
disown
```

**4. Add the public hostname** — back in the dashboard, on the tunnel's
**Public Hostname** tab → **Add a public hostname**:

- **Subdomain**: e.g. `llm`
- **Domain**: pick yours from the dropdown
- **Type**: **`HTTP`** — not `HTTPS`. This app serves plain HTTP; setting
  `HTTPS` here makes cloudflared try to TLS-handshake against a plain HTTP
  port, which fails every request with `tls: first record does not look
  like a TLS handshake` (visible in `~/cloudflared.log`) and a 502 to the
  client.
- **URL**: `localhost:<PORT>` — must exactly match `PORT` in your `.env`
  (default `8000`). If you ever change `PORT`, update this field too, or
  you'll get a 502 even though your server is healthy locally.
- **Save**.

**5. Test:**

```bash
curl https://llm.yourdomain.com/health
curl https://llm.yourdomain.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

### Updating or rotating the tunnel token

The token only lives in the `cloudflared tunnel run --token <TOKEN>` command
you started — it's never stored in this repo (and shouldn't be committed
anywhere). To get it again, or a fresh one:

- **Reuse the existing token** (e.g. restarting after a reboot): dashboard →
  **Networks → Tunnels → llm-api** → open the connector install instructions
  again (same place as step 2 above) — it shows the same token. Copy it and
  reuse it in the `tunnel run --token` command.
- **Rotate to a new token** (e.g. the old one may have leaked): in the
  tunnel's connector settings, use **Refresh token** — this invalidates the
  old one immediately, so any currently-running `cloudflared tunnel run`
  process using it will disconnect.
- Either way, apply it by restarting the connector:

```bash
pkill -f "cloudflared tunnel run"
sleep 1
nohup ~/cloudflared tunnel run --token <TOKEN> > ~/cloudflared.log 2>&1 &
disown
sleep 5 && tail -n 20 ~/cloudflared.log
```

Treat the token like a credential (anyone holding it can run a connector that
receives traffic meant for your tunnel) — don't paste it into the repo, a
commit, or anywhere public.

### Keeping both processes running in the background

Two long-running processes need to survive your SSH session ending: the API
server and the tunnel connector. Same `nohup ... & disown` pattern for both:

```bash
# API server
cd ~/LLM_infereceAPI
nohup python3 -m app.main > server.log 2>&1 &
disown

# Tunnel connector
nohup ~/cloudflared tunnel run --token <TOKEN> > ~/cloudflared.log 2>&1 &
disown
```

Check both are alive at any time:

```bash
ps aux | grep -E "app.main|cloudflared" | grep -v grep
```

Neither currently survives a server reboot/container restart on its own —
after one, re-run both `nohup` commands above manually. (With sudo/systemd
access, `sudo ~/cloudflared service install --token <TOKEN>` plus a systemd
unit for the Python app would make both persist across reboots.)

### Troubleshooting quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| Cloudflare error 1033 | `cloudflared tunnel run` isn't running | `ps aux \| grep cloudflared`; restart it with the token command above |
| Cloudflare error 502, but `curl http://localhost:<PORT>/health` works locally | Public Hostname **Type** is `HTTPS` instead of `HTTP`, or the **URL** port doesn't match your app's `PORT` | Fix both fields on the dashboard's Public Hostname tab |
| Public curl hangs / `~/cloudflared.log` shows repeated QUIC errors, no `Registered tunnel connection` | Outbound UDP blocked/unreliable | Restart the connector with `--protocol http2` |
| Public URL returns "Invalid or missing API key" even with the right key | `.env`'s `API_KEY` changed after the server last started | Restart the API server — env vars are read once at startup, not hot-reloaded |
| `curl` right after a restart returns nothing / 502 briefly | Model is still loading (lifespan startup isn't done) | Wait for `Application startup complete` in `server.log` before testing |

### Harden it before leaving it running

- Keep `API_KEY` set in `.env` — the app rejects requests without the correct
  `Authorization: Bearer` header once it's set.
- Consider adding a **Cloudflare Access** application (Zero Trust dashboard) in
  front of the tunnel hostname, requiring login (Google/GitHub/OTP) before
  traffic even reaches your API — a second layer beyond the API key.
- Consider a Cloudflare rate-limiting rule on the hostname to cap requests/min
  — GPU inference is expensive to let strangers hammer, and this server only
  runs `MAX_CONCURRENT_REQUESTS` generations at once; beyond `MAX_QUEUE_SIZE`
  queued on top of that, requests get HTTP 503 (see "Concurrency, queueing,
  and idle VRAM release" above).
