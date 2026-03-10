# NorthStar - Installation Guide

## Hardware Required (non-negotiable)
- GPU: NVIDIA RTX 4060 8GB minimum
- NPU: Intel AI Boost (Intel Core Ultra)
- RAM: 32 GB minimum
- Storage: 50 GB free SSD
- OS: Windows 11

## Software Required (all free)
- Rust toolchain: rustup.rs
- CUDA Toolkit 12.x: developer.nvidia.com
- Node.js 18+: nodejs.org
- Python 3.10+
- llama-server: github.com/ggerganov/llama.cpp/releases

## API Keys Required (free accounts)
- Exchange API keys (Kraken, Coinbase, Binance etc.)
- NVIDIA NIM API key: build.nvidia.com

## AI Model Files (~9GB total, all free from HuggingFace)
- Nemotron Nano 9B Q5_K_M (~5.5 GB)
- OpenReasoning 1.5B Q8_0 (~1.6 GB)
- Phi-3 mini int4 OpenVINO (~2.0 GB)

All download links in docs/MODEL_SETUP.md after purchase.

## Steps
1. Purchase at kittick.gumroad.com/l/northstarcrypto
2. Email kittickds@icloud.com with your GitHub username
3. Accept repo invite and clone
4. Copy .env.example to .env and add your API keys
5. Run: cd rust_core && cargo build --release
6. Run: cd codezero-team-hub && npm install
7. Run: start_all.bat
8. Open: http://localhost:3000

Start with PAPER_TRADING=1 to test before going live.
Full guide: docs/OPERATOR_RUNBOOK.md
