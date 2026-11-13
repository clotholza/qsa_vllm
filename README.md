# YuLan QSA vLLM

BF16 paged GDN/QSA inference plugin for NVIDIA A800, using vLLM 0.30.0 and TP2.

Run from the source checkout with a compatible HF checkpoint and a Python 3.12
environment containing the dependencies in `requirements.txt`.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
QSA_MODEL_DIR=/path/to/qsa-hf \
QSA_PYTHON_BIN=/path/to/venv/bin/python \
QSA_CUDA_VISIBLE_DEVICES=0,1 \
bash serve.sh
```

See `env_paths.sh` for serving options and `NOTICE.md` for source attribution.
