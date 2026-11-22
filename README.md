# YuLan QSA vLLM

BF16 paged GDN/QSA inference plugin for NVIDIA A800, using vLLM 0.30.0 and TP2
by default. The repository includes the YuLan tool parser and chat template.

## Start from a clone

Use a compatible YuLan QSA Hugging Face checkpoint and a CUDA development
toolchain. The plugin compiles its CUDA top-k kernel on first load.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .

CUDA_DEVICE_ORDER=PCI_BUS_ID \
QSA_MODEL_DIR=/path/to/qsa-hf \
QSA_PYTHON_BIN="$PWD/.venv/bin/python" \
QSA_CUDA_VISIBLE_DEVICES=0,1 \
bash serve.sh
```

`serve.sh` loads the plugin from the checkout and enables
`--enable-auto-tool-choice --tool-call-parser yulan` by default. With vLLM 0.30.0
already installed, run the same script with that environment's Python.
An installed plugin also registers `yulan` for direct `vllm serve` launches
using those two flags.

## Chat Completions and PinchBench

Use `http://127.0.0.1:8000/v1` as the OpenAI base URL and `yulan-qsa`
as the model name. Send tool definitions in `tools` and use
`"tool_choice": "auto"`. The parser accepts Qwen3-Coder XML and Hermes JSON.
Streaming responses emit each completed call with its index, ID, name, and
complete JSON arguments. A successful automatic call ends with
`finish_reason="tool_calls"`.

YuLan QSA checkpoints without a tokenizer chat template use the bundled
tool-aware template. Set `QSA_CHAT_TEMPLATE` to override the selected template,
or `QSA_ENABLE_AUTO_TOOL_CHOICE=false` to disable automatic tool parsing.
See `env_paths.sh` for other serving options and `NOTICE.md` for source attribution.

Run the parser and Chat Completions regression tests without a GPU.

```bash
python -m unittest discover -s tests -v
```

Set `QSA_TEST_TOKENIZER` to a local YuLan checkpoint directory to run the same
tests with its tokenizer.
