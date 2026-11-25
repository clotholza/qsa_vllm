"""Regression coverage for vLLM's loading of YuLan tokenizer.json."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from vllm.plugins import load_general_plugins
from vllm.tokenizers import get_tokenizer


class TokenizerTests(unittest.TestCase):
    def test_vllm_preserves_serialized_yulan_normalizer(self):
        load_general_plugins()
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text(
                json.dumps({"model_type": "yulan_qsa"}), encoding="utf-8"
            )
            (checkpoint / "tokenizer_config.json").write_text(json.dumps({
                "tokenizer_class": "LlamaTokenizer",
                "eos_token": "<|im_end|>",
                "unk_token": "<unk>",
                "chat_template": (
                    "<|im_start|>user\n{{ messages[0].content }}<|im_end|>\n"
                    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
                ),
            }), encoding="utf-8")

            tokens = ["<unk>", "<|im_start|>", "<|im_end|>", "▁", "\n"]
            tokens.extend(sorted(set("userhelloassistant你好")))
            reference = Tokenizer(BPE(
                vocab={token: index for index, token in enumerate(tokens)},
                merges=[],
                unk_token="<unk>",
            ))
            reference.add_special_tokens(tokens[:3])
            reference.normalizer = normalizers.Sequence([
                normalizers.Prepend("▁"),
                normalizers.Replace(" ", "▁"),
                normalizers.Replace("　", "▁"),
            ])
            reference.save(str(checkpoint / "tokenizer.json"))

            tokenizer = get_tokenizer(checkpoint)
            self.assertEqual(tokenizer.eos_token_id, 2)
            for content in ("hello", "你　好"):
                with self.subTest(content=content):
                    rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    self.assertEqual(
                        tokenizer.encode(rendered, add_special_tokens=False),
                        reference.encode(rendered, add_special_tokens=False).ids,
                    )
