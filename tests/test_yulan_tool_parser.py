"""CPU regressions for the YuLan parser and vLLM 0.30 Chat Completions."""

import json
import os
import unittest
from types import SimpleNamespace

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.plugins import load_general_plugins
from vllm.renderers.hf import resolve_chat_template
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.hf import get_cached_tokenizer
from vllm.tool_parsers import ToolParserManager

ARGUMENTS = {"path": "/tmp/test.txt", "content": "你好\nhello", "count": 3}
XML = (
    "<tool_call>\n<function=write>\n"
    "<parameter=path>\n/tmp/test.txt\n</parameter>\n"
    "<parameter=content>\n你好\nhello\n</parameter>\n"
    "<parameter=count>\n3\n</parameter>\n</function>\n</tool_call>"
)
HERMES = (
    '<tool_call>{"name":"write","arguments":'
    + json.dumps(ARGUMENTS, ensure_ascii=False)
    + "}</tool_call>"
)


def chat_request(*, stream=True, tool_choice="auto"):
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "test"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                },
            },
        }],
        stream=stream,
        tool_choice=tool_choice,
    )


def load_tokenizer():
    model = os.environ.get("QSA_TEST_TOKENIZER")
    if model:
        return get_tokenizer(model, trust_remote_code=True)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    return get_cached_tokenizer(PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        additional_special_tokens=["<tool_call>", "</tool_call>"],
    ))


class ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_general_plugins()
        cls.parser_cls = ToolParserManager.get_tool_parser("yulan")
        cls.tokenizer = load_tokenizer()
        cls.request = chat_request()

    def assert_stream(self, chunks, expected, content="", parser=None):
        parser = parser or self.parser_cls(self.tokenizer)
        previous = ""
        deltas = []
        for chunk in chunks:
            current = previous + chunk
            deltas.append(parser.extract_tool_calls_streaming(
                previous, current, chunk, (), (), (), self.request
            ))
            previous = current
        deltas.append(parser.finish_streaming())
        prose = []
        calls = []
        for delta in filter(None, deltas):
            prose.append(delta.content or "")
            for call in delta.tool_calls:
                self.assertEqual(call.index, len(calls))
                self.assertTrue(call.id)
                self.assertEqual(call.type, "function")
                calls.append((call.function.name, json.loads(call.function.arguments)))
        self.assertEqual("".join(prose), content)
        self.assertEqual(calls, [("write", args) for args in expected])
        self.assertEqual(len(parser.prev_tool_call_arr), len(expected))
        self.assertEqual(parser.get_remaining_unstreamed_args(), "")
        return parser

    def test_chunk_boundaries(self):
        for text in (XML, HERMES):
            self.assert_stream(list(text), [ARGUMENTS])
            for cut in range(1, len(text)):
                with self.subTest(format=text[:24], cut=cut):
                    self.assert_stream([text[:cut], text[cut:]], [ARGUMENTS])

    def test_multiple_calls_and_prose(self):
        for text in (XML + XML, HERMES + HERMES, XML + HERMES):
            response = "Before. " + text + " After. <tool_cal"
            for size in (7, 31, len(response)):
                chunks = [response[i:i + size] for i in range(0, len(response), size)]
                self.assert_stream(chunks, [ARGUMENTS, ARGUMENTS], "Before.  After. <tool_cal")

    def test_missing_closing_tags_and_reuse(self):
        parser = self.assert_stream([XML.replace("</function>", "")], [ARGUMENTS])
        self.assert_stream([HERMES.removesuffix("</tool_call>")], [ARGUMENTS], parser=parser)
        self.assert_stream(["Literal <tool", "", "_cal"], [], "Literal <tool_cal", parser)

    def test_json_with_literal_markers(self):
        arguments = {**ARGUMENTS, "content": 'quoted "string" </tool_call><tool_call>'}
        response = (
            '<tool_call>{"arguments": ' + json.dumps(arguments) + ', "name": "write"}'
            '</tool_call>'
        )
        self.assert_stream(list(response), [arguments])

    def test_non_streaming_formats_and_plain_text(self):
        parser = self.parser_cls(self.tokenizer)
        for text in (XML, HERMES, XML.replace("</function>", "")):
            result = parser.extract_tool_calls(text, self.request)
            self.assertTrue(result.tools_called)
            self.assertEqual(result.tool_calls[0].function.name, "write")
            self.assertEqual(json.loads(result.tool_calls[0].function.arguments), ARGUMENTS)
        text = "Literal <tool_call> and <tool_cal"
        result = parser.extract_tool_calls(text, self.request)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, text)

    def test_request_adjustment_and_template_fallback(self):
        parser = self.parser_cls(self.tokenizer)
        self.assertFalse(parser.adjust_request(chat_request()).skip_special_tokens)
        disabled = chat_request(tool_choice="none")
        self.assertTrue(parser.adjust_request(disabled).skip_special_tokens)
        delta = parser.extract_tool_calls_streaming("", XML, XML, (), (), (), disabled)
        self.assertEqual(delta.content, XML)
        self.assertFalse(delta.tool_calls)
        self.assertIsNone(parser.finish_streaming())

        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel()))
        template = resolve_chat_template(
            tokenizer, None, [{"type": "function"}],
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="yulan_qsa")),
        )
        self.assertIn("<function=example_function_name>", template)
        tokenizer.chat_template = "{{ messages[0].content }}"
        self.assertEqual(
            resolve_chat_template(tokenizer, None, [], model_config=None),
            tokenizer.chat_template,
        )


class ChatCompletionsTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        load_general_plugins()
        cls.tokenizer = load_tokenizer()
        model_config = SimpleNamespace(
            model="test",
            hf_config=SimpleNamespace(model_type="yulan_qsa"),
            get_diff_sampling_param=dict,
            generation_config="vllm",
        )
        cls.serving = OpenAIServingChat(
            SimpleNamespace(model_config=model_config, renderer=None, input_processor=None),
            None,
            "assistant",
            online_renderer=None,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=True,
            tool_parser="yulan",
        )

    async def engine_outputs(self, chunks):
        for i, chunk in enumerate(chunks):
            finished = i == len(chunks) - 1
            yield RequestOutput(
                request_id="test",
                prompt=None,
                prompt_token_ids=[0],
                prompt_logprobs=None,
                outputs=[CompletionOutput(
                    index=0,
                    text=chunk,
                    token_ids=self.tokenizer.encode(chunk) or [0],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="stop" if finished else None,
                )],
                finished=finished,
            )

    async def sse_choices(self, chunks):
        choices = []
        async for event in self.serving.chat_completion_stream_generator(
            chat_request(), self.engine_outputs(chunks), "test", "test", [],
            self.tokenizer, RequestResponseMetadata(request_id="test"),
        ):
            payload = event.removeprefix("data: ").strip()
            if payload != "[DONE]":
                data = json.loads(payload)
                self.assertIn("choices", data)
                choices.extend(data["choices"])
        return choices

    async def test_streaming_finish_on_last_nonempty_chunk(self):
        choices = await self.sse_choices(
            ["Before. <tool_", "call>" + XML.removeprefix("<tool_call>"), HERMES, " <tool_cal"]
        )
        self.assertEqual(choices[-1]["finish_reason"], "tool_calls")
        self.assertEqual(
            "".join(choice["delta"].get("content") or "" for choice in choices),
            "Before.  <tool_cal",
        )
        calls = {}
        for choice in choices:
            for call in choice["delta"].get("tool_calls", []):
                self.assertTrue(call["id"])
                calls[call["index"]] = json.loads(call["function"]["arguments"])
        self.assertEqual(calls, {0: ARGUMENTS, 1: ARGUMENTS})

        prose = await self.sse_choices(["Literal <tool_cal"])
        self.assertEqual(prose[-1]["finish_reason"], "stop")
        self.assertEqual(prose[-1]["delta"]["content"], "Literal <tool_cal")

        stopped = await self.sse_choices([HERMES.removesuffix("</tool_call>")])
        self.assertEqual(stopped[-1]["finish_reason"], "tool_calls")
        self.assertEqual(
            json.loads(stopped[-1]["delta"]["tool_calls"][0]["function"]["arguments"]),
            ARGUMENTS,
        )

    async def test_non_streaming_response(self):
        request = chat_request(stream=False)
        parser = self.serving.parser_cls(self.tokenizer, request.tools)
        response = await self.serving.chat_completion_full_generator(
            request, self.engine_outputs(["Before. " + HERMES]), "test", "test", [],
            self.tokenizer, RequestResponseMetadata(request_id="test"), parser=parser,
        )
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(choice.message.content, "Before. ")
        self.assertEqual(json.loads(choice.message.tool_calls[0].function.arguments), ARGUMENTS)
