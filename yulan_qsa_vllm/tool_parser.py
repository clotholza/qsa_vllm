# SPDX-License-Identifier: Apache-2.0
"""Tool-call parsing for YuLan's Qwen3-Coder XML and Hermes JSON formats."""

import json
import re
from collections.abc import Sequence

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
from vllm.tool_parsers.utils import Tool, partial_tag_overlap

_TOOL_START = "<tool_call>"
_TOOL_END = "</tool_call>"
_FUNCTION_START = "<function="
_FUNCTION_END = "</function>"
_WHITESPACE = re.compile(r"\s*")
# A tool end marker inside a JSON string is part of the argument value.
_JSON_TOOL_END = re.compile(
    r'"(?:[^"\\]|\\.)*+(?:"|\\?\Z)|(?P<end></tool_call>)', re.DOTALL
)


@ToolParserManager.register_module("yulan")
class YulanToolParser(ToolParser):
    # vLLM 0.30 calls finish_streaming on the final generation delta, even
    # when that delta contains text and there is no extra empty chunk.
    engine_based_streaming = True

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self._qwen: Qwen3EngineToolParser | None = None
        self._pending = ""
        self._request: ChatCompletionRequest | None = None
        self._stream_finished = False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def _extract_xml(
        self, payload: str, request: ChatCompletionRequest
    ) -> list[ToolCall]:
        if self._qwen is None:
            self._qwen = Qwen3EngineToolParser(self.model_tokenizer, self.tools)
        if _FUNCTION_END not in payload:
            payload += _FUNCTION_END
        result = self._qwen.extract_tool_calls(
            _TOOL_START + payload + _TOOL_END, request
        )
        return result.tool_calls

    @staticmethod
    def _extract_json(payload: str) -> list[ToolCall]:
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if not isinstance(obj, dict):
            return []
        name = obj.get("name")
        arguments = obj.get("arguments")
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            return []
        return [
            ToolCall(
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
        ]

    def _parse_frames(
        self, text: str, request: ChatCompletionRequest, *, finished: bool
    ) -> tuple[str, list[ToolCall], str]:
        """Consume complete tool frames and return any unresolved suffix."""
        content: list[str] = []
        calls: list[ToolCall] = []
        pos = 0
        while pos < len(text):
            start = text.find(_TOOL_START, pos)
            if start == -1:
                held = 0 if finished else partial_tag_overlap(text[pos:], _TOOL_START)
                end = len(text) - held
                content.append(text[pos:end])
                pos = end
                break

            content.append(text[pos:start])
            body = start + len(_TOOL_START)
            payload_start = _WHITESPACE.match(text, body).end()
            is_json = text.startswith("{", payload_start)
            is_xml = text.startswith(_FUNCTION_START, payload_start)
            if not (is_json or is_xml):
                if not finished and _FUNCTION_START.startswith(text[payload_start:]):
                    pos = start
                    break
                content.append(_TOOL_START)
                pos = body
                continue

            end = text.find(_TOOL_END, body)
            if is_json and end != -1:
                end = next(
                    (
                        match.start()
                        for match in _JSON_TOOL_END.finditer(text, body)
                        if match.lastgroup == "end"
                    ),
                    -1,
                )

            if end == -1 and not finished:
                pos = start
                break

            pos = len(text) if end == -1 else end + len(_TOOL_END)
            payload = text[body:] if end == -1 else text[body:end]
            parsed = (
                self._extract_json(payload)
                if is_json
                else self._extract_xml(payload, request)
            )
            if parsed:
                calls.extend(parsed)
            else:
                content.append(text[start:pos])

        return "".join(content), calls, text[pos:]

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        if request.tool_choice == "none":
            content, calls = model_output, []
        else:
            content, calls, _ = self._parse_frames(model_output, request, finished=True)
            # The legacy XML parser also accepts a function with no outer tag.
            if not calls and _TOOL_START not in model_output:
                prefix, marker, payload = model_output.partition(_FUNCTION_START)
                if marker:
                    calls = self._extract_xml(marker + payload, request)
                    if calls:
                        content = prefix
        return ExtractedToolCallInformation(
            tools_called=bool(calls),
            tool_calls=calls,
            content=(content or None) if calls else content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Engine-based vLLM serving leaves previous_text empty on every
        # delta. Only its end-of-stream hook marks the instance for reuse.
        if self._stream_finished:
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = []
            self.current_tool_id = -1
            self.current_tool_name_sent = False
            self._stream_finished = False
        if request.tool_choice == "none":
            self._request = None
            return DeltaMessage(content=delta_text) if delta_text else None

        self._request = request
        self._pending += delta_text
        return self._drain(finished=False)

    def finish_streaming(self) -> DeltaMessage | None:
        delta = self._drain(finished=True)
        self._request = None
        self._stream_finished = True
        return delta

    def _drain(self, *, finished: bool) -> DeltaMessage | None:
        if self._request is None or not self._pending:
            return None

        content, calls, self._pending = self._parse_frames(
            self._pending, self._request, finished=finished
        )
        deltas = []
        for call in calls:
            name = call.function.name
            arguments = call.function.arguments
            index = len(self.prev_tool_call_arr)
            self.prev_tool_call_arr.append({"name": name, "arguments": arguments})
            self.streamed_args_for_tool.append(arguments)
            self.current_tool_id = index
            self.current_tool_name_sent = True
            deltas.append(
                DeltaToolCall(
                    index=index,
                    id=call.id,
                    type="function",
                    function=DeltaFunctionCall(name=name, arguments=arguments),
                )
            )
        if content or deltas:
            return DeltaMessage(content=content or None, tool_calls=deltas)
        return None
