from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

from openai.resources.chat.completions.completions import (
    AsyncCompletions,
    Completions,
)


class OpenAIInterceptor:
    """
    Intercepts OpenAI Chat Completions calls.

    It records:
    - Complete messages sent to OpenAI
    - Model and request parameters
    - Structured response format
    - Assistant/Judge response
    - Prompt, completion and total token usage
    - Request ID
    - API errors

    This tool patches OpenAI SDK resource methods temporarily. Calling
    disable(), or leaving the context manager, restores the original methods.
    """

    def __init__(
        self,
        log_file: str | Path,
        *,
        clear_existing: bool = True,
        print_requests: bool = False,
        print_responses: bool = False,
        capture_full_messages: bool = True,
        capture_full_response: bool = True,
    ) -> None:
        self.log_file = Path(log_file)
        self.clear_existing = clear_existing
        self.print_requests = print_requests
        self.print_responses = print_responses
        self.capture_full_messages = capture_full_messages
        self.capture_full_response = capture_full_response

        self._enabled = False
        self._write_lock = threading.Lock()
        self._original_methods: dict[tuple[type[Any], str], Any] = {}

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert SDK, Pydantic and ordinary Python values to JSON."""
        if value is None:
            return None

        if hasattr(value, "model_dump"):
            try:
                return value.model_dump(mode="json")
            except TypeError:
                return value.model_dump()

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {
                str(key): OpenAIInterceptor._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                OpenAIInterceptor._json_safe(item)
                for item in value
            ]

        if isinstance(value, (str, int, float, bool)):
            return value

        return str(value)

    @classmethod
    def _extract_usage(cls, response: Any) -> dict[str, Any] | None:
        usage = getattr(response, "usage", None)

        if usage is None:
            return None

        safe_usage = cls._json_safe(usage)

        if isinstance(safe_usage, dict):
            return safe_usage

        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(
                usage,
                "completion_tokens",
                None,
            ),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    @classmethod
    def _extract_assistant_output(cls, response: Any) -> Any:
        choices = getattr(response, "choices", None)

        if not choices:
            return None

        outputs: list[dict[str, Any]] = []

        for choice in choices:
            message = getattr(choice, "message", None)

            if message is None:
                outputs.append(
                    {
                        "choice": cls._json_safe(choice),
                    }
                )
                continue

            output: dict[str, Any] = {
                "role": getattr(message, "role", None),
                "content": getattr(message, "content", None),
                "finish_reason": getattr(
                    choice,
                    "finish_reason",
                    None,
                ),
            }

            parsed = getattr(message, "parsed", None)
            if parsed is not None:
                output["parsed"] = cls._json_safe(parsed)

            refusal = getattr(message, "refusal", None)
            if refusal is not None:
                output["refusal"] = cls._json_safe(refusal)

            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls is not None:
                output["tool_calls"] = cls._json_safe(tool_calls)

            outputs.append(output)

        return outputs

    def _write(self, record: dict[str, Any]) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        complete_record = {
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            **record,
        }

        line = json.dumps(
            complete_record,
            ensure_ascii=False,
            default=str,
        )

        with self._write_lock:
            with self.log_file.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    @staticmethod
    def _display(
        heading: str,
        record: dict[str, Any],
    ) -> None:
        print(f"\n{'=' * 20} {heading} {'=' * 20}")
        print(
            json.dumps(
                record,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    def _build_request_record(
        self,
        interaction_id: str,
        method_name: str,
        kwargs: dict[str, Any],
        *,
        asynchronous: bool,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "interaction_id": interaction_id,
            "event": "request",
            "api": "chat.completions",
            "method": method_name,
            "asynchronous": asynchronous,
            "model": kwargs.get("model"),
            "temperature": self._json_safe(
                kwargs.get("temperature")
            ),
            "top_p": self._json_safe(kwargs.get("top_p")),
            "max_tokens": self._json_safe(
                kwargs.get("max_tokens")
            ),
            "max_completion_tokens": self._json_safe(
                kwargs.get("max_completion_tokens")
            ),
            "seed": self._json_safe(kwargs.get("seed")),
            "stop": self._json_safe(kwargs.get("stop")),
            "response_format": self._json_safe(
                kwargs.get("response_format")
            ),
            "tools": self._json_safe(kwargs.get("tools")),
            "tool_choice": self._json_safe(
                kwargs.get("tool_choice")
            ),
            "logprobs": self._json_safe(
                kwargs.get("logprobs")
            ),
            "top_logprobs": self._json_safe(
                kwargs.get("top_logprobs")
            ),
            "stream": self._json_safe(kwargs.get("stream")),
        }

        if self.capture_full_messages:
            record["messages"] = self._json_safe(
                kwargs.get("messages")
            )
        else:
            messages = kwargs.get("messages") or []
            record["message_count"] = len(messages)

        return record

    def _build_response_record(
        self,
        interaction_id: str,
        method_name: str,
        response: Any,
        *,
        asynchronous: bool,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "interaction_id": interaction_id,
            "event": "response",
            "api": "chat.completions",
            "method": method_name,
            "asynchronous": asynchronous,
            "model": getattr(response, "model", None),
            "request_id": getattr(response, "_request_id", None),
            "usage": self._extract_usage(response),
            "assistant_output": self._extract_assistant_output(
                response
            ),
        }

        if self.capture_full_response:
            record["full_response"] = self._json_safe(response)

        return record

    def _patch_sync_method(self, method_name: str) -> None:
        resource_class = Completions
        original = getattr(resource_class, method_name)

        self._original_methods[(resource_class, method_name)] = (
            original
        )

        @wraps(original)
        def wrapper(
            resource: Completions,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            interaction_id = str(uuid.uuid4())

            request_record = self._build_request_record(
                interaction_id,
                method_name,
                kwargs,
                asynchronous=False,
            )
            self._write(request_record)

            if self.print_requests:
                self._display("OPENAI REQUEST", request_record)

            try:
                response = original(resource, *args, **kwargs)
            except Exception as error:
                error_record = {
                    "interaction_id": interaction_id,
                    "event": "error",
                    "api": "chat.completions",
                    "method": method_name,
                    "asynchronous": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }

                self._write(error_record)

                if self.print_responses:
                    self._display("OPENAI ERROR", error_record)

                raise

            response_record = self._build_response_record(
                interaction_id,
                method_name,
                response,
                asynchronous=False,
            )
            self._write(response_record)

            if self.print_responses:
                self._display("OPENAI RESPONSE", response_record)

            return response

        setattr(resource_class, method_name, wrapper)

    def _patch_async_method(self, method_name: str) -> None:
        resource_class = AsyncCompletions
        original = getattr(resource_class, method_name)

        self._original_methods[(resource_class, method_name)] = (
            original
        )

        @wraps(original)
        async def wrapper(
            resource: AsyncCompletions,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            interaction_id = str(uuid.uuid4())

            request_record = self._build_request_record(
                interaction_id,
                method_name,
                kwargs,
                asynchronous=True,
            )
            self._write(request_record)

            if self.print_requests:
                self._display(
                    "OPENAI ASYNC REQUEST",
                    request_record,
                )

            try:
                response = await original(
                    resource,
                    *args,
                    **kwargs,
                )
            except Exception as error:
                error_record = {
                    "interaction_id": interaction_id,
                    "event": "error",
                    "api": "chat.completions",
                    "method": method_name,
                    "asynchronous": True,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }

                self._write(error_record)

                if self.print_responses:
                    self._display(
                        "OPENAI ASYNC ERROR",
                        error_record,
                    )

                raise

            response_record = self._build_response_record(
                interaction_id,
                method_name,
                response,
                asynchronous=True,
            )
            self._write(response_record)

            if self.print_responses:
                self._display(
                    "OPENAI ASYNC RESPONSE",
                    response_record,
                )

            return response

        setattr(resource_class, method_name, wrapper)

    def enable(self) -> None:
        if self._enabled:
            return

        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        if self.clear_existing and self.log_file.exists():
            self.log_file.unlink()

        for method_name in ("create", "parse"):
            if hasattr(Completions, method_name):
                self._patch_sync_method(method_name)

            if hasattr(AsyncCompletions, method_name):
                self._patch_async_method(method_name)

        self._enabled = True

    def disable(self) -> None:
        if not self._enabled:
            return

        for (
            resource_class,
            method_name,
        ), original_method in self._original_methods.items():
            setattr(
                resource_class,
                method_name,
                original_method,
            )

        self._original_methods.clear()
        self._enabled = False

    def __enter__(self) -> "OpenAIInterceptor":
        self.enable()
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception: Any,
        traceback: Any,
    ) -> None:
        self.disable()