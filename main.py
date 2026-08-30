from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register


PLUGIN_ID = "astrbot_plugin_message_merger"
EXTRA_PREFIX = "message_merger"


@dataclass
class BufferedMessage:
    sequence: tuple[float, int]
    text: str


@dataclass
class Burst:
    messages: list[BufferedMessage] = field(default_factory=list)
    current_event: AstrMessageEvent | None = None
    pipeline_task: asyncio.Task[Any] | None = None
    tools_started: bool = False
    updated_at: float = field(default_factory=time.monotonic)


@register(PLUGIN_ID, "ahuai", "零等待合并用户未说完的连续消息", "1.2.1")
class MessageMergerPlugin(Star):
    """Merge follow-up user messages without delaying the initial request."""

    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config = config
        self._bursts: dict[str, Burst] = {}
        self._arrival_sequence = 0

    @filter.regex(r"^[\s\S]*$", priority=10000)
    async def stamp_message_arrival(self, event: AstrMessageEvent) -> None:
        """Record adapter arrival order before other handlers can delay events."""
        if self._enabled():
            self._ensure_arrival_sequence(event)

    @filter.regex(r"^[\s\S]*$", priority=-10000)
    async def capture_incoming_message(self, event: AstrMessageEvent) -> None:
        """Capture LLM-bound messages before the default pipeline starts."""
        if self._is_early_llm_candidate(event):
            self._capture(event)

    @filter.on_waiting_llm_request(priority=1000)
    async def register_pipeline_task(self, event: AstrMessageEvent) -> None:
        if event.get_extra(self._extra_key("captured"), False):
            key = event.get_extra(self._extra_key("key"), None)
        else:
            self._capture(event)
            key = event.get_extra(self._extra_key("key"), None)

        if not isinstance(key, str):
            return
        burst = self._bursts.get(key)
        is_current_burst = burst is not None and burst.current_event is event
        is_detached_burst = bool(event.get_extra(self._extra_key("detached"), False))
        if not is_current_burst and not is_detached_burst:
            event.set_extra(self._extra_key("stale"), True)
            event.stop_event()
            return

        messages = event.get_extra(self._extra_key("merged_messages"), None)
        if isinstance(messages, list):
            parts = [item.strip() for item in messages if isinstance(item, str) and item.strip()]
            if parts:
                merged = self._join_messages(parts)
                event.message_str = merged
                message_obj = getattr(event, "message_obj", None)
                if message_obj is not None:
                    message_obj.message_str = merged

        if is_current_burst and burst is not None:
            self._bind_pipeline_task(burst, key, event)

    def _capture(self, event: AstrMessageEvent) -> None:
        if not self._enabled():
            return
        text = self._message_text(event)
        key = self._conversation_key(event)
        if not text or not key or not self._mergeable(event, text):
            return

        self._prune_bursts()
        max_messages = self._safe_int("max_messages", 8, minimum=0)
        max_chars = self._safe_int("max_chars", 4000, minimum=0)
        sequence = self._ensure_arrival_sequence(event)
        buffered_message = BufferedMessage(sequence=sequence, text=text)

        burst = self._bursts.get(key)
        start_new_burst = burst is None
        if burst is not None:
            start_new_burst = self._must_start_new_burst(burst) or self._would_exceed_limits(
                burst,
                buffered_message,
                max_messages,
                max_chars,
            )
            if start_new_burst:
                previous = burst.current_event
                if previous is not None:
                    previous.set_extra(self._extra_key("detached"), True)

        if start_new_burst:
            burst = None
        if burst is None:
            burst = Burst()
            self._bursts[key] = burst
        else:
            previous = burst.current_event
            if previous is not None and previous is not event:
                previous.set_extra(self._extra_key("stale"), True)
                previous_task = burst.pipeline_task
                burst.pipeline_task = None
                if previous_task is not None and not previous_task.done():
                    previous_task.cancel()

        burst.messages.append(buffered_message)
        burst.messages.sort(key=lambda item: item.sequence)

        burst.current_event = event
        burst.updated_at = time.monotonic()
        event.set_extra(self._extra_key("captured"), True)
        event.set_extra(self._extra_key("key"), key)
        event.set_extra(
            self._extra_key("merged_messages"),
            [item.text for item in burst.messages],
        )
        self._bind_pipeline_task(burst, key, event)

    @filter.on_llm_request(priority=1000)
    async def merge_request(self, event: AstrMessageEvent, request: Any) -> None:
        if event.get_extra(self._extra_key("stale"), False):
            event.stop_event()
            return

    @filter.on_using_llm_tool(priority=1000)
    async def mark_tool_started(
        self,
        event: AstrMessageEvent,
        tool: Any,
        tool_args: dict[str, Any] | None,
    ) -> None:
        key = event.get_extra(self._extra_key("key"), None)
        if not isinstance(key, str):
            return
        burst = self._bursts.get(key)
        if burst is not None and burst.current_event is event:
            burst.tools_started = True

    @filter.on_decorating_result(priority=1000)
    async def suppress_stale_result(self, event: AstrMessageEvent) -> None:
        if event.get_extra(self._extra_key("stale"), False):
            result = event.get_result()
            if result is not None and getattr(result, "chain", None) is not None:
                result.chain = []
            event.stop_event()
            return

    def _is_early_llm_candidate(self, event: AstrMessageEvent) -> bool:
        if not self._enabled():
            return False
        stopped_getter = getattr(event, "is_stopped", None)
        if callable(stopped_getter) and stopped_getter():
            return False
        if bool(getattr(event, "_has_send_oper", False)):
            return False
        return bool(
            getattr(event, "call_llm", False)
            or getattr(event, "is_wake", False)
            or getattr(event, "is_at_or_wake_command", False)
        )

    def _prune_bursts(self) -> None:
        cutoff = time.monotonic() - 120.0
        expired = [
            key
            for key, burst in self._bursts.items()
            if burst.updated_at < cutoff
            and (burst.pipeline_task is None or burst.pipeline_task.done())
        ]
        for key in expired:
            self._bursts.pop(key, None)

    def _must_start_new_burst(self, burst: Burst) -> bool:
        task = burst.pipeline_task
        if burst.tools_started or (task is not None and task.done()):
            return True
        previous = burst.current_event
        if previous is None:
            return False
        result_getter = getattr(previous, "get_result", None)
        result = result_getter() if callable(result_getter) else None
        content_type = getattr(result, "result_content_type", None)
        content_type_name = getattr(content_type, "name", str(content_type or ""))
        return "STREAMING" in content_type_name.upper()

    def _would_exceed_limits(
        self,
        burst: Burst,
        message: BufferedMessage,
        max_messages: int,
        max_chars: int,
    ) -> bool:
        if max_messages > 0 and len(burst.messages) + 1 > max_messages:
            return True
        if max_chars > 0:
            current_chars = sum(len(item.text) for item in burst.messages)
            if current_chars + len(message.text) > max_chars:
                return True
        return False

    def _on_pipeline_done(
        self,
        key: str,
        event: AstrMessageEvent,
        task: asyncio.Task[Any],
    ) -> None:
        burst = self._bursts.get(key)
        if (
            burst is not None
            and burst.current_event is event
            and burst.pipeline_task is task
        ):
            self._bursts.pop(key, None)

    def _bind_pipeline_task(
        self,
        burst: Burst,
        key: str,
        event: AstrMessageEvent,
    ) -> None:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is None or burst.pipeline_task is task:
            return
        burst.pipeline_task = task
        task.add_done_callback(
            lambda completed, burst_key=key, current_event=event: self._on_pipeline_done(
                burst_key,
                current_event,
                completed,
            )
        )

    def _join_messages(self, messages: list[str]) -> str:
        return "\n".join(messages)

    def _ensure_arrival_sequence(
        self,
        event: AstrMessageEvent,
    ) -> tuple[float, int]:
        key = self._extra_key("arrival_sequence")
        existing = event.get_extra(key, None)
        if (
            isinstance(existing, tuple)
            and len(existing) == 2
            and isinstance(existing[0], int | float)
            and isinstance(existing[1], int)
        ):
            return float(existing[0]), existing[1]

        self._arrival_sequence += 1
        try:
            created_at = float(getattr(event, "created_at", time.time()))
        except (TypeError, ValueError):
            created_at = time.time()
        sequence = (created_at, self._arrival_sequence)
        event.set_extra(key, sequence)
        return sequence

    def _mergeable(self, event: AstrMessageEvent, text: str) -> bool:
        message_obj = getattr(event, "message_obj", None)
        original_text = getattr(message_obj, "message_str", None)
        prefix_text = (
            original_text.strip()
            if isinstance(original_text, str) and original_text.strip()
            else text
        )

        if bool(self._config_get("ignore_prefixes_enabled", True)):
            prefixes = self._config_get("ignore_prefixes", ["/", "!"])
            if isinstance(prefixes, list) and any(
                prefix_text.startswith(item)
                for item in prefixes
                if isinstance(item, str) and item
            ):
                return False

        group_getter = getattr(event, "get_group_id", None)
        group_id = group_getter() if callable(group_getter) else None
        is_group_message = group_id is not None and bool(str(group_id).strip())
        components = getattr(message_obj, "message", None)
        if is_group_message and bool(
            self._config_get("required_prefixes_enabled", False)
        ):
            required_prefixes = self._config_get("required_prefixes", [])
            matches_required_prefix = isinstance(required_prefixes, list) and any(
                prefix_text.startswith(item)
                for item in required_prefixes
                if isinstance(item, str) and item
            )
            if not matches_required_prefix and not self._mentions_bot(event, components):
                return False

        if not isinstance(components, (list, tuple)):
            return True
        allowed = {"Plain", "At"}
        return all(type(component).__name__ in allowed for component in components)

    def _mentions_bot(self, event: AstrMessageEvent, components: Any) -> bool:
        if not isinstance(components, (list, tuple)):
            return False
        self_id_getter = getattr(event, "get_self_id", None)
        self_id = self_id_getter() if callable(self_id_getter) else None
        if self_id is None or not str(self_id).strip():
            return False
        return any(
            type(component).__name__ == "At"
            and str(getattr(component, "qq", "")) == str(self_id)
            for component in components
        )

    def _message_text(self, event: AstrMessageEvent) -> str:
        text = getattr(event, "message_str", "")
        return text.strip() if isinstance(text, str) else ""

    def _conversation_key(self, event: AstrMessageEvent) -> str | None:
        sender_id = self._sender_id(event)
        if not sender_id:
            return None
        group_getter = getattr(event, "get_group_id", None)
        group_id = group_getter() if callable(group_getter) else None
        scope = f"group:{group_id}" if group_id is not None else "private"
        return f"{event.unified_msg_origin}|{scope}|user:{sender_id}"

    def _sender_id(self, event: AstrMessageEvent) -> str | None:
        getter = getattr(event, "get_sender_id", None)
        if callable(getter):
            value = getter()
            if value is not None and str(value).strip():
                return str(value).strip()
        message_obj = getattr(event, "message_obj", None)
        sender = getattr(message_obj, "sender", None)
        for name in ("user_id", "uin", "sender_id", "id"):
            value = getattr(sender, name, None)
            if value is not None and str(value).strip():
                return str(value).strip()
        raw = getattr(message_obj, "raw_message", None)
        if isinstance(raw, dict):
            value = raw.get("user_id") or raw.get("sender_id")
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    def _enabled(self) -> bool:
        return bool(self._config_get("enabled", True))

    def _safe_int(self, key: str, default: int, minimum: int) -> int:
        try:
            return max(minimum, int(self._config_get(key, default)))
        except (TypeError, ValueError):
            logger.warning("[%s] invalid integer config %s; using %s", PLUGIN_ID, key, default)
            return default

    def _config_get(self, key: str, default: Any) -> Any:
        if self.config is None:
            return default
        getter = getattr(self.config, "get", None)
        return getter(key, default) if callable(getter) else getattr(self.config, key, default)

    def _extra_key(self, name: str) -> str:
        return f"{EXTRA_PREFIX}.{name}"
