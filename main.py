from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register


PLUGIN_ID = "astrbot_plugin_message_merger"
EXTRA_PREFIX = "message_merger"


@dataclass
class Burst:
    messages: list[str] = field(default_factory=list)
    current_event: AstrMessageEvent | None = None
    updated_at: float = field(default_factory=time.monotonic)


@register(PLUGIN_ID, "Codex", "立即回复并在后续消息到达时乐观合并", "1.0.1")
class MessageMergerPlugin(Star):
    """Merge follow-up user messages without delaying the initial request."""

    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config = config
        self._bursts: dict[str, Burst] = {}

    @filter.regex(r"^[\s\S]*$")
    async def capture_incoming_message(self, event: AstrMessageEvent):
        """Capture before the event reaches the per-conversation LLM queue."""
        if self._is_early_llm_candidate(event):
            self._capture(event)
        if False:
            # Regex handlers are async generators in AstrBot's plugin pipeline.
            yield None

    @filter.on_waiting_llm_request(priority=1000)
    async def collect_message(self, event: AstrMessageEvent) -> None:
        # Some adapters decide call_llm after regular handlers; capture those here.
        if event.get_extra(self._extra_key("captured"), False):
            return
        self._capture(event)

    def _capture(self, event: AstrMessageEvent) -> None:
        if not self._enabled():
            return
        text = self._message_text(event)
        key = self._conversation_key(event)
        if not text or not key or not self._mergeable(event, text):
            return

        self._prune_bursts()
        burst = self._bursts.get(key)
        if burst is None:
            burst = Burst()
            self._bursts[key] = burst
        else:
            previous = burst.current_event
            if previous is not None and previous is not event:
                # The previous LLM result may still be running; suppress it at send time.
                previous.set_extra(self._extra_key("stale"), True)

        max_messages = self._safe_int("max_messages", 8, minimum=0)
        max_chars = self._safe_int("max_chars", 4000, minimum=0)
        burst.messages.append(text)
        if max_messages > 0 and len(burst.messages) > max_messages:
            burst.messages = burst.messages[-max_messages:]
        while (
            max_chars > 0
            and sum(len(item) for item in burst.messages) > max_chars
            and len(burst.messages) > 1
        ):
            burst.messages.pop(0)

        burst.current_event = event
        burst.updated_at = time.monotonic()
        event.set_extra(self._extra_key("captured"), True)
        event.set_extra(self._extra_key("key"), key)
        event.set_extra(self._extra_key("merged_messages"), list(burst.messages))

    @filter.on_llm_request(priority=1000)
    async def merge_request(self, event: AstrMessageEvent, request: Any) -> None:
        if event.get_extra(self._extra_key("stale"), False):
            event.stop_event()
            return

        messages = event.get_extra(self._extra_key("merged_messages"), None)
        if not isinstance(messages, list) or len(messages) < 2:
            return
        parts = [item.strip() for item in messages if isinstance(item, str) and item.strip()]
        if len(parts) >= 2:
            request.prompt = self._join_messages(parts)

    @filter.on_decorating_result(priority=1000)
    async def suppress_stale_result(self, event: AstrMessageEvent) -> None:
        if event.get_extra(self._extra_key("stale"), False):
            result = event.get_result()
            if result is not None and getattr(result, "chain", None) is not None:
                result.chain = []
            event.stop_event()
            return

        # Clean up before sending because another plugin may stop after_message_sent hooks.
        key = event.get_extra(self._extra_key("key"), None)
        if not isinstance(key, str):
            return
        burst = self._bursts.get(key)
        if burst is not None and burst.current_event is event:
            self._bursts.pop(key, None)

    def _is_early_llm_candidate(self, event: AstrMessageEvent) -> bool:
        if not self._enabled():
            return False
        private_getter = getattr(event, "is_private_chat", None)
        if callable(private_getter) and private_getter():
            return True
        return bool(
            getattr(event, "call_llm", False)
            or getattr(event, "is_wake", False)
            or getattr(event, "is_at_or_wake_command", False)
        )

    def _prune_bursts(self) -> None:
        cutoff = time.monotonic() - 120.0
        expired = [key for key, burst in self._bursts.items() if burst.updated_at < cutoff]
        for key in expired:
            self._bursts.pop(key, None)

    def _join_messages(self, messages: list[str]) -> str:
        return "\n".join(messages)

    def _mergeable(self, event: AstrMessageEvent, text: str) -> bool:
        prefixes = self._config_get("ignore_prefixes", ["/", "!"])
        if isinstance(prefixes, list) and any(
            text.startswith(item) for item in prefixes if isinstance(item, str)
        ):
            return False
        message_obj = getattr(event, "message_obj", None)
        components = getattr(message_obj, "message", None)
        if not isinstance(components, (list, tuple)):
            return True
        allowed = {"Plain", "At", "Reply"}
        return all(type(component).__name__ in allowed for component in components)

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
