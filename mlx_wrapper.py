"""
MLXChatModel — LangChain BaseChatModel wrapper around mlx-lm.

Wraps mlx_lm.load() + mlx_lm.stream_generate() so LangGraph nodes can call
it like any other LangChain chat model (llm.invoke(messages)).
"""
from __future__ import annotations

from typing import Any, Iterator, List, Optional

import mlx.core as mx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from mlx_lm.sample_utils import make_sampler
from pydantic import PrivateAttr


class MLXChatModel(BaseChatModel):
    """LangChain-compatible chat model backed by a local mlx-lm model."""

    model_path: str = "mlx-community/Qwen2.5-14B-Instruct-4bit"
    max_tokens: int = 2048
    temperature: float = 0.1

    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load model + tokenizer into unified memory (lazy, called once)."""
        if self._model is None:
            import mlx_lm
            self._model, self._tokenizer = mlx_lm.load(self.model_path)

    def memory_stats(self) -> dict:
        """Return current Metal/unified memory usage in GB."""
        return {
            "active_gb":  round(mx.get_active_memory()  / 1e9, 2),
            "peak_gb":    round(mx.get_peak_memory()     / 1e9, 2),
            "cache_gb":   round(mx.get_cache_memory()    / 1e9, 2),
        }

    # ── Prompt formatting ─────────────────────────────────────────────────────

    def _format_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages → Qwen chat-template string."""
        self._load()
        chat = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                chat.append({"role": "system",    "content": msg.content})
            elif isinstance(msg, HumanMessage):
                chat.append({"role": "user",      "content": msg.content})
            elif isinstance(msg, AIMessage):
                chat.append({"role": "assistant", "content": msg.content})
        return self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

    # ── LangChain required interface ──────────────────────────────────────────

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Non-streaming generation (used by llm.invoke())."""
        import mlx_lm
        self._load()
        prompt = self._format_prompt(messages)
        sampler = make_sampler(temp=self.temperature)
        response = mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response))]
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Streaming generation — yields tokens as they are produced."""
        import mlx_lm
        self._load()
        prompt = self._format_prompt(messages)
        sampler = make_sampler(temp=self.temperature)
        for chunk in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
        ):
            # mlx-lm 0.21+ yields GenerationResponse with .text
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    @property
    def _llm_type(self) -> str:
        return "mlx-chat"
