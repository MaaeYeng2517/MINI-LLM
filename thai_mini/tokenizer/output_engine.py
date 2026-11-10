"""Output Engine for Mini-LLM TH.

Provides comprehensive output handling for generated text including:
- Token ID decoding via Turing Decoder
- Unicode validation and NFC normalization
- Terminal display with proper Thai rendering
- Export to TXT, JSON, JSONL, CSV formats
- Error handling and auto-folder creation
- Chatbot conversation formatting
"""

import json
import csv
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Union, Iterator, TextIO, Tuple
from datetime import datetime
from enum import Enum
import unicodedata

from .turing_decoder import TuringDecoder
from .thai_unicode import (
    ThaiUnicodeEngine,
    normalize_thai_nfc,
    validate_thai_text,
    segment_thai_graphemes,
    is_thai_char,
    analyze,
)


class OutputFormat(Enum):
    """Supported output formats."""
    TXT = "txt"
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"


class OutputError(Exception):
    """Base exception for output engine errors."""
    pass


class DecodeError(OutputError):
    """Error during token decoding."""
    pass


class ValidationError(OutputError):
    """Error during Unicode validation."""
    pass


class ExportError(OutputError):
    """Error during export."""
    pass


@dataclass
class GenerationResult:
    """Result of a text generation."""
    prompt: str
    generated_text: str
    token_ids: List[int]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_info: Dict[str, Any] = field(default_factory=dict)
    generation_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    token_ids: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatConversation:
    """A chat conversation with multiple messages."""
    messages: List[ChatMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_message(self, role: str, content: str, token_ids: Optional[List[int]] = None, **metadata):
        msg = ChatMessage(role=role, content=content, token_ids=token_ids, metadata=metadata)
        self.messages.append(msg)
        return msg

    def get_last_user_message(self) -> Optional[ChatMessage]:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None

    def get_last_assistant_message(self) -> Optional[ChatMessage]:
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class OutputEngine:
    """Main output engine for Mini-LLM TH."""

    def __init__(
        self,
        decoder: Optional[TuringDecoder] = None,
        unicode_engine: Optional[ThaiUnicodeEngine] = None,
        output_dir: Union[str, Path] = "outputs",
        auto_create_dirs: bool = True,
        default_format: OutputFormat = OutputFormat.JSONL,
        validate_unicode: bool = True,
        normalize_output: bool = True,
    ):
        self.decoder = decoder or TuringDecoder()
        self.unicode_engine = unicode_engine or ThaiUnicodeEngine()
        self.output_dir = Path(output_dir)
        self.auto_create_dirs = auto_create_dirs
        self.default_format = default_format
        self.validate_unicode = validate_unicode
        self.normalize_output = normalize_output

        if self.auto_create_dirs:
            self._ensure_output_dirs()

    def _ensure_output_dirs(self):
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "txt").mkdir(exist_ok=True)
        (self.output_dir / "json").mkdir(exist_ok=True)
        (self.output_dir / "jsonl").mkdir(exist_ok=True)
        (self.output_dir / "csv").mkdir(exist_ok=True)
        (self.output_dir / "chat").mkdir(exist_ok=True)

    def set_decoder(self, decoder: TuringDecoder):
        """Set the decoder to use."""
        self.decoder = decoder

    def load_decoder_state(self, state: Dict[str, Any]):
        """Load decoder vocabulary state."""
        self.decoder.load_state(state)

    def decode_tokens(
        self,
        token_ids: List[int],
        filter_special: bool = True,
        validate: bool = None,
        normalize: bool = None,
    ) -> str:
        """Decode token IDs to text with Unicode validation and normalization."""
        if not token_ids:
            return ""

        validate = self.validate_unicode if validate is None else validate
        normalize = self.normalize_output if normalize is None else normalize

        try:
            text = self.decoder.decode(token_ids, filter_special=filter_special)
        except Exception as e:
            raise DecodeError(f"Failed to decode tokens: {e}") from e

        if normalize:
            text = normalize_thai_nfc(text)

        if validate:
            is_valid, invalid_chars = validate_thai_text(text)
            if not is_valid:
                raise ValidationError(
                    f"Invalid Unicode characters found: {invalid_chars[:10]}..."
                    if len(invalid_chars) > 10 else f"Invalid Unicode characters: {invalid_chars}"
                )

        return text

    def decode_stream(
        self,
        token_ids: List[int],
        filter_special: bool = True,
    ) -> Iterator[str]:
        """Decode token IDs incrementally for streaming output."""
        if not token_ids:
            return

        decoded = ""
        for i, token_id in enumerate(token_ids):
            try:
                partial = self.decoder.decode(token_ids[:i+1], filter_special=filter_special)
                if self.normalize_output:
                    partial = normalize_thai_nfc(partial)
                if partial != decoded:
                    yield partial
                    decoded = partial
            except Exception as e:
                raise DecodeError(f"Failed to decode token at position {i}: {e}") from e

    def create_generation_result(
        self,
        prompt: str,
        token_ids: List[int],
        model_info: Optional[Dict[str, Any]] = None,
        generation_params: Optional[Dict[str, Any]] = None,
        **metadata,
    ) -> GenerationResult:
        """Create a GenerationResult from token IDs."""
        generated_text = self.decode_tokens(token_ids)
        return GenerationResult(
            prompt=prompt,
            generated_text=generated_text,
            token_ids=token_ids,
            model_info=model_info or {},
            generation_params=generation_params or {},
            metadata=metadata,
        )

    def display_to_terminal(
        self,
        text: str,
        prefix: str = "",
        show_graphemes: bool = False,
        color: bool = False,
    ) -> str:
        """Format text for terminal display with proper Thai rendering."""
        if not text:
            return ""

        normalized = normalize_thai_nfc(text) if self.normalize_output else text

        if show_graphemes:
            graphemes = segment_thai_graphemes(normalized)
            display = " | ".join(graphemes)
        else:
            display = normalized

        if prefix:
            display = f"{prefix}{display}"

        return display

    def print_generation(
        self,
        result: GenerationResult,
        show_tokens: bool = False,
        show_graphemes: bool = False,
    ):
        """Print generation result to terminal."""
        print(f"\n{'='*60}")
        print(f"Prompt: {result.prompt}")
        print(f"{'-'*60}")
        print(f"Generated: {self.display_to_terminal(result.generated_text, show_graphemes=show_graphemes)}")
        if show_tokens:
            print(f"Tokens: {result.token_ids}")
        print(f"Time: {result.timestamp}")
        print(f"{'='*60}\n")

    def export(
        self,
        data: Union[GenerationResult, ChatConversation, List[GenerationResult], List[ChatConversation], List[ChatMessage]],
        filepath: Union[str, Path],
        format: Optional[OutputFormat] = None,
        **kwargs,
    ) -> Path:
        """Export data to file in specified format."""
        format = format or self.default_format
        filepath = Path(filepath)

        if self.auto_create_dirs:
            filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == OutputFormat.TXT:
            return self._export_txt(data, filepath, **kwargs)
        elif format == OutputFormat.JSON:
            return self._export_json(data, filepath, **kwargs)
        elif format == OutputFormat.JSONL:
            return self._export_jsonl(data, filepath, **kwargs)
        elif format == OutputFormat.CSV:
            return self._export_csv(data, filepath, **kwargs)
        else:
            raise ExportError(f"Unsupported format: {format}")

    def _export_txt(
        self,
        data: Union[GenerationResult, ChatConversation, List[GenerationResult], List[ChatConversation], List[ChatMessage]],
        filepath: Path,
        include_metadata: bool = False,
        separator: str = "\n" + "="*60 + "\n",
    ) -> Path:
        """Export to plain text file."""
        with open(filepath, "w", encoding="utf-8") as f:
            if isinstance(data, GenerationResult):
                self._write_generation_txt(f, data, include_metadata)
            elif isinstance(data, ChatConversation):
                self._write_conversation_txt(f, data, include_metadata)
            elif isinstance(data, list):
                if data and isinstance(data[0], GenerationResult):
                    for i, item in enumerate(data):
                        if i > 0:
                            f.write(separator)
                        self._write_generation_txt(f, item, include_metadata)
                elif data and isinstance(data[0], ChatConversation):
                    for i, item in enumerate(data):
                        if i > 0:
                            f.write(separator)
                        self._write_conversation_txt(f, item, include_metadata)
                elif data and isinstance(data[0], ChatMessage):
                    self._write_messages_txt(f, data, include_metadata)
                else:
                    raise ExportError("Unsupported list type for TXT export")
            else:
                raise ExportError(f"Unsupported data type: {type(data)}")
        return filepath

    def _write_generation_txt(self, f: TextIO, result: GenerationResult, include_metadata: bool):
        f.write(f"Prompt: {result.prompt}\n")
        f.write(f"Generated: {result.generated_text}\n")
        if include_metadata:
            f.write(f"Timestamp: {result.timestamp}\n")
            if result.model_info:
                f.write(f"Model: {json.dumps(result.model_info, ensure_ascii=False)}\n")
            if result.generation_params:
                f.write(f"Params: {json.dumps(result.generation_params, ensure_ascii=False)}\n")
            if result.metadata:
                f.write(f"Metadata: {json.dumps(result.metadata, ensure_ascii=False)}\n")

    def _write_conversation_txt(self, f: TextIO, conv: ChatConversation, include_metadata: bool):
        f.write(f"Conversation (created: {conv.created_at})\n")
        if include_metadata and conv.metadata:
            f.write(f"Metadata: {json.dumps(conv.metadata, ensure_ascii=False)}\n")
        f.write("\n")
        for msg in conv.messages:
            f.write(f"[{msg.role.upper()}] {msg.content}\n")
            if include_metadata and msg.metadata:
                f.write(f"  Metadata: {json.dumps(msg.metadata, ensure_ascii=False)}\n")

    def _write_messages_txt(self, f: TextIO, messages: List[ChatMessage], include_metadata: bool):
        for msg in messages:
            f.write(f"[{msg.role.upper()}] {msg.content}\n")
            if include_metadata and msg.metadata:
                f.write(f"  Metadata: {json.dumps(msg.metadata, ensure_ascii=False)}\n")

    def _export_json(
        self,
        data: Union[GenerationResult, ChatConversation, List[GenerationResult], List[ChatConversation], List[ChatMessage]],
        filepath: Path,
        indent: int = 2,
    ) -> Path:
        """Export to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            if isinstance(data, GenerationResult):
                json.dump(data.to_dict(), f, ensure_ascii=False, indent=indent)
            elif isinstance(data, ChatConversation):
                json.dump(data.to_dict(), f, ensure_ascii=False, indent=indent)
            elif isinstance(data, list):
                if data and isinstance(data[0], GenerationResult):
                    json.dump([item.to_dict() for item in data], f, ensure_ascii=False, indent=indent)
                elif data and isinstance(data[0], ChatConversation):
                    json.dump([item.to_dict() for item in data], f, ensure_ascii=False, indent=indent)
                elif data and isinstance(data[0], ChatMessage):
                    json.dump([item.to_dict() for item in data], f, ensure_ascii=False, indent=indent)
                else:
                    raise ExportError("Unsupported list type for JSON export")
            else:
                raise ExportError(f"Unsupported data type: {type(data)}")
        return filepath

    def _export_jsonl(
        self,
        data: Union[GenerationResult, ChatConversation, List[GenerationResult], List[ChatConversation], List[ChatMessage]],
        filepath: Path,
    ) -> Path:
        """Export to JSONL file (one JSON object per line)."""
        with open(filepath, "w", encoding="utf-8") as f:
            if isinstance(data, GenerationResult):
                f.write(json.dumps(data.to_dict(), ensure_ascii=False) + "\n")
            elif isinstance(data, ChatConversation):
                for msg in data.messages:
                    f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")
            elif isinstance(data, list):
                if data and isinstance(data[0], GenerationResult):
                    for item in data:
                        f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
                elif data and isinstance(data[0], ChatConversation):
                    for item in data:
                        conv_dict = item.to_dict()
                        f.write(json.dumps(conv_dict, ensure_ascii=False) + "\n")
                elif data and isinstance(data[0], ChatMessage):
                    for item in data:
                        f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
                else:
                    raise ExportError("Unsupported list type for JSONL export")
            else:
                raise ExportError(f"Unsupported data type: {type(data)}")
        return filepath

    def _export_csv(
        self,
        data: Union[GenerationResult, ChatConversation, List[GenerationResult], List[ChatConversation], List[ChatMessage]],
        filepath: Path,
        include_token_ids: bool = False,
    ) -> Path:
        """Export to CSV file."""
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            if isinstance(data, GenerationResult):
                self._write_generation_csv(f, data, include_token_ids)
            elif isinstance(data, ChatConversation):
                self._write_conversation_csv(f, data, include_token_ids)
            elif isinstance(data, list):
                if data and isinstance(data[0], GenerationResult):
                    writer = csv.writer(f)
                    writer.writerow(["prompt", "generated_text", "token_ids", "timestamp", "model_info", "generation_params", "metadata"])
                    for item in data:
                        self._write_generation_csv(f, item, include_token_ids)
                elif data and isinstance(data[0], ChatConversation):
                    writer = csv.writer(f)
                    writer.writerow(["role", "content", "timestamp", "token_ids", "metadata"])
                    for item in data:
                        self._write_conversation_csv(f, item, include_token_ids)
                elif data and isinstance(data[0], ChatMessage):
                    writer = csv.writer(f)
                    writer.writerow(["role", "content", "timestamp", "token_ids", "metadata"])
                    for item in data:
                        row = [
                            item.role,
                            item.content,
                            item.timestamp,
                            json.dumps(item.token_ids) if include_token_ids and item.token_ids else "",
                            json.dumps(item.metadata, ensure_ascii=False) if item.metadata else "",
                        ]
                        writer.writerow(row)
                else:
                    raise ExportError("Unsupported list type for CSV export")
            else:
                raise ExportError(f"Unsupported data type: {type(data)}")
        return filepath

    def _write_generation_csv(self, f: TextIO, result: GenerationResult, include_token_ids: bool):
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["prompt", "generated_text", "token_ids", "timestamp", "model_info", "generation_params", "metadata"])
        writer.writerow([
            result.prompt,
            result.generated_text,
            json.dumps(result.token_ids) if include_token_ids else "",
            result.timestamp,
            json.dumps(result.model_info, ensure_ascii=False) if result.model_info else "",
            json.dumps(result.generation_params, ensure_ascii=False) if result.generation_params else "",
            json.dumps(result.metadata, ensure_ascii=False) if result.metadata else "",
        ])

    def _write_conversation_csv(self, f: TextIO, conv: ChatConversation, include_token_ids: bool):
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["role", "content", "timestamp", "token_ids", "metadata"])
        for msg in conv.messages:
            writer.writerow([
                msg.role,
                msg.content,
                msg.timestamp,
                json.dumps(msg.token_ids) if include_token_ids and msg.token_ids else "",
                json.dumps(msg.metadata, ensure_ascii=False) if msg.metadata else "",
            ])

    def save_generation(
        self,
        result: GenerationResult,
        filename: Optional[str] = None,
        format: Optional[OutputFormat] = None,
    ) -> Path:
        """Save a generation result with auto-generated filename."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generation_{timestamp}.{format.value if format else self.default_format.value}"

        subdir = format.value if format else self.default_format.value
        filepath = self.output_dir / subdir / filename
        return self.export(result, filepath, format)

    def save_conversation(
        self,
        conversation: ChatConversation,
        filename: Optional[str] = None,
        format: Optional[OutputFormat] = None,
    ) -> Path:
        """Save a conversation with auto-generated filename."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_{timestamp}.{format.value if format else self.default_format.value}"

        filepath = self.output_dir / "chat" / filename
        return self.export(conversation, filepath, format)

    def load_generation(self, filepath: Union[str, Path]) -> GenerationResult:
        """Load a generation result from file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return GenerationResult(**data)

    def load_conversation(self, filepath: Union[str, Path]) -> ChatConversation:
        """Load a conversation from file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = [ChatMessage(**m) for m in data.get("messages", [])]
        return ChatConversation(
            messages=messages,
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )

    def analyze_output(self, text: str) -> Dict[str, Any]:
        """Analyze generated text for Thai language properties."""
        return analyze(text)

    def validate_output(self, text: str) -> Tuple[bool, List[str]]:
        """Validate output text for Unicode correctness."""
        return validate_thai_text(text)


class ChatbotOutputEngine(OutputEngine):
    """Specialized output engine for chatbot interactions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_conversation = ChatConversation()

    def add_user_message(self, content: str, token_ids: Optional[List[int]] = None) -> ChatMessage:
        """Add a user message to the conversation."""
        return self.current_conversation.add_message("user", content, token_ids)

    def add_assistant_message(self, content: str, token_ids: Optional[List[int]] = None) -> ChatMessage:
        """Add an assistant message to the conversation."""
        return self.current_conversation.add_message("assistant", content, token_ids)

    def add_system_message(self, content: str, token_ids: Optional[List[int]] = None) -> ChatMessage:
        """Add a system message to the conversation."""
        return self.current_conversation.add_message("system", content, token_ids)

    def get_conversation(self) -> ChatConversation:
        """Get the current conversation."""
        return self.current_conversation

    def clear_conversation(self):
        """Clear the current conversation."""
        self.current_conversation = ChatConversation()

    def generate_response(
        self,
        model,
        tokenizer,
        prompt: str,
        device,
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        top_k: int = 24,
        repetition_penalty: float = 1.15,
    ) -> GenerationResult:
        """Generate a response and add to conversation."""
        from .generate import generate_text

        token_ids = tokenizer.encode(prompt)
        if not token_ids:
            token_ids = [0]

        import torch
        idx = torch.tensor([token_ids], dtype=torch.long, device=device)
        seq_len = int(getattr(model, "seq_len", idx.size(1)))
        generated = list(token_ids)

        model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                cur = idx[:, -seq_len:]
                logits = model(cur)
                logits = logits[0] if isinstance(logits, (tuple, list)) else logits
                next_logits = (logits[:, -1, :] if logits.ndim == 3 else logits).to(torch.float32)
                next_logits = next_logits / max(float(temperature), 1e-5)

                if repetition_penalty and repetition_penalty != 1.0:
                    seen = set(generated[-96:])
                    for t in seen:
                        if 0 <= t < next_logits.size(-1):
                            v = next_logits[0, t]
                            next_logits[0, t] = v / repetition_penalty if v > 0 else v * repetition_penalty

                if top_k and top_k > 0:
                    k = min(int(top_k), next_logits.size(-1))
                    vals, _ = torch.topk(next_logits, k)
                    cutoff = vals[:, -1].unsqueeze(-1)
                    next_logits = torch.where(next_logits < cutoff, torch.full_like(next_logits, float("-inf")), next_logits)

                probs = torch.softmax(next_logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                nid = int(nxt.item())
                generated.append(nid)
                idx = torch.cat([idx, nxt], dim=1)

        generated_text = self.decode_tokens(generated[len(token_ids):])
        self.add_user_message(prompt, token_ids)
        self.add_assistant_message(generated_text, generated[len(token_ids):])

        return GenerationResult(
            prompt=prompt,
            generated_text=generated_text,
            token_ids=generated,
            generation_params={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            },
        )

    def export_chat(
        self,
        filename: Optional[str] = None,
        format: Optional[OutputFormat] = None,
    ) -> Path:
        """Export the current conversation."""
        return self.save_conversation(self.current_conversation, filename, format)


DEFAULT_OUTPUT_ENGINE = OutputEngine()
DEFAULT_CHAT_ENGINE = ChatbotOutputEngine()


def decode_tokens(token_ids: List[int], decoder: Optional[TuringDecoder] = None) -> str:
    """Decode token IDs using default engine."""
    engine = OutputEngine(decoder=decoder) if decoder else DEFAULT_OUTPUT_ENGINE
    return engine.decode_tokens(token_ids)


def save_generation(result: GenerationResult, filepath: Union[str, Path], format: Optional[OutputFormat] = None) -> Path:
    """Save generation using default engine."""
    return DEFAULT_OUTPUT_ENGINE.export(result, filepath, format)


def export_chat(conversation: ChatConversation, filepath: Union[str, Path], format: Optional[OutputFormat] = None) -> Path:
    """Export chat using default engine."""
    return DEFAULT_OUTPUT_ENGINE.export(conversation, filepath, format)


__all__ = [
    "OutputFormat",
    "OutputError",
    "DecodeError",
    "ValidationError",
    "ExportError",
    "GenerationResult",
    "ChatMessage",
    "ChatConversation",
    "OutputEngine",
    "ChatbotOutputEngine",
    "DEFAULT_OUTPUT_ENGINE",
    "DEFAULT_CHAT_ENGINE",
    "decode_tokens",
    "save_generation",
    "export_chat",
]