import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from django.conf import settings


def _load_env_from_files():
    base_dir = Path(getattr(settings, "BASE_DIR", "") or Path.cwd()) if settings.configured else Path.cwd()
    candidates = [base_dir / ".env", base_dir / ",env"]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    if os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    if os.environ.get("GOOGLE_API") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API"]
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def _compact_error(error: Exception) -> str:
    text = str(error).splitlines()[0].strip()
    text = re.sub(r"\s+", " ", text)
    return text[:180]


class GeminiClient:
    def __init__(self):
        _load_env_from_files()
        self.enabled = False
        self.model_name = ""
        self.available_models: List[str] = []
        self.last_error = ""
        self._init_error = ""
        self._client = None
        self._configured_model = os.environ.get("GEMINI_MODEL", "").strip()
        api_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            self._init_error = "GEMINI_API_KEY missing"
            return
        try:
            from google import genai  # type: ignore

            self._client = genai.Client(api_key=api_key)
            self.available_models = self._list_generate_models()
            if not self.available_models:
                self._init_error = "No generateContent models available for this key"
                return
            selected = self._select_startup_model()
            if not selected:
                self._init_error = "Unable to select Gemini model"
                return
            self.model_name = selected
            self.enabled = True
            print("Gemini client initialized: OK")
            print(f"Gemini model selected: {self.model_name} (generateContent supported)")
        except Exception as error:  # pragma: no cover
            self._init_error = _compact_error(error)
            self.last_error = self._init_error

    def _list_generate_models(self) -> List[str]:
        if not self._client:
            return []
        try:
            models = []
            for item in self._client.models.list():
                name = getattr(item, "name", "")
                methods = list(getattr(item, "supported_actions", []) or [])
                if "generateContent" in methods and name:
                    models.append(str(name).replace("models/", ""))
            return sorted(set(models))
        except Exception as error:
            self._init_error = f"ListModels failed: {_compact_error(error)}"
            self.last_error = self._init_error
            return []

    def _select_startup_model(self) -> Optional[str]:
        preferred = [
            self._configured_model,
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
        ]
        for name in [item for item in preferred if item]:
            if name in self.available_models:
                return name
        return self.available_models[0] if self.available_models else None

    def _candidate_models(self) -> List[str]:
        ordered = []
        for name in [
            self._configured_model,
            self.model_name,
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
        ]:
            if name and name in self.available_models and name not in ordered:
                ordered.append(name)
        for name in self.available_models:
            if name not in ordered:
                ordered.append(name)
        return ordered[:6]

    def _generate_text(self, prompt: str) -> Tuple[Optional[str], str, str]:
        if not self.enabled or not self._client:
            return None, "", self._init_error or "Gemini disabled"
        last_error = ""
        for model in self._candidate_models():
            try:
                response = self._client.models.generate_content(model=model, contents=prompt)
                text = self._extract_text(response)
                text = text.strip()
                if text:
                    self.last_error = ""
                    self.model_name = model
                    return text, model, ""
            except Exception as error:
                last_error = _compact_error(error)
                continue
        self.last_error = last_error or "Gemini returned empty response"
        return None, "", self.last_error

    @staticmethod
    def _extract_text(response) -> str:
        candidates = getattr(response, "candidates", None) or []
        chunks: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(str(text))
        if chunks:
            return "\n".join(chunks).strip()
        return getattr(response, "text", "") or ""

    def classify_router(self, partner_text: str) -> Tuple[Optional[str], str, str]:
        prompt = (
            "Classify the user input into one label only: Personal, Contextual, or Open-domain.\n"
            "Respond with only one of these exact labels.\n"
            f"Input: {partner_text}"
        )
        text, model, error = self._generate_text(prompt)
        if not text:
            return None, model, error
        label = text.strip().splitlines()[0].strip()
        if label in {"Personal", "Contextual", "Open-domain"}:
            return label, model, ""
        return None, model, "Router output invalid"

    def refine_evidence(self, partner_text: str, evidence_lines: List[str]) -> Tuple[Optional[List[str]], str, str]:
        if not evidence_lines:
            return [], "", ""
        joined = "\n".join(f"- {line}" for line in evidence_lines[:10])
        prompt = (
            "Refine evidence for AAC reply generation.\n"
            "Return JSON object: {\"evidence\": [\"line1\", \"line2\", ...]} with 1-6 concise lines.\n"
            f"Query: {partner_text}\n"
            f"Candidate evidence:\n{joined}"
        )
        text, model, error = self._generate_text(prompt)
        if not text:
            return None, model, error
        try:
            start = text.find("{")
            end = text.rfind("}")
            payload = json.loads(text[start : end + 1]) if start != -1 and end != -1 else json.loads(text)
            evidence = [str(item).strip() for item in payload.get("evidence", []) if str(item).strip()]
            return evidence[:6], model, ""
        except Exception:
            return None, model, "Evidence JSON parse failed"

    def generate_candidates(
        self,
        partner_text: str,
        evidence_block: str,
        style: Dict,
        pb_exemplars: List[str],
        face_summary: str,
    ) -> Tuple[Optional[List[str]], str, str]:
        exemplars = "\n".join(f"- {item}" for item in pb_exemplars[:5]) or "- none"
        prompt = (
            "Generate exactly 3 natural AAC response options.\n"
            "Return strict JSON object with keys option_1, option_2, option_3.\n"
            "No markdown, no labels, no reasoning text.\n"
            "Option styles: polite, casual, short.\n"
            f"Face cue: {face_summary}\n"
            f"Style preferences: {json.dumps(style)}\n"
            f"Phrase exemplars:\n{exemplars}\n"
            f"Partner message: {partner_text}\n"
            f"Evidence: {evidence_block}\n"
        )
        text, model, error = self._generate_text(prompt)
        if not text:
            return None, model, error
        try:
            start = text.find("{")
            end = text.rfind("}")
            payload = json.loads(text[start : end + 1]) if start != -1 and end != -1 else json.loads(text)
            options = [
                str(payload.get("option_1", "")).strip(),
                str(payload.get("option_2", "")).strip(),
                str(payload.get("option_3", "")).strip(),
            ]
            if all(options):
                return options, model, ""
            return None, model, "Candidate JSON missing options"
        except Exception:
            return None, model, "Candidate JSON parse failed"

    def generate_speak_suggestions(
        self,
        grouped: Dict[str, List[str]],
        style: Dict,
        face_summary: str,
    ) -> Tuple[Optional[Dict[str, List[str]]], str, str]:
        if not grouped:
            return {}, "", ""
        prompt = (
            "You are refining AAC proactive suggestions.\n"
            "Keep the same top-level groups and keep each group's item count exactly the same.\n"
            "Rewrite items to sound natural, concise, and varied, while staying grounded in the provided content.\n"
            "Do not invent new plans or facts.\n"
            "Return strict JSON only with the same group keys mapped to arrays of strings.\n"
            f"Face cue: {face_summary}\n"
            f"Style preferences: {json.dumps(style)}\n"
            f"Grouped suggestions: {json.dumps(grouped)}\n"
        )
        text, model, error = self._generate_text(prompt)
        if not text:
            return None, model, error
        try:
            start = text.find("{")
            end = text.rfind("}")
            payload = json.loads(text[start : end + 1]) if start != -1 and end != -1 else json.loads(text)
            normalized: Dict[str, List[str]] = {}
            for key, original_items in grouped.items():
                candidate_items = payload.get(key, [])
                if not isinstance(candidate_items, list):
                    return None, model, f"Speak suggestions for group '{key}' were not a list"
                cleaned = [str(item).strip() for item in candidate_items if str(item).strip()]
                if len(cleaned) != len(original_items):
                    return None, model, f"Speak suggestions count changed for group '{key}'"
                normalized[key] = cleaned
            return normalized, model, ""
        except Exception:
            return None, model, "Speak suggestion JSON parse failed"

    def health_ping(self) -> Dict[str, object]:
        started = time.perf_counter()
        text, model, error = self._generate_text("ping")
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": bool(text),
            "model": model,
            "latency_ms": latency_ms,
            "error": error if not text else "",
            "text_preview": (text or "")[:60],
        }


gemini_client = GeminiClient()
