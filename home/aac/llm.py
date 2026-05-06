import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from django.conf import settings


# Latency-tier configuration (Bonus #4: latency-optimised fallback).
# Primary is the lighter / faster model; we race it against a wall-clock SLA
# and fall through to the heavier model on timeout OR parse failure.
PRIMARY_TIER = ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]
FALLBACK_TIER = ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.0-flash"]
LATENCY_SLA_SECONDS = 5.0


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
        # Primary tier first (latency-optimised), then fallback tier.
        for name in PRIMARY_TIER + FALLBACK_TIER + [self._configured_model, self.model_name]:
            if name and name in self.available_models and name not in ordered:
                ordered.append(name)
        for name in self.available_models:
            if name not in ordered:
                ordered.append(name)
        return ordered[:6]

    def _tier_models(self, tier: str) -> List[str]:
        wanted = PRIMARY_TIER if tier == "primary" else FALLBACK_TIER
        return [name for name in wanted if name in self.available_models]

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
        polarity: str = "",
        tone: str = "",
        verbosity: str = "",
        tone_variants: Optional[List[str]] = None,
    ) -> Tuple[Optional[List[str]], str, str]:
        """Generate AAC candidate replies.

        Two modes:
          - tone_variants given (camera off / no emotion known): produce one
            option per tone, so the user can pick the tone they want.
          - tone_variants None (camera on, emotion known): produce 3 options
            ALL in the single ``tone`` (matched to detected facial emotion).
        """
        exemplars = "\n".join(f"- {item}" for item in pb_exemplars[:5]) or "- none"
        polarity_rule = ""
        if polarity == "positive":
            polarity_rule = "Each option MUST start with an affirmative word (Yes/Sure/Okay)."
        elif polarity == "negative":
            polarity_rule = "Each option MUST start with a negative word (No/Sorry/Not)."
        elif polarity == "clarify":
            polarity_rule = "Each option MUST be a short clarifying question."
        len_rule = "Keep each option <= 14 words." if verbosity == "short" else ""

        if tone_variants:
            tones = list(tone_variants)[:3]
            while len(tones) < 3:
                tones.append("neutral")
            tone_block = (
                f"Produce exactly 3 options, ONE PER TONE in this order: "
                f"option_1={tones[0]}, option_2={tones[1]}, option_3={tones[2]}. "
                "Tones should be clearly different in word choice and rhythm."
            )
            tone_constraint = f"per-option tones: {tones}"
        else:
            tone_block = "Produce exactly 3 options, ALL written in the same tone."
            tone_constraint = tone or "neutral"

        prompt = (
            "Generate AAC response options.\n"
            "Return strict JSON object with keys option_1, option_2, option_3.\n"
            "No markdown, no labels, no reasoning text.\n"
            f"{tone_block}\n"
            f"Polarity constraint: {polarity_rule or 'none'}\n"
            f"Tone constraint: {tone_constraint}\n"
            f"Length constraint: {len_rule or 'normal'}\n"
            f"Face cue: {face_summary}\n"
            f"Style preferences: {json.dumps(style)}\n"
            f"Phrase exemplars:\n{exemplars}\n"
            f"Partner message: {partner_text}\n"
            f"Evidence: {evidence_block}\n"
        )
        return self._generate_candidates_with_sla(prompt)

    # ----------------------------------------------------------
    # Latency-tier fallback (Bonus #4)
    # ----------------------------------------------------------
    def _parse_candidates(self, text: str) -> Optional[List[str]]:
        if not text:
            return None
        try:
            start = text.find("{")
            end = text.rfind("}")
            payload = json.loads(text[start : end + 1]) if start != -1 and end != -1 else json.loads(text)
            options = [
                str(payload.get("option_1", "")).strip(),
                str(payload.get("option_2", "")).strip(),
                str(payload.get("option_3", "")).strip(),
                str(payload.get("option_4", "")).strip(),
            ]
            options = [o for o in options if o]
            if len(options) >= 3:
                return options[:3]
            return None
        except Exception:
            return None

    def _try_tier(self, prompt: str, tier: str, deadline: float) -> Tuple[Optional[List[str]], str, str]:
        """Try every model in the given tier, racing each one against the
        remaining wall-clock budget. Returns (options, model, error)."""
        models = self._tier_models(tier)
        if not models:
            return None, "", f"{tier}_tier_empty"
        last_error = ""
        for model in models:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                last_error = "deadline_exceeded"
                break
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    lambda m=model: self._client.models.generate_content(model=m, contents=prompt)
                )
                try:
                    response = future.result(timeout=remaining)
                except FutureTimeoutError:
                    last_error = f"timeout_after_{remaining:.1f}s"
                    continue
                except Exception as exc:
                    last_error = _compact_error(exc)
                    continue
            text = self._extract_text(response).strip()
            options = self._parse_candidates(text)
            if options:
                self.model_name = model
                return options, model, ""
            last_error = "parse_failed"
        return None, "", last_error

    def _generate_candidates_with_sla(self, prompt: str) -> Tuple[Optional[List[str]], str, str]:
        if not self.enabled or not self._client:
            return None, "", self._init_error or "Gemini disabled"
        started = time.perf_counter()
        deadline = started + LATENCY_SLA_SECONDS
        options, model, err = self._try_tier(prompt, "primary", deadline)
        if options:
            return options, model, ""
        # Fall through to the heavier tier on timeout OR parse failure.
        # Allocate at least 3s to the fallback tier even if primary used up SLA.
        fallback_deadline = max(time.perf_counter() + 3.0, deadline)
        options, model, err2 = self._try_tier(prompt, "fallback", fallback_deadline)
        if options:
            return options, model, f"primary_failed:{err}"
        return None, "", f"primary:{err}|fallback:{err2}"

    def nli_judge(self, candidate: str, evidence: str) -> Tuple[float, str, str]:
        """Return (score in [0,1], label, raw_text). entail=1.0, neutral=0.5,
        contradict=0.0. Used by the offline evaluation harness; safe to call
        when the LLM is disabled (returns 0.5)."""
        if not self.enabled or not self._client:
            return 0.5, "neutral", "llm_disabled"
        prompt = (
            "You are an NLI judge for an AAC reply.\n"
            "Decide whether the CANDIDATE is entailed by, neutral to, or "
            "contradicts the EVIDENCE.\n"
            "Respond with ONLY one word: entail, neutral, or contradict.\n"
            f"EVIDENCE: {evidence[:600]}\n"
            f"CANDIDATE: {candidate[:300]}\n"
        )
        text, model, err = self._generate_text(prompt)
        if not text:
            return 0.5, "neutral", err or "no_text"
        label = text.strip().splitlines()[0].strip().lower()
        if "entail" in label:
            return 1.0, "entail", text
        if "contradict" in label:
            return 0.0, "contradict", text
        return 0.5, "neutral", text

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
