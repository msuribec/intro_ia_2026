from __future__ import annotations

import io

from gtts import gTTS

from expense_auditor.constants import GTTS_SUPPORTED_LANGS


class AudioService:
    def generate_audio(self, tip: str, lang_code: str) -> tuple[bytes, str]:
        lang = lang_code if lang_code in GTTS_SUPPORTED_LANGS else "es"
        tts = gTTS(text=tip, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read(), lang

