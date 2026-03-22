from __future__ import annotations

import json
import re
from typing import Any


def parse_model_json_response(raw_response: str) -> Any:
    raw = raw_response.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)

