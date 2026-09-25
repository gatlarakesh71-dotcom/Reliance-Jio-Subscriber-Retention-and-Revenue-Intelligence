from __future__ import annotations

import json
from typing import Dict

from config import get_llm
from prompts import ROUTER_PROMPT


class IntentRouter:
    def __init__(self):
        self.llm = get_llm()

    def route(self, query: str) -> Dict[str, str]:
        response = self.llm.invoke(ROUTER_PROMPT.format(query=query))
        text = getattr(response, "content", str(response))

        try:
            parsed = json.loads(text)
            intent = str(parsed.get("intent", "TEXT_TO_SQL")).strip().upper()
            reason = str(parsed.get(
                "reason", "Routed by LLM intent classification."))
            if intent not in {"TEXT_TO_SQL", "EXCEL_RAG"}:
                intent = "TEXT_TO_SQL"
            return {"intent": intent, "reason": reason}
        except Exception:
            # Fallback router for safety
            if any(keyword in query.lower() for keyword in ["offer", "sheet", "campaign", "target", "matrix", "policy", "guideline", "quarter", "excel"]):
                return {"intent": "EXCEL_RAG", "reason": "Fallback route: document-oriented business question detected."}
            return {"intent": "TEXT_TO_SQL", "reason": "Fallback route: analytical question detected."}
