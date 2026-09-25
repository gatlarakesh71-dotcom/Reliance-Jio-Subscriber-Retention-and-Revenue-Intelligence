from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


class ExcelRAG:
    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir

    def _iter_excel_files(self):
        if not self.docs_dir.exists():
            return []
        return sorted(self.docs_dir.glob("*.xlsx")) + sorted(self.docs_dir.glob("*.xls")) + sorted(self.docs_dir.glob("*.csv"))

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query = (query or "").lower().strip()
        if not query:
            return []

        results: List[Dict[str, Any]] = []
        for file_path in self._iter_excel_files():
            try:
                if file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
            except Exception:
                continue

            rows = df.astype(str).fillna("").to_dict(orient="records")
            for index, row in enumerate(rows):
                row_text = " ".join(str(value)
                                    for value in row.values()).lower()
                if query in row_text:
                    results.append(
                        {
                            "file": file_path.name,
                            "sheet": "Sheet1",
                            "row_index": index,
                            "content": row,
                        }
                    )
                    if len(results) >= top_k:
                        return results

        return results

    def get_context(self, query: str, top_k: int = 5) -> str:
        hits = self.search(query, top_k=top_k)
        if not hits:
            return "No matching Excel or CSV records were found for the provided query."

        lines = []
        for hit in hits:
            lines.append(
                f"File: {hit['file']} | Row: {hit['row_index']} | Content: {hit['content']}")
        return "\n".join(lines)
