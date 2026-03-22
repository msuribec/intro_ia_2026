from __future__ import annotations

import io

import pandas as pd


def parse_categories(file) -> list[str]:
    content = file.read().decode("utf-8")
    file.seek(0)
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(io.StringIO(content))
        for column in df.columns:
            if column.strip().lower() in (
                "category",
                "categoria",
                "categories",
                "categorias",
            ):
                return df[column].dropna().str.strip().tolist()
        return df.iloc[:, 0].dropna().str.strip().tolist()
    return [line.strip() for line in content.splitlines() if line.strip()]

