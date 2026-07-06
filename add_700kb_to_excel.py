# -*- coding: utf-8 -*-
"""
This script adds the required new row to the Excel file:
Data size = 700KB, Alg.1 = 80, Alg.2 = 320, Alg.3 = 700

Run:
    python add_700kb_to_excel.py

Then run project3_main.py again. The charts will update automatically.
"""

from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "project3.xlsx"


def extract_number(value) -> float:
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        raise ValueError(f"Cannot extract numeric data size from: {value}")
    return float(match.group())


def main():
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_FILE}")

    df = pd.read_excel(EXCEL_FILE)
    df.columns = [str(col).strip() for col in df.columns]

    data_size_col = df.columns[0]
    algorithm_cols = list(df.columns[1:4])

    # Do not duplicate 700KB if it already exists
    existing_sizes = df[data_size_col].apply(extract_number).tolist()
    if 700 in existing_sizes:
        print("700KB row already exists. No duplicate row was added.")
    else:
        new_row = {
            data_size_col: "700KB",
            algorithm_cols[0]: 80,
            algorithm_cols[1]: 320,
            algorithm_cols[2]: 700,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False)
        print("700KB row was added successfully.")

    print(f"Saved Excel file: {EXCEL_FILE}")


if __name__ == "__main__":
    main()
