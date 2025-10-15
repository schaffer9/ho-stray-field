from typing import Any
import os
import csv
from pathlib import Path


def write_csv_row(file: Path, data: dict[str, Any]) -> None:
    fieldnames = list(data.keys())
    with open(file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if os.stat(file).st_size == 0:
            writer.writeheader()

        writer.writerow(data)