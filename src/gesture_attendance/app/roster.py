"""Import the instructor's class roster from CSV or Excel (columns: name + student ID, any header spelling)."""
from __future__ import annotations

import csv
import io
import re

NAME_KEYS = ("name", "student name", "full name", "studentname", "fullname")
ID_KEYS = ("student id", "student_id", "studentid", "id", "student number", "university id", "uid")


def _norm(h) -> str:
    return re.sub(r"\s+", " ", str(h or "").strip().lower().replace("_", " "))


def _find(header: list[str], keys) -> int | None:
    norm = [_norm(h) for h in header]
    for k in keys:
        k = _norm(k)
        if k in norm:
            return norm.index(k)
    return None


def _from_table(rows: list[list]) -> list[tuple[str, str]]:
    rows = [r for r in rows if any(str(c or "").strip() for c in r)]
    if not rows:
        return []
    header = [str(c or "") for c in rows[0]]
    i_name, i_id = _find(header, NAME_KEYS), _find(header, ID_KEYS)
    if i_name is None or i_id is None:
        raise ValueError(f"roster needs a name column and a student ID column; got header {header}")
    out = []
    for r in rows[1:]:
        sid = r[i_id] if i_id < len(r) else None
        name = r[i_name] if i_name < len(r) else None
        if sid is None or str(sid).strip() == "":
            continue
        if isinstance(sid, float) and sid.is_integer():  # Excel stores IDs as numbers
            sid = int(sid)
        out.append((str(sid).strip(), str(name or "").strip()))
    return out


def parse_roster(data: bytes, filename: str) -> list[tuple[str, str]]:
    """Returns [(student_id, name)]."""
    if filename.lower().endswith((".xlsx", ".xlsm")):
        from openpyxl import load_workbook
        ws = load_workbook(io.BytesIO(data), read_only=True, data_only=True).active
        return _from_table([list(r) for r in ws.iter_rows(values_only=True)])
    text = data.decode("utf-8-sig", errors="replace")
    dialect = csv.Sniffer().sniff(text.splitlines()[0]) if text.strip() else csv.excel
    return _from_table(list(csv.reader(io.StringIO(text), dialect)))
