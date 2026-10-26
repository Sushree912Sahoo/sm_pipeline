"""
farm_store.py - saved farms for the dashboard (one small SQLite file, no server needed).

  FarmStore(path).list_ids() / get(id) / upsert(dict) / delete(id)
  FarmStore.import_csv(farms.csv)   add or update farms from the pipeline's farm list
  FarmStore.export_csv(farms.csv)   write all saved farms as the pipeline's farm list
                                    (the old file is copied to farm_inputs/_backup first)
"""
import csv
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

# same columns and order as farm_inputs/farms.csv
COLUMNS = ["farm_id", "lat", "lon", "area_ha", "crop", "irrigation_type", "well_depth_m",
           "well_radius_m", "furrow_width_m", "furrow_length_m", "sprinkler_discharge_ls",
           "sprinkler_spacing_m", "lateral_spacing_m"]
TEXT_COLS = {"farm_id", "crop", "irrigation_type"}


def _clean(key, v):
    if v is None:
        return None
    if isinstance(v, float) and v != v:
        return None
    if key in TEXT_COLS:
        v = str(v).strip()
        return v or None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class FarmStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            cols = ", ".join(f"{k} {'TEXT' if k in TEXT_COLS else 'REAL'}" for k in COLUMNS[1:])
            c.execute(f"CREATE TABLE IF NOT EXISTS farms (farm_id TEXT PRIMARY KEY, {cols}, updated TEXT)")

    def _conn(self):
        return sqlite3.connect(self.path)

    def list_ids(self):
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT farm_id FROM farms ORDER BY farm_id")]

    def get(self, farm_id):
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            r = c.execute("SELECT * FROM farms WHERE farm_id = ?", (farm_id,)).fetchone()
        return {k: r[k] for k in COLUMNS} if r else None

    def upsert(self, farm):
        row = {k: _clean(k, farm.get(k)) for k in COLUMNS}
        if not row["farm_id"]:
            raise ValueError("a farm needs a farm id")
        names = COLUMNS + ["updated"]
        vals = [row[k] for k in COLUMNS] + [datetime.now().isoformat(timespec="seconds")]
        with self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO farms ({', '.join(names)}) "
                      f"VALUES ({', '.join('?' * len(names))})", vals)
        return row

    def delete(self, farm_id):
        with self._conn() as c:
            c.execute("DELETE FROM farms WHERE farm_id = ?", (farm_id,))

    def all_rows(self):
        return [self.get(i) for i in self.list_ids()]

    def import_csv(self, csv_path):
        n = 0
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("farm_id"):
                    self.upsert(r)
                    n += 1
        return n

    def export_csv(self, csv_path):
        """Writes all saved farms to csv_path. Returns the backup path (or None if there was no old file).
        Raises PermissionError if the file is open in another program (for example Excel)."""
        csv_path = Path(csv_path)
        backup = None
        if csv_path.exists():
            bdir = csv_path.parent / "_backup"
            bdir.mkdir(exist_ok=True)
            backup = bdir / f"farms_before_dashboard_export_{datetime.now():%Y%m%d_%H%M%S}.csv"
            shutil.copy2(csv_path, backup)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(COLUMNS)
            for r in self.all_rows():
                w.writerow(["" if r[k] is None else r[k] for k in COLUMNS])
        return backup
