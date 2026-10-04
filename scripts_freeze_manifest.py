"""Write results/data_manifest.json: sha256 of every input data file and every output table.

Run after `panel-b`, `oracle-b` and `revision-b`. Seeds are fixed in the code (see revision_b.py,
panel_b.py, oracle_b.py); this manifest freezes the data/table state the paper's numbers come from.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


groups = {
    "inputs_data": sorted((ROOT / "data").glob("*.parquet")),
    "inputs_confirmatory_raw": sorted((ROOT / "results" / "confirmatory" / "raw").glob("*.txt")),
    "inputs_repeat_raw": sorted(p for d in ("noise", "noise_haiku", "pinned", "pinned_repeat")
                                for p in (ROOT / "results" / "recollect" / d).glob("*.txt")),
    "inputs_interventions": sorted((ROOT / "results" / "interv4").glob("*")),
    "outputs_tables": sorted((ROOT / "results" / "tables").glob("*_b.csv")),
}
manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "files": {g: {str(p.relative_to(ROOT)): sha(p) for p in ps if p.is_file()} for g, ps in groups.items()}}
out = ROOT / "results" / "data_manifest.json"
out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
print("wrote", out, sum(len(v) for v in manifest["files"].values()), "files")
