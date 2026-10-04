"""Seed artifacts/tmp/sha256_cache.json from a hashes file (md5 sha256 size path per line, paths relative to
artifacts/data, as written by the single-pass hasher), using each file's current size and mtime."""
import json
import sys
from pathlib import Path

root = Path("artifacts/data")
cache_path = Path("artifacts/tmp/sha256_cache.json")
cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
n = 0
for line in Path(sys.argv[1]).read_text().splitlines():
    parts = line.split(maxsplit=3)
    if len(parts) != 4 or len(parts[0]) != 32:
        continue
    _, sha, size, rel = parts
    p = (root / rel).resolve()
    st = p.stat()
    if st.st_size != int(size):
        raise SystemExit(f"size changed since hashing: {p}")
    cache[str(p)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha}
    n += 1
cache_path.parent.mkdir(parents=True, exist_ok=True)
cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
print("seeded", n)
