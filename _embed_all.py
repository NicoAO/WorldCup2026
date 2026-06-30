"""Re-embed every file in index.html's <script id="embedded-files"> block from
disk. Keeps the existing key set; 'glue.py' is sourced from _new_glue.py."""
import json
import re
from pathlib import Path

root = Path(__file__).parent
html = (root / "index.html").read_text(encoding="utf-8")

m = re.search(
    r'(<script type="application/json" id="embedded-files">)(.*?)(</script>)',
    html, flags=re.S,
)
if not m:
    raise SystemExit("could not find embedded-files block")

data = json.loads(m.group(2))
changed = []
for key in list(data):
    disk = "_new_glue.py" if key == "glue.py" else key
    content = (root / disk).read_text(encoding="utf-8")
    if content != data[key]:
        changed.append(key)
    data[key] = content

new_json = json.dumps(data, ensure_ascii=False)
new_html = html[: m.start(2)] + new_json + html[m.end(2):]
(root / "index.html").write_text(new_html, encoding="utf-8")
print("re-embedded", len(data), "files; updated:", ", ".join(changed) or "(none)")
