"""Injects data/results.json into the HTML template and writes dashboard.html."""
import json
r = json.load(open("data/results.json"))
r.pop("desc", None)
page = open("dashboard/template.html").read().replace("__DATA__", json.dumps(r, separators=(",", ":")))
open("dashboard.html", "w").write(page)
print(len(page) // 1024, "KB")
