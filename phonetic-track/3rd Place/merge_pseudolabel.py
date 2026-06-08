import json

SCORE_THRESHOLD = 0.7

# ── Load phon_train_all_filtered.json ────────────────────────────────────────

train_rows = []
with open("phon_train_all_filtered.json", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            train_rows.append(json.loads(line))

# ── Load pseudolabel_phon.json, filter by pseudo_score >= 0.7 ───────────────

pseudo_rows = []
with open("pseudolabel_phon.json", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("pseudo_score", 0) >= SCORE_THRESHOLD:
            pseudo_rows.append(r)

# ── Merge and write ───────────────────────────────────────────────────────────

merged_rows = train_rows + pseudo_rows

with open("phon_train_all_filtered_pseudo07.json", "w", encoding="utf-8") as f:
    for r in merged_rows:
        out = {
            "audio_filepath": r.get("audio_filepath", ""),
            "duration":       r.get("duration", ""),
            "text":           r.get("text", ""),
            "word":           r.get("word", ""),
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"phon_train_all_filtered.json  -> {len(train_rows):,} records")
print(f"pseudolabel_phon.json (>=0.7) -> {len(pseudo_rows):,} records")
print(f"phon_train_all_filtered_pseudo07.json -> {len(merged_rows):,} records total")