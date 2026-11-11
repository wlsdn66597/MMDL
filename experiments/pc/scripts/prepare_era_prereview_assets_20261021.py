import csv
import importlib.util
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/era_prereview_20261021"
OUT.mkdir(exist_ok=True)
def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
audit = read(ROOT / "reports/era_abcd_review_20261020/audit_full.json")
spec = importlib.util.spec_from_file_location("selector", ROOT / "reports/era_selection_kit_20261015/select_questions.py")
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)
history = selector.read_log(audit["selections"]["C"]["path"])
reasons = {"1003": "2026-10-21 통합 준비: clock의 소리 문제로 장식 이미지 없이도 답할 수 있어 이미지 필요성 기준 미충족.",
           "10413": "2026-10-21 통합 준비: food web의 kelp bass 먹이 관계와 primary consumer 판정의 원본 과학적 타당성 불확실. 정답을 임의 변경하지 않고 대체."}
for row in history:
    if row["원본문항ID"] in reasons:
        row.update({"상태": "제외", "제외범위": "문항", "메모": row["메모"] + " | " + reasons[row["원본문항ID"]]})
source, fingerprint = selector.load_sources(ROOT / "reports/era_dataset_examples_20261014", "C")
for number, path, items in selector.grouped(source):
    for item in items:
        candidate_exclusions = {
            "10592": "2026-10-21 보충 후보 직접 확인: 질문에서 Missouri를 명시하며 지도에는 수도 정보가 없다. 지도 없이 수도 지식만으로 답하므로 이미지 필요성 기준 미충족.",
            "10605": "2026-10-21 보충 후보 지문 확인: hint와 보기만으로 scientific name을 식별할 수 있어 이미지 필요성 기준 미충족."
        }
        if item["source_id"] in candidate_exclusions:
            excluded = selector.new_log("C", number, item, fingerprint)
            excluded.update({"상태": "제외", "제외범위": "문항", "메모": candidate_exclusions[item["source_id"]]})
            history.append(excluded)
log, selected, info = selector.select("C", source, fingerprint, 50, history)
with (OUT / "C_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=selector.HEADERS)
    w.writeheader()
    w.writerows(log)
existing = {r["source_id"] for r in (json.loads(x) for x in Path(audit["archive_identities"]["C"]["manifest"]).read_text(encoding="utf-8-sig").splitlines())}
new = [r for r in selected if r["source_id"] not in existing]
(OUT / "replacement_candidates.json").write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
print("New candidates:", json.dumps(new, ensure_ascii=False))
images = read(ROOT / "reports/era_abcd_review_20261020/image_fingerprints_full.json")
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 23)
def boards(rows, name, cols=2, h=760):
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    for start in range(0,len(rows),4):
        batch = rows[start:start+4]
        canvas = Image.new("RGB", (1800, h*2), "white")
        draw = ImageDraw.Draw(canvas)
        for idx, r in enumerate(batch):
            x, y = (idx%2)*900, (idx//2)*h
            draw.text((x+10,y+8), r["sample_id"], font=font, fill="black")
            im = Image.open(r["image_path"]).convert("RGB")
            im.thumbnail((880,h-52))
            canvas.paste(im, (x+10+(880-im.width)//2,y+48))
        canvas.save(folder / f"{start//4+1:02d}.jpg", quality=94)
boards([r for r in images if r["owner"] in "AB"], "ab_boards")
ids = ["C_1003","C_10413","C_10426","C_10430","C_10502","C_10507","C_10517","C_10586","C_10589"]
boards([r for sid in ids for r in images if r["sample_id"]==sid], "fix_boards")
