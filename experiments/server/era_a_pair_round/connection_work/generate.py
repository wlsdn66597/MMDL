"""Create the frozen 24-family binding-transfer diagnostic. Render once and distribute the PNGs."""
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "spec"
SEED = 915422073
NAMES = ["Cedar", "Opal", "Quartz", "Birch", "Orchid", "Fern", "Onyx", "Maple",
         "Slate", "Flint", "Beryl", "Violet", "Elm", "Iris", "Coral", "Hazel"]
COLORS = ["#12558a", "#a14313"]
INK = "#20303c"
GRID = "#c9d4db"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_json(path, obj):
    content = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf8") != content:
        raise ValueError(f"Existing spec differs: {path}")
    path.write_text(content, encoding="utf8")


def title(draw, x, y, s, size=26, anchor="lm", fill=INK):
    font = ImageFont.truetype("C:/Windows/Fonts/calibri.ttf", size)
    draw.text((x, y), str(s), anchor=anchor, font=font, fill=fill)


def panel(draw, names, vals, reversed_cols):
    title(draw, 52, 54, "Instrument log", 42)
    cols = [(430, "Before", 0), (660, "After", 1)]
    if reversed_cols:
        cols = [(430, "After", 1), (660, "Before", 0)]
    draw.rounded_rectangle((52, 106, 774, 371), radius=12, fill="#eaf1f4")
    title(draw, 82, 154, "Condition", 29)
    for x, label, _ in cols:
        title(draw, x, 154, label, 29, "mm")
    for y in (198, 285):
        draw.line((62, y, 763, y), fill=GRID, width=2)
    for i, (name, row) in enumerate(zip(names, vals)):
        y = 240 + i * 90
        title(draw, 82, y, name, 37)
        for x, _, j in cols:
            title(draw, x, y, row[j], 39, "mm")


def series(draw, names, vals, times):
    title(draw, 50, 37, "Sensor readings", 37)
    for j, (name, x) in enumerate(zip(names, (380, 590))):
        draw.line((x, 37, x+32, 37), fill=COLORS[j], width=5)
        title(draw, x+42, 37, name, 26)
    xlo, xhi, ylo = 98, 726, 378
    xpos = [xlo+(xhi-xlo)*(t-times[0])/(times[-1]-times[0]) for t in times]
    ypos = lambda v: ylo - 2.6*v
    for v in (0, 20, 40, 60, 80, 100):
        y = ypos(v)
        draw.line((xlo, y, xhi, y), fill=GRID)
        title(draw, xlo-15, y, v, 20, "rm")
    draw.line((xlo, 102, xlo, ylo), fill=INK, width=2)
    draw.line((xlo, ylo, xhi, ylo), fill=INK, width=2)
    for x,t in zip(xpos, times):
        title(draw, x, ylo+18, t, 22, "mm")
    title(draw, 412, 429, "Elapsed time (min)", 23, "mm")
    for i, row in enumerate(vals):
        points=[(x,ypos(v)) for x,v in zip(xpos,row)]
        draw.line(points, fill=COLORS[i], width=4)
        for (x,y),v in zip(points,row):
            draw.ellipse((x-5,y-5,x+5,y+5), fill=COLORS[i])
            # Upper and lower labels are offset in opposite directions and stay within the frame.
            title(draw, x+8 if x==xlo else x, y+15 if i==0 else y-17, v, 22, "mm", COLORS[i])


def make_pair(rng, i):
    shape = "panel" if i < 12 else "series"
    names = rng.sample(NAMES, 2)
    target = rng.choice(names)
    rev = rng.choice([True, False])
    start = rng.randint(1, 8)
    span = rng.choice([2, 4, 6])
    times = [start, start+span//2, start+span]
    while True:
        if shape == "panel":
            vals = [rng.sample(range(12, 93), 2) for _ in range(2)]
            answers = [r[-1] - r[0] for r in vals]
        else:
            vals = []
            for low, high in ((15, 43), (59, 91)):
                ends = [rng.randrange(low, high), rng.randrange(low, high)]
                mid = rng.randrange(low, high)
                vals.append([ends[0], mid, ends[1]])
            if any((r[-1]-r[0]) % span for r in vals):
                continue
            answers = [(r[-1]-r[0])//span for r in vals]
        distractors = [-answers[0], -answers[1]]
        options = answers + distractors
        if len(set(options)) == 4 and 0 not in options and len(set(sum(vals, [])))==len(sum(vals, [])):
            break
    rng.shuffle(options)
    if shape == "panel":
        question = f"For condition {target}, what is the signed change in the instrument reading from before to after?"
    else:
        question = (f"For sensor {target}, what is the average rate of change in its reading "
                    f"from {times[0]} min to {times[-1]} min, in units per minute?")
    pair = f"K{i+1:02d}"
    blind, gold = [], []
    for variant in range(2):
        order = names if variant==0 else names[::-1]
        j = order.index(target)
        rid = f"{pair}-{variant+1}"
        path = SPEC / "images" / f"{rid}.png"
        image = Image.new("RGB", (828, 458), "white")
        draw = ImageDraw.Draw(image)
        if shape == "panel":
            panel(draw, order, vals, rev)
        else:
            series(draw, order, vals, times)
        image.save(path)
        item = dict(id=rid, image="images/"+path.name, image_sha256=sha(path), question=question,
                    options=list(map(str, options)))
        ref = dict(**item, pair=pair, task=shape, label=target, marks=order, readings=vals, times=times if shape=="series" else None,
                   before=vals[j][0], after=vals[j][-1], span=span if shape=="series" else 1, result=answers[j],
                   gold="ABCD"[options.index(answers[j])], reversed_columns=rev if shape=="panel" else None)
        blind.append(item)
        gold.append(ref)
    assert gold[0]["options"] == gold[1]["options"] and gold[0]["question"] == gold[1]["question"]
    assert gold[0]["image_sha256"] != gold[1]["image_sha256"] and gold[0]["gold"] != gold[1]["gold"]
    return blind, gold


def main():
    rng = random.Random(SEED)
    (SPEC/"images").mkdir(exist_ok=True, parents=True)
    blind, gold = [], []
    for i in range(24):
        a,b = make_pair(rng,i)
        blind.extend(a); gold.extend(b)
    # Spread both orderings and domains across batches; do not send pairs consecutively by design.
    blind.sort(key=lambda r: hashlib.sha256(("challenge-shuffle"+r["id"]).encode()).digest())
    assert len({r["image_sha256"] for r in blind})==48
    assert all(r["after"]-r["before"]==r["result"]*r["span"] for r in gold)
    frozen_json(SPEC/"inputs.json",blind)
    frozen_json(SPEC/"gold.json",gold)
    frozen_json(SPEC/"manifest.json",dict(seed=SEED,items=48,pairs=24,tasks={"panel":12,"series":12},
        inputs_sha256=sha(SPEC/"inputs.json"),gold_sha256=sha(SPEC/"gold.json"),
        generator_sha256=sha(Path(__file__)), font_sha256=sha(Path("C:/Windows/Fonts/calibri.ttf"))))


if __name__ == "__main__":
    main()
