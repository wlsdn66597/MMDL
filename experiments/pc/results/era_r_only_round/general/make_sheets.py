"""Write local image inspection pages; filenames are not used as training or solver inputs."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps

HERE=Path(__file__).resolve().parent
def main():
    items=[json.loads(s) for s in (HERE/"candidates.jsonl").read_text(encoding="utf8").splitlines()]
    for name,setsize,cols,w,h in [("ChartQA",4,2,550,350),("ScienceQA",9,3,315,242)]:
        group=sorted((r for r in items if r["source"].startswith(name)),key=lambda r:r["id"])
        for start in range(0,len(group),setsize):
            page=Image.new("RGB",(cols*w,((setsize+cols-1)//cols)*h),"white");d=ImageDraw.Draw(page)
            for j,r in enumerate(group[start:start+setsize]):
                x=j%cols*w;y=j//cols*h
                im=ImageOps.contain(Image.open(HERE.parent/r["image"]).convert("RGB"),(w-4,h-36))
                page.paste(im,(x+(w-im.width)//2,y+33));d.text((x+2,y+2),r['id']+" "+r['gold']+" "+r['options'][ord(r['gold'])-65][:38],fill="black")
                d.text((x+2,y+15),r['question'][:66],fill="black")
            page.save(HERE/f"{name.lower()}_{start//setsize+1}.jpg",quality=90)
    print(len(items))
if __name__=="__main__": main()
