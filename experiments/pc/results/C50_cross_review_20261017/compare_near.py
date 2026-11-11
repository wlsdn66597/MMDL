import json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
report=json.loads((OUT/'automated_audit.json').read_text(encoding='utf-8'))
def path(s):
    if s.startswith('C_'):return OUT/'source_C/images'/f"{s[2:]}.png"
    return ROOT/'results/MMDL_ERA_Dataset_D50_package/images'/f"{s.split('_')[-1]}.png"
rs=[]
for pair in report['near_pairs_dhash_le4']:
    aa=Image.open(path(pair['left'])).convert('RGB');bb=Image.open(path(pair['right'])).convert('RGB')
    r=dict(pair);r['same_dimensions']=aa.size==bb.size
    if aa.size==bb.size:
        a=np.asarray(aa).astype(np.int16);b=np.asarray(bb).astype(np.int16);diff=np.abs(a-b);mx=diff.max(2)
        r.update(mean_absolute_channel_difference=float(diff.mean()),pixel_fraction_any_difference=float((mx>0).mean()),pixel_fraction_difference_gt10=float((mx>10).mean()),max_channel_difference=int(diff.max()))
    rs.append(r)
rs.sort(key=lambda r:r.get('mean_absolute_channel_difference',999))
(OUT/'near_image_metrics.json').write_text(json.dumps(rs,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(rs,ensure_ascii=False,indent=2))
# Human/model visual adjudication panels: all candidate pairs, four pairs per page.
folder=OUT/'duplicate_panels';folder.mkdir(exist_ok=True)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
for start in range(0,len(rs),4):
    sheet=Image.new('RGB',(1600,2100),'white');draw=ImageDraw.Draw(sheet)
    for n,r in enumerate(rs[start:start+4]):
        y=n*525
        for j,key in enumerate(['left','right']):
            im=Image.open(path(r[key])).convert('RGB');im.thumbnail((775,470));sheet.paste(im,(j*800+(800-im.width)//2,y+50+(470-im.height)//2))
            draw.text((j*800+10,y+10),f"{r[key]} | dHash {r['dhash_distance']} | MAE {r.get('mean_absolute_channel_difference',-1):.3f}",font=font,fill='black')
        draw.line((0,y+524,1600,y+524),fill='#999999',width=2)
    sheet.save(folder/f'{start+1:02d}-{min(start+4,len(rs)):02d}.jpg',quality=95)
Image.open(OUT/'source_C/images/10426.png').crop((270,65,390,200)).resize((720,810)).save(OUT/'10426_detail.png')
