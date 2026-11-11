"""Versioned image identity registry and fail-closed train/evaluation overlap checks.

Dependencies: Pillow, numpy. Perceptual hashes propose candidates; never merge them.
Run --help for commands. Original assignment numbers and question/answer text are immutable.
"""
import argparse
import csv
import hashlib
import itertools
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROFILE = {'version':'visual-fingerprint-v1', 'exact':'EXIF-transposed RGBA; dimensions + pixels SHA256',
           'dhash':'64-bit; grayscale white composite; 9x8 BICUBIC; right > left',
           'phash':'64-bit; grayscale white composite; 32x32 LANCZOS; unnormalized DCT-II; coefficients abs<1e-8 zeroed; top-left 8x8 > median including DC',
           'dhash_threshold':4, 'phash_threshold':6}
VERSION = 'visual-groups-v1'

def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def jsonl(path, data): Path(path).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data),encoding='utf-8')
def sha(data): return hashlib.sha256(data).hexdigest()
def bits(array): return f'{int("".join("1" if x else "0" for x in array.flatten()),2):016x}'
def hamming(a,b): return (int(a,16)^int(b,16)).bit_count()
def key(a,b): return tuple(sorted((a,b)))

def fingerprint(path):
    blob=Path(path).read_bytes()
    with Image.open(path) as source:
        rgba=ImageOps.exif_transpose(source).convert('RGBA')
        size=list(rgba.size)
        pixel=sha(f'RGBA:{size[0]}x{size[1]}:'.encode()+rgba.tobytes())
        rgb=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
        gray=rgb.convert('L')
        small=np.asarray(gray.resize((9,8),Image.Resampling.BICUBIC))
        dh=bits(small[:,1:]>small[:,:-1])
        data=np.asarray(gray.resize((32,32),Image.Resampling.LANCZOS),dtype=float)
        # Same unnormalized DCT-II convention as scipy.fftpack.dct(type=2).
        matrix=2*np.cos(np.pi*np.arange(32)[:,None]*(2*np.arange(32)[None,:]+1)/64)
        low=(matrix@data@matrix.T)[:8,:8]
        low[np.abs(low)<1e-8]=0
        ph=bits(low>np.median(low))
    return dict(file_sha256=sha(blob),pixel_sha256=pixel,width=size[0],height=size[1],dhash=dh,phash=ph)

def distance(a,b):
    return {'dhash_distance':hamming(a['dhash'],b['dhash']),'phash_distance':hamming(a['phash'],b['phash'])}
def is_candidate(a,b):
    d=distance(a,b)
    return d['dhash_distance']<=PROFILE['dhash_threshold'] or d['phash_distance']<=PROFILE['phash_threshold']

def image_path(record, manifest):
    name=record.get('image_path')
    if not name:
        found=[x.get('path') for m in record.get('messages',[]) if isinstance(m.get('content'),list)
               for x in m['content'] if x.get('type')=='image']
        if len(found)!=1: raise ValueError(f'{record.get("sample_id")}: exactly one image is required')
        name=found[0]
    base=Path(manifest).resolve().parent
    path=(base/name).resolve()
    if not path.is_relative_to(base): raise ValueError(f'Image path escapes manifest directory: {name}')
    if path.is_file(): return path
    # Explicit compatibility for the supplied C/D source packages only.
    sid=str(record.get('source_id',''))
    fallback=base/'images'/f'{sid}.png'
    if sid.isdigit() and name==f'train/{sid}/image.png' and fallback.is_file(): return fallback
    raise ValueError(f'Missing image: {path}')

def parse_named(values):
    result={}
    for value in values:
        if '=' not in value: raise ValueError('Use NAME=JSONL_PATH')
        name,path=value.split('=',1)
        if not name or not name.replace('_','').isalnum() or name in result: raise ValueError(f'Invalid/repeated name: {name}')
        result[name]=Path(path).resolve()
    return result

def collect(packages):
    data=[]; originals={}; seen=set()
    for owner,manifest in packages.items():
        source=rows(manifest); originals[owner]=source
        for row in source:
            sid=row['sample_id']
            if sid in seen: raise ValueError(f'Duplicate sample_id: {sid}')
            seen.add(sid); path=image_path(row,manifest); fp=fingerprint(path)
            if row.get('image_sha256') and row['image_sha256']!=fp['file_sha256']:
                raise ValueError(f'Stale declared image SHA256: {sid}')
            data.append(dict(sample_id=sid,package=owner,source_dataset=row.get('source_dataset',row.get('dataset')),
                             source_id=row.get('source_id'),image_group_no=row.get('image_group_no'),
                             image_path=str(path),**fp))
    return data,originals

def pair_metrics(a,b):
    result=distance(a,b)
    if (a['width'],a['height'])==(b['width'],b['height']):
        with Image.open(a['image_path']) as x,Image.open(b['image_path']) as y:
            aa=np.asarray(ImageOps.exif_transpose(x).convert('RGB'),dtype=np.int16)
            bb=np.asarray(ImageOps.exif_transpose(y).convert('RGB'),dtype=np.int16)
            delta=np.abs(aa-bb)
            result.update(mean_absolute_channel_difference=float(delta.mean()),pixel_fraction_any_difference=float(np.any(delta!=0,axis=2).mean()))
    return result

def scan(data):
    return [dict(left=a['sample_id'],right=b['sample_id'],left_sha256=a['file_sha256'],right_sha256=b['file_sha256'],
                 exact_pixels=a['pixel_sha256']==b['pixel_sha256'],**pair_metrics(a,b))
            for a,b in itertools.combinations(data,2) if a['pixel_sha256']==b['pixel_sha256'] or is_candidate(a,b)]

def panels(data,pairs,out):
    by_id={r['sample_id']:r for r in data}; out.mkdir(parents=True,exist_ok=True)
    fontpath=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(fontpath),22) if fontpath.exists() else ImageFont.load_default()
    for start in range(0,len(pairs),4):
        canvas=Image.new('RGB',(1600,2080),'white'); draw=ImageDraw.Draw(canvas)
        for j,p in enumerate(pairs[start:start+4]):
            for col,field in enumerate(['left','right']):
                sid=p[field]; y=j*520
                label=f'{start+j+1}. {sid} | d={p["dhash_distance"]}, p={p["phash_distance"]}'
                draw.text((col*800+10,y+5),label,fill='black',font=font)
                with Image.open(by_id[sid]['image_path']) as im:
                    thumb=ImageOps.exif_transpose(im).convert('RGB');thumb.thumbnail((780,465))
                    canvas.paste(thumb,(col*800+(800-thumb.width)//2,y+45))
                draw.line((0,y+519,1600,y+519),fill='#bbbbbb')
        canvas.save(out/f'{start+1:03d}-{min(start+4,len(pairs)):03d}.jpg',quality=90)

def validated_decisions(data,decisions):
    by_id={r['sample_id']:r for r in data}; seen={}; result=[]
    for r in decisions:
        a,b=by_id[r['left']],by_id[r['right']]
        if a['sample_id']==b['sample_id']:raise ValueError('A decision cannot compare a sample with itself')
        if r['left_sha256']!=a['file_sha256'] or r['right_sha256']!=b['file_sha256']:
            raise ValueError(f'Stale image-bound decision: {r["left"]}, {r["right"]}')
        if r['verdict'] not in ['same','different'] or not r.get('reviewer') or not r.get('reason'):
            raise ValueError('Decision needs verdict same/different, reviewer and reason')
        k=key(a['pixel_sha256'],b['pixel_sha256'])
        if k in seen and seen[k]!=r['verdict']:raise ValueError('Contradictory pair decisions')
        if k in seen:continue
        if k[0]==k[1] and r['verdict']=='different':raise ValueError('Cannot separate exact same pixels')
        seen[k]=r['verdict']
        result.append(dict(r,left_pixel_sha256=a['pixel_sha256'],right_pixel_sha256=b['pixel_sha256']))
    return result

def assign(data,decisions,previous=None):
    parents={r['pixel_sha256']:r['pixel_sha256'] for r in data}
    def find(x):
        while parents[x]!=x:parents[x]=parents[parents[x]];x=parents[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    prior={}
    if previous:
        if previous['hash_profile']!=PROFILE:raise ValueError('Previous hash profile differs')
        for row in previous['images']:prior[row['pixel_sha256']]=row['visual_group_id']
        old=defaultdict(list)
        for px in parents:
            if px in prior:old[prior[px]].append(px)
        for values in old.values():
            for px in values[1:]:union(values[0],px)
    for r in decisions:
        if r['verdict']=='same':union(r['left_pixel_sha256'],r['right_pixel_sha256'])
    for r in decisions:
        if r['verdict']=='different' and find(r['left_pixel_sha256'])==find(r['right_pixel_sha256']):
            raise ValueError('A different decision contradicts a transitive same-image group')
    components=defaultdict(list)
    for r in data:components[find(r['pixel_sha256'])].append(r)
    groups=[]
    for root,members in sorted(components.items()):
        oldids={prior[r['pixel_sha256']] for r in members if r['pixel_sha256'] in prior}
        if len(oldids)>1:raise ValueError('Merging existing published groups requires an explicit migration; old IDs will not be silently changed')
        gid=next(iter(oldids)) if oldids else 'VG_'+root[:24]
        for r in members:r['visual_group_id']=gid
        groups.append(dict(visual_group_id=gid,member_ids=sorted(r['sample_id'] for r in members),
                           pixel_sha256s=sorted({r['pixel_sha256'] for r in members}),
                           basis='reviewed_same' if len({r['pixel_sha256'] for r in members})>1 else ('exact_pixels' if len(members)>1 else 'singleton')))
    if len({g['visual_group_id'] for g in groups})!=len(groups):raise ValueError('Group ID collision')
    return groups

def build(args):
    packages=parse_named(args.package);out=Path(args.output).resolve()
    if out.exists():raise ValueError('Output already exists; choose a new version directory')
    data,originals=collect(packages)
    if not args.previous_registry and any(r.get('visual_group_id') for source in originals.values() for r in source):
        raise ValueError('Already grouped input requires --previous-registry to preserve published visual group IDs')
    decisions=validated_decisions(data,load(args.decisions) if args.decisions else [])
    previous=load(args.previous_registry) if args.previous_registry else None
    if previous:
        prior_ids={x['sample_id'] for x in previous['images']}
        if not prior_ids <= {x['sample_id'] for x in data}:raise ValueError('Incremental build requires all previous samples plus additions')
        old_pixels={x['sample_id']:x['pixel_sha256'] for x in previous['images']}
        if any(r['sample_id'] in old_pixels and old_pixels[r['sample_id']]!=r['pixel_sha256'] for r in data):
            raise ValueError('A published sample image changed; review and explicitly migrate the registry')
        # All prior adjudications must be retained or explicitly migrate outside this command.
        previous_decisions=validated_decisions(data,previous['decisions'])
        decisions=validated_decisions(data,previous_decisions+decisions)
    groups=assign(data,decisions,previous)
    candidates=scan(data); by_id={r['sample_id']:r for r in data}
    decision_map={key(d['left_pixel_sha256'],d['right_pixel_sha256']):d['verdict'] for d in decisions}
    for pair in candidates:
        a,b=by_id[pair['left']],by_id[pair['right']]
        pair['status']='same' if a['visual_group_id']==b['visual_group_id'] else decision_map.get(key(a['pixel_sha256'],b['pixel_sha256']),'pending')
    registry_id='VGR_'+sha(json.dumps({'profile':PROFILE,'members':sorted((r['sample_id'],r['file_sha256'],r['visual_group_id']) for r in data),
        'decisions':sorted((d['left_pixel_sha256'],d['right_pixel_sha256'],d['verdict']) for d in decisions)},sort_keys=True).encode())[:24]
    out.mkdir(parents=True)
    manifest_hashes={name:sha(path.read_bytes()) for name,path in packages.items()}
    for name,source in originals.items():
        dest=out/name; (dest/'images').mkdir(parents=True)
        paired=[];plain=[];era=[]
        for row in source:
            info=by_id[row['sample_id']];path=Path(info['image_path']);image=f'images/{path.name}'
            if (dest/image).exists() and sha((dest/image).read_bytes())!=info['file_sha256']:raise ValueError('Image filename collision')
            shutil.copy2(path,dest/image)
            item=dict(row,visual_group_id=info['visual_group_id'],visual_group_registry_version=VERSION,
                      visual_group_registry_id=registry_id,
                      visual_group_registry='visual_group_registry.json',image_pixel_sha256=info['pixel_sha256'],
                      image_dhash=info['dhash'],image_phash=info['phash'],image_fingerprint_version=PROFILE['version'])
            if item['image_path']!=image:item['original_image_path']=item['image_path'];item['image_path']=image
            paired.append(item)
            for variant,target in [('general',plain),('era',era)]:
                target.append(dict(sample_id=item['sample_id'],visual_group_id=info['visual_group_id'],
                    visual_group_registry_id=registry_id,
                    messages=[{'role':'user','content':[{'type':'image','path':image},{'type':'text','text':item['input_prompt']}]},
                              {'role':'assistant','content':item['response_'+variant]}]))
            info['image_path']=f'{name}/{image}'
        jsonl(dest/'paired_train.draft.jsonl',paired);jsonl(dest/'train_plain.draft.jsonl',plain);jsonl(dest/'train_era.draft.jsonl',era)
    registry=dict(schema_version=VERSION,registry_id=registry_id,hash_profile=PROFILE,source_manifest_sha256=manifest_hashes,
                  groups=groups,images=data,decisions=decisions,candidates=candidates,
                  unresolved_pairs=sum(p['status']=='pending' for p in candidates),
                  scope='C/D submitted images only; not all ScienceQA, not human final content approval')
    dump(out/'visual_group_registry.json',registry)
    for name in packages:shutil.copy2(out/'visual_group_registry.json',out/name/'visual_group_registry.json')
    with (out/'visual_groups.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','owner','source_id','image_group_no','visual_group_id','image_path','file_sha256','pixel_sha256'])
        for r in data:w.writerow([r['sample_id'],r['package'],r['source_id'],r['image_group_no'],r['visual_group_id'],r['image_path'],r['file_sha256'],r['pixel_sha256']])
    report=dict(rows=len(data),groups=len(groups),multi_sample_groups=sum(len(g['member_ids'])>1 for g in groups),
                candidate_statuses=dict(Counter(p['status'] for p in candidates)),
                assignment_numbers_preserved=all(next(x for x in originals[r['package']] if x['sample_id']==r['sample_id']).get('image_group_no')==r['image_group_no'] for r in data),
                no_split_performed=True,source_manifest_sha256=manifest_hashes)
    dump(out/'build_report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))

def check_records(partitions,registry):
    if registry['hash_profile']!=PROFILE:raise ValueError('Unsupported registry hash profile')
    known={r['pixel_sha256']:r['visual_group_id'] for r in registry['images']}
    different={key(d['left_pixel_sha256'],d['right_pixel_sha256']) for d in registry['decisions'] if d['verdict']=='different'}
    data=[];invalid=[]
    for split,manifest in partitions.items():
        source=rows(manifest)
        if not source:invalid.append({'split':split,'reason':'Empty partition'})
        seen=set()
        for row in source:
            sid=row.get('sample_id')
            try:
                if not sid or sid in seen:raise ValueError('Missing/repeated sample_id within partition')
                seen.add(sid);path=image_path(row,manifest);fp=fingerprint(path)
                if row.get('image_sha256') and row['image_sha256']!=fp['file_sha256']:raise ValueError('Declared file SHA256 mismatch')
                if row.get('image_pixel_sha256') and row['image_pixel_sha256']!=fp['pixel_sha256']:raise ValueError('Declared pixel SHA256 mismatch')
                gid=known.get(fp['pixel_sha256']);declared=row.get('visual_group_id')
                if declared and declared!=gid:raise ValueError('Declared visual_group_id is unknown or inconsistent with actual image')
                data.append(dict(sample_id=sid,split=split,visual_group_id=gid or 'UNREGISTERED_'+fp['pixel_sha256'],registered=bool(gid),**fp))
            except (ValueError,OSError) as e:invalid.append({'sample_id':sid,'split':split,'reason':str(e)})
    leaks=[];pending=[]
    for a,b in itertools.combinations(data,2):
        if a['split']==b['split']:continue
        pair=dict(left=a['sample_id'],left_split=a['split'],right=b['sample_id'],right_split=b['split'])
        if a['sample_id']==b['sample_id'] or a['visual_group_id']==b['visual_group_id'] or a['pixel_sha256']==b['pixel_sha256']:
            leaks.append(dict(pair,reason='same sample, verified visual group, or exact pixels'))
        elif is_candidate(a,b) and key(a['pixel_sha256'],b['pixel_sha256']) not in different:
            pending.append(dict(pair,**distance(a,b),reason='Unreviewed similar-image candidate; do not treat as a confirmed duplicate'))
    status='invalid' if invalid else ('leak' if leaks else ('review_required' if pending else 'pass'))
    return dict(status=status,exit_code={'pass':0,'leak':1,'review_required':2,'invalid':3}[status],
                scope='Verified groups + exact pixels + perceptual candidates across supplied partitions; no guarantee against arbitrary crops, edits or shared templates',
                partitions={k:len(rows(v)) for k,v in partitions.items()},verified_records=len(data),
                unregistered_samples=[r['sample_id'] for r in data if not r['registered']],
                invalid_records=invalid,leaks=leaks,review_required=pending,hash_profile=PROFILE)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('scan',help='Fingerprint packages and emit candidate pairs without automatic near-image merging')
    s.add_argument('--package',action='append',required=True);s.add_argument('--output',required=True);s.add_argument('--panels',action='store_true')
    b=sub.add_parser('build',help='Create separately enriched C/D packages and shared registry')
    b.add_argument('--package',action='append',required=True);b.add_argument('--decisions');b.add_argument('--previous-registry');b.add_argument('--output',required=True)
    c=sub.add_parser('check',help='Check actual image bytes and verified groups across train/dev/test manifests')
    c.add_argument('--partition',action='append',required=True);c.add_argument('--registry',required=True);c.add_argument('--report',required=True)
    args=p.parse_args(argv)
    try:
        if args.command=='scan':
            out=Path(args.output)
            if out.exists():raise ValueError('Choose a new scan output directory')
            data,_=collect(parse_named(args.package));pairs=scan(data)
            pairs.sort(key=lambda x:(x.get('mean_absolute_channel_difference',999),x['left'],x['right']))
            dump(out/'inventory.json',{'hash_profile':PROFILE,'images':data});dump(out/'candidates.json',pairs)
            if args.panels:panels(data,pairs,out/'panels')
            print(json.dumps({'images':len(data),'candidates':len(pairs)},indent=2))
        elif args.command=='build':build(args)
        else:
            partitions=parse_named(args.partition)
            if len(partitions)<2:raise ValueError('At least two distinct partitions are required')
            report=check_records(partitions,load(args.registry));dump(args.report,report)
            print(json.dumps({k:v for k,v in report.items() if k not in ['hash_profile','scope']},ensure_ascii=False,indent=2))
            return report['exit_code']
        return 0
    except (ValueError,KeyError,OSError,json.JSONDecodeError) as e:
        print(f'ERROR: {e}',file=sys.stderr)
        if args.command=='check':dump(args.report,{'status':'invalid','exit_code':3,'error':str(e)})
        return 3

if __name__=='__main__':sys.exit(main())
