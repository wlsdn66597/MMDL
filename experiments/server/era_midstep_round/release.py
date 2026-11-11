"""Publish a checksum-addressed release to the already running experiment."""
import argparse
from pathlib import Path
from server_run import ROOT, read, save, sha, stamp


def main():
    p=argparse.ArgumentParser()
    p.add_argument("name",choices=["teacher-ready","eval-ready","train-ready","blind-gate"])
    p.add_argument("files",nargs="+")
    p.add_argument("--gate",type=Path)
    a=p.parse_args()
    assert (a.name=="blind-gate")==(a.gate is not None)
    hashes={}
    for rel in a.files:
        f=(ROOT/rel).resolve()
        assert f.is_relative_to(ROOT) and f.is_file()
        hashes[str(f.relative_to(ROOT))]=sha(f)
    identity=dict(hashes=hashes)
    if a.gate:
        g=read(a.gate)
        assert type(g["promote"]) is bool and isinstance(g["reason"],str)
        identity.update(promote=g["promote"],reason=g["reason"])
    dest=ROOT/"signals"/(a.name+".json")
    if dest.exists():
        existing=read(dest)
        assert all(existing[k]==v for k,v in identity.items())
    else:
        save(dest,dict(time_kst=stamp(),**identity))
    print(a.name, len(hashes))


if __name__=="__main__":
    main()
