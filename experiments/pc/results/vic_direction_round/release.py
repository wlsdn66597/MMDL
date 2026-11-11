"""Seal the inputs for a tmux stage. Preserve any already issued release."""
import argparse
from pathlib import Path
from campaign import ROOT, read, save, sha, timestamp


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=("train","eval","decision"))
    cli.add_argument("files",nargs="+")
    cli.add_argument("--verdict",type=Path)
    a=cli.parse_args()
    assert (a.stage=="decision")==(a.verdict is not None)
    hashes={}
    for name in a.files:
        path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT)
        hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    identity=dict(hashes=hashes)
    if a.verdict:
        v=read(a.verdict)
        assert type(v["promote"]) is bool
        identity.update(promote=v["promote"],reason=v["reason"])
    path=ROOT/"signals"/(a.stage+".json")
    if path.exists():
        old=read(path)
        assert all(old[k]==val for k,val in identity.items())
    else:save(path,dict(time_kst=timestamp(),**identity))
    print(a.stage,len(hashes),flush=True)


if __name__=="__main__":
    main()
