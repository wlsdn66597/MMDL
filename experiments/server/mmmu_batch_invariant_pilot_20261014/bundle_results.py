"""Bundle pilot evidence only; never include venv, caches, or model weights."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    status = json.loads((ROOT/'status.json').read_text())
    if status['state'] not in ('complete','mismatch','error','paused'):
        raise RuntimeError('Do not archive a live/incomplete pilot as final evidence.')
    files = [p for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink()
             and not any(s in ('.venv','cache','__pycache__','test_tmp') for s in p.relative_to(ROOT).parts)
             and p.name not in ('evidence_manifest.json','controller.lock')]
    manifest = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    evidence = ROOT/'evidence_manifest.json'
    evidence.write_text(json.dumps(dict(state=status['state'], files=manifest), indent=2)+'\n')
    archive = ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[evidence]:
            assert p.resolve().is_relative_to(ROOT.resolve())
            z.write(p, (Path(ROOT.name)/p.relative_to(ROOT)).as_posix())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps(dict(path=str(archive), sha256=digest, files=len(files)+1)))


if __name__ == '__main__':
    main()
