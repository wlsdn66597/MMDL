"""Verify the immutable downloaded ZIP before safe, non-overwriting extraction."""
from pathlib import Path, PurePosixPath
import hashlib, json, stat, zipfile

BASE = Path('C:/project/MMDL/results')
NAME = 'mmmu_batch_invariant_extended_20261014'
archive = BASE / (NAME + '.zip')
expected = archive.with_suffix('.zip.sha256').read_text(encoding='utf-8').split()[0]
def sha(data): return hashlib.sha256(data).hexdigest()
actual = sha(archive.read_bytes())
assert actual == expected, 'ZIP hash mismatch'
with zipfile.ZipFile(archive) as z:
    members = z.infolist()
    names = [i.filename for i in members]
    assert len(names) == len(set(names))
    assert len(names) == len(set(n.casefold() for n in names))
    for i in members:
        p = PurePosixPath(i.filename)
        assert not p.is_absolute() and '..' not in p.parts
        assert p.parts[0] == NAME and '\\' not in i.filename and ':' not in i.filename
        assert not stat.S_ISLNK(i.external_attr >> 16)
        dest = BASE.joinpath(*p.parts)
        assert dest.resolve().is_relative_to((BASE / NAME).resolve())
        assert not any(x.is_symlink() for x in [dest, *dest.parents])
    manifest = json.loads(z.read(NAME + '/evidence_manifest.json'))
    assert manifest['state'] == 'complete'
    expected_names = {NAME + '/' + n for n in manifest['files']} | {NAME + '/evidence_manifest.json'}
    assert set(names) == expected_names
    for n, h in manifest['files'].items():
        assert sha(z.read(NAME + '/' + n)) == h, n
    # Check every existing destination before writing any file.
    preserved = []
    for i in members:
        dest = BASE.joinpath(*PurePosixPath(i.filename).parts)
        if dest.exists():
            assert dest.is_file() and dest.read_bytes() == z.read(i), str(dest)
            preserved.append(i.filename)
    parity = BASE / NAME / 'input_parity.json'
    if parity.exists():
        assert parity.read_bytes() == z.read(NAME + '/provenance/input_parity.json')
        preserved.append(str(parity))
    for i in members:
        dest = BASE.joinpath(*PurePosixPath(i.filename).parts)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f: f.write(z.read(i))
    result = dict(passed=True, archive=str(archive), sha256=actual,
                  manifest_files=len(manifest['files']), zip_members=len(members),
                  preserved_existing=preserved, safe_paths=True, no_symlinks=True)
    out = BASE / NAME / 'transfer_verification.json'
    with out.open('x', encoding='utf-8') as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
