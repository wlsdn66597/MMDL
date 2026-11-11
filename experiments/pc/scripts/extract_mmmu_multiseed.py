"""Extract only regular files/directories within the named campaign directory."""
from pathlib import Path, PurePosixPath
import sys
import tarfile

archive = Path(sys.argv[1]).resolve()
root = archive.parent
campaign = 'mmmu_multiseed_20261011'
with tarfile.open(archive, 'r:gz') as source:
    for member in source.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or not path.parts or path.parts[0] != campaign or '..' in path.parts or '\\' in member.name:
            raise ValueError('Unexpected archive path: ' + member.name)
        if not (member.isfile() or member.isdir()):
            raise ValueError('Unexpected archive member: ' + member.name)
        target = (root / member.name).resolve()
        if not target.is_relative_to(root / campaign):
            raise ValueError('Archive target escapes campaign: ' + member.name)
    source.extractall(root, filter='data')
print('Verified archive paths and extracted campaign.')
