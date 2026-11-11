"""Verify the downloaded immutable server archive before safe extraction."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
archive = root / 'results/mmmu_visual_tokens_seed42_20261010.tar.gz'
expected = '69abdc7189c72959dc9495dc23de6527fde7667631c8adde2c7f7142f0bd0577'
h = hashlib.sha256()
with archive.open('rb') as file:
    for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
        h.update(chunk)
actual = h.hexdigest()
assert actual == expected, (actual, expected)
destination = (root / 'results').resolve()
with tarfile.open(archive, 'r:gz') as file:
    members = file.getmembers()
    for member in members:
        target = (destination / member.name).resolve()
        assert target.is_relative_to(destination / 'mmmu_visual_tokens_seed42_20261010'), member.name
        assert member.isfile() or member.isdir(), member.name
    file.extractall(destination, filter='data')
record = {'server_sha256': expected, 'local_sha256': actual, 'match': True,
          'archive_bytes': archive.stat().st_size, 'archive_members': len(members)}
(destination / 'mmmu_visual_tokens_seed42_20261010/transfer_verification.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record))
