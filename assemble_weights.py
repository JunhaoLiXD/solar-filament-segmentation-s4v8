"""Rebuild weights/ from weight_parts/ using weight_parts.json. Standard library only; run from anywhere: python assemble_weights.py
Every part and every rebuilt file is checked against its SHA-256; a file already present with the right hash is left alone.
A mismatch stops with an error and leaves no partial weight file behind."""
import hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    man = json.loads((ROOT / 'weight_parts.json').read_text(encoding='utf-8'))
    (ROOT / 'weights').mkdir(exist_ok=True)
    built = kept = 0
    for w in man['files']:
        dst = ROOT / 'weights' / w['name']
        if dst.exists() and dst.stat().st_size == w['size'] and sha_file(dst) == w['sha256']:
            kept += 1
            continue
        tmp = dst.with_name(dst.name + '.tmp')
        h = hashlib.sha256()
        try:
            with open(tmp, 'wb') as out:
                for part in w['parts']:
                    data = (ROOT / 'weight_parts' / part['name']).read_bytes()
                    if sha_bytes(data) != part['sha256']:
                        raise ValueError(f"bad part {part['name']}: hash differs")
                    out.write(data)
                    h.update(data)
            if h.hexdigest() != w['sha256']:
                raise ValueError(f"bad file {w['name']}: hash differs after joining")
            os.replace(tmp, dst)
        finally:
            tmp.unlink(missing_ok=True)
        built += 1
    print(f"weights ready: {built} rebuilt, {kept} already correct, {len(man['files'])} total")


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as e:
        sys.exit(f'assemble failed: {e}')
