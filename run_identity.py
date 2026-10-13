"""Identity of numerical inputs and scripts for checkpoint reuse."""
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def source_identity():
    digest = hashlib.sha256()
    for path in sorted([*ROOT.glob('*.py'), ROOT/'udm_data.txt', ROOT/'requirements.txt']):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
