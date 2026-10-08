"""Download only the pinned MiniLM files needed by the development experiment."""
from pathlib import Path

import requests

from paretojev.utils.io import atomic_write_json, read_json
from paretojev.utils.provenance import sha256_file


if __name__ == '__main__':
    report = Path('reports/microlens-sasrec-development-2026-10-08')
    semantic = read_json(report / 'registration.json')['semantic']
    root = Path('artifacts/microlens-real-sasrec-dev/minilm')
    root.mkdir(parents=True, exist_ok=True)
    names = ['config.json', 'tokenizer.json', 'tokenizer_config.json',
             'special_tokens_map.json', 'vocab.txt', 'model.safetensors']
    reference = read_json(report / 'semantic.json') if (report / 'semantic.json').exists() else None
    for name in names:
        path = root / name
        if not path.exists():
            url = f"https://huggingface.co/{semantic['model']}/resolve/{semantic['revision']}/{name}"
            temporary = path.with_suffix(path.suffix + '.part')
            with requests.get(url, stream=True, timeout=90) as response:
                response.raise_for_status()
                with temporary.open('wb') as handle:
                    for chunk in response.iter_content(1 << 20):
                        handle.write(chunk)
            temporary.replace(path)
        if reference and sha256_file(path) != reference['weights_and_tokenizer_sha256'][str(path)]:
            raise ValueError(f'Downloaded file differs from the recorded artifact: {name}')
        print(name, sha256_file(path), flush=True)
    revision_path = root / 'revision.json'
    if revision_path.exists():
        existing = read_json(revision_path)
        if existing.get('model') != semantic['model'] or existing.get('revision') != semantic['revision']:
            raise ValueError('Existing semantic model provenance names a different revision')
    else:
        atomic_write_json(revision_path, {'model': semantic['model'], 'revision': semantic['revision']})
