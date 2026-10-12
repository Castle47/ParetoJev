"""Disk staging preserves the exact registered permutation and batch boundaries."""
from pathlib import Path
import numpy as np
from ..utils.io import read_json, atomic_write_json
from ..utils.provenance import sha256_file


def stage_epochs(data, normalized_weights, root, *, seed, epochs, identity, chunk_size=32768):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    receipt = root/'manifest.json'
    if receipt.exists():
        saved = read_json(receipt)
        if saved['identity'] != identity or saved['seed'] != seed or saved['epochs'] != epochs:
            raise ValueError('Staged protocol changed')
        for name, digest in saved['files'].items():
            if sha256_file(root/name) != digest: raise ValueError('Staged bytes changed')
        return [root/f'epoch-{epoch}.npy' for epoch in range(epochs)]
    n = len(data['x'])
    dtype = np.dtype([('x', '<f4', (data['x'].shape[1],)), ('y', 'u1', (7,)),
        ('history', '<i4', (data['history'].shape[1],)), ('item', '<i4'), ('user', '<i4'),
        ('weight', normalized_weights.dtype)])
    rng = np.random.default_rng(seed); paths = []; files = {}
    for epoch in range(epochs):
        order = rng.permutation(n)
        path = root/f'epoch-{epoch}.npy'; temporary = path.with_suffix('.partial.npy')
        dest = np.lib.format.open_memmap(temporary, mode='w+', dtype=dtype, shape=(n,))
        for start in range(0, n, chunk_size):
            idx = order[start:start+chunk_size]
            block = np.empty(len(idx), dtype=dtype)
            for key in ('x','y','history','item','user'): block[key] = data[key][idx]
            block['weight'] = normalized_weights[idx]
            dest[start:start+len(idx)] = block
        dest.flush(); del dest
        temporary.replace(path); paths.append(path); files[path.name] = sha256_file(path)
        print(f'Staged identical epoch {epoch+1}/{epochs}', flush=True)
    atomic_write_json(receipt, dict(identity=identity, seed=seed, epochs=epochs, rows=n, files=files))
    return paths


def sequential_batches(path, size):
    info = np.load(path, mmap_mode='r'); dtype, offset, n = info.dtype, info.offset, len(info)
    del info
    with Path(path).open('rb') as handle:
        handle.seek(offset)
        for start in range(0, n, size):
            count = min(size, n-start)
            block = np.fromfile(handle, dtype=dtype, count=count)
            if len(block) != count: raise ValueError('Truncated staged epoch')
            yield {key: np.array(block[key], copy=True) for key in dtype.names}
