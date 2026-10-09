"""Checksum-pinned downloads and guarded extraction for large research data."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

import requests

from paretojev.utils.io import atomic_write_json, read_json


def hashes(path: Path) -> dict:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            md5.update(chunk); sha256.update(chunk)
    return {'md5': md5.hexdigest(), 'sha256': sha256.hexdigest(), 'bytes': path.stat().st_size}


def require_space(directory: Path, required_bytes: int) -> None:
    ancestor = directory.resolve()
    while not ancestor.exists(): ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    if free < required_bytes:
        raise OSError(f'Insufficient storage: {free / 2**30:.2f} GiB free; need {required_bytes / 2**30:.2f} GiB. No data removed.')


def download_asset(asset: dict, directory: str | Path, *, reserve_bytes: int = 2 * 2**30, session=None) -> dict:
    """Resume exact byte ranges, retain interrupted data, and require published MD5.

    No retry or TLS bypass. Content-length is not an integrity substitute. An
    ignored Range response is rejected rather than appended or silently restarted.
    """
    directory = Path(directory)
    name = asset['name']
    if Path(name).name != name or not re.fullmatch('[a-f0-9]{32}', asset['md5']):
        raise ValueError('Need a safe filename and pinned published checksum')
    final = directory/name; partial = directory/(name+'.part'); identity = directory/(name+'.source.json')
    if any(path.is_symlink() for path in (final, partial, identity)):
        raise ValueError('Download files must not be symlinks')
    if identity.exists() and read_json(identity) != asset:
        raise ValueError('Partial/download identity differs; refusing mixed bytes')
    if final.exists():
        observed = hashes(final)
        if observed['md5'] != asset['md5']: raise ValueError('Existing final file fails published checksum')
        return {'path': str(final.resolve()), **asset, **observed, 'status': 'verified_existing'}
    require_space(directory, reserve_bytes + (1 << 20))
    directory.mkdir(parents=True, exist_ok=True)
    if partial.exists() and not identity.exists():
        raise ValueError('Unattributed partial file; refusing to resume')
    atomic_write_json(identity, asset)
    if partial.exists() and hashes(partial)['md5'] == asset['md5']:
        os.replace(partial, final)
        return {'path': str(final.resolve()), **asset, **hashes(final), 'status': 'verified_partial_complete'}
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {'Accept-Encoding': 'identity'}
    if offset: headers['Range'] = f'bytes={offset}-'
    client = session or requests.Session()
    try:
        with client.get(asset['url'],headers=headers,stream=True,timeout=(10,60)) as response:
            response.raise_for_status()
            if response.headers.get('Content-Encoding','identity') not in ('','identity'):
                raise ValueError('Byte-range transfer must not use content encoding')
            total = None
            if offset:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range',''))
                if response.status_code != 206 or not match or int(match[1]) != offset:
                    raise ValueError('Server did not honor the exact resume range; partial file preserved')
                if int(match[2]) < offset or int(match[3]) <= int(match[2]):
                    raise ValueError('Invalid Content-Range total/end')
                total = int(match[3])
            elif response.status_code != 200:
                raise ValueError('A fresh download requires a full HTTP 200 response')
            elif response.headers.get('Content-Length'):
                total = int(response.headers['Content-Length'])
            if total is not None: require_space(directory, reserve_bytes + max(0,total-offset))
            with partial.open('ab' if offset else 'wb') as handle:
                for chunk in response.iter_content(1 << 20):
                    if not chunk: continue
                    require_space(directory,reserve_bytes + len(chunk))
                    handle.write(chunk)
                handle.flush(); os.fsync(handle.fileno())
            if total is not None and partial.stat().st_size != total:
                raise ValueError('Interrupted/short transfer; partial file retained')
    finally:
        if session is None: client.close()
    observed = hashes(partial)
    if observed['md5'] != asset['md5']:
        raise ValueError('Published checksum mismatch; unverified partial retained, not promoted')
    os.replace(partial, final)
    return {'path': str(final.resolve()), **asset, **observed, 'status': 'downloaded_verified'}


def safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or not path.parts:
        raise ValueError('Unsafe archive member')
    if path.parts[0] != 'KuaiRand-27K':
        raise ValueError('Archive must contain KuaiRand-27K, not a different variant')
    return path


def extract_27k(archive_path: str | Path, destination: str | Path, *, expected_md5: str,
                reserve_bytes: int = 2 * 2**30) -> dict:
    """Per-file atomic extraction; immutable completed files allow safe resume."""
    archive_path, destination = Path(archive_path), Path(destination)
    observed = hashes(archive_path)
    if observed['md5'] != expected_md5: raise ValueError('Archive checksum mismatch')
    require_space(destination, reserve_bytes)
    destination.mkdir(parents=True,exist_ok=True)
    manifest_path = destination/'extraction_manifest.json'
    report = read_json(manifest_path) if manifest_path.exists() else {
        'archive_md5': expected_md5, 'archive_sha256': observed['sha256'], 'status': 'in_progress', 'files': {}}
    if report['archive_sha256'] != observed['sha256']: raise ValueError('Extraction source changed')
    atomic_write_json(manifest_path,report)
    seen = set(); root = destination.resolve()
    with tarfile.open(archive_path,'r|gz') as archive:
        for member in archive:
            relative = safe_member_path(member.name)
            name = str(relative)
            if member.isdir(): continue
            if not member.isfile() or name in seen: raise ValueError('Only distinct regular archive files are accepted')
            seen.add(name)
            target = destination/relative
            target.resolve().relative_to(root)
            if target.exists():
                saved = report['files'].get(name)
                if not saved or hashes(target) != saved:
                    raise ValueError('Existing extracted file differs or has no verified provenance')
                continue
            require_space(destination, reserve_bytes + member.size)
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary = target.with_name(target.name+'.extracting')
            marker = temporary.with_name(temporary.name+'.source.json')
            expected = {'archive_sha256': observed['sha256'], 'member': name, 'bytes': member.size}
            if temporary.is_symlink() or marker.is_symlink():
                raise ValueError('Extraction temporary files must not be symlinks')
            temporary.resolve().relative_to(root)
            if (temporary.exists() and not marker.exists()) or (marker.exists() and read_json(marker) != expected):
                raise ValueError('Unattributed or mismatched extraction temporary file')
            atomic_write_json(marker, expected)
            with archive.extractfile(member) as source, temporary.open('wb') as out:
                for chunk in iter(lambda: source.read(1 << 20), b''):
                    require_space(destination,reserve_bytes + len(chunk));out.write(chunk)
                out.flush();os.fsync(out.fileno())
            if temporary.stat().st_size != member.size: raise ValueError('Incomplete archive member')
            digest = hashes(temporary)
            report['files'][name] = digest
            atomic_write_json(manifest_path, report)
            os.replace(temporary,target)
            print(f'Extracted {name}: {member.size} bytes',flush=True)
    if set(report['files']) != seen: raise ValueError('Archive/extraction manifest membership differs')
    report['status'] = 'complete';atomic_write_json(manifest_path, report)
    return report
