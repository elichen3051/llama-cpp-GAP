#!/usr/bin/env python3
"""Replay one frozen manifest with two native builds and require identical metrics."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.kld_metrics_io import load_kld_metrics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-bin-dir', type=Path, required=True)
    parser.add_argument('--candidate-bin-dir', type=Path, required=True)
    parser.add_argument('--lane', choices=('vlm', 'llm'), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--library-path', help='Use the same explicit LD_LIBRARY_PATH for both runs')
    parser.add_argument('native_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    native_args = args.native_args
    if native_args[:1] == ['--']:
        native_args = native_args[1:]
    if any(arg.split('=', 1)[0] in ('--manifest', '--output-metrics') for arg in native_args):
        parser.error('the verifier owns the manifest and metric output paths')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    manifest = args.manifest.resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error('the manifest is empty')
    inputs = {}
    for row in rows:
        paths = [row[key] for key in ('tokens_in', 'formatted_chat') if key in row]
        paths.extend(row.get('images', []))
        for name in paths:
            path = Path(name)
            if not path.is_absolute():
                parser.error('manifest input paths must be absolute')
            inputs[str(path)] = sha256(path)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    if args.library_path is not None:
        env['LD_LIBRARY_PATH'] = args.library_path
    receipt = {'accepted': False, 'lane': args.lane, 'source_manifest': str(manifest),
               'source_manifest_sha256': sha256(manifest), 'input_sha256': inputs,
               'library_path': env.get('LD_LIBRARY_PATH'), 'runs': []}
    for label, directory in [('baseline', args.baseline_bin_dir), ('candidate', args.candidate_bin_dir)]:
        binary = directory.resolve() / f'llama-{args.lane}-kld'
        run = out / label
        run.mkdir()
        entries = [{**row, 'output_metrics': str(run / f'{index:06d}.vlmk')} for index, row in enumerate(rows)]
        current = run / 'manifest.jsonl'
        current.write_text(''.join(json.dumps(row) + '\n' for row in entries))
        command = [str(binary), '--manifest', str(current), *native_args]
        start = time.monotonic()
        with (run / 'native.log').open('x') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=args.timeout)
        for name, digest in inputs.items():
            if sha256(name) != digest:
                raise ValueError(f'input changed during replay: {name}')
        receipt['runs'].append({'label': label, 'command': command, 'binary_sha256': sha256(binary),
                                'wall_seconds': time.monotonic() - start})
    comparisons = []
    for index in range(len(rows)):
        baseline = out / 'baseline' / f'{index:06d}.vlmk'
        candidate = out / 'candidate' / f'{index:06d}.vlmk'
        if not baseline.is_file() or baseline.stat().st_size <= 20:
            raise ValueError(f'missing or empty baseline metric record: {baseline}')
        for path in (baseline, candidate):
            metrics, header = load_kld_metrics(path)
            if header['npos'] <= 0:
                raise ValueError(f'no scored targets: {path}')
            if any(not np.isfinite(values).all() for values in metrics.values() if values.dtype.kind == 'f'):
                raise ValueError(f'non-finite metric: {path}')
        equal = baseline.read_bytes() == candidate.read_bytes()
        comparisons.append({'row': index, 'bytes_equal': equal,
                            'baseline_sha256': sha256(baseline), 'candidate_sha256': sha256(candidate)})
    receipt['comparisons'] = comparisons
    receipt['accepted'] = all(item['bytes_equal'] for item in comparisons)
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if not receipt['accepted']:
        raise SystemExit('native metric bytes differ; inspect receipt.json')
    print(json.dumps({'accepted': True, 'rows': len(rows), 'receipt': str(out / 'receipt.json')}))


if __name__ == '__main__':
    main()
