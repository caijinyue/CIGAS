#!/usr/bin/env python3
"""Portable single-process entry point. Paths are resolved from the caller's cwd."""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=['binary', 'multilabel'])
    p.add_argument('action', choices=['train', 'evaluate'])
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--pretrained', help='Local RETFound initialization checkpoint (binary training)')
    p.add_argument('--binary-checkpoint', type=Path, help='Binary initialization for multilabel training')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--epochs', type=int)
    p.add_argument('--dry-run', action='store_true')
    opts = p.parse_args()
    cfg = json.loads((ROOT / 'configs' / f'{opts.experiment}.json').read_text())
    train = opts.action == 'train'
    if not train and not opts.checkpoint:
        p.error('evaluate requires --checkpoint')
    if train and opts.experiment == 'binary' and not opts.pretrained and not opts.checkpoint:
        p.error('binary train requires --pretrained (or --checkpoint for resume)')
    if train and opts.experiment == 'multilabel' and not opts.binary_checkpoint and not opts.checkpoint:
        p.error('multilabel train requires --binary-checkpoint (or --checkpoint for resume)')
    entry = 'train_binary' if train and opts.experiment == 'binary' else 'train_multilabel'
    values = cfg['training'].copy()
    if not train:
        values.update(cfg['evaluation'])
        values['tta'] = 'tent'
    values.update(data_path=str(opts.data_root.resolve() / 'DOVS'), output_dir=str(opts.output_dir.resolve()),
                  log_dir=str(opts.output_dir.resolve() / 'logs'), device=opts.device,
                  num_workers=opts.num_workers, resume=str(opts.checkpoint.resolve()) if opts.checkpoint else '',
                  eval=not train)
    if train and opts.pretrained:
        values['finetune'] = str(Path(opts.pretrained).resolve())
    if train and opts.binary_checkpoint:
        values['single_pretrain'] = str(opts.binary_checkpoint.resolve())
    if not train:
        values['task'] = cfg['evaluation']['task']
        values['single_pretrain'] = ''
    if opts.batch_size is not None: values['batch_size'] = opts.batch_size
    if opts.epochs is not None: values['epochs'] = opts.epochs
    if opts.dry_run:
        print(json.dumps({'entry': entry, 'args': values}, indent=2)); return
    for file in [opts.checkpoint, opts.binary_checkpoint, Path(opts.pretrained) if opts.pretrained else None]:
        if file and not file.is_file(): p.error(f'File not found: {file}')
    import torch
    module = importlib.import_module(entry)
    args = module.get_args_parser().parse_args([])
    vars(args).update(values)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    criterion = torch.nn.BCEWithLogitsLoss() if opts.experiment == 'multilabel' else torch.nn.CrossEntropyLoss()
    module.main(args, criterion)

if __name__ == '__main__': main()
