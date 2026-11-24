# Copyright (C) 2026 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

"""
Hero GIF, step 1: read the recorded training runs from the released TensorBoard logs.

train_classification.py and train_classification_with_puzzle.py log, after every epoch, the CAMs of the
first eight training images and the CAM mIoU on the PASCAL VOC 2012 train set. This script copies what the
GIF needs out of the event files of the two ResNet-50 runs (classification loss only, and Puzzle-CAM):

    hero_gif/frames/<run>/CAM_<image>_epoch<NN>.jpg     the logged CAM of each chosen image, padding removed
    hero_gif/curves.json                                train mIoU of both runs after every epoch

    python hero_gif/extract_logs.py --logs experiments/tensorboards

The event files are part of experiments.zip (see the README). They are parsed with a minimal protobuf
reader, so the tensorboard package is not required.
"""
import argparse
import glob
import io
import json
import os
import struct

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--logs', default=os.path.join(ROOT, 'experiments', 'tensorboards'), help='folder with one sub-folder per run')
args = ap.parse_args()

V = json.load(open(os.path.join(HERE, 'scenes.json'), encoding='utf-8'))['training']


def varint(buf, i):
    v, shift = 0, 0
    while True:
        b = buf[i]; i += 1
        v |= (b & 0x7F) << shift
        if not b & 0x80:
            return v, i
        shift += 7


def fields(buf):
    """(field number, wire type, value) of one protobuf message."""
    i = 0
    while i < len(buf):
        key, i = varint(buf, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(buf, i)
        elif wt == 1:
            v = buf[i:i + 8]; i += 8
        elif wt == 2:
            n, i = varint(buf, i)
            v = buf[i:i + n]; i += n
        elif wt == 5:
            v = buf[i:i + 4]; i += 4
        else:
            raise ValueError('wire type %d' % wt)
        yield f, wt, v


def events(path):
    """(step, tag, value) of every summary value: a float for a scalar, the encoded bytes for an image."""
    with open(path, 'rb') as fh:
        while True:
            head = fh.read(12)
            if len(head) < 12:
                return
            rec = fh.read(struct.unpack('<Q', head[:8])[0]); fh.read(4)
            step, summary = 0, None
            for f, wt, v in fields(rec):
                if f == 2 and wt == 0:
                    step = v
                elif f == 5 and wt == 2:
                    summary = v
            for f, wt, v in fields(summary or b''):
                if f != 1 or wt != 2:
                    continue
                tag, val = '', None
                for g, gwt, gv in fields(v):
                    if g == 1 and gwt == 2:
                        tag = gv.decode('utf-8')
                    elif g == 2 and gwt == 5:
                        val = struct.unpack('<f', gv)[0]
                    elif g == 4 and gwt == 2:
                        val = next(hv for h, hwt, hv in fields(gv) if h == 4 and hwt == 2)
                if val is not None:
                    yield step, tag, val


runs = {}
for name, folder in V['runs'].items():
    scalars, images = {}, {}
    for path in sorted(glob.glob(os.path.join(args.logs, folder, 'events.*'))):
        for step, tag, val in events(path):
            (images if tag.startswith('CAM/') else scalars).setdefault(tag, {})[step] = val
    runs[name] = (scalars, images)

steps = sorted(runs['puzzle'][0]['Evaluation/train_mIoU'])               # one evaluation per epoch
assert steps == sorted(runs['baseline'][0]['Evaluation/train_mIoU'])
out = dict(epochs=list(range(1, len(steps) + 1)), steps=steps, runs={}, sizes={})
for name, (scalars, _) in runs.items():
    out['runs'][name] = dict(log=V['runs'][name], train_mIoU=[round(scalars['Evaluation/train_mIoU'][s], 4) for s in steps])

for i in V['images']:
    frames = {name: [np.asarray(Image.open(io.BytesIO(images['CAM/%d' % i][s])).convert('RGB')) for s in steps] for name, (_, images) in runs.items()}
    stack = np.stack([f for fs in frames.values() for f in fs]).astype(np.float32)
    # The evaluation images are cropped from the top-left corner and padded. The padding shows only the upsampled CAM,
    # which is smooth, so the image ends at the last row and column with texture (second differences) in any frame.
    rows = np.abs(stack[:, :, 2:] - 2 * stack[:, :, 1:-1] + stack[:, :, :-2]).mean(axis=(2, 3)).max(axis=0)
    cols = np.abs(stack[:, 2:] - 2 * stack[:, 1:-1] + stack[:, :-2]).mean(axis=(1, 3)).max(axis=0)
    h, w = len(rows), len(cols)
    while h > 0 and rows[h - 1] < 1.0:
        h -= 1
    while w > 0 and cols[w - 1] < 1.0:
        w -= 1
    out['sizes'][str(i)] = [w, h]
    for name, fs in frames.items():
        os.makedirs(os.path.join(HERE, 'frames', name), exist_ok=True)
        for e, f in zip(out['epochs'], fs):
            Image.fromarray(f[:h, :w]).save(os.path.join(HERE, 'frames', name, 'CAM_%d_epoch%02d.jpg' % (i, e)), quality=92)
    print('image %d: %d x %d, %d epochs per run' % (i, w, h, len(steps)))

json.dump(out, open(os.path.join(HERE, 'curves.json'), 'w', encoding='utf-8'), indent=1)
for name, r in out['runs'].items():
    print('%-9s train mIoU %.2f -> %.2f (best %.2f)' % (name, r['train_mIoU'][0], r['train_mIoU'][-1], max(r['train_mIoU'])))
