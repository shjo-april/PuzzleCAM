# Copyright (C) 2026 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

"""
Hero GIF, step 2 (CPU only): compose res/hero.gif from the paper figures and the recorded training runs.

Four scenes: the idea (paper Fig. 1), the CAMs of two training images after every epoch with and without the
puzzle module (recorded runs, written by extract_logs.py), the loss ablation (paper Fig. 3 and Table 1), and the
results. Every image and number comes from res/, scenes.json, frames/ and curves.json, so no model is needed.

    python hero_gif/build_gif.py                  # writes res/hero.gif
    python hero_gif/build_gif.py --preview DIR    # writes every frame to DIR as a PNG instead

The committed GIF was rendered on Windows with Segoe UI. On other systems the script falls back to the
DejaVu Sans bundled with matplotlib: same layout, slightly different glyphs.
"""
import argparse
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(ROOT, 'res', 'hero.gif'))
ap.add_argument('--preview', metavar='DIR', help='write the frames as PNGs to DIR and skip the GIF')
args = ap.parse_args()

W, H = 1280, 960
TEAL, TEAL_BG, MINT_LINE = (0, 102, 102), (232, 245, 244), (150, 205, 200)
INK, MUTED, LINE, WHITE, GREY = (16, 24, 31), (91, 105, 117), (220, 229, 233), (255, 255, 255), (178, 188, 196)
TAB_DIM = (191, 217, 217)                                          # inactive scene tab
BASE = (0, 110, 175)                                               # baseline: classification loss only / AffinityNet
M, TOP, NOTE_Y = 24, 214, 938                                      # side margin · top of the panels · baseline note
V = json.load(open(os.path.join(HERE, 'scenes.json'), encoding='utf-8'))
C = json.load(open(os.path.join(HERE, 'curves.json'), encoding='utf-8'))
EPOCHS = len(C['epochs'])

WIN_FONTS = 'C:/Windows/Fonts'
FALLBACK = {'segoeui.ttf': 'DejaVuSans.ttf', 'seguisb.ttf': 'DejaVuSans-Bold.ttf', 'segoeuib.ttf': 'DejaVuSans-Bold.ttf'}


def F(name, size):
    path = os.path.join(WIN_FONTS, name)
    if not os.path.exists(path):
        import matplotlib
        path = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', FALLBACK[name])
    return ImageFont.truetype(path, size)


F_TITLE, F_BADGE, F_SUBT, F_TAB = F('segoeuib.ttf', 40), F('seguisb.ttf', 16), F('segoeui.ttf', 17), F('segoeuib.ttf', 16)
F_HEAD, F_H, F_HS, F_HSUB = F('segoeuib.ttf', 22), F('segoeuib.ttf', 20), F('segoeui.ttf', 16), F('segoeui.ttf', 12)
F_HB, F_HBSUB = F('segoeuib.ttf', 17), F('segoeuib.ttf', 12)
F_MSG, F_NOTE, F_TICK = F('segoeuib.ttf', 20), F('segoeui.ttf', 15), F('segoeui.ttf', 14)
F_NUM, F_BIG, F_LAB = F('segoeuib.ttf', 32), F('segoeuib.ttf', 34), F('segoeuib.ttf', 18)


def logo(name, height):
    im = Image.open(os.path.join(HERE, 'logos', name)).convert('RGBA')
    return im.resize((round(im.width * height / im.height), height), Image.LANCZOS)


LOGO_L, LOGO_R = logo('gyn.png', 44), logo('kaist_soc.png', 44)


def load(rel, size):
    return Image.open(os.path.join(ROOT, rel)).convert('RGB').resize(size, Image.LANCZOS)


def rrect(dr, box, r, fill=None, outline=None, width=1):
    dr.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def rich(dr, x, y, s, font, sub, fill, anchor='l'):
    """Text with subscripts written as _{...}; y is the baseline. Returns the width."""
    parts = re.split(r'_\{([^}]*)\}', s)                           # even items: normal text, odd items: subscripts
    widths = [dr.textlength(p, font=sub if i % 2 else font) for i, p in enumerate(parts)]
    x -= {'l': 0, 'm': sum(widths) / 2, 'r': sum(widths)}[anchor]
    for i, (p, w) in enumerate(zip(parts, widths)):
        dr.text((x, y + (round(sub.size * 0.3) if i % 2 else 0)), p, font=sub if i % 2 else font, fill=fill, anchor='ls')
        x += w
    return sum(widths)


def wrap(dr, text, font, width):
    lines, cur = [], ''
    for word in text.split():
        t = (cur + ' ' + word).strip()
        if cur and dr.textlength(t, font=font) > width:
            lines.append(cur); cur = word
        else:
            cur = t
    return lines + [cur]


def line(dr, y, text, font, fill):
    """One centered line; the build stops if it does not fit between the side margins."""
    assert dr.textlength(text, font=font) <= W - 2 * M, text
    dr.text((W / 2, y), text, font=font, fill=fill, anchor='mm')


def canvas(scene, head, note):
    """White frame with the header: logos, title and venue, subtitle, scene tabs, the scene headline and its note."""
    im = Image.new('RGB', (W, H), WHITE); dr = ImageDraw.Draw(im)
    im.paste(LOGO_L, (34, 20), LOGO_L)
    im.paste(LOGO_R, (W - 34 - LOGO_R.width, 20), LOGO_R)
    tw = dr.textlength(V['title'], font=F_TITLE); bw = dr.textlength(V['badge'], font=F_BADGE) + 28
    x0 = (W - (tw + 14 + bw)) / 2
    dr.text((x0, 6), V['title'], font=F_TITLE, fill=TEAL)
    rrect(dr, (x0 + tw + 14, 22, x0 + tw + 14 + bw, 50), 14, fill=TEAL)
    dr.text((x0 + tw + 14 + bw / 2, 36), V['badge'], font=F_BADGE, fill=WHITE, anchor='mm')
    line(dr, 74, V['subtitle'], F_SUBT, MUTED)
    tabs = V['tabs']                                               # fixed position and size, only the current one is solid
    tw_ = (W - 2 * M - 16 * (len(tabs) - 1)) / len(tabs); x = M
    for i, t in enumerate(tabs):
        rrect(dr, (x, 96, x + tw_, 122), 8, fill=TEAL if i == scene else TAB_DIM)
        dr.text((x + tw_ / 2, 109), t, font=F_TAB, fill=WHITE if i == scene else TEAL, anchor='mm')
        x += tw_ + 16
    line(dr, 154, head, F_HEAD, INK)
    line(dr, NOTE_Y, note, F_NOTE, MUTED)
    return im, dr


def framed(im, dr, pic, x, y, col, width=3):
    im.paste(pic, (x, y))
    dr.rectangle((x - 2, y - 2, x + pic.width + 1, y + pic.height + 1), outline=col, width=width)


def placeholder(dr, x, y, w, h, label):
    rrect(dr, (x, y, x + w - 1, y + h - 1), 10, fill=TEAL_BG, outline=MINT_LINE)
    dr.text((x + w / 2, y + h / 2), label, font=F_HEAD, fill=MINT_LINE, anchor='mm')


def badge(dr, cx, cy, sym, col):
    dr.ellipse((cx - 21, cy - 21, cx + 21, cy + 21), fill=WHITE, outline=col, width=3)
    dr.text((cx, cy - 1), sym, font=F_HEAD, fill=col, anchor='mm')


# ---------------------------------------------------------------- scene 1: the idea (paper Fig. 1)
PW, PH, GAP = 394, 593, 24
IDEA = [(load(p, (PW, PH)), t) for p, t in V['idea']['panels']]
IDEA_X = [(W - (3 * PW + 2 * GAP)) // 2 + k * (PW + GAP) for k in range(3)]
IDEA_MSG = [("A classifier's CAM focuses on the most discriminative parts of an object", BASE),
            ('Tile the image into four patches: each patch gets its own CAM', BASE),
            ('Merged back, the CAMs of the patches differ from the CAM of the whole image', BASE),
            ('Puzzle-CAM trains the two to match, so the CAM covers the whole object', TEAL)]


def idea_frame(s):
    im, dr = canvas(0, 'The CAM of an image and the CAMs of its patches do not match', V['idea']['note'])
    shown = [True, s >= 1, s >= 3]
    for k, (pic, title) in enumerate(IDEA):
        x, col = IDEA_X[k], TEAL if k == 2 else BASE
        dr.text((x, TOP - 10), title, font=F_H, fill=col if shown[k] else GREY, anchor='ls')
        if not shown[k]:
            placeholder(dr, x, TOP, PW, PH, title[:3])
        elif k == 1 and s == 1:                                    # the four patches, apart
            g = 18; tw, th = (PW - g) // 2, (PH - g) // 2
            for i, (bx, by) in enumerate([(0, 0), (PW // 2, 0), (0, PH // 2), (PW // 2, PH // 2)]):
                tile = pic.crop((bx, by, bx + PW // 2, by + PH // 2)).resize((tw, th), Image.LANCZOS)
                framed(im, dr, tile, x + (i % 2) * (PW - tw), TOP + (i // 2) * (PH - th), col, 2)
        else:
            framed(im, dr, pic, x, TOP, col)
    if s >= 2:
        badge(dr, IDEA_X[1] - GAP / 2, TOP + PH / 2, '≠', BASE)
    if s >= 3:
        badge(dr, IDEA_X[2] - GAP / 2, TOP + PH / 2, '→', TEAL)
    line(dr, TOP + PH + 52, IDEA_MSG[s][0], F_MSG, IDEA_MSG[s][1])
    return im


# ---------------------------------------------------------------- scene 2: the recorded training runs
CW, CH = 420, 315                                                  # one CAM (the logged images are 500 x 375)
RUNS = [('baseline', 'Classification loss only', 'L_{cls}', BASE), ('puzzle', 'Puzzle-CAM (Ours)', 'L_{cls} + L_{p-cls} + αL_{re}', TEAL)]
CAMS = {(run, i, e): load('hero_gif/frames/%s/CAM_%d_epoch%02d.jpg' % (run, i, e), (CW, CH))
        for run, *_ in RUNS for i in V['training']['images'] for e in C['epochs']}
RX0, RX1 = M + 2 * CW + 12 + 24, W - M                             # right column: epoch, curve, numbers


def curve(im, e, box, lo=38, hi=49, ss=3):
    """Train mIoU of both runs up to epoch e. The lines are drawn on a larger layer and scaled down to smooth them."""
    x0, y0, x1, y1 = box
    dr = ImageDraw.Draw(im)
    Y = lambda v: y1 - (y1 - y0) * (v - lo) / (hi - lo)
    X = lambda ep: x0 + (x1 - x0) * (ep - 1) / (EPOCHS - 1)
    for v in range(lo, hi + 1, 2):
        dr.line((x0, Y(v), x1, Y(v)), fill=LINE)
        dr.text((x0 - 10, Y(v)), str(v), font=F_TICK, fill=MUTED, anchor='rm')
    for ep in (1, 5, 10, 15):
        dr.text((X(ep), y1 + 16), str(ep), font=F_TICK, fill=MUTED, anchor='mm')
    dr.text(((x0 + x1) / 2, y1 + 38), 'epoch', font=F_TICK, fill=MUTED, anchor='mm')
    layer = Image.new('RGBA', (W * ss, H * ss), (0, 0, 0, 0)); ld = ImageDraw.Draw(layer)
    for run, _, _, col in RUNS:
        pts = [(X(k + 1) * ss, Y(v) * ss) for k, v in enumerate(C['runs'][run]['train_mIoU'][:e])]
        if len(pts) > 1:
            ld.line(pts, fill=col + (255,), width=4 * ss, joint='curve')
        for k, (px, py) in enumerate(pts):
            r = (7 if k == len(pts) - 1 else 4) * ss
            ld.ellipse((px - r, py - r, px + r, py + r), fill=col + (255,), outline=WHITE + (255,), width=2 * ss)
    im.paste(Image.alpha_composite(im.convert('RGBA'), layer.resize((W, H), Image.LANCZOS)).convert('RGB'))


def training_frame(e, final=False):
    im, dr = canvas(1, 'With and without the puzzle module: the same images after every epoch', V['training']['note'])
    for j, (run, title, losses, col) in enumerate(RUNS):
        x = M + j * (CW + 12)
        dr.text((x, TOP - 10), title, font=F_H, fill=col, anchor='ls')
        rich(dr, x + dr.textlength(title, font=F_H) + 12, TOP - 11, losses, F_HS, F_HSUB, MUTED)
        for r, i in enumerate(V['training']['images']):
            framed(im, dr, CAMS[run, i, e], x, TOP + r * (CH + 12), col)
    dr.text((RX0, TOP + 16), 'Epoch', font=F_H, fill=MUTED, anchor='lm')
    dr.text((RX1, TOP + 16), '%d / %d' % (e, EPOCHS), font=F_NUM, fill=INK, anchor='rm')
    rrect(dr, (RX0, TOP + 44, RX1, TOP + 52), 4, fill=TEAL_BG)
    rrect(dr, (RX0, TOP + 44, RX0 + (RX1 - RX0) * e / EPOCHS, TOP + 52), 4, fill=TEAL)
    dr.text((RX0, TOP + 86), 'CAM mIoU on the train set (%)', font=F_HB, fill=INK, anchor='lm')
    curve(im, e, (RX0 + 36, TOP + 112, RX1 - 8, TOP + 430))
    dr = ImageDraw.Draw(im)
    for j, (run, title, _, col) in enumerate(RUNS):
        y = TOP + 520 + j * 56
        dr.text((RX0, y), title, font=F_LAB, fill=col, anchor='lm')
        dr.text((RX1, y), '%.2f' % C['runs'][run]['train_mIoU'][e - 1], font=F_NUM, fill=col, anchor='rm')
    a, b = (C['runs'][run]['train_mIoU'][-1] for run, *_ in RUNS)
    msg = ('After %d epochs: CAM mIoU %.2f with the classification loss only, %.2f with Puzzle-CAM (recorded run)' % (EPOCHS, a, b) if final else
           'The baseline stays on small parts, while Puzzle-CAM spreads over the objects')
    line(dr, 900, msg, F_MSG, TEAL)
    return im


# ---------------------------------------------------------------- scene 3: the loss ablation (paper Fig. 3 and Table 1)
LW, LH = 440, 329
LOSS = [(load(p, (LW, LH)), t, v, m) for p, t, v, m in V['losses']['rows']]
LX0, LX1 = M + 2 * LW + 16 + 24, W - M


def losses_frame(s):
    im, dr = canvas(2, 'What each loss adds to the CAM', V['losses']['note'])
    for k, (pic, title, _, _) in enumerate(LOSS):
        x, y = M + (k % 2) * (LW + 16), TOP + (k // 2) * (LH + 44)
        col = TEAL if k == 3 else BASE
        rich(dr, x, y - 10, title, F_H, F_HBSUB, col if k <= s else GREY)
        if k <= s:
            framed(im, dr, pic, x, y, col)
        else:
            placeholder(dr, x, y, LW, LH, title[:3])
    dr.text((LX0, TOP + 10), 'CAM mIoU (%)', font=F_H, fill=INK, anchor='lm')
    for k, (_, title, v, _) in enumerate(LOSS):
        y = TOP + 36 + k * 100
        on, cur = k <= s, k == s
        fg = WHITE if cur else INK if on else GREY
        rrect(dr, (LX0, y, LX1, y + 88), 12, fill=TEAL if cur else TEAL_BG if on else WHITE, outline=None if cur else MINT_LINE if on else LINE)
        rich(dr, LX0 + 16, y + 37, title, F_HB, F_HBSUB, fg)
        if on:
            delta = 'baseline' if k == 0 else '%+.2f over the baseline' % (v - LOSS[0][2])
            dr.text((LX0 + 16, y + 62), delta.replace('-', '−'), font=F_TICK, fill=(215, 238, 236) if cur else MUTED, anchor='lm')
            dr.text((LX1 - 16, y + 44), '%.2f' % v, font=F_NUM, fill=WHITE if cur else TEAL, anchor='rm')
    for n, t in enumerate(wrap(dr, LOSS[s][3], F_MSG, LX1 - LX0)):
        dr.text((LX0, TOP + 472 + n * 31), t, font=F_MSG, fill=TEAL if s == 3 else INK, anchor='lm')
    return im


# ---------------------------------------------------------------- scene 4: results
def results_frame():
    R = V['results']
    im, dr = canvas(3, 'Segmentation from image-level labels only', R['note'])
    n = len(R['columns']); tw = (W - 2 * M - 8 * (n - 1)) // n; th = round(tw * 3 / 4)
    for j, (folder, title) in enumerate(R['columns']):
        x = M + j * (tw + 8)
        col = TEAL if folder == 'puzzle-cam' else BASE if folder == 'baseline' else MUTED
        dr.text((x + tw / 2, TOP - 10), title, font=F_H, fill=col, anchor='ms')
        for r, i in enumerate(R['examples']):
            framed(im, dr, load('res/presentation/%s/%d.png' % (folder, i), (tw, th)), x, TOP + r * (th + 10), col if col != MUTED else LINE)
    y = TOP + len(R['examples']) * (th + 10) + 14
    k = len(R['cards']); cw = (W - 2 * M - 16 * (k - 1)) / k
    for j, (value, metric, desc) in enumerate(R['cards']):
        x = M + j * (cw + 16)
        rrect(dr, (x, y, x + cw, y + 140), 12, fill=TEAL_BG, outline=MINT_LINE)
        dr.text((x + cw / 2, y + 42), value, font=F_BIG, fill=TEAL, anchor='mm')
        dr.text((x + cw / 2, y + 86), metric, font=F_HB, fill=INK, anchor='mm')
        dr.text((x + cw / 2, y + 113), desc, font=F_NOTE, fill=MUTED, anchor='mm')
    line(dr, y + 140 + 36, R['line'], F_MSG, TEAL)
    return im


frames = []
add = lambda im, ms: frames.append((im, ms))
for s, ms in enumerate((1700, 1900, 2400, 3800)):
    add(idea_frame(s), ms)
add(training_frame(1), 1600)
for e in C['epochs'][1:]:
    add(training_frame(e), 450)
add(training_frame(EPOCHS, final=True), 4000)
for s, ms in enumerate((1700, 1900, 2300, 3800)):
    add(losses_frame(s), ms)
add(results_frame(), 7000)

# GIF palette: the flat UI colors and the main logo colors are reserved so text and logos stay clean; the rest of each frame is quantized.
def logo_colors(im, n=10):
    flat = Image.new('RGB', im.size, WHITE); flat.paste(im, (0, 0), im)
    p = flat.quantize(colors=n, method=Image.MEDIANCUT).getpalette()[:3 * n]
    return [tuple(p[i:i + 3]) for i in range(0, len(p), 3)]


RESERVED = list(dict.fromkeys([TEAL, TEAL_BG, MINT_LINE, INK, MUTED, LINE, WHITE, (0, 0, 0), GREY, TAB_DIM, BASE, (215, 238, 236)]
                              + logo_colors(LOGO_L) + logo_colors(LOGO_R)))


def to_palette(f):
    k = len(RESERVED)
    q = f.quantize(colors=256 - k, method=Image.MEDIANCUT)
    p = (q.getpalette() or [])[:3 * (256 - k)]
    p = [v for c in RESERVED for v in c] + p + [0] * (3 * (256 - k) - len(p))
    pi = Image.new('P', (1, 1)); pi.putpalette(p)
    out = f.quantize(palette=pi, dither=Image.FLOYDSTEINBERG)
    idx, rgb = np.array(out), np.array(f.convert('RGB'))
    for j, col in enumerate(RESERVED):
        idx[(rgb == col).all(-1)] = j
    fixed = Image.frombytes('P', out.size, idx.astype(np.uint8).tobytes()); fixed.putpalette(p)
    return fixed


total_ms = sum(ms for _, ms in frames)
if args.preview:
    os.makedirs(args.preview, exist_ok=True)
    for i, (f, _) in enumerate(frames):
        f.save(os.path.join(args.preview, f'frame_{i:03d}.png'))
    json.dump([ms for _, ms in frames], open(os.path.join(args.preview, 'durations.json'), 'w'))
    print(f'{len(frames)} frames ({total_ms / 1000:.1f} s) written to {args.preview}')
else:
    before = open(args.out, 'rb').read() if os.path.exists(args.out) else None
    pal = [to_palette(f) for f, _ in frames]
    pal[0].save(args.out, save_all=True, append_images=pal[1:], duration=[ms for _, ms in frames], loop=0, optimize=True, disposal=1)
    after = open(args.out, 'rb').read()
    state = 'new file' if before is None else 'identical to the previous file' if before == after else 'differs from the previous file'
    print(f'{args.out}: {len(frames)} frames, {total_ms / 1000:.1f} s, {len(after) / 1e6:.1f} MB ({state})')
