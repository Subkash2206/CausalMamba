"""
make_fig3_isic_qualitative.py — Crop the ISIC qualitative panel to a 2-row comparison.

Takes isic_sample_medium.png (3 rows: Clean / Input-LP / Feature-LP, 6 cols:
Input + 5 held-out models) and keeps only the top row (Clean) and the bottom row
(Feature-LP), dropping the middle Input-LP row. The original image is left
untouched; the crop is written to paper_v2/figures/fig3_isic_qualitative.png.

Row boundaries are detected from the white separator bands in the rendered panel,
so the script is robust to regeneration at other sizes/DPIs.

Usage:
    python interventions/experiments/make_fig3_isic_qualitative.py
"""

import os

import numpy as np
from PIL import Image

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIG_DIR = os.path.join(_REPO, 'interventions', 'results', 'paper_v2', 'figures')
SRC = os.path.join(FIG_DIR, 'isic_sample_medium.png')
DST = os.path.join(FIG_DIR, 'fig3_isic_qualitative.png')
GAP_PX = 20          # white gap between the two kept rows (matches inter-row look)
MIN_TALL = 500       # image-panel band is >500 px tall at 300 dpi
MAX_TITLE = 200      # per-row title band is narrower than this


def find_row_bands(a):
    """Return [(y_start, y_end), ...] for each of the 3 content rows."""
    ink = (a.min(axis=2) < 245).mean(axis=1)
    white = ink < 0.002
    # contiguous white bands
    wb, start = [], None
    for y, is_w in enumerate(white):
        if is_w and start is None:
            start = y
        if not is_w and start is not None:
            wb.append((start, y - 1))
            start = None
    if start is not None:
        wb.append((start, len(ink) - 1))
    # content bands = complement of white bands
    cb, prev = [], -1
    for s, e in wb:
        if s > prev + 1:
            cb.append((prev + 1, s - 1))
        prev = e
    if prev < len(ink) - 1:
        cb.append((prev + 1, len(ink) - 1))
    # a row = title band (<MIN_TALL/2 px) followed by a tall panel band (>MIN_TALL)
    rows = []
    for i, (s, e) in enumerate(cb):
        h = e - s + 1
        if h > MIN_TALL:
            # rewind across the short title band that precedes this panel
            r_start = s
            if i > 0 and (cb[i - 1][1] - cb[i - 1][0] + 1) < MAX_TITLE:
                r_start = cb[i - 1][0]
            rows.append((r_start, e))
    return rows


def main():
    im = Image.open(SRC).convert('RGB')
    a = np.asarray(im)
    rows = find_row_bands(a)
    assert len(rows) >= 3, f'expected 3 rows, found {len(rows)} in {SRC}'
    clean, feat = rows[0], rows[2]
    print(f'  source {im.size}  rows: {rows}')
    print(f'  keep Clean {clean}, Feature-LP {feat}; drop middle row')

    top = im.crop((0, 0, im.width, clean[1] + 1))          # top margin + Clean
    bot = im.crop((0, feat[0], im.width, im.height))       # Feature-LP + bottom margin
    gap = Image.new('RGB', (im.width, GAP_PX), (255, 255, 255))
    out = Image.new('RGB', (im.width, top.height + gap.height + bot.height),
                    (255, 255, 255))
    out.paste(top, (0, 0))
    out.paste(gap, (0, top.height))
    out.paste(bot, (0, top.height + gap.height))
    out.save(DST, dpi=(300, 300))
    print(f'  saved {DST}  size {out.size} @300 dpi (original left as-is)')


if __name__ == '__main__':
    main()
