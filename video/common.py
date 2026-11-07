"""Shared look for every segment: canvas size, palette, fonts, header and
caption band, world-to-pixel view, frame writer. Used by render2d.py (inside
the roadmaptracking Docker image) and cards.py (host); both have matplotlib,
whose bundled DejaVu fonts are used so the typography matches everywhere."""
import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
SS = 2                      # supersampling factor for the scene layer
BAND = 120                  # caption band height at the bottom

BG, TEXT, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GREEN, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
RED, PURPLE, TEAL = '#d6453d', '#7b4fd1', '#1b9aaa'
POLY_FILL, POLY_EDGE, ROAD = '#f1f0ec', '#2b2b2b', '#9fb0c4'
GROUP_COLORS = [BLUE, GREEN, PURPLE]


def rgb(h, a=255):
    h = h.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


_FONT_DIR = None


def font(size, bold=False):
    global _FONT_DIR
    if _FONT_DIR is None:
        import matplotlib
        _FONT_DIR = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(os.path.join(_FONT_DIR, name), size)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


class View:
    """Affine world -> pixel map that fits `bounds` into `box` (x0, y0, x1, y1),
    pixel coordinates on the supersampled canvas."""

    def __init__(self, bounds, box, margin=0.06):
        minx, miny, maxx, maxy = bounds
        x0, y0, x1, y1 = box
        bw, bh = (x1 - x0) * (1 - 2 * margin), (y1 - y0) * (1 - 2 * margin)
        self.s = min(bw / (maxx - minx), bh / (maxy - miny))
        self.cx, self.cy = (minx + maxx) / 2, (miny + maxy) / 2
        self.px, self.py = (x0 + x1) / 2, (y0 + y1) / 2

    def p(self, x, y):
        return (self.px + (x - self.cx) * self.s, self.py - (y - self.cy) * self.s)

    def pts(self, seq):
        return [self.p(x, y) for x, y in seq]


class Scene:
    """Supersampled RGBA scene layer with alpha-aware drawing helpers."""

    def __init__(self, bg=BG):
        self.im = Image.new('RGBA', (W * SS, H * SS), rgb(bg))
        self.d = ImageDraw.Draw(self.im)

    def layer(self):
        return Image.new('RGBA', self.im.size, (0, 0, 0, 0))

    def blend(self, layer):
        self.im = Image.alpha_composite(self.im, layer)
        self.d = ImageDraw.Draw(self.im)

    def polygon(self, pts, fill=None, outline=None, width=0, alpha=255):
        if fill is not None and alpha < 255:
            lay = self.layer()
            ImageDraw.Draw(lay).polygon(pts, fill=rgb(fill, alpha))
            self.blend(lay)
            fill = None
        if fill is not None or outline is not None:
            self.d.polygon(pts, fill=rgb(fill) if fill else None,
                           outline=rgb(outline) if outline else None,
                           width=width * SS if outline else 0)

    def line(self, pts, color, width=2, alpha=255):
        if alpha < 255:
            lay = self.layer()
            ImageDraw.Draw(lay).line(pts, fill=rgb(color, alpha), width=width * SS, joint='curve')
            self.blend(lay)
        else:
            self.d.line(pts, fill=rgb(color), width=width * SS, joint='curve')

    def dashed(self, a, b, color, width=2, dash=14, gap=10):
        (x0, y0), (x1, y1) = a, b
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1e-6:
            return
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        s = 0.0
        while s < L:
            e = min(L, s + dash * SS)
            self.d.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)],
                        fill=rgb(color), width=width * SS)
            s = e + gap * SS

    def disc(self, c, r, color, outline=None, ow=2, alpha=255):
        x, y = c
        r *= SS
        box = [x - r, y - r, x + r, y + r]
        if alpha < 255:
            lay = self.layer()
            ImageDraw.Draw(lay).ellipse(box, fill=rgb(color, alpha))
            self.blend(lay)
        else:
            self.d.ellipse(box, fill=rgb(color), outline=rgb(outline) if outline else None,
                           width=ow * SS if outline else 0)

    def ring(self, c, r, color, width=3):
        x, y = c
        r *= SS
        self.d.ellipse([x - r, y - r, x + r, y + r], outline=rgb(color), width=width * SS)

    def diamond(self, c, r, color, outline='#ffffff'):
        x, y = c
        r *= SS
        self.d.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)],
                       fill=rgb(color), outline=rgb(outline), width=2 * SS)

    def label(self, c, text, size=20, color=TEXT, bold=True, dx=0, dy=0, bg=None):
        f = font(size * SS, bold)
        x, y = c[0] + dx * SS, c[1] + dy * SS
        if bg:
            w = self.d.textlength(text, font=f)
            pad = 5 * SS
            self.d.rounded_rectangle([x - pad, y - pad, x + w + pad, y + size * SS + pad],
                                     radius=5 * SS, fill=rgb(bg, 225))
        self.d.text((x, y), text, font=f, fill=rgb(color))

    def finish(self):
        return self.im.convert('RGB').resize((W, H), Image.LANCZOS)


def legend(d, items, x=1500, y=160):
    """items: list of (kind, color, text); kind in disc, ring, diamond, fill, line."""
    f = font(19)
    wmax = max(d.textlength(t, font=f) for _, _, t in items) + 70
    hgt = 22 + len(items) * 34
    d.rounded_rectangle([x - 16, y - 12, x + wmax, y + hgt], radius=10, fill=rgb('#ffffff', 225), outline=rgb(GRID))
    for i, (kind, color, text) in enumerate(items):
        cy = y + 8 + i * 34
        cx = x + 14
        if kind == 'disc':
            d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=rgb(color), outline=rgb('#ffffff'), width=2)
        elif kind == 'ring':
            d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], outline=rgb(color), width=3)
        elif kind == 'diamond':
            d.polygon([(cx, cy - 10), (cx + 10, cy), (cx, cy + 10), (cx - 10, cy)], fill=rgb(color))
        elif kind == 'fill':
            d.rectangle([cx - 12, cy - 9, cx + 12, cy + 9], fill=rgb(color, 110), outline=rgb(color))
        elif kind == 'line':
            d.line([(cx - 13, cy), (cx + 13, cy)], fill=rgb(color), width=4)
        elif kind == 'dash':
            for k in range(3):
                d.line([(cx - 13 + k * 10, cy), (cx - 7 + k * 10, cy)], fill=rgb(color), width=3)
        d.text((x + 40, cy - 11), text, font=f, fill=rgb(TEXT))


def chrome(im, title, subtitle='', caption='', tag='', tiles=None, pane_titles=None, legend_items=None):
    """Header (top-left), optional stat tiles (top-right), optional pane
    titles, and the caption band. Draws on the final 1920x1080 image."""
    d = ImageDraw.Draw(im, 'RGBA')
    # header
    d.text((40, 28), title, font=font(34, True), fill=rgb(TEXT))
    if subtitle:
        d.text((40, 76), subtitle, font=font(21), fill=rgb(MUTED))
    # stat tiles, right-aligned
    if tiles:
        x = W - 40
        for label, value, color in reversed(tiles):
            fv, fl = font(30, True), font(16)
            w = max(d.textlength(value, font=fv), d.textlength(label, font=fl)) + 36
            d.rounded_rectangle([x - w, 24, x, 106], radius=10, fill=rgb('#ffffff', 235), outline=rgb(GRID))
            d.text((x - w + 18, 34), value, font=fv, fill=rgb(color))
            d.text((x - w + 18, 76), label, font=fl, fill=rgb(MUTED))
            x -= w + 14
    if legend_items:
        legend(d, legend_items)
    if pane_titles:
        for (px, py, text, color) in pane_titles:
            f = font(24, True)
            w = d.textlength(text, font=f)
            d.rounded_rectangle([px - 14, py - 8, px + w + 14, py + 36], radius=8, fill=rgb('#ffffff', 235), outline=rgb(GRID))
            d.text((px, py), text, font=f, fill=rgb(color))
    # caption band
    d.rectangle([0, H - BAND, W, H], fill=rgb(BG))
    d.line([(0, H - BAND), (W, H - BAND)], fill=rgb(GRID), width=2)
    if caption:
        lines = caption.split('\n')
        if len(lines) == 1 and d.textlength(caption, font=font(30)) > W - 80:
            words = caption.split(' ')
            best = None
            for i in range(1, len(words)):
                a, b = ' '.join(words[:i]), ' '.join(words[i:])
                wa, wb = d.textlength(a, font=font(27)), d.textlength(b, font=font(27))
                if max(wa, wb) <= W - 80 and (best is None or abs(wa - wb) < best[0]):
                    best = (abs(wa - wb), a, b)
            if best:
                lines = [best[1], best[2]]
        f = font(30 if len(lines) == 1 else 27)
        y = H - BAND + (BAND - len(lines) * (f.size + 8)) / 2
        for ln in lines:
            d.text((40, y), ln, font=f, fill=rgb(TEXT))
            y += f.size + 8
    if tag:
        f = font(17)
        w = d.textlength(tag, font=f)
        d.text((W - 40 - w, H - 34), tag, font=f, fill=rgb(MUTED))
    return im


class Writer:
    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        for f in os.listdir(outdir):
            if f.endswith('.jpg'):
                os.remove(os.path.join(outdir, f))
        self.n = 0

    def write(self, im):
        im.save(os.path.join(self.outdir, f'{self.n:05d}.jpg'), quality=93, subsampling=0)
        self.n += 1


def caption_at(schedule, t):
    """schedule: list of (start_seconds, caption). Returns the active caption."""
    cur = ''
    for start, text in schedule:
        if t >= start:
            cur = text
    return cur
