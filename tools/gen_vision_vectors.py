#!/usr/bin/env python3
"""Regenerate the image decoder test vectors (tests/data/vision) and fuzz seeds (fuzz/corpus/{png,bmp,pnm}).

Every file is written by code in this script from pixel arrays it made up (stdlib zlib + struct only), next to
a .raw file holding the pixels a decoder must produce: four little-endian u32 (width, height, channels,
depth) and then the samples (u8, or little-endian u16 for depth 16). The PNG writer here is independent of
the C++ one: it applies every filter type, splits IDAT, writes Adam7, palettes, tRNS and low bit depths.
When Pillow is importable each file is also decoded with it and compared (a check of this script).
Usage: python3 tools/gen_vision_vectors.py   (from the repository root; deterministic)"""
import os, random, struct, zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "data", "vision")
FUZZ = os.path.join(ROOT, "fuzz", "corpus")
rnd = random.Random(1234)

def chunk(t, body): return struct.pack(">I", len(body)) + t + body + struct.pack(">I", zlib.crc32(t + body))

def paeth(a, b, c):
    p = a + b - c; pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c

def filt(rows, bpp, pick):
    out = bytearray(); prev = bytes(len(rows[0])) if rows else b""
    for y, row in enumerate(rows):
        f = pick(y); out.append(f)
        for i, v in enumerate(row):
            a = row[i - bpp] if i >= bpp else 0; b = prev[i]; c = prev[i - bpp] if i >= bpp else 0
            pred = [0, a, b, (a + b) // 2, paeth(a, b, c)][f]
            out.append((v - pred) & 255)
        prev = row
    return bytes(out)

def pack_rows(px, w, h, ch, depth, xs, ys):
    """px[y][x] = tuple of ch samples. Returns packed scanlines for the sub-image picked by xs, ys."""
    rows = []
    for y in ys:
        if depth == 16: rows.append(b"".join(struct.pack(">H", s) for x in xs for s in px[y][x]))
        elif depth == 8: rows.append(bytes(s for x in xs for s in px[y][x]))
        else:
            bits = "".join(format(px[y][x][0], "0%db" % depth) for x in xs); bits += "0" * (-len(bits) % 8)
            rows.append(bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)))
    return rows

def png(px, w, h, ctype, depth, interlace=False, pick=lambda y: y % 5, plte=None, trns=None, idat_split=0, level=6, extra=b""):
    ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]; bpp = max(1, ch * depth // 8)
    if not interlace: raw = filt(pack_rows(px, w, h, ch, depth, range(w), range(h)), bpp, pick)
    else:
        raw = b""
        for x0, y0, dx, dy in [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]:
            xs, ys = range(x0, w, dx), range(y0, h, dy)
            if len(xs) and len(ys): raw += filt(pack_rows(px, w, h, ch, depth, xs, ys), bpp, pick)
    z = zlib.compress(raw, level)
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 1 if interlace else 0)) + extra
    if plte is not None: out += chunk(b"PLTE", bytes(v for e in plte for v in e))
    if trns is not None: out += chunk(b"tRNS", trns)
    parts = [z] if not idat_split else [z[i:i + idat_split] for i in range(0, len(z), idat_split)] or [b""]
    for p in parts: out += chunk(b"IDAT", p)
    return out + chunk(b"IEND", b"")

def raw_file(w, h, c, depth, samples):
    return struct.pack("<IIII", w, h, c, depth) + (bytes(samples) if depth == 8 else b"".join(struct.pack("<H", s) for s in samples))

def image(w, h, ch, maxv, smooth=True):
    if smooth: return [[tuple((x * 7 + y * 13 + k * 31 + (x * y) % 5) % (maxv + 1) for k in range(ch)) for x in range(w)] for y in range(h)]
    return [[tuple(rnd.randrange(maxv + 1) for _ in range(ch)) for x in range(w)] for y in range(h)]

vectors = {}   # name -> (file bytes, expected raw)
def add(name, data, w, h, c, depth, samples): vectors[name] = (data, raw_file(w, h, c, depth, samples))

# ---- PNG ----
for ctype, ch in [(0, 1), (2, 3), (4, 2), (6, 4)]:
    for depth in (8, 16):
        for inter in (False, True):
            for (w, h) in [(1, 1), (5, 7), (33, 17)]:
                px = image(w, h, ch, (1 << depth) - 1, smooth=(w != 5))
                data = png(px, w, h, ctype, depth, inter, idat_split=37 if w == 33 else 0)
                add("png_c%d_d%d_%s_%dx%d.png" % (ctype, depth, "adam7" if inter else "plain", w, h), data, w, h, ch, depth, [s for row in px for p in row for s in p])
for depth in (1, 2, 4):   # low-depth grey: scaled to 8 bits
    for inter in (False, True):
        w, h = 13, 9; px = image(w, h, 1, (1 << depth) - 1, smooth=False); scale = 255 // ((1 << depth) - 1)
        add("png_gray_d%d_%s.png" % (depth, "adam7" if inter else "plain"), png(px, w, h, 0, depth, inter), w, h, 1, 8, [p[0] * scale for row in px for p in row])
for depth in (1, 2, 4, 8):   # palette, with and without tRNS
    n = min(1 << depth, 40); pal = [(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(n)]
    for inter in (False, True):
        w, h = 19, 11; px = [[(rnd.randrange(n),) for x in range(w)] for y in range(h)]
        add("png_pal_d%d_%s.png" % (depth, "adam7" if inter else "plain"), png(px, w, h, 3, depth, inter, plte=pal), w, h, 3, 8, [v for row in px for p in row for v in pal[p[0]]])
        alpha = bytes(rnd.randrange(256) for _ in range(max(1, n // 2)))   # shorter than the palette: the rest are opaque
        add("png_pal_trns_d%d_%s.png" % (depth, "adam7" if inter else "plain"), png(px, w, h, 3, depth, inter, plte=pal, trns=alpha), w, h, 4, 8,
            [v for row in px for p in row for v in (*pal[p[0]], alpha[p[0]] if p[0] < len(alpha) else 255)])
# colour-key transparency
px = image(9, 6, 1, 255, smooth=False); px[2][3] = (77,); px[5][8] = (77,)
add("png_gray_key.png", png(px, 9, 6, 0, 8, trns=struct.pack(">H", 77)), 9, 6, 2, 8, [v for row in px for p in row for v in (p[0], 0 if p[0] == 77 else 255)])
px = image(9, 6, 3, 255, smooth=False); px[1][1] = (1, 2, 3); px[4][7] = (1, 2, 3)
add("png_rgb_key.png", png(px, 9, 6, 2, 8, trns=struct.pack(">HHH", 1, 2, 3)), 9, 6, 4, 8, [v for row in px for p in row for v in (*p, 0 if p == (1, 2, 3) else 255)])
px = image(4, 4, 3, 65535, smooth=False); px[0][0] = (500, 600, 700)
add("png_rgb16_key.png", png(px, 4, 4, 2, 16, trns=struct.pack(">HHH", 500, 600, 700)), 4, 4, 4, 16, [v for row in px for p in row for v in (*p, 0 if p == (500, 600, 700) else 65535)])
# one filter type for the whole image (each of the five), stored (level 0) and best compression, ancillary chunks
px = image(21, 10, 3, 255)
flat = [s for row in px for p in row for s in p]
for f in range(5): add("png_filter%d.png" % f, png(px, 21, 10, 2, 8, pick=lambda y, f=f: f), 21, 10, 3, 8, flat)
add("png_stored.png", png(px, 21, 10, 2, 8, level=0), 21, 10, 3, 8, flat)
add("png_level9.png", png(px, 21, 10, 2, 8, level=9), 21, 10, 3, 8, flat)
add("png_ancillary.png", png(px, 21, 10, 2, 8, extra=chunk(b"gAMA", struct.pack(">I", 45455)) + chunk(b"tEXt", b"Comment\0made by gen_vision_vectors")), 21, 10, 3, 8, flat)
big = image(160, 100, 3, 255)   # long matches and distances for inflate
add("png_big.png", png(big, 160, 100, 2, 8, level=9), 160, 100, 3, 8, [s for row in big for p in row for s in p])

# ---- PNM ----
px = image(7, 5, 3, 255, smooth=False); s = [v for row in px for p in row for v in p]
add("pnm_p6.ppm", b"P6\n# a comment\n7 5\n255\n" + bytes(s), 7, 5, 3, 8, s)
add("pnm_p3.ppm", ("P3\n7 5 # trailing comment\n255\n" + " ".join(map(str, s)) + "\n").encode(), 7, 5, 3, 8, s)
g = [rnd.randrange(256) for _ in range(35)]
add("pnm_p5.pgm", b"P5 7 5 255\n" + bytes(g), 7, 5, 1, 8, g)
add("pnm_p2.pgm", ("P2\n7 5\n255\n" + "\n".join(map(str, g))).encode(), 7, 5, 1, 8, g)
g16 = [rnd.randrange(65536) for _ in range(35)]
add("pnm_p5_16.pgm", b"P5\n7 5\n65535\n" + b"".join(struct.pack(">H", v) for v in g16), 7, 5, 1, 16, g16)
g15 = [rnd.randrange(16) for _ in range(35)]   # maxval 15 is scaled to 0..255
add("pnm_p5_max15.pgm", b"P5\n7 5\n15\n" + bytes(g15), 7, 5, 1, 8, [(v * 255 + 7) // 15 for v in g15])

# ---- BMP ----
def bmp(w, h, bpp, rows, palette=b"", comp=0, masks=b"", top_down=False, hdr=40, used=0):
    stride = (w * bpp + 31) // 32 * 4
    body = b"".join(r + bytes(stride - len(r)) for r in (rows if top_down else rows[::-1]))
    if hdr == 12: dib = struct.pack("<IHHHH", 12, w, h, 1, bpp)
    else:
        dib = struct.pack("<IiiHHIIiiII", hdr, w, -h if top_down else h, 1, bpp, comp, len(body), 2835, 2835, used, 0)
        if hdr > 40: dib += masks + bytes(hdr - 40 - len(masks)); masks = b""
    off = 14 + len(dib) + len(masks) + len(palette)
    return b"BM" + struct.pack("<IHHI", off + len(body), 0, 0, off) + dib + masks + palette + body
w, h = 11, 6
px = image(w, h, 3, 255, smooth=False); s = [v for row in px for p in row for v in p]
add("bmp_24.bmp", bmp(w, h, 24, [bytes(v for p in row for v in p[::-1]) for row in px]), w, h, 3, 8, s)
add("bmp_24_topdown.bmp", bmp(w, h, 24, [bytes(v for p in row for v in p[::-1]) for row in px], top_down=True), w, h, 3, 8, s)
add("bmp_24_core.bmp", bmp(w, h, 24, [bytes(v for p in row for v in p[::-1]) for row in px], hdr=12), w, h, 3, 8, s)
add("bmp_32_rgb.bmp", bmp(w, h, 32, [bytes(v for p in row for v in (*p[::-1], 9)) for row in px]), w, h, 3, 8, s)
pxa = image(w, h, 4, 255, smooth=False); sa = [v for row in pxa for p in row for v in p]
m32 = struct.pack("<IIII", 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
add("bmp_32_bitfields_v4.bmp", bmp(w, h, 32, [bytes(v for p in row for v in (p[2], p[1], p[0], p[3])) for row in pxa], comp=3, masks=m32, hdr=108), w, h, 4, 8, sa)
m_odd = struct.pack("<III", 0x000003FF, 0x000FFC00, 0x3FF00000)   # 10-bit channels, R lowest
p10 = [[tuple(rnd.randrange(1024) for _ in range(3)) for x in range(w)] for y in range(h)]
add("bmp_32_bitfields_10bit.bmp", bmp(w, h, 32, [b"".join(struct.pack("<I", p[0] | p[1] << 10 | p[2] << 20) for p in row) for row in p10], comp=3, masks=m_odd),
    w, h, 3, 8, [(v * 255 + 511) // 1023 for row in p10 for p in row for v in p])
p5 = [[tuple(rnd.randrange(32) for _ in range(3)) for x in range(w)] for y in range(h)]
add("bmp_16_555.bmp", bmp(w, h, 16, [b"".join(struct.pack("<H", p[0] << 10 | p[1] << 5 | p[2]) for p in row) for row in p5]), w, h, 3, 8, [(v * 255 + 15) // 31 for row in p5 for p in row for v in p])
p565 = [[(rnd.randrange(32), rnd.randrange(64), rnd.randrange(32)) for x in range(w)] for y in range(h)]
add("bmp_16_565.bmp", bmp(w, h, 16, [b"".join(struct.pack("<H", p[0] << 11 | p[1] << 5 | p[2]) for p in row) for row in p565], comp=3, masks=struct.pack("<III", 0xF800, 0x07E0, 0x001F)),
    w, h, 3, 8, [v for row in p565 for p in row for v in ((p[0] * 255 + 15) // 31, (p[1] * 255 + 31) // 63, (p[2] * 255 + 15) // 31)])
for bpp in (1, 4, 8):
    n = min(1 << bpp, 20); pal = [(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(n)]
    idx = [[rnd.randrange(n) for x in range(w)] for y in range(h)]
    rows = []
    for row in idx:
        bits = "".join(format(v, "0%db" % bpp) for v in row); bits += "0" * (-len(bits) % 8)
        rows.append(bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)))
    add("bmp_pal%d.bmp" % bpp, bmp(w, h, bpp, rows, palette=b"".join(bytes((p[2], p[1], p[0], 0)) for p in pal), used=n if n != (1 << bpp) else 0), w, h, 3, 8, [v for row in idx for i in row for v in pal[i]])

os.makedirs(OUT, exist_ok=True)
for name, (data, raw) in sorted(vectors.items()):
    open(os.path.join(OUT, name), "wb").write(data); open(os.path.join(OUT, name + ".raw"), "wb").write(raw)
for kind, pre in [("png", "png_"), ("bmp", "bmp_"), ("pnm", "pnm_")]:
    d = os.path.join(FUZZ, kind); os.makedirs(d, exist_ok=True)
    for name, (data, raw) in sorted(vectors.items()):
        if name.startswith(pre) and len(data) < 4000: open(os.path.join(d, name), "wb").write(data)
# zlib streams for the inflate fuzz target: stored, fixed and dynamic blocks
d = os.path.join(FUZZ, "inflate"); os.makedirs(d, exist_ok=True)
text = (b"the quick brown fox jumps over the lazy dog. " * 20) + bytes(range(256))
for lvl in (0, 1, 6, 9): open(os.path.join(d, "text_l%d.z" % lvl), "wb").write(zlib.compress(text, lvl))
co = zlib.compressobj(9, zlib.DEFLATED, 15, 9, zlib.Z_FIXED); open(os.path.join(d, "fixed.z"), "wb").write(co.compress(text) + co.flush())
open(os.path.join(d, "empty.z"), "wb").write(zlib.compress(b""))
print("wrote %d vectors to %s" % (len(vectors), OUT))

try:
    from PIL import Image
    import io
    bad = 0
    for name, (data, raw) in sorted(vectors.items()):
        w, h, c, depth = struct.unpack("<IIII", raw[:16])
        try: im = Image.open(io.BytesIO(data)); im.load()
        except OSError as e: print("Pillow cannot read", name, "-", e); continue
        if depth == 16:
            if c != 1: continue   # Pillow reduces 16-bit colour to 8 bits
            got = list(im.convert("I").tobytes()); want = [b for s in struct.unpack("<%dH" % (w * h), raw[16:]) for b in struct.pack("<i", s)]
        else:
            got = list(im.convert({1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}[c]).tobytes()); want = list(raw[16:])
        if got != want: bad += 1; print("PILLOW DISAGREES:", name)
    print("Pillow cross-check: %d mismatches" % bad)
except ImportError:
    print("Pillow not installed: cross-check skipped")
