# -*- coding: utf-8 -*-
"""Источники для маски облаков: готовые маски геостационарных спутников и снимки GMGSI.

Всё открыто и без ключей:
  goes18, goes19  NOAA GOES-18/19 ABI-L2-ACMF (s3://noaa-goes18, s3://noaa-goes19), полный диск каждые 10 мин
  himawari        NOAA Himawari-9 AHI-L2-FLDK-Clouds, AHI-CMSK (s3://noaa-himawari9), данные JMA через NOAA Open Data
  iodc, fes       EUMETSAT Meteosat CLM через открытый WMS view.eumetsat.int (msg_iodc:clm, msg_fes:clm), CC-BY-4.0
  gcc             NASA LaRC SatCORPS GCC-GEO v2, cloud_phase (свой алгоритм по всем геостационарам)
  gmgsi           NOAA/NESDIS GMGSI LW/VIS/SW (s3://noaa-gmgsi-pds), для плотности облаков и запасной ИК-формулы
  gfs             NOAA GFS 0,25° (через UCAR THREDDS NCSS), температура поверхности, осаждённая вода, суша — для ночного фильтра

Крупные HDF5-файлы (Himawari 380 МБ, GCC 1,3 ГБ, GOES 25 МБ) не качаются целиком: из файла читается только
оглавление и сжатые куски нужной переменной параллельными HTTP Range-запросами.

Каждая функция fetch_* возвращает словарь {(W, H): массив H x W} для запрошенных сеток или бросает Missing,
если данных за этот час нет (опоздали, не выложены) — тогда сведение обходится без источника.
"""
import io, os, re, sys, time, zlib, threading, datetime as dt
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import h5py

# Контакт в User-Agent: так просят NSF Unidata (THREDDS) и другие открытые серверы — написать, а не заблокировать.
CONTACT = os.environ.get('QST_CONTACT', 'iverrum@gmail.com')
UA = {'User-Agent': 'QSTiUM-edu/1.0 (+https://qstium.com; %s; Earth module cloud mask, education)' % CONTACT}
TIMEOUT = (15, 60)       # соединение, тишина в чтении (с); общий срок источника — в build_mask.py
LW_BAD = 240             # GMGSI LW: 240..255 (≤ 178 K) — край диска IODC / заливка, а не облака (все такие точки южнее 55° ю. ш.)


class Missing(Exception):
    """Данных за этот час у источника нет (ещё не выложены или не будет)."""


class BadRange(IOError):
    """Сервер не выполнил Range-запрос (200 со всем файлом, чужой диапазон, короткое тело)."""


# ---------------------------------------------------------------- HTTP и учёт скачанного
_local = threading.local()
_lock = threading.Lock()
BYTES = {}          # источник -> байт скачано


def session():
    s = getattr(_local, 's', None)
    if s is None:
        s = requests.Session()
        s.headers.update(UA)
        retry = Retry(total=3, connect=3, read=1, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=None, raise_on_status=False)
        ad = HTTPAdapter(pool_connections=8, pool_maxsize=32, max_retries=retry)
        s.mount('https://', ad); s.mount('http://', ad)
        _local.s = s
    return s


def count(src, n):
    with _lock:
        BYTES[src] = BYTES.get(src, 0) + n


def http_get(url, src, headers=None, timeout=TIMEOUT, ok404=False):
    """Тело ответа. С заголовком Range требует 206 и Content-Range с тем же началом и читает не больше
    запрошенного: сервер, отдавший весь файл (GCC — 1,3 ГБ) вместо куска, обрывается сразу."""
    headers = headers or {}
    want = None
    m = re.match(r'bytes=(\d+)-(\d+)$', headers.get('Range', ''))
    if m:
        want = (int(m.group(1)), int(m.group(2)) - int(m.group(1)) + 1)
    r = session().get(url, headers=headers, timeout=timeout, stream=True)
    try:
        if r.status_code == 404 and ok404:
            return None
        r.raise_for_status()
        if want:
            cr = r.headers.get('Content-Range', '')
            if r.status_code != 206 or not cr.startswith('bytes %d-' % want[0]):
                raise BadRange('%s: Range %s -> %d %r' % (src, headers['Range'], r.status_code, cr))
        buf = bytearray()
        for chunk in r.iter_content(1 << 16):
            buf += chunk
            if want and len(buf) > want[1]:
                raise BadRange('%s: тело длиннее запрошенного (%d > %d)' % (src, len(buf), want[1]))
        if want and len(buf) != want[1]:
            raise BadRange('%s: тело короче запрошенного (%d < %d)' % (src, len(buf), want[1]))
        count(src, len(buf))
        return bytes(buf)
    finally:
        r.close()


def s3_list(bucket, prefix):
    """Ключи s3://bucket/prefix* (анонимный листинг по HTTPS)."""
    keys, token = [], None
    while True:
        q = {'list-type': '2', 'prefix': prefix}
        if token: q['continuation-token'] = token
        r = session().get('https://%s.s3.amazonaws.com/' % bucket, params=q, timeout=60)
        r.raise_for_status()
        x = r.text
        keys += re.findall(r'<Key>(.*?)</Key>', x)
        m = re.search(r'<NextContinuationToken>(.*?)</NextContinuationToken>', x)
        if not m: return keys
        token = m.group(1)


# ---------------------------------------------------------------- чтение HDF5 по кускам через HTTP Range
class RangeFile(io.RawIOBase):
    """Файл по HTTP для h5py: читает блоками по Range-запросу, держит последние блоки в памяти."""

    def __init__(self, url, src, block=256 * 1024, keep=512):
        super().__init__()
        self.url, self.src, self.block, self.keep = url, src, block, keep
        r = session().head(url, allow_redirects=True, timeout=60)
        if r.status_code == 404: raise Missing(url)
        r.raise_for_status()
        self.size = int(r.headers['Content-Length'])
        self.pos = 0
        self.cache = OrderedDict()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _blk(self, i):
        b = self.cache.get(i)
        if b is not None:
            self.cache.move_to_end(i); return b
        a = i * self.block; z = min(a + self.block, self.size) - 1
        b = http_get(self.url, self.src, {'Range': 'bytes=%d-%d' % (a, z)})    # длина проверена: ровно z - a + 1
        self.cache[i] = b
        if len(self.cache) > self.keep: self.cache.popitem(last=False)
        return b

    def readinto(self, buf):
        mv = memoryview(buf).cast('B')
        n = min(len(mv), max(self.size - self.pos, 0))
        got = 0
        while got < n:
            p = self.pos + got; i = p // self.block; b = self._blk(i); o = p - i * self.block
            k = min(len(b) - o, n - got)
            if k <= 0:                       # блок короче смещения — иначе вечный цикл
                self.cache.pop(i, None)
                raise BadRange('%s: блок %d короче ожидаемого (%d байт)' % (self.src, i, len(b)))
            mv[got:got + k] = b[o:o + k]; got += k
        self.pos += n
        return n


def h5_open(url, src):
    rf = RangeFile(url, src)
    return h5py.File(rf, 'r'), rf


def h5_read(h, rf, name, threads=16, gap=256 * 1024, maxgroup=8 * 2 ** 20):
    """Переменная целиком, но сжатые куски — параллельными Range-запросами сгруппированных соседей.
    Поддержаны фильтры deflate / shuffle / fletcher32; иначе — обычное чтение h5py."""
    d = h[name]
    if d.chunks is None:
        return d[()]
    pl = d.id.get_create_plist()
    filters = [pl.get_filter(i)[0] for i in range(pl.get_nfilters())]
    if any(f not in (1, 2, 3) for f in filters):
        return d[()]
    infos = []
    if hasattr(d.id, 'chunk_iter'):
        d.id.chunk_iter(lambda ci: infos.append((ci.byte_offset, ci.size, ci.chunk_offset, ci.filter_mask)))
    else:
        for i in range(d.id.get_num_chunks()):
            ci = d.id.get_chunk_info(i)
            infos.append((ci.byte_offset, ci.size, ci.chunk_offset, ci.filter_mask))
    infos.sort()
    groups = []
    for it in infos:
        if groups and it[0] - groups[-1][1] < gap and it[0] + it[1] - groups[-1][0] < maxgroup:
            groups[-1][1] = it[0] + it[1]; groups[-1][2].append(it)
        else:
            groups.append([it[0], it[0] + it[1], [it]])
    out = np.full(d.shape, d.fillvalue, d.dtype)
    isz = d.dtype.itemsize

    def get(g):
        return g, http_get(rf.url, rf.src, {'Range': 'bytes=%d-%d' % (g[0], g[1] - 1)})

    def decode(raw, mask):
        for j in range(len(filters) - 1, -1, -1):
            if mask & (1 << j): continue
            f = filters[j]
            if f == 1: raw = zlib.decompress(raw)
            elif f == 2 and isz > 1: raw = np.frombuffer(raw, np.uint8).reshape(isz, -1).T.tobytes()
            elif f == 3: raw = raw[:-4]
        return np.frombuffer(raw, d.dtype).reshape(d.chunks)

    with ThreadPoolExecutor(threads) as ex:
        for g, b in ex.map(get, groups):
            for off, size, co, mask in g[2]:
                a = decode(b[off - g[0]:off - g[0] + size], mask)
                sl = tuple(slice(o, min(o + c, s)) for o, c, s in zip(co, d.chunks, d.shape))
                out[sl] = a[tuple(slice(0, x.stop - x.start) for x in sl)]
    return out


def _attr(h, k, default=None):
    v = h.attrs.get(k, default)
    if isinstance(v, bytes): v = v.decode()
    if isinstance(v, np.ndarray) and v.size == 1: v = v.item()
    return v


# ---------------------------------------------------------------- перевод точек снимка в доли облачности на сетках
class Binner:
    """Копит по ячейкам сетки число годных точек и число облачных; доля = облачные / годные при n >= nmin."""

    def __init__(self, grids, nmin2048=4):
        self.g = list(grids)
        self.n = {g: np.zeros(g[0] * g[1]) for g in self.g}
        self.c = {g: np.zeros(g[0] * g[1]) for g in self.g}
        self.nmin = {g: max(1, round(nmin2048 * (2048.0 / g[0]) ** 2)) for g in self.g}

    def add(self, lat, lon, cloudy, valid):
        lat = np.asarray(lat).ravel(); lon = np.asarray(lon).ravel()
        ok = np.asarray(valid).ravel() & np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) < 72.5) & (np.abs(lon) <= 180)
        la = lat[ok]; lo = lon[ok]; cl = np.asarray(cloudy).ravel()[ok].astype(np.float64)
        for (W, H) in self.g:
            ri = np.clip(np.floor((72.5 - la) / 145.0 * H).astype(np.int64), 0, H - 1)
            ci = np.floor((lo + 180.0) / 360.0 * W).astype(np.int64) % W
            idx = ri * W + ci
            self.n[(W, H)] += np.bincount(idx, minlength=W * H)
            self.c[(W, H)] += np.bincount(idx, weights=cl, minlength=W * H)

    def result(self):
        out = {}
        for (W, H) in self.g:
            n = self.n[(W, H)].reshape(H, W)
            with np.errstate(invalid='ignore', divide='ignore'):
                f = np.where(n >= self.nmin[(W, H)], self.c[(W, H)].reshape(H, W) / n, np.nan).astype(np.float32)
            # Шов 180°: сетка источника может не доходить до ±180 (GCC на 4096 — пустые первый и последний столбцы,
            # там, где спутники уже не голосуют, это рисовало тёмную полосу ИК-формулой). Берём соседа через шов.
            for c, nbs in ((0, (1, W - 1)), (W - 1, (W - 2, 0))):
                for nb in nbs:
                    e = ~np.isfinite(f[:, c])
                    if not e.any(): break
                    f[e, c] = f[e, nb]
            out[(W, H)] = f
        return out


def _geos_fixed_grid(x, y, req, rpol, Hs, lon0):
    """GOES-R ABI: углы сканирования x (по строке), y (по столбцу), рад -> широта, долгота."""
    X = x[None, :]; Y = y[:, None]
    a = np.sin(X) ** 2 + np.cos(X) ** 2 * (np.cos(Y) ** 2 + (req ** 2 / rpol ** 2) * np.sin(Y) ** 2)
    b = -2 * Hs * np.cos(X) * np.cos(Y)
    c = Hs ** 2 - req ** 2
    with np.errstate(invalid='ignore'):
        rs = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
        sx = rs * np.cos(X) * np.cos(Y); sy = -rs * np.sin(X); sz = rs * np.cos(X) * np.sin(Y)
        lat = np.degrees(np.arctan((req ** 2 / rpol ** 2) * sz / np.sqrt((Hs - sx) ** 2 + sy ** 2)))
        lon = lon0 - np.degrees(np.arctan(sy / (Hs - sx)))
    return lat, (lon + 180) % 360 - 180


def _cgms_grid(cols, rows, step, off, lon0, H=42164.0, req=6378.137, rpol=6356.7523):
    """JMA / CGMS (Himawari): номер столбца и строки -> широта, долгота. Совпадает с полями Latitude/Longitude файла."""
    e = req ** 2 / rpol ** 2
    x = ((cols - off) * step)[None, :]; y = ((rows - off) * step)[:, None]
    cxy = np.cos(x) * np.cos(y)
    q = np.cos(y) ** 2 + e * np.sin(y) ** 2
    with np.errstate(invalid='ignore'):
        sd = np.sqrt((H * cxy) ** 2 - q * (H ** 2 - req ** 2))
        sn = (H * cxy - sd) / q
        s1 = H - sn * cxy; s2 = sn * np.sin(x) * np.cos(y); s3 = -sn * np.sin(y)
        lon = np.degrees(np.arctan(s2 / s1)) + lon0
        lat = np.degrees(np.arctan(e * s3 / np.hypot(s1, s2)))
    return lat, (lon + 180) % 360 - 180


# ---------------------------------------------------------------- GOES-18 / GOES-19 ACM
def fetch_goes(sat, t, grids):
    """sat: 'goes18' | 'goes19'. Полный диск, скан, начатый в HH:00 (или ближайший в пределах часа)."""
    doy = t.timetuple().tm_yday
    bucket = 'noaa-' + sat
    keys = [k for k in s3_list(bucket, 'ABI-L2-ACMF/%04d/%03d/%02d/' % (t.year, doy, t.hour)) if k.endswith('.nc')]
    if not keys: raise Missing('%s %s' % (sat, t))
    want = '_s%04d%03d%02d00' % (t.year, doy, t.hour)
    k = next((k for k in keys if want in k), sorted(keys)[0])
    h, rf = h5_open('https://%s.s3.amazonaws.com/%s' % (bucket, k), sat)
    pr = h['goes_imager_projection'].attrs
    req, rpol = float(pr['semi_major_axis'][0]), float(pr['semi_minor_axis'][0])
    Hs = float(pr['perspective_point_height'][0]) + req
    lon0 = float(pr['longitude_of_projection_origin'][0])

    def coord(n):
        v = h[n]; a = v[()].astype(np.float64)
        return a * float(v.attrs.get('scale_factor', [1])[0]) + float(v.attrs.get('add_offset', [0])[0])
    x, y = coord('x'), coord('y')
    acm = h5_read(h, rf, 'ACM').view(np.uint8).astype(np.int16)
    dqf = h5_read(h, rf, 'DQF').view(np.uint8).astype(np.int16)
    t0 = _attr(h, 'time_coverage_start')
    h.close()
    acm[(dqf != 0) & (dqf != 6)] = -1
    acm[acm > 3] = -1
    b = Binner(grids)
    step = 400
    for r0 in range(0, y.size, step):
        lat, lon = _geos_fixed_grid(x, y[r0:r0 + step], req, rpol, Hs, lon0)
        cl = acm[r0:r0 + step]
        b.add(lat, lon, cl >= 2, cl >= 0)
    return b.result(), {'файл': os.path.basename(k), 'начало': t0, 'долгота': lon0}


# ---------------------------------------------------------------- Himawari-9 CMSK
HIM_STEP = np.radians(2 ** 16 / 20466275.0)   # 2 км, CFAC = 20466275


def fetch_himawari(t, grids):
    pre = 'AHI-L2-FLDK-Clouds/%04d/%02d/%02d/%02d00/' % (t.year, t.month, t.day, t.hour)
    keys = [k for k in s3_list('noaa-himawari9', pre) if '/AHI-CMSK_' in k]
    if not keys: raise Missing('himawari %s' % t)
    h, rf = h5_open('https://noaa-himawari9.s3.amazonaws.com/' + keys[0], 'himawari')
    cm = h5_read(h, rf, 'CloudMask').astype(np.int16)
    t0 = _attr(h, 'time_coverage_start')
    h.close()
    n = cm.shape[1]
    off = (n - 1) / 2.0
    cols = np.arange(n, dtype=np.float64)
    b = Binner(grids)
    step = 500
    for r0 in range(0, cm.shape[0], step):
        rows = np.arange(r0, min(r0 + step, cm.shape[0]), dtype=np.float64)
        lat, lon = _cgms_grid(cols, rows, HIM_STEP, off, 140.7)
        cl = cm[r0:r0 + step]
        b.add(lat, lon, cl >= 2, (cl >= 0) & (cl <= 3))
    return b.result(), {'файл': os.path.basename(keys[0]), 'начало': t0, 'долгота': 140.7}


# ---------------------------------------------------------------- EUMETSAT CLM (Meteosat IODC 45,5° в.д. и 0°)
CLM_SPAN = {'iodc': ('msg_iodc', -35.5, 118.5, 45.5), 'fes': ('msg_fes', -77.0, 77.0, 0.0)}
CLM_PX = 8192   # ширина картинки WMS на 360°: по 4 точки на ячейку сетки 2048


def fetch_clm(which, t, grids):
    layer, lo0, lo1, sub = CLM_SPAN[which]
    c0 = int(np.floor((lo0 + 180) / 360 * CLM_PX)); c1 = int(np.ceil((lo1 + 180) / 360 * CLM_PX))
    c0 -= c0 % 4; c1 += (-c1) % 4
    L0 = -180 + c0 * 360 / CLM_PX; L1 = -180 + c1 * 360 / CLM_PX
    HP = 824 * 4                                           # 4 точки на ячейку 2048 x 824 (2 — на 4096 x 1648)
    # Плитки 8 x 2 (~412 x 1750 точек): большую полосу WMS иногда рисует дольше минуты и рвёт соединение,
    # маленькие — 3–6 с; края плиток лежат на границах точек, так что картинка та же, что одним куском.
    nr, nc = 8, 2
    rows = HP // nr
    cm = c0 + ((c1 - c0) // nc) // 4 * 4
    cols = [(c0, cm), (cm, c1)]
    iso = t.strftime('%Y-%m-%dT%H:%M:00Z')
    from PIL import Image

    def tile(ij):
        bi, cj = ij
        la1 = 72.5 - bi * rows * 145 / HP; la0 = 72.5 - (bi + 1) * rows * 145 / HP
        a, b_ = cols[cj]
        u = ('https://view.eumetsat.int/geoserver/ows?service=WMS&version=1.3.0&request=GetMap&layers=%s:clm&styles='
             '&crs=CRS:84&bbox=%.6f,%.6f,%.6f,%.6f&width=%d&height=%d&format=image/png&time=%s') % (
            layer, -180 + a * 360 / CLM_PX, la0, -180 + b_ * 360 / CLM_PX, la1, b_ - a, rows, iso)
        for attempt in range(3):
            try:
                b = http_get(u, which, timeout=(15, 45))
                break
            except requests.RequestException:
                if attempt == 2: raise
                time.sleep(2 + 3 * attempt)
        if not b.startswith(b'\x89PNG'): raise Missing('%s %s: %s' % (which, iso, b[:200]))
        return np.array(Image.open(io.BytesIO(b)).convert('RGBA')).astype(np.int16)

    with ThreadPoolExecutor(4) as ex:
        parts = list(ex.map(tile, [(i, j) for i in range(nr) for j in range(nc)]))
    im = np.concatenate([np.concatenate(parts[i * nc:(i + 1) * nc], 1) for i in range(nr)], 0)
    R, G, B, A = im[..., 0], im[..., 1], im[..., 2], im[..., 3]
    cloud = (R > 200) & (G > 200) & (B > 200) & (A > 0)
    clear = (A > 0) & ~cloud & ((G > 100) | (B > 100))
    valid = cloud | clear
    if valid.mean() < 0.05: raise Missing('%s %s: пустой кадр' % (which, iso))
    out = {}
    for (W, H) in grids:
        s = CLM_PX // W                                   # точек картинки на ячейку по каждой оси
        a0, a1 = c0 // s, c1 // s
        def binsum(x):
            return x.reshape(H, s, a1 - a0, s).sum((1, 3))
        n = binsum(valid.astype(np.int32)); c = binsum(cloud.astype(np.int32))
        f = np.full((H, W), np.nan, np.float32)
        with np.errstate(invalid='ignore', divide='ignore'):
            f[:, a0:a1] = np.where(n >= s * s / 2, c / np.maximum(n, 1), np.nan)
        out[(W, H)] = f
    return out, {'слой': layer + ':clm', 'время': iso, 'долгота': sub}


# ---------------------------------------------------------------- NASA LaRC SatCORPS GCC v2
def fetch_gcc(t, grids):
    doy = t.timetuple().tm_yday
    url = ('https://satcorps.larc.nasa.gov/prod/GCC-GEO/visst-pixel-netcdf-v2/%04d/%02d/%02d/'
           'satcorps-gcc.v02.geo.glob-comp.%04d%03d.%02d00.3km.nc') % (t.year, t.month, t.day, t.year, doy, t.hour)
    h, rf = h5_open(url, 'gcc')
    lat = h5_read(h, rf, 'lat').astype(np.float64)
    lon = h5_read(h, rf, 'lon').astype(np.float64)
    ph = h5_read(h, rf, 'cloud_phase')
    h.close()
    if not (np.isfinite(lat).all() and np.abs(lat).max() <= 90 and np.isfinite(lon).all() and np.abs(lon).max() <= 360):
        raise IOError('gcc: координаты вне Земли (lat %.3g..%.3g) — битое чтение' % (np.nanmin(lat), np.nanmax(lat)))
    ph = ph[0] if ph.ndim == 3 else ph
    rows = np.where(np.abs(lat) < 72.5)[0]
    b = Binner(grids)
    lon = (lon + 180) % 360 - 180
    for r0 in range(rows.min(), rows.max() + 1, 500):
        rr = np.arange(r0, min(r0 + 500, rows.max() + 1))
        p = ph[rr]
        LA = np.broadcast_to(lat[rr][:, None], p.shape); LO = np.broadcast_to(lon[None, :], p.shape)
        b.add(LA, LO, np.isin(p, [1, 2, 6, 7]), p != 127)
    return b.result(), {'файл': os.path.basename(url)}


# ---------------------------------------------------------------- GMGSI (NOAA/NESDIS), каналы LW / VIS / SW
GMGSI = {'lw': 'GMGSI_LW', 'vis': 'GMGSI_VIS', 'sw': 'GMGSI_SW', 'wv': 'GMGSI_WV'}


def gmgsi_hours(back_hours=30, now=None):
    """Часы (datetime UTC), за которые на S3 уже есть кадр GMGSI LW, по возрастанию."""
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    out = set()
    days = sorted({(now - dt.timedelta(hours=h)).date() for h in range(0, back_hours + 1, 6)} | {now.date()})
    for d in days:
        for k in s3_list('noaa-gmgsi-pds', 'GMGSI_LW/%04d/%02d/%02d/' % (d.year, d.month, d.day)):
            m = re.match(r'GMGSI_LW/(\d{4})/(\d\d)/(\d\d)/(\d\d)/.*\.nc$', k)
            if m: out.add(dt.datetime(*map(int, m.groups())))
    lim = now - dt.timedelta(hours=back_hours)
    return sorted(h for h in out if h >= lim)


def fetch_gmgsi(ch, t, grids):
    """float32 H x W: среднее по годным точкам Меркатора 4999 x 3000, попавшим в ячейку (шкала 0..255 как в файле).
    NaN — в ячейке нет ни одной годной точки (у LW это край диска IODC у Антарктиды: значения >= LW_BAD).
    Ячейка, куда не попала ни одна точка файла (зазор сеток), берёт соседа по строке."""
    keys = [k for k in s3_list('noaa-gmgsi-pds', '%s/%04d/%02d/%02d/%02d/' % (GMGSI[ch], t.year, t.month, t.day, t.hour)) if k.endswith('.nc')]
    if not keys: raise Missing('gmgsi %s %s' % (ch, t))
    b = http_get('https://noaa-gmgsi-pds.s3.amazonaws.com/' + sorted(keys)[-1], 'gmgsi_' + ch, timeout=(15, 60))
    with h5py.File(io.BytesIO(b), 'r') as h:
        lat = h['lat'][:, 0].astype(np.float64)
        lon = h['lon'][0, :].astype(np.float64)
        dv = h['data']
        v = dv[0].astype(np.float32)
        fill = dv.attrs.get('_FillValue')
        if fill is not None: v[v == np.asarray(fill).astype(np.float32).ravel()[0]] = np.nan
        vr = dv.attrs.get('valid_range')
        if vr is not None: v[(v < vr[0]) | (v > vr[1])] = np.nan
        if 'scale_factor' in dv.attrs or 'add_offset' in dv.attrs:
            v = v * float(np.ravel(dv.attrs.get('scale_factor', [1]))[0]) + float(np.ravel(dv.attrs.get('add_offset', [0]))[0])
    good = np.isfinite(v)
    if ch == 'lw':
        good &= v < LW_BAD
    if good.mean() < 0.5:                               # файл из одной заливки — не подставлять «тёплые» нули
        raise Missing('gmgsi %s %s: годных точек %.0f %%' % (ch, t, 100 * good.mean()))
    vz = np.where(good, v, 0).astype(np.float64)
    # Столбец 0 файла — долгота 179,9996 (сразу за последним 179,90): кладём его в последнюю ячейку, а не выбрасываем,
    # иначе на 4096 последний столбец пуст.
    L = (lon + 180.0) % 360.0
    out = {}
    for (W, H) in grids:
        ci = np.floor(L / 360.0 * W) % W; ri = np.floor((72.5 - lat) / 145.0 * H)
        okc = np.isfinite(ci); okr = (ri >= 0) & (ri < H)
        idx = (ri[okr].astype(np.int64)[:, None] * W + ci[okc].astype(np.int64)[None, :]).ravel()
        s = np.bincount(idx, weights=vz[np.ix_(okr, okc)].ravel(), minlength=W * H).reshape(H, W)
        n = np.bincount(idx, weights=good[np.ix_(okr, okc)].ravel().astype(np.float64), minlength=W * H).reshape(H, W)
        nall = np.bincount(idx, minlength=W * H).reshape(H, W)
        with np.errstate(invalid='ignore', divide='ignore'):
            m = np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)
        hole = nall == 0
        for d in (1, -1, 2, -2, 3, -3):                 # зазор сеток — соседом по строке (через 180° тоже)
            if not hole.any(): break
            nb = np.roll(m, d, 1); nh = np.roll(hole, d, 1)
            put = hole & ~nh
            m[put] = nb[put]; hole &= ~put
        out[(W, H)] = m
    return out, {'файл': os.path.basename(sorted(keys)[-1])}


# ---------------------------------------------------------------- есть ли уже данные источника за час (дёшево)
def available(src, t):
    """True, если данные источника за час t уже выложены. Ошибка сети — False (проверим в следующий раз)."""
    try:
        doy = t.timetuple().tm_yday
        if src in ('goes18', 'goes19'):
            want = '_s%04d%03d%02d00' % (t.year, doy, t.hour)
            return any(want in k for k in s3_list('noaa-' + src, 'ABI-L2-ACMF/%04d/%03d/%02d/' % (t.year, doy, t.hour)))
        if src == 'himawari':
            pre = 'AHI-L2-FLDK-Clouds/%04d/%02d/%02d/%02d00/' % (t.year, t.month, t.day, t.hour)
            return any('/AHI-CMSK_' in k for k in s3_list('noaa-himawari9', pre))
        if src == 'gcc':
            url = ('https://satcorps.larc.nasa.gov/prod/GCC-GEO/visst-pixel-netcdf-v2/%04d/%02d/%02d/'
                   'satcorps-gcc.v02.geo.glob-comp.%04d%03d.%02d00.3km.nc') % (t.year, t.month, t.day, t.year, doy, t.hour)
            return session().head(url, allow_redirects=True, timeout=(10, 30)).status_code == 200
        if src in CLM_SPAN:
            layer, lo0, lo1, _ = CLM_SPAN[src]
            u = ('https://view.eumetsat.int/geoserver/ows?service=WMS&version=1.3.0&request=GetMap&layers=%s:clm&styles='
                 '&crs=CRS:84&bbox=%g,-60,%g,60&width=64&height=32&format=image/png&time=%s') % (
                layer, lo0 + 20, lo1 - 20, t.strftime('%Y-%m-%dT%H:%M:00Z'))
            b = session().get(u, timeout=(10, 30)).content
            if not b.startswith(b'\x89PNG'): return False
            from PIL import Image
            return np.array(Image.open(io.BytesIO(b)).convert('RGBA'))[..., 3].mean() > 10
    except Exception:
        return False
    return False


# ---------------------------------------------------------------- GFS 0,25° для ночного фильтра
GFS_VARS = (('tsk', 'Temperature_surface'), ('pw', 'Precipitable_water_entire_atmosphere_single_layer'),
            ('land', 'Land_cover_0__sea_1__land_surface'))
GFS_NCSS = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p25deg/Best'


def fetch_gfs(t, bbox=(-15.0, 130.0, -5.0, 73.0)):
    """Краткосрочный прогноз GFS на час t, только окно bbox = (запад, восток, юг, север). -> (lat1d, lon1d, {ключ: 2D})."""
    w, e, s, n = bbox
    q = '&'.join('var=' + v for _, v in GFS_VARS)
    u = '%s?%s&north=%g&south=%g&west=%g&east=%g&time=%s&accept=netcdf4' % (GFS_NCSS, q, n, s, w, e, t.strftime('%Y-%m-%dT%H:00:00Z'))
    b = http_get(u, 'gfs', timeout=(15, 90))
    if not b.startswith(b'\x89HDF'): raise Missing('gfs: ' + b[:200].decode('latin-1'))
    return read_gfs(io.BytesIO(b))


def read_gfs(f):
    with h5py.File(f, 'r') as h:
        la = h['latitude'][:].astype(np.float64); lo = h['longitude'][:].astype(np.float64)
        out = {}
        for k, v in GFS_VARS:
            a = h[v][()].astype(np.float32)
            out[k] = a.reshape(a.shape[-2], a.shape[-1])
    return la, lo, out
