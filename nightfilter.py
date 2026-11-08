# -*- coding: utf-8 -*-
"""Ночной фильтр для Средней Азии (метод C исследования, перенесён из method_multichannel.py с параметрами P_GFS).

Ночью маска Meteosat IODC над сушей Средней Азии видит лишние облака (на стенде: Казахстан ночью 22,7 % ложных).
Многоканальная проверка по одному часу GMGSI (LW 10,7 мкм, SW 3,9 мкм, VIS) и прогнозу GFS (температура поверхности,
осаждённая вода, суша) даёт вероятность облака C. Там, где ночью C <= 0,05, облако снимается.
Действует только в зоне ZONE (Средняя Азия и соседи) и только ночью — в остальных местах стенд выгоды не показал.

Считается на вырезке CROP с запасом ~30° вокруг зоны: фильтры окна (до ~19°) у краёв вырезки не дотягиваются до зоны.
"""
import numpy as np
from scipy.ndimage import percentile_filter, uniform_filter, maximum_filter, minimum_filter, gaussian_filter, median_filter, map_coordinates

THR = 0.05
ZONE_LON = (25.0, 33.0, 92.0, 100.0)     # плавный вход 25…33° в.д., выход 92…100° в.д.
ZONE_LAT = (28.0, 34.0, 60.0, 66.0)
NIGHT_SZ = (90.0, 96.0)                   # зенитный угол Солнца: от 90° фильтр включается, к 96° — полностью
CROP = (-15.0, 130.0, -5.0, 72.5)         # запад, восток, юг, север (окно GFS берётся тем же)
BIAS_PCT, BIAS_WIN = 15, 21
P = {"t0": 8.908, "t0s": 0.299, "t1": 17.815, "pw": 1.019, "bmin": -0.809, "bmax": 0.299, "e0": 8.0, "e0s": 8.0, "e1": 10.0,
     "r0": 6.0, "r0s": 6.0, "r1": 10.0, "R0": 14.134, "R0s": 5.421, "R1": 4.16, "V0": 138.203, "V0s": 60.45, "V1": 148.46,
     "C0": 238.0, "C1": 230.0, "S0": 95.653, "S0s": 12.647, "S1": 13.939, "Sw": 0.495, "N0": 10.067, "N1": 5.0, "Nw": 0.7,
     "gam": 0.792, "ta": 99.0, "ta1": 3.0, "tn": -3.0}


def ramp(x, a, b):
    return np.clip((x - a) / (b - a), 0, 1)


def zone_weight(grid, sz):
    """0..1: ночь и Средняя Азия."""
    a, b, c, d = ZONE_LON; e, f, g, h = ZONE_LAT
    z = ramp(grid.lon, a, b) * ramp(-grid.lon, -d, -c) * ramp(grid.lat, e, f) * ramp(-grid.lat, -h, -g)
    return (z * ramp(sz, *NIGHT_SZ)).astype(np.float32)


def g2k(c):
    c = c.astype(np.float32)
    return np.where(c <= 176, 330 - c / 2, 418 - c)


def _down(a, k=4):
    H, W = a.shape
    return a[:H - H % k, :W - W % k].reshape(H // k, k, W // k, k).mean((1, 3))


def _up(a, shape, k=4):
    out = np.kron(a, np.ones((k, k), np.float32))[:shape[0], :shape[1]]
    if out.shape != tuple(shape):        # остаток строк/столбцов, не кратный 4, — повтором края
        out = np.pad(out, ((0, shape[0] - out.shape[0]), (0, shape[1] - out.shape[1])), mode='edge')
    return out


def gfs_on(lat, lon, gla, glo, fields):
    """Билинейно поля GFS (gla, glo — оси файла, долгота 0…360 или −180…180, возможно с переходом через 0) на точки."""
    lo = (np.asarray(glo, np.float64) + 180) % 360 - 180
    oc = np.argsort(lo); lo = lo[oc]
    la = np.asarray(gla, np.float64); orr = np.argsort(la); la = la[orr]
    r = np.interp(lat, la, np.arange(la.size)); c = np.interp(lon, lo, np.arange(lo.size))
    out = {}
    for k, a in fields.items():
        a = np.asarray(a, np.float32)[orr][:, oc]
        out[k] = map_coordinates(a, [r, c], order=1, mode='nearest').astype(np.float32)
    return out


def _crop_slices(grid):
    w, e, s, n = CROP
    rs = np.where((grid.lat1 >= s) & (grid.lat1 <= n))[0]
    cs = np.where((grid.lon1 >= w) & (grid.lon1 <= e))[0]
    r0, c0 = rs.min() - rs.min() % 4, cs.min() - cs.min() % 4    # блоки 4x4 те же, что на полной сетке
    return slice(r0, rs.max() + 1), slice(c0, cs.max() + 1)


def probability(lw, sw, vis, sz, grid, gfs):
    """Вероятность облака C (0..1) на вырезке CROP. gfs = (lat1d, lon1d, {'tsk','pw','land'}). -> (C, срезы)."""
    rs, cs = _crop_slices(grid)
    ch = {}
    bad = None
    for k, a in (('lw', lw), ('sw', sw), ('vis', vis)):
        a = np.asarray(a).astype(np.float32).copy(); a[:, 0] = a[:, 1]; a = a[rs, cs]
        nb = ~np.isfinite(a)
        if k == 'lw': nb |= a >= 240                     # край диска / заливка GMGSI (см. mosaic.LW_BAD)
        if nb.any():                                     # подставить медиану, а ячейки без данных не снимать (C = 1)
            a[nb] = np.nanmedian(np.where(nb, np.nan, a)) if (~nb).any() else 0.0
            bad = nb if bad is None else (bad | nb)
        ch[k] = a
    lat = grid.lat[rs, cs]; lon = grid.lon[rs, cs]; szc = sz[rs, cs]
    g = gfs_on(lat, lon, *gfs)
    Tl, Ts = g2k(ch['lw']), g2k(ch['sw'])
    mu = np.cos(np.radians(np.minimum(szc, 87.0)))
    f = {'Tl': Tl, 'sz': szc, 'lat': lat}
    dsl = Ts - Tl
    f['dsl'] = dsl
    sd = _down(gaussian_filter(dsl, 1.0))
    f['dslb'] = _up(gaussian_filter(minimum_filter(sd, 9, mode='wrap'), 3, mode='wrap'), dsl.shape)
    f['dslm'] = _up(median_filter(sd, 9, mode='wrap'), dsl.shape)
    visn = ch['vis'] / np.maximum(mu, 0.15)
    vd = _down(visn)
    f['visb'] = _up(gaussian_filter(minimum_filter(vd, 9, mode='wrap'), 3, mode='wrap'), dsl.shape)
    f['visn'] = visn
    m = uniform_filter(Tl, 3, mode='wrap'); f['std3'] = np.sqrt(np.maximum(uniform_filter(Tl * Tl, 3, mode='wrap') - m * m, 0))
    td = _down(Tl)
    f['env'] = _up(gaussian_filter(maximum_filter(td, 15, mode='wrap'), 5, mode='wrap'), Tl.shape) - Tl
    f['land'] = _up(gaussian_filter(_down(g['land']), 1.0, mode='wrap'), Tl.shape)
    dt_ = g['tsk'] - Tl
    b = gaussian_filter(percentile_filter(_down(dt_), BIAS_PCT, size=BIAS_WIN, mode='wrap'), 4, mode='wrap')
    f['dt'] = dt_; f['gbias'] = _up(b, Tl.shape); f['pw'] = g['pw']
    C = _combine(f)
    if bad is not None:
        C[maximum_filter(bad, 9)] = 1.0
    return C, (rs, cs)


def _combine(f):
    sz = f['sz']; L = f['land']
    q = lambda k: L * P[k] + (1 - L) * P[k + 's']
    wd = ramp(-sz, -88, -78)
    wn = ramp(sz, 90, 100)
    warm = ramp(f['Tl'], 235, 250)
    bias = np.clip(f['gbias'], P['bmin'], P['bmax'])
    x = f['dt'] - P['pw'] * f['pw'] / 10.0 - bias
    sT = ramp(x, q('t0') + L * wn * P['tn'], q('t0') + L * wn * P['tn'] + P['t1'])
    sT = np.maximum(sT, ramp(x, P['ta'], P['ta'] + P['ta1']) * ramp(f['env'], q('e0'), q('e0') + P['e1']))
    sR = wd * warm * ramp(f['dsl'] - f['dslb'], q('R0'), q('R0') + P['R1'])
    sV = wd * ramp(f['visn'] - f['visb'], q('V0'), q('V0') + P['V1'])
    sC = ramp(-f['Tl'], -P['C0'], -P['C1'])
    sS = P['Sw'] * ramp(f['std3'], q('S0'), q('S0') + P['S1'])
    sN = P['Nw'] * wn * warm * ramp(np.abs(f['dsl'] - f['dslm']), P['N0'], P['N0'] + P['N1'])
    p = 1 - (1 - sT) * (1 - sR) * (1 - sV) * (1 - sC) * (1 - sS) * (1 - sN)
    edge = 1 - ramp(np.abs(f['lat']), 65, 72.5)
    return (np.clip(p, 0, 1) ** P['gam'] * edge).astype(np.float32)


def veto(lw, sw, vis, sz, grid, gfs):
    """0..1: насколько снять облако в ячейке (ночь, зона, C <= THR). Параметры подобраны на сетке 2048 x 824:
    считать на ней, для 4096 — растянуть (upscale)."""
    z = zone_weight(grid, sz)
    v = np.zeros((grid.H, grid.W), np.float32)
    if not (z > 0).any():
        return v
    C, (rs, cs) = probability(lw, sw, vis, sz, grid, gfs)
    v[rs, cs] = (C <= THR).astype(np.float32)
    return z * v


def upscale(v, shape):
    k = shape[0] // v.shape[0]
    return np.kron(v, np.ones((k, k), np.float32))[:shape[0], :shape[1]]


def apply(a, lw, sw, vis, sz, grid, gfs, v=None):
    """Снимает облака в зоне ночью там, где C <= THR. -> (новая альфа, доля сетки, где фильтр что-то снял)."""
    if v is None:
        v = veto(lw, sw, vis, sz, grid, gfs)
    if v.shape != a.shape:
        v = upscale(v, a.shape)
    out = a * (1 - v)
    return out.astype(np.float32), float(((a > 0.2) & (out <= 0.2)).mean())
