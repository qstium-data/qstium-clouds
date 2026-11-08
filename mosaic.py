# -*- coding: utf-8 -*-
"""Сведение масок в одну карту непрозрачности облаков (метод B исследования, перенесён из method_pro_masks.py).

1. Доля облачности ячейки — взвешенное голосование всех масок, которые видят ячейку:
   вес спутника = clip((72° − угол обзора) / 15°), так швов между спутниками нет;
   IODC затухает за 6° до краёв своего слоя WMS (118,5° в.д. и 35,5° з.д.), 0° — до ±77°;
   GCC (другой алгоритм по всем геостационарам) голосует всюду с весом 0,5.
   Где нет ни одной маски — нынешняя ИК-формула сайта (по строке, из GMGSI LW).
2. Покрытие cov = clip((доля − 0,5) / 0,3).
3. Плотность s = max(насколько ячейка холоднее ясного фона по ИК, насколько ярче ясного фона в видимом днём);
   ясный фон — среднее по ясным по маске ячейкам в гауссовом окне ~±4°; где ясных мало — плавно к фону широты
   (фон широты — 70-й перцентиль ясных ячеек строки, только по строкам с ≥ 20 ясными, медиана + гаусс по строкам).
   LW ≥ 240 (≤ 178 K) — край диска IODC / заливка GMGSI, а не облако: такие ячейки считаются без данных.
4. Альфа = cov · (0,45 + 0,55 · s^0,6). Где вес масок мал (< 0,1 — самый край дисков без GCC), альфа плавно переходит
   в ИК-формулу, а не скачком. К полюсам (65…70°) всё гаснет до нуля: дальше GCC нет, спутники не видят.
"""
import datetime as dt
import numpy as np

VZA_MAX = 72.0          # угол обзора, после которого маске не верим
VZA_FADE = 15.0
GCC_W = 0.5             # вес GCC в голосовании
F0, F1 = 0.5, 0.8       # доля облачности: начало и полная непрозрачность
FLOOR = 0.45            # минимальная непрозрачность облачной ячейки
EDGE_FADE = {'iodc': (-35.5, 118.5), 'fes': (-77.0, 77.0)}   # края слоёв WMS EUMETSAT
SATS = ('goes18', 'goes19', 'himawari', 'iodc', 'fes')
CODES = {'ik': 0, 'goes18': 1, 'goes19': 2, 'himawari': 3, 'iodc': 4, 'gcc': 5, 'fes': 6}
W_FULL = 0.1            # суммарный вес масок, с которого ИК-формула уже не подмешивается (≈ последние 1,5° угла обзора)
POLE = (65.0, 70.0)     # затухание к полюсам: GCC кончается на ~70°, спутники там уже не голосуют
LW_BAD = 240            # GMGSI LW ≥ 240 (≤ 178 K) — нет данных (край диска IODC у Антарктиды), см. sources.LW_BAD


class Grid:
    """Равнопромежуточная сетка CRS:84, долгота −180…180, широта 72,5…−72,5, строка 0 — север, центры ячеек."""

    def __init__(self, W=2048, H=824):
        self.W, self.H = W, H
        self.key = (W, H)
        self.k = W / 2048.0
        self.lat1 = (72.5 - (np.arange(H) + 0.5) * 145.0 / H).astype(np.float32)
        self.lon1 = (-180 + (np.arange(W) + 0.5) * 360.0 / W).astype(np.float32)
        self.lat = np.broadcast_to(self.lat1[:, None], (H, W)).astype(np.float32)
        self.lon = np.broadcast_to(self.lon1[None, :], (H, W)).astype(np.float32)


def sun_zenith(t, grid):
    """Зенитный угол Солнца (°), приближение NOAA."""
    lat, lon = grid.lat1, grid.lon1
    doy = t.timetuple().tm_yday
    hr = t.hour + t.minute / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hr - 12) / 24.0)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    ha = np.radians((hr * 60 + eqt + 4 * lon[None, :]) / 4.0 - 180)
    la = np.radians(lat[:, None])
    c = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return np.degrees(np.arccos(np.clip(c, -1, 1))).astype(np.float32)


def geo_vza(grid, sublon):
    """Угол обзора геостационарного спутника над долготой sublon (°); за краем диска 90."""
    la = np.radians(grid.lat); dl = np.radians(grid.lon - sublon)
    cg = np.cos(la) * np.cos(dl)
    g = np.arccos(np.clip(cg, -1, 1))
    r = 6378.137 / 42164.16
    v = np.degrees(np.arctan2(np.sin(g), cg - r))
    return np.where(cg > r, v, 90.0).astype(np.float32)


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def clean_lw(lw):
    """LW (0..255, uint8 или float с NaN) -> float32, NaN там, где данных нет (заливка / край диска, >= LW_BAD)."""
    v = np.asarray(lw).astype(np.float32).copy()
    v[~(v < LW_BAD)] = np.nan
    return v


def ir_formula(lw, grid, edge=True):
    """Нынешняя формула сайта (earth-clouds.js): только ИК, фон по строке. Запасной вариант, где масок нет.
    Ячейки без данных LW — 0. edge=False — без своего затухания у полюсов (его даёт alpha())."""
    v = clean_lw(lw)              # столбец 0 — настоящий (sources.fetch_gmgsi кладёт долготу 180° в шов как надо)
    s = np.sort(v, axis=1)                                      # NaN уходят в конец строки
    nf = np.isfinite(v).sum(1)
    k = np.clip(np.ceil(0.35 * nf).astype(int) - 1, 0, grid.W - 1)   # без NaN — тот же 35-й перцентиль, что на сайте
    bg = np.where(nf > 0, s[np.arange(grid.H), k], 80.0)
    bg = np.maximum(bg, 80)[:, None]
    diff = np.clip((v - bg - 2) / 40, 0, 1)
    out = np.nan_to_num(np.maximum(diff ** 0.7, _smooth(165, 228, v)), nan=0.0)
    if edge:
        out = out * (1 - _smooth(65, 72.5, np.abs(grid.lat)))
    return out


def vote(layers, gcc, grid, with_weight=False):
    """layers: [(имя, доля H x W с NaN, угол обзора H x W)], gcc: доля или None.
    -> (доля с NaN там, где масок нет;  код главного источника по ячейке;  {имя: доля сетки, где он главный})
    with_weight=True — четвёртым ещё суммарный вес масок по ячейке (для плавного перехода к ИК-формуле в alpha)."""
    shape = (grid.H, grid.W)
    num = np.zeros(shape, np.float32); den = np.zeros(shape, np.float32)
    best = np.full(shape, 999.0, np.float32); src = np.zeros(shape, np.int8)
    for k, f, v in layers:
        v = np.nan_to_num(v, nan=999.0)
        w = np.clip((VZA_MAX - v) / VZA_FADE, 0, 1) * np.isfinite(f)
        if k in EDGE_FADE:
            a, b = EDGE_FADE[k]
            w = w * np.clip((b - grid.lon) / 6.0, 0, 1) * np.clip((grid.lon - a) / 6.0, 0, 1)
        num += w * np.nan_to_num(f); den += w
        t = (w > 0) & (v < best); best[t] = v[t]; src[t] = CODES[k]
    if gcc is not None:
        w = np.isfinite(gcc) * GCC_W
        num += w * np.nan_to_num(gcc); den += w; src[(src == 0) & (w > 0)] = CODES['gcc']
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = np.where(den > 0, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)
    names = {v: k for k, v in CODES.items()}
    u, n = np.unique(src, return_counts=True)
    share = {names[int(c)]: float(m) / src.size for c, m in zip(u, n)}   # где источник главный (ik — ни одной маски)
    if with_weight:
        return frac, src, share, den
    return frac, src, share


def _local_sum(x, w, r):
    """Взвешенные суммы x*w и w в окрестности ~(2r+1)^2 ячеек, гауссово окно той же ширины, что квадрат ±r
    (sigma = r/sqrt(3)). Квадратное окно давало ступеньки там, где ясных ячеек единицы.
    По долготе окно замкнуто (wrap), по широте — край повторяется. Возвращает суммы в «ячейках»."""
    from scipy.ndimage import gaussian_filter
    n = (2 * r + 1) ** 2
    g = lambda a: gaussian_filter(a.astype(np.float32), r / np.sqrt(3.0), mode=('nearest', 'wrap'), truncate=3.0) * n
    return g(x * w), g(w)


def lw_kelvin(v):
    v = np.asarray(v).astype(np.float32)
    return np.where(v <= 176, 330 - v / 2, 418 - v)


def row_background(tk, clear, grid):
    """Фон широты: 70-й перцентиль ясных ячеек строки (тёплое ясное небо). Строки, где ясных < 20 (на 2048),
    не в счёт — их заполняет интерполяция соседних; затем медиана и гаусс по строкам (~±1°).
    Прямоугольное окно по 13 строкам превращало один выброс в две ступеньки по всей широте."""
    cnt = clear.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        rowp = np.nanpercentile(np.where(clear, tk, np.nan), 70, axis=1)
    rowp[cnt < 20 * grid.k] = np.nan
    ok = np.isfinite(rowp)
    if not ok.any():
        f = tk[np.isfinite(tk)]
        return np.full(grid.H, float(np.percentile(f, 70)) if f.size else 270.0, np.float32)
    j = np.arange(grid.H)
    rowp = np.interp(j, j[ok], rowp[ok])                 # за краями — ближайшая годная строка
    from scipy.ndimage import median_filter, gaussian_filter1d
    k = 2 * int(round(6 * grid.k)) + 1
    rowp = median_filter(rowp, size=k, mode='nearest')
    return gaussian_filter1d(rowp, k / 4.0, mode='nearest').astype(np.float32)


def opacity(lw, vis, sz, frac, grid):
    """0..1 «сила» облака: холоднее ясного фона по ИК или ярче его в видимом днём. Без данных LW — 0 по ИК."""
    lw = clean_lw(lw)
    tk = lw_kelvin(lw)                                 # NaN там, где LW нет
    okt = np.isfinite(tk)
    clear = (np.nan_to_num(frac, nan=1.0) < 0.05) & okt
    r = int(round(24 * grid.k))                       # окно ~±4°
    nmin = 20 * grid.k * grid.k
    num, den = _local_sum(np.where(okt, tk, 0), clear.astype(np.float32), r)
    rowp = row_background(tk, clear, grid)
    # Местный фон, а где ясных ячеек в окне мало — плавно к фону широты (он входит как nmin «ясных» ячеек).
    # Жёсткое переключение по порогу давало прямоугольники размером с окно там, где почти всё в облаках.
    tbg = (num + nmin * rowp[:, None]) / (den + nmin)
    s_ir = np.nan_to_num(np.clip((tbg - tk) / 40.0, 0, 1), nan=0.0)
    if vis is None:
        return s_ir
    vis = np.asarray(vis).astype(np.float32)
    okv = np.isfinite(vis)
    num, den = _local_sum(np.where(okv, vis, 0), (clear & okv).astype(np.float32), r)
    vbg = (num + nmin * 40.0) / (den + nmin)
    day = np.clip((88.0 - sz) / 18.0, 0, 1)
    s_vis = np.nan_to_num(np.clip((vis - vbg) / 90.0, 0, 1), nan=0.0) * day
    return np.maximum(s_ir, s_vis)


def alpha(frac, lw, vis, sz, grid, weight=None):
    """Итоговая непрозрачность 0..1 (float32 H x W). weight — суммарный вес масок из vote(with_weight=True):
    где он меньше W_FULL, к альфе масок плавно подмешивается ИК-формула (без него — только там, где масок нет)."""
    s = opacity(lw, vis, sz, frac, grid)
    z = np.nan_to_num(frac, nan=0.0)
    cov = np.clip((z - F0) / (F1 - F0), 0, 1)
    a = cov * (FLOOR + (1 - FLOOR) * s ** 0.6)
    if weight is not None:
        wm = np.clip(np.nan_to_num(weight) / W_FULL, 0, 1)
        if (wm < 1).any():
            a = wm * a + (1 - wm) * ir_formula(lw, grid, edge=False)
    else:
        nomask = ~np.isfinite(frac)
        if nomask.any():
            a[nomask] = ir_formula(lw, grid, edge=False)[nomask]
    edge = 1 - _smooth(POLE[0], POLE[1], np.abs(grid.lat))
    return (a * edge).astype(np.float32)


def sat_layers(fr, sublons, grid):
    """fr: {имя: доля}, sublons: {имя: долгота подспутниковой точки} -> список для vote()."""
    return [(k, fr[k], geo_vza(grid, sublons[k])) for k in SATS if k in fr]
