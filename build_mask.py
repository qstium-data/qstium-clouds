# -*- coding: utf-8 -*-
"""Сборка карты облаков для модуля «Земля»: мозаика готовых масок геостационаров + плотность по GMGSI.

Один запуск = обработать часы за последние 26 ч, для которых уже есть кадр GMGSI LW, а готовой карты ещё нет
(из них не больше --max-hours самых новых). Свежие часы (моложе --refresh-hours), собранные без какого-то
источника, пересобираются, если этот источник с тех пор появился (GCC и Himawari иногда опаздывают); источник,
который при такой пересборке всё равно не скачался, второй раз не ждём.

Выход на каждый час и размер сетки — непрозрачность облака одним каналом (0..255 = 0..1), WebP без потерь
(или PNG), и общий manifest.json. Локально — в --out-dir; с --upload — ещё и в Cloudflare Workers KV:
  m/<ширина>/<ISO>.webp   кадр, живёт 30 ч (expiration_ttl)
  m/manifest.json         список кадров за 26 ч и сведения об источниках (пишется после каждого собранного часа)
Переменные окружения для --upload: CF_ACCOUNT_ID, CF_API_TOKEN, CF_KV_NAMESPACE_ID.

Примеры:
  python build_mask.py                                  # последние 3 часа в ./out
  python build_mask.py --hour 2026-11-07T08 --preview   # один час и наглядная картинка
  python build_mask.py --upload --max-hours 4           # как в GitHub Actions
"""
import argparse, datetime as dt, io, json, os, sys, time, traceback, warnings
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sources as S
import mosaic as M
import nightfilter as NF

warnings.filterwarnings('ignore', message='All-NaN slice')     # строки широты без ясных ячеек — так и задумано
warnings.filterwarnings('ignore', message='Mean of empty slice')

SIZES = {2048: (2048, 824), 4096: (4096, 1648)}
KEEP_H = 26                       # сколько часов держит манифест
TTL_S = 30 * 3600                 # сколько живёт кадр в KV
MANIFEST = 'm/manifest.json'
MASKS = ('goes18', 'goes19', 'himawari', 'iodc', 'fes', 'gcc')
MIN_CLOUDY = 0.15                 # доля сетки с альфой > 0.2 ниже этого — кадр битый (на Земле облачно 60–70 %)
# Подписи: продукт изменён (пересчёт на сетку, голосование, своя альфа) — «Derived from», а не имя исходного продукта
# (условия NOAA Open Data Dissemination); EUMETSAT — формула ст. 6.3 их Data Policy; SatCORPS — их форма благодарности.
SRC = {
    'goes18': ('noaa', 'Derived from NOAA GOES-18 ABI L2 Clear Sky Mask (NOAA Open Data Dissemination)'),
    'goes19': ('noaa', 'Derived from NOAA GOES-19 ABI L2 Clear Sky Mask (NOAA Open Data Dissemination)'),
    'himawari': ('noaa', 'Derived from Himawari-9 (JMA) AHI Cloud Mask by NOAA/NESDIS (NOAA Open Data Dissemination)'),
    'iodc': ('eumetsat', 'Contains modified EUMETSAT Meteosat data %d (CC BY 4.0)'),
    'fes': ('eumetsat', 'Contains modified EUMETSAT Meteosat data %d (CC BY 4.0)'),
    'gcc': ('satcorps', 'Derived from NASA LaRC Satellite ClOud and Radiation Property retrieval System (SatCORPS) GCC-GEO'),
    'gmgsi': ('noaa', 'Derived from NOAA/NESDIS GMGSI (NOAA Open Data Dissemination)'),
    'gfs': ('unidata', 'NOAA NCEP GFS via NSF Unidata THREDDS'),
}
LINKS = {
    'noaa': 'https://www.noaa.gov/information-technology/open-data-dissemination',
    'eumetsat': 'https://creativecommons.org/licenses/by/4.0/',
    'satcorps': 'https://satcorps.larc.nasa.gov',
    'unidata': 'https://www.unidata.ucar.edu',
}
NOTICE = ('Cloud opacity map is a QSTiUM derived product; it is not original NOAA, JMA, EUMETSAT or NASA data '
          'and is not endorsed by them.')
LOG = print


def iso(t):
    return t.strftime('%Y-%m-%dT%H:00:00Z')


def parse_hour(s):
    s = s.strip().rstrip('Z').replace(' ', 'T')
    for f in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M', '%Y-%m-%dT%H', '%Y%m%dT%H'):
        try:
            return dt.datetime.strptime(s, f).replace(minute=0, second=0)
        except ValueError:
            pass
    raise SystemExit('не понял час: ' + s)


def utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def credits(srcs, year):
    """-> (подписи: [строка], ссылки: [{текст, url}]) без повторов, в порядке источников."""
    lines, links = [], []
    for s in srcs:
        kind, text = SRC[s]
        text = text % year if '%d' in text else text
        if text not in lines:
            lines.append(text)
            links.append({'текст': text, 'url': LINKS[kind]})
    return lines, links


# ---------------------------------------------------------------- хранилища: папка и KV
class Store:
    """Папка --out-dir и, с --upload, Cloudflare KV. Манифест читается из KV, если он включён (там правда)."""

    def __init__(self, out_dir, kv=None):
        self.out, self.kv = out_dir, kv

    def get_manifest(self):
        b = None
        if self.kv:
            b = self.kv.get(MANIFEST)            # ошибка сети/прав — исключение: пустым манифестом ничего не затираем
        else:
            p = os.path.join(self.out, MANIFEST)
            if os.path.exists(p):
                b = open(p, 'rb').read()
        if not b:
            return {'кадры': [], 'сведения': {}}
        m = json.loads(b)
        m.setdefault('кадры', []); m.setdefault('сведения', {})
        return m

    def put(self, key, data, ttl=None, ctype='application/octet-stream'):
        p = os.path.join(self.out, *key.replace(':', '').split('/'))     # в имени файла Windows двоеточие нельзя
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as f:
            f.write(data)
        if self.kv:
            self.kv.put(key, data, ttl=ttl, content_type=ctype)


# ---------------------------------------------------------------- загрузка источников одного часа
def fetch_all(t, grids, use, workers=8, deadline=300, only=None):
    """Все источники часа параллельно. -> (данные {имя: (результат, сведения)}, отчёт {имя: {...}})."""
    sz = M.sun_zenith(t, M.Grid(*grids[0]))
    night = bool((NF.zone_weight(M.Grid(*grids[0]), sz) > 0).any()) and 'night' in use
    jobs = [k for k in MASKS if k in use] + ['lw', 'vis'] + (['sw', 'gfs'] if night else [])

    def run(k):
        t0 = time.time()
        try:
            if k in ('goes18', 'goes19'): r = S.fetch_goes(k, t, grids)
            elif k == 'himawari': r = S.fetch_himawari(t, grids)
            elif k in ('iodc', 'fes'): r = S.fetch_clm(k, t, grids)
            elif k == 'gcc': r = S.fetch_gcc(t, grids)
            elif k in ('lw', 'vis', 'sw'): r = S.fetch_gmgsi(k, t, grids)
            elif k == 'gfs': r = (S.fetch_gfs(t, NF.CROP), {})
            return k, r, {'ok': True, 'сек': round(time.time() - t0, 1)}
        except Exception as e:
            why = 'нет данных' if isinstance(e, S.Missing) else type(e).__name__
            return k, None, {'ok': False, 'сек': round(time.time() - t0, 1), 'ошибка': '%s: %s' % (why, str(e)[:240])}

    data, rep = {}, {}
    ex = ThreadPoolExecutor(workers)
    futs = {ex.submit(run, k): k for k in jobs}
    # общий срок: зависший источник не держит остальных — он «опоздал», голосуют остальные
    done, late = wait(futs, timeout=deadline)
    for f in done:
        k, r, info = f.result()
        rep[k] = info
        if r is not None:
            data[k] = r
    for f in late:
        rep[futs[f]] = {'ok': False, 'сек': deadline, 'ошибка': 'опоздал: дольше %d с' % deadline}
    ex.shutdown(wait=False, cancel_futures=True)
    for k in jobs:                               # байты считаются по имени источника в sources.BYTES
        rep[k]['МБ'] = round(sum(v for s, v in S.BYTES.items() if s == k or s == 'gmgsi_' + k) / 1e6, 2)
    return data, rep


# ---------------------------------------------------------------- сведение и кодирование
def build_hour(t, data, grids, use):
    """-> {(W,H): альфа uint8}, сведения для манифеста."""
    out, info = {}, {}
    v = None
    if 'night' in use and 'sw' in data and 'gfs' in data and 'vis' in data:
        g0 = M.Grid(2048, 824)
        k0 = (2048, 824)
        v = NF.veto(data['lw'][0][k0], data['sw'][0][k0], data['vis'][0][k0], M.sun_zenith(t, g0), g0, data['gfs'][0])
    for (W, H) in grids:
        g = M.Grid(W, H)
        sz = M.sun_zenith(t, g)
        fr = {k: data[k][0][(W, H)] for k in M.SATS if k in data}
        sub = {k: float(data[k][1]['долгота']) for k in fr}
        gcc = data['gcc'][0][(W, H)] if 'gcc' in data else None
        frac, src, share, wt = M.vote(M.sat_layers(fr, sub, g), gcc, g, with_weight=True)
        lw = data['lw'][0][(W, H)]
        vis = data['vis'][0][(W, H)] if 'vis' in data else None
        a = M.alpha(frac, lw, vis, sz, g, weight=wt)
        nf = None
        if v is not None:
            a, nf = NF.apply(a, None, None, None, None, None, None, v=v)
        out[(W, H)] = np.clip(np.rint(a * 255), 0, 255).astype(np.uint8)
        if (W, H) == (2048, 824):
            info = {'маски': sorted(fr) + (['gcc'] if gcc is not None else []),
                    'главный': {k: round(v, 3) for k, v in share.items()},
                    'без_масок': round(float((~np.isfinite(frac)).mean()), 4),
                    'видимый': vis is not None,
                    'ночной_фильтр': None if nf is None else round(nf, 5),
                    'облачно': round(float((a > 0.2).mean()), 3)}
    return out, info


def encode(a, fmt, quality=0):
    """quality=0 — WebP без потерь; 1..100 — с потерями (q90: −29 % веса, средняя ошибка ~1/255)."""
    im = Image.fromarray(a, 'L')
    buf = io.BytesIO()
    if fmt == 'webp':
        if quality:
            im.save(buf, 'WEBP', quality=quality, method=4)
        else:
            im.save(buf, 'WEBP', lossless=True, quality=100, method=4, exact=True)   # method 6: в 13 раз дольше ради 0,1 %
    else:
        im.save(buf, 'PNG', optimize=True)
    return buf.getvalue()


def preview(a, data, grid_key, t, path):
    """Наглядная картинка: сверху альфа белым поверх ИК/видимого (тёмно-синий фон), снизу исходный ИК."""
    W, H = grid_key
    lw = np.nan_to_num(np.asarray(data['lw'][0][grid_key], np.float32), nan=0.0)
    vis = np.nan_to_num(np.asarray(data['vis'][0][grid_key], np.float32), nan=0.0) if 'vis' in data else np.zeros_like(lw)
    day = np.clip((88.0 - M.sun_zenith(t, M.Grid(W, H))) / 18.0, 0, 1)
    base = np.maximum(lw * 0.55, vis * day * 0.8) / 255.0
    bg = np.stack([0.05 + 0.25 * base, 0.12 + 0.3 * base, 0.25 + 0.35 * base], -1)
    al = (a.astype(np.float32) / 255.0)[..., None]
    top = bg * (1 - al) + al * 1.0
    bot = np.repeat((lw / 255.0)[..., None], 3, -1)
    img = (np.concatenate([top, np.ones((6, W, 3)), bot], 0) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# ---------------------------------------------------------------- какие часы делать
def plan(manifest, max_hours, refresh_h, now, use):
    """Часы с кадром GMGSI LW за последние KEEP_H, которых нет в манифесте, и свежие неполные, если пропавший
    источник появился. Из всех кандидатов — max_hours самых новых (старые дыры тоже дойдут до очереди)."""
    have = set(manifest.get('кадры', []))
    hours = sorted((h for h in S.gmgsi_hours(KEEP_H, now) if h > now - dt.timedelta(hours=KEEP_H)), reverse=True)
    todo = []
    for h in hours:
        if len(todo) >= max_hours:
            break
        k = iso(h)
        if k not in have:
            todo.append((h, 'новый')); continue
        if now - h > dt.timedelta(hours=refresh_h):
            continue
        inf = manifest.get('сведения', {}).get(k, {})
        got = set(inf.get('маски', []))
        tried = inf.get('пересборка', {})                  # источник уже подводил при пересборке — не ждём его снова
        missing = [s for s in MASKS if s in use and s not in got and not tried.get(s)]
        late = [s for s in missing if S.available(s, h)]
        if late:
            todo.append((h, 'появились: ' + ','.join(late)))
    return todo


def frame_ok(info):
    """Правдоподобен ли кадр: на Земле облачно ~60–70 %, кадр почти без облаков — битый вход."""
    if info.get('облачно', 1.0) < MIN_CLOUDY:
        return 'облачно только %.1f %% сетки (порог %.0f %%)' % (100 * info['облачно'], 100 * MIN_CLOUDY)
    return None


def write_manifest(store, manifest, want, fmt):
    """Отрезать кадры старше KEEP_H и записать манифест. -> список кадров."""
    edge = utcnow() - dt.timedelta(hours=KEEP_H)
    frames = sorted(x for x in set(manifest['кадры']) if parse_hour(x) > edge)
    out = {
        'в': 1,
        'обновлено': utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'формат': fmt,
        'размеры': want,
        'сетка': {'crs': 'CRS:84', 'запад': -180, 'восток': 180, 'север': 72.5, 'юг': -72.5,
                  'высота': {str(w): SIZES[w][1] for w in SIZES}},
        'значение': 'один канал (в WebP R=G=B): 0..255 = непрозрачность облака 0..1; строка 0 — север',
        'оговорка': NOTICE,
        'кадры': frames,
        'сведения': {x: manifest['сведения'][x] for x in frames if x in manifest.get('сведения', {})},
    }
    store.put(MANIFEST, json.dumps(out, ensure_ascii=False, separators=(',', ':')).encode('utf-8'),
              ctype='application/json; charset=utf-8')
    manifest.clear(); manifest.update(out)
    return frames


def do_hour(h, why, manifest, store, grids, want, use, a):
    """Собрать и записать один час. -> (запись отчёта, изменился ли манифест)."""
    k = iso(h); rec = {'причина': why}
    S.BYTES.clear()
    t0 = time.time()
    data, rep = fetch_all(h, grids, use)
    rec['загрузка_сек'] = round(time.time() - t0, 1); rec['источники'] = rep
    rec['скачано_МБ'] = round(sum(S.BYTES.values()) / 1e6, 1)
    if 'lw' not in data:
        LOG('%s: нет GMGSI LW — пропуск (%s)' % (k, rep.get('lw', {}).get('ошибка', '')))
        rec['итог'] = 'нет LW'; return rec, False
    old = manifest.get('сведения', {}).get(k)
    got = [s for s in MASKS if s in data]
    n_old = len(old.get('маски', [])) if old else 0
    if old and ((why.startswith('появились') and len(got) <= n_old) or (why == 'по запросу' and len(got) < n_old and not a.force)):
        # Не заменять кадр более бедным. Опоздавший источник, который так и не скачался, запоминаем — второй раз не ждём.
        if why.startswith('появились'):
            tried = old.setdefault('пересборка', {})
            for s in why.split(': ', 1)[1].split(','):
                if s not in data:
                    tried[s] = tried.get(s, 0) + 1
        LOG('%s: масок не прибавилось (было %d, теперь %s) — оставляю прежний кадр%s' % (
            k, n_old, ','.join(got), '' if why.startswith('появились') else ' (затереть: --force)'))
        rec['итог'] = 'без изменений'; return rec, why.startswith('появились')
    t1 = time.time()
    alphas, info = build_hour(h, data, grids, use)
    rec['сведение_сек'] = round(time.time() - t1, 1)
    bad = frame_ok(info)
    if bad:
        LOG('%s: кадр отклонён — %s; прежний (если был) остаётся' % (k, bad))
        rec['итог'] = 'отклонён: ' + bad; return rec, False
    info['нет'] = {s: rep[s]['ошибка'] for s in MASKS if s in rep and not rep[s]['ok']}
    info['собрано'] = utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    used = list(info['маски']) + ['gmgsi'] + (['gfs'] if info.get('ночной_фильтр') is not None else [])
    info['подписи'], info['ссылки'] = credits(used, h.year)
    t2 = time.time()
    sizes = {}
    for (W, H), al in alphas.items():
        if W not in want:
            continue
        b = encode(al, a.format, a.quality)
        key = 'm/%d/%s.%s' % (W, k, a.format)
        store.put(key, b, ttl=TTL_S, ctype='image/' + a.format)
        sizes[W] = len(b)
        if a.preview:
            preview(al, data, (W, H), h, os.path.join(a.out_dir, 'm', str(W), k.replace(':', '') + '.preview.png'))
    rec['запись_сек'] = round(time.time() - t2, 1); rec['байт'] = sizes
    info['байт'] = sizes
    manifest.setdefault('сведения', {})[k] = info
    if k not in manifest['кадры']:
        manifest['кадры'].append(k)
    rec['итог'] = 'собран'; rec['маски'] = info['маски']; rec['всего_сек'] = round(time.time() - t0, 1)
    LOG('%s: собран за %.0f с, скачано %.1f МБ, маски %s, нет %s, %s' % (
        k, rec['всего_сек'], rec['скачано_МБ'], ','.join(info['маски']) or '—',
        ','.join(info['нет']) or '—', ', '.join('%d: %d КБ' % (w, n // 1024) for w, n in sizes.items())))
    for s, r in rep.items():
        LOG('    %-8s %-5s %6.1f с %7.2f МБ %s' % (s, 'ок' if r['ok'] else 'НЕТ', r['сек'], r['МБ'], r.get('ошибка', '')))
    return rec, True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=os.path.join(os.getcwd(), 'out'))
    ap.add_argument('--upload', action='store_true', help='писать в Cloudflare KV (нужны CF_ACCOUNT_ID, CF_API_TOKEN, CF_KV_NAMESPACE_ID)')
    ap.add_argument('--max-hours', type=int, default=3, help='сколько часов собрать за запуск (самые новые из недостающих)')
    ap.add_argument('--refresh-hours', type=int, default=3, help='до какого возраста часа пересобирать при опоздавшем источнике')
    ap.add_argument('--hour', action='append', default=[], help='собрать этот час (можно несколько), независимо от манифеста')
    ap.add_argument('--force', action='store_true', help='с --hour: заменить кадр, даже если масок теперь меньше')
    ap.add_argument('--budget-min', type=float, default=17.0,
                    help='не начинать новый час после стольких минут (час — до ~5,5 мин; срок задачи в Actions 25 мин)')
    ap.add_argument('--sizes', default='2048', help='2048 и/или 4096 через запятую')
    ap.add_argument('--format', default='webp', choices=['webp', 'png'])
    ap.add_argument('--quality', type=int, default=0, help='WebP с потерями 1..100 (0 — без потерь, по умолчанию)')
    ap.add_argument('--skip', default='', help='не брать эти источники: goes18,goes19,himawari,iodc,fes,gcc,night')
    ap.add_argument('--preview', action='store_true', help='сохранить наглядный PNG рядом с кадром (только в папку)')
    ap.add_argument('--report', default='', help='JSON с отчётом прогона (секунды и МБ по шагам)')
    a = ap.parse_args()
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

    want = sorted({int(x) for x in a.sizes.split(',') if x})
    if any(w not in SIZES for w in want):
        raise SystemExit('размеры: только ' + ', '.join(map(str, SIZES)))
    grids = [SIZES[w] for w in sorted(set(want) | {2048})]      # 2048 считается всегда: на ней сведения и ночной фильтр
    skip = {x for x in a.skip.split(',') if x}
    use = (set(MASKS) | {'night'}) - skip
    kv = None
    if a.upload:
        miss = [v for v in ('CF_ACCOUNT_ID', 'CF_API_TOKEN', 'CF_KV_NAMESPACE_ID') if not os.environ.get(v)]
        if miss:
            raise SystemExit('для --upload нужны переменные окружения: ' + ', '.join(miss))
        import cfkv
        kv = cfkv.KV()
    store = Store(a.out_dir, kv)
    now = utcnow()
    T0 = time.time()
    run = {'начало': now.strftime('%Y-%m-%dT%H:%M:%SZ'), 'часы': {}}

    def save_report():
        run['всего_сек'] = round(time.time() - T0, 1)
        if a.report:
            os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
            with open(a.report, 'w', encoding='utf-8') as f:
                json.dump(run, f, ensure_ascii=False, indent=1)

    manifest = store.get_manifest()
    if a.hour:
        todo = [(parse_hour(h), 'по запросу') for h in a.hour]
    else:
        todo = plan(manifest, a.max_hours, a.refresh_hours, now, use)
    run['план_сек'] = round(time.time() - T0, 1)
    LOG('план: %s' % (', '.join('%s (%s)' % (iso(h), why) for h, why in todo) or 'нового нет'))

    built = failed = 0
    for h, why in todo:
        k = iso(h)
        if time.time() - T0 > a.budget_min * 60:
            LOG('%s: бюджет %.0f мин исчерпан — оставляю на следующий запуск' % (k, a.budget_min))
            run['часы'][k] = {'причина': why, 'итог': 'отложен: бюджет времени'}
            continue
        try:
            rec, changed = do_hour(h, why, manifest, store, grids, want, use, a)
            if changed:                                    # манифест — сразу: обрыв запуска не теряет собранное
                write_manifest(store, manifest, want, a.format)
            if rec['итог'] == 'собран':
                built += 1
        except Exception as e:                             # ошибка одного часа (KV, сведение) не роняет остальные
            failed += 1
            traceback.print_exc()
            rec = {'причина': why, 'итог': 'ошибка: %s: %s' % (type(e).__name__, str(e)[:300])}
            LOG('%s: %s' % (k, rec['итог']))
        run['часы'][k] = rec
        save_report()

    frames = sorted(manifest['кадры'])
    if not a.hour or built:                                # отрезать устаревшие кадры даже без новых часов
        frames = write_manifest(store, manifest, want, a.format)
    run['кадров_в_манифесте'] = len(frames)
    save_report()
    LOG('итог: собрано %d, ошибок %d, в манифесте %d кадров, %.0f с' % (built, failed, len(frames), run['всего_сек']))
    return 1 if failed and not built else 0


if __name__ == '__main__':
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int): print(e.code, file=sys.stderr)
    except BaseException:
        traceback.print_exc(); code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)          # не ждать потоков опоздавших источников: всё нужное уже записано
