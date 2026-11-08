# -*- coding: utf-8 -*-
"""Cloudflare Workers KV через REST API (без wrangler). Переменные окружения:
CF_ACCOUNT_ID, CF_API_TOKEN (право «Workers KV Storage: Edit»), CF_KV_NAMESPACE_ID.
Сетевые ошибки, 429 и 5xx повторяются с нарастающей паузой (2, 5, 10, 20 с)."""
import os, time, urllib.parse
import requests

API = 'https://api.cloudflare.com/client/v4/accounts/%s/storage/kv/namespaces/%s/values/%s'
RETRY_STATUS = (429, 500, 502, 503, 504)
PAUSES = (2, 5, 10, 20)


class KV:
    def __init__(self, account=None, token=None, namespace=None):
        self.account = account or os.environ['CF_ACCOUNT_ID']
        self.token = token or os.environ['CF_API_TOKEN']
        self.ns = namespace or os.environ['CF_KV_NAMESPACE_ID']
        self.s = requests.Session()
        self.s.headers['Authorization'] = 'Bearer ' + self.token

    def _url(self, key):
        return API % (self.account, self.ns, urllib.parse.quote(key, safe=''))

    def _call(self, what, fn):
        """fn() -> Response; повтор при сетевой ошибке и временном статусе. -> последний Response."""
        err = None
        for i in range(len(PAUSES) + 1):
            try:
                r = fn()
                if r.status_code not in RETRY_STATUS:
                    return r
                err = '%d %s' % (r.status_code, r.text[:200])
            except requests.RequestException as e:
                r, err = None, '%s: %s' % (type(e).__name__, e)
            if i < len(PAUSES):
                time.sleep(PAUSES[i])
        if r is not None:
            return r
        raise RuntimeError('KV %s: %s' % (what, err))

    def get(self, key):
        """bytes или None, если ключа нет. Любая другая ошибка — исключение (чтобы не затереть манифест пустым)."""
        r = self._call('get ' + key, lambda: self.s.get(self._url(key), timeout=60))
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content

    def put(self, key, data, ttl=None, content_type='application/octet-stream'):
        u = self._url(key) + ('?expiration_ttl=%d' % ttl if ttl else '')
        r = self._call('put ' + key, lambda: self.s.put(u, data=data, headers={'Content-Type': content_type}, timeout=120))
        try:
            ok = r.ok and r.json().get('success')
        except ValueError:
            ok = False
        if not ok:
            raise RuntimeError('KV put %s: %s %s' % (key, r.status_code, r.text[:300]))
