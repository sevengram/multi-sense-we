# -*- coding: utf-8 -*-

import json
import urllib

import tornado.httpclient
import tornado.httputil

user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36'


def urlencode(dic):
    return urllib.urlencode(dict_str(dic))


def dict_str(src, encoding='utf8'):
    return {k: v.encode(encoding) if type(v) is unicode else v for k, v in src.iteritems()}


type_methods = {
    'json': json.dumps,
    'form': urlencode,
    'raw': lambda a: a
}


def _send_dict(url, method, data, data_type, headers):
    _headers = headers or {}
    if data_type == 'form':
        _headers['Content-Type'] = 'application/x-www-form-urlencoded; charset=UTF-8'
    client = tornado.httpclient.HTTPClient()
    req = tornado.httpclient.HTTPRequest(
        url=url,
        method=method,
        body=type_methods.get(data_type)(data),
        headers=tornado.httputil.HTTPHeaders(_headers)
    )
    return client.fetch(req)


def post_dict(url, data, data_type='form', headers=None):
    return _send_dict(url, 'POST', data, data_type, headers)


def put_dict(url, data, data_type='form', headers=None):
    return _send_dict(url, 'PUT', data, data_type, headers)


def get_dict(url, data, headers=None):
    client = tornado.httpclient.HTTPClient()
    req = tornado.httpclient.HTTPRequest(
        url=url + '?' + urlencode(data),
        method='GET',
        headers=tornado.httputil.HTTPHeaders(headers or {})
    )
    return client.fetch(req)


def build_url(base_url, params):
    return base_url + '?' + urllib.urlencode(params)
