# -*- coding:utf-8 -*-

import tornado.gen
from tornado.web import HTTPError


class BaseHandler(tornado.web.RequestHandler):
    def send_response(self, data=None, err_code=0, err_msg=''):
        resp = {'err_code': err_code,
                'err_msg': err_msg,
                'data': data or ''}
        self.write(resp)
        self.finish()

    def write_error(self, status_code, **kwargs):
        self.write({'err_code': status_code, 'err_msg': kwargs.get('msg', '')})

    def data_received(self, chunk):
        pass


class AnswerHandler(BaseHandler):
    def initialize(self, msg_queue):
        self.msg_queue = msg_queue

    @tornado.gen.coroutine
    def post(self, *args, **kwargs):
        qid = self.get_argument('qid')
        key = self.get_argument('key')
        self.msg_queue.put({'qid': qid, 'key': key})
        self.send_response()
