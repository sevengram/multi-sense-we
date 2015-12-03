# -*- coding:utf-8 -*-

import Queue
import json
import threading

import http

question_url = 'http://115.28.254.167/learning/questions'

task_url = 'http://115.28.254.167/learning/tasks'


class QuestionThread(threading.Thread):
    def __init__(self, question, index, tid, qmap):
        threading.Thread.__init__(self)
        self.question = question
        self.tid = tid
        self.qmap = qmap
        self.index = index

    def run(self):
        resp = http.post_dict(question_url, {
            'tid': self.tid,
            'stem': self.question['stem'],
            'options': json.dumps(self.question['options'])
        })
        resp_data = json.loads(resp.body)
        self.qmap[resp_data['data']['qid']] = self.index


class UserClassifier(object):
    def __init__(self):
        self.qmap = {}
        self.tid = None
        self.msg_queue = Queue.Queue()

    def create_task(self):
        resp = http.post_dict(task_url, {
            'type': 1,
            'resp_url': '',  # FIXME
            'title': 'Word Embedding Task'
        })
        resp_data = json.loads(resp.body)
        self.tid = resp_data['data']['tid']

    def ask(self, questions):
        threads = []
        with self.msg_queue.mutex:
            self.msg_queue.queue.clear()
        for i, q in enumerate(questions):
            t = QuestionThread(q, self.tid, self.qmap, i)
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        result = [None] * len(self.qmap)
        while self.qmap:
            answer = self.msg_queue.get()
            qid = answer['qid']
            key = answer['key']
            result[self.qmap[qid]] = key
            del self.qmap[qid]
        return result
