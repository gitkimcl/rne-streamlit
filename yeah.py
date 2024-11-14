# https://github.com/langchain-ai/langgraph/issues/101

import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx
import asyncio


_threads_queues = weakref.WeakKeyDictionary()

class MyThreadPoolExecutor(ThreadPoolExecutor):

    def _adjust_thread_count(self):
        # if idle threads are available, don't spin new threads
        if self._idle_semaphore.acquire(timeout=0):
            return

        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = '%s_%d' % (self._thread_name_prefix or self,
                                     num_threads)
            t = threading.Thread(name=thread_name, target=_worker,
                                 args=(weakref.ref(self, weakref_cb),
                                       self._work_queue,
                                       self._initializer,
                                       self._initargs))
            add_script_run_ctx(t)
            t.start()
            self._threads.add(t)
            _threads_queues[t] = self._work_queue