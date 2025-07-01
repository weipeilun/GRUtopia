# encoding: utf-8

import cv2
import time
import random
import logging
import numpy as np
from system.tools.processor_config import config
from system.processor.abstract_processor import AbstractProcessor
from system.constants import CONFIG_KEY_MOCK, CONFIG_KEY_FREQUENCY
        

class MockProcessor(AbstractProcessor):
    _PID = CONFIG_KEY_MOCK
    
    def init(self):
        self.frame_id = 0
        self.frame_interval = 1 / config[CONFIG_KEY_MOCK][CONFIG_KEY_FREQUENCY]
    
    # simplify run_loop for better performance
    def run_loop(self):
        start_time = time.time()
        while self._running:
            if self.interrupt_check():
                logging.info("processor %s detect interrupt" % self._PID)
                break

            data = self.process(start_time)
            cur_time = time.time()
            # print(cur_time - start_time)
            start_time = cur_time
            self.push(data)

    def process(self, start_time):
        if self.debug_time:
            self.counter2 -= 1
            if self.counter2 <= 0:
                logging.debug('camera running at fps=%s, timestamp=%s' % (self.camera_frequency, time.time()))
                self.counter2 = self.camera_frequency

        if random.random() > 0.5:
            frame = np.zeros((1, 2, 768, 1024, 3), dtype=np.uint8)
        else:
            frame = np.ones((1, 2, 768, 1024, 3), dtype=np.uint8) * 255
        cur_time = time.time()
        sleep_time = max(self.frame_interval - cur_time + start_time, 0.)
        # print(sleep_time)
        time.sleep(sleep_time)

        self.frame_id += 1
        # jpg比png压缩后数据更小，序列化/反序列化更快
        return self.frame_id, frame
