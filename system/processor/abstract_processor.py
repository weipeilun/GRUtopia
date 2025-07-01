# encoding: utf-8

import threading
import multiprocessing
import time
import logging
import setproctitle
from system.constants import (
    CONFIG_KEY_IN_SOURCE,
    CONFIG_KEY_OUT_PORT,
    CONFIG_KEY_OUTPUT_FREQUENCY,
    CONFIG_KEY_FREQUENCY,
    CONFIG_KEY_IN_QUEUE,
    CONFIG_KEY_TORCH_DEVICE,
    CONFIG_KEY_TENSORFLOW_DEVICE,
    CONFIG_KEY_PADDLE_DEVICE,
    CONFIG_KEY_ONNX_DEVICE,
    CONFIG_KEY_DEFAULT_TORCH_DEVICE,
    CONFIG_KEY_DEFAULT_TENSORFLOW_DEVICE,
    CONFIG_KEY_DEFAULT_PADDLE_DEVICE,
    CONFIG_KEY_DEFAULT_ONNX_DEVICE,
)
from system.tools.processor_config import config, gpu_config
from system.tools import interrupt
from system.tools import config_formatter
from queue import Empty, Queue
import msgpack
import traceback


class AbstractProcessor(multiprocessing.Process):

    _PID = 'abstract'

    def __init__(self, in_queue_list=None, out_queue_info=None, interrupt_check=interrupt.interrupt_callback, listen_interval=0.01, do_validate=True, debug_time=False, log_level=logging.INFO):
        assert self._PID in config, 'processor id %s not in config' % self._PID

        self.interrupt_check = interrupt_check
        self.listen_interval = listen_interval
        self.debug_time = debug_time
        self.log_level = log_level

        self.data_sources = []
        self.data_addresses = []
        self.do_validates = []
        self.data_intervals = []
        if CONFIG_KEY_IN_SOURCE in config[self._PID]:
            for source_config in config[self._PID][CONFIG_KEY_IN_SOURCE]:
                pid, ip, validate = config_formatter.parse_in_source(source_config, do_validate)

                assert pid in config, 'processor id %s not in config' % pid
                assert CONFIG_KEY_OUT_PORT in config[pid], 'config key %s not in %s config' % (CONFIG_KEY_OUT_PORT, pid)

                self.data_sources.append(pid)
                self.do_validates.append(validate)

                # zmq subscribe
                self.data_addresses.append("tcp://%s:%s" % (ip, config[pid][CONFIG_KEY_OUT_PORT]))
                # some processors have different processing frequency and output frequency
                self.data_intervals.append(1 / float(config[pid][CONFIG_KEY_OUTPUT_FREQUENCY]) if CONFIG_KEY_OUTPUT_FREQUENCY in config[pid] else 1 / float(config[pid][CONFIG_KEY_FREQUENCY]) if CONFIG_KEY_FREQUENCY in config[pid] else None)

        self.p_interval = 1 / float(config[self._PID][CONFIG_KEY_FREQUENCY]) if CONFIG_KEY_FREQUENCY in config[self._PID] else None

        # zmq publish
        self.out_address = None
        self.zmq_socket = None
        if CONFIG_KEY_OUT_PORT in config[self._PID]:
            self.out_address = "tcp://*:%s" % config[self._PID][CONFIG_KEY_OUT_PORT]

        self.data_cache = dict()

        # 用于数据校验, last_time: {processor_to_get: time}
        self.start_time = time.time()
        start_time_truncated = self.truncate_time(self.start_time)
        self.last_time = {name: start_time_truncated for name in config.keys()}

        # in_queue是否取array数据，减少性能损耗
        self.in_queue_get_array_dict = dict()
        # 检查依赖queue数据的进程
        if CONFIG_KEY_IN_QUEUE in config[self._PID]:
            # assert in_queue_list is not None and len(in_queue_list) == len(config[self._PID][CONFIG_KEY_IN_QUEUE]), f'{self._PID} in_queue_list must be set according to config.'

            self_freq = config[self._PID][CONFIG_KEY_FREQUENCY]
            # 输出的数据必须frequency满足整数倍
            for in_pid_info in config[self._PID][CONFIG_KEY_IN_QUEUE]:
                in_pid, is_get_array = config_formatter.parse_in_queue(in_pid_info)
                self.in_queue_get_array_dict[in_pid] = is_get_array

                in_freq = config[in_pid][CONFIG_KEY_FREQUENCY]
                if self_freq >= 1:
                    assert in_freq % self_freq == 0, f'{in_pid} freq ({in_freq}) must be an integer multiple of {self._PID} freq ({self_freq}).'

        self.in_queue_list = in_queue_list
        self.in_pid_queue_dict = dict()
        self.out_queue_info = out_queue_info
        self.pid_push_counter = dict()
        if in_queue_list is not None:
            for pid, queue in in_queue_list:
                self.in_pid_queue_dict[pid] = Queue()
        if out_queue_info is not None:
            out_pid2consumer_idx_dict, out_queue = out_queue_info
            for pid in out_pid2consumer_idx_dict.keys():
                push_queue_frame = config[self._PID][CONFIG_KEY_FREQUENCY] // config[pid][CONFIG_KEY_FREQUENCY]
                self.pid_push_counter[pid] = (push_queue_frame, push_queue_frame)

        self.torch_device = config[self._PID][CONFIG_KEY_TORCH_DEVICE] if CONFIG_KEY_TORCH_DEVICE in config[self._PID] else gpu_config[CONFIG_KEY_DEFAULT_TORCH_DEVICE]
        self.tensorflow_device = config[self._PID][CONFIG_KEY_TENSORFLOW_DEVICE] if CONFIG_KEY_TENSORFLOW_DEVICE in config[self._PID] else gpu_config[CONFIG_KEY_DEFAULT_TENSORFLOW_DEVICE]
        self.paddle_device = config[self._PID][CONFIG_KEY_PADDLE_DEVICE] if CONFIG_KEY_PADDLE_DEVICE in config[self._PID] else gpu_config[CONFIG_KEY_DEFAULT_PADDLE_DEVICE]
        self.onnx_device = config[self._PID][CONFIG_KEY_ONNX_DEVICE] if CONFIG_KEY_ONNX_DEVICE in config[self._PID] else gpu_config[CONFIG_KEY_DEFAULT_ONNX_DEVICE]

        self._running = False

        # threading.Thread.__init__(self)
        # self._stop_event = threading.Event()
        multiprocessing.Process.__init__(self)
        # super(AbstractProcessor, self).__init__()

    def _listen(self, p_name, p_address):
        # get data from zmq thread
        import zmq
        zmq_context = zmq.Context.instance()
        p_socket = zmq_context.socket(zmq.SUB)

        p_socket.connect(p_address)
        p_socket.setsockopt(zmq.RCVTIMEO, int(self.listen_interval * 1000))
        p_socket.setsockopt(zmq.SUBSCRIBE, b"")
        logging.info('%s subscriber connected to address %s' % (self._PID, p_address))

        while self._running:
            if self.interrupt_check():
                logging.info(f"{self._PID} _listen detect interrupt")
                break

            try:
                msg = p_socket.recv()
                unpacked_msg = msgpack.unpackb(msg)
                if unpacked_msg is not None:
                    self.data_cache[p_name] = unpacked_msg
            except zmq.ZMQError as e:
                if e.errno != zmq.ETERM and e.errno != zmq.EAGAIN:  # Interrupted or timeout
                    raise

    def _pull(self):
        # get data from high_performance_queue thread
        while self._running:
            if self.interrupt_check():
                logging.info(f"{self._PID} _listen detect interrupt")
                break

            for pid, in_hp_queue in self.in_queue_list:
                try:
                    data = in_hp_queue.get(block=False, ignore_array=not self.in_queue_get_array_dict[pid])
                    self.in_pid_queue_dict[pid].put(data)
                except Empty:
                    pass
            time.sleep(0.003)

    def run(self):
        setproctitle.setproctitle(f'teleoperate-{self._PID}')

        logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%Y-%m-%d,%H:%M:%S')
        logger = logging.getLogger()
        logger.setLevel(self.log_level)

        # numba_logger = logging.getLogger('numba')
        # numba_logger.setLevel(logging.WARNING)

        self._running = True

        self.init()

        import zmq
        zmq_context = zmq.Context.instance()
        self.zmq_socket = zmq_context.socket(zmq.PUB)
        if self.out_address is not None and self.zmq_socket is not None:
            try:
                self.zmq_socket.bind(self.out_address)
                logging.info('%s publisher bind to address %s' % (self._PID, self.out_address))
            except zmq.error.ZMQError as e:
                logging.error('%s. publisher=%s; address=%s' % (e.strerror, self._PID, self.out_address))
                raise e

        if len(self.data_sources) > 0 and len(self.data_addresses) > 0:
            for data_source, data_address in zip(self.data_sources, self.data_addresses):
                if data_source not in self.in_pid_queue_dict:
                    t = threading.Thread(target=self._listen, args=(data_source, data_address), daemon=True)
                    t.start()

        if self.in_queue_list is not None and len(self.in_queue_list) > 0:
            threading.Thread(target=self._pull, daemon=True).start()

        self.run_loop()

        self.stop()

    def run_loop(self):
        previous_t = time.time()
        while self._running:
            if self.interrupt_check():
                logging.info("processor %s detect interrupt" % self._PID)
                break

            for pid, in_queue in self.in_pid_queue_dict.items():
                if in_queue.qsize() > 100:
                    logging.warning(f'{self._PID} image queue from {pid} size={in_queue.qsize()}')

            now = time.time()
            if self.debug_time:
                start_t = now

            data, queues_not_empty = self.pull()
            # queue里有数据不要等，取空为止
            if not queues_not_empty and self.p_interval is not None and now - previous_t < self.p_interval:
                time.sleep(self.listen_interval)
                continue

            if self.debug_time:
                pulled_t = time.time()

            if self._is_valid(data):
                if self.debug_time:
                    checked_t = time.time()

                try:
                    data = self.process(data)
                except Exception as e:
                    data = None
                    error_info = traceback.format_exc()
                    logging.error('process error: pid=%s, exception=%s, trace=%s', self._PID, str(e), error_info)

                if self.debug_time:
                    processed_t = time.time()

                self.push(data)

                if self.debug_time:
                    pushed_t = time.time()
                    in_queue_info = 'in queue: '
                    for pid, in_queue in self.in_pid_queue_dict.items():
                        in_queue_info += f' {pid}: {in_queue.qsize()},'

                    debug_str = self.debug_str()

                    logging.info(f'{self._PID} time elapse pull:{pulled_t - start_t:.4f}, validation:{checked_t - pulled_t:.4f}, '
                                 f'process:{processed_t - checked_t:.4f}, push:{pushed_t - processed_t:.4f}, sum:{pushed_t - start_t:.4f}; {in_queue_info}; {debug_str}')
            previous_t = now

    def debug_str(self):
        return ''

    def stop(self):
        self._running = False

    def init(self):
        pass

    def pull(self):
        data = dict()

        now = time.time()
        # 如果有queue且所有queue都已经取空，要sleep，否则线程无限循环会打满cpu
        # 如果没有queue，或有queue但没有全部取空，不要sleep
        queues_not_empty = False
        for pid, in_queue in self.in_pid_queue_dict.items():
            try:
                msg = in_queue.get(block=False)
                data[pid] = (now, msg)
                queues_not_empty = True
            except Empty:
                data[pid] = None

        if len(self.data_sources) > 0 and len(self.data_intervals) > 0:
            for data_source, do_validate, data_interval in zip(self.data_sources, self.do_validates, self.data_intervals):
                # 用queue传输的数据不走cache
                if data_source not in self.in_pid_queue_dict:
                    cur_data = self.data_cache.get(data_source, None)
                    if cur_data is not None:
                        previous_t, msg = cur_data
                        # 数据有效性校验：数据时间间隔超过传感器频率*2认定为无效数据
                        if do_validate and data_interval is not None and now - previous_t >= data_interval * 2:
                            data[data_source] = None
                        else:
                            data[data_source] = (previous_t, msg)
                    else:
                        data[data_source] = None
                logging.debug('processor %s pulled from processor %s' % (self._PID, data_source))
        return data, queues_not_empty

    def push(self, data):
        if data is not None:
            if self.out_queue_info is not None:
                out_pid2consumer_idx_dict, out_queue = self.out_queue_info
                consumer_to_push_to_idx_list = []
                for out_pid, consumer_idx in out_pid2consumer_idx_dict.items():
                    counter, start = self.pid_push_counter[out_pid]
                    if counter <= 1:
                        consumer_to_push_to_idx_list.append(consumer_idx)
                        self.pid_push_counter[out_pid] = (start, start)
                    else:
                        self.pid_push_counter[out_pid] = (counter - 1, start)
                out_queue.put(data, broadcast_to=consumer_to_push_to_idx_list)

                # 用zmq不要发大数组
                if not out_queue.ignore_array:
                    data = data[0]

            if self.zmq_socket is not None and data is not None:
                now = time.time()
                data_packed = msgpack.packb((now, data))
                self.zmq_socket.send(data_packed)
            # logging.debug('processor %s push %s' % (self._PID, data))
        else:
            if self.zmq_socket is not None:
                data_packed = msgpack.packb(None)
                self.zmq_socket.send(data_packed)

    def process(self, data):
        # 建议np.ndarray传输前后使用tobytes和frombuffer
        return data

    def refresh_cache(func):
        def do_refresh(self, data):
            res = func(self, data)
            for key in self.data_sources:
                if key in data:
                    d = data[key]
                    if d is not None:
                        if isinstance(d, list):
                            d = d[-1]
                        time_round = self.truncate_time(d[0])
                        last_time = self.last_time[key]
                        if last_time != time_round:
                            self.last_time[key] = time_round
            return res
        return do_refresh

    @refresh_cache
    def _is_valid(self, data):
        if data is None or len(data) <= 0:
            return True

        # 用queue输入的数据不做数据有效性校验
        if len(self.in_pid_queue_dict) > 0:
            for pid in self.in_pid_queue_dict.keys():
                if data[pid] is not None:
                    return True
            return False

        if CONFIG_KEY_IN_SOURCE in config[self._PID]:
            has_valid = False
            for key, do_validate in zip(self.data_sources, self.do_validates):
                if not do_validate or self.is_data_repeat(data, key):
                    has_valid = True
                    break
            if not has_valid:
                return has_valid

        has_valid = False
        for value in data.values():
            if value is not None:
                has_valid = True
                break
        return has_valid

    def is_data_repeat(self, data, key):
        if key in data:
            d = data[key]
            if d is not None:
                time_round = self.truncate_time(d[0])
                last_time = self.last_time[key]
                return last_time != time_round
        return False

    @staticmethod
    def truncate_time(time_float):
        return round(time_float, 4)