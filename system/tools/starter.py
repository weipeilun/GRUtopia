import importlib
import inspect
import multiprocessing

import numpy as np
from system.tools import config_formatter
from system.tools import high_performance_queue
from system.tools.processor_config import config
from system.constants import CONFIG_KEY_IN_QUEUE, CONFIG_KEY_OUT_QUEUE_SHAPE, CONFIG_KEY_OUT_QUEUE_DTYPE, CONFIG_KEY_CAMERA_GROUP, CONFIG_KEY_ISAAC_SIM, CONFIG_KEY_VALID_CAMERAS
from multiprocessing import freeze_support


def gen_pid_processor_dict():
    pid_processor_dict = dict()

    import system.processor as processor
    for module_name in processor.__dir__():
        if '_processor' in module_name:
            module_full_name = f'system.processor.{module_name}'
            module = importlib.import_module(module_full_name)
            # try:
            for cls_name, cls in inspect.getmembers(module, inspect.isclass):
                if 'Processor' in cls_name and 'AbstractProcessor' not in cls_name:
                    pid_processor_dict[cls._PID] = cls
            # except ModuleNotFoundError:
            #     pass
    return pid_processor_dict


# high_performance_queue设置shared_memory大小
def set_config_by_topology(producer_consumers_dict, start_processors):
    headers = [CONFIG_KEY_ISAAC_SIM]
    ignores = dict()

    # ignores也要走high_performance_queue，所以要检查配置
    for pid, header_pid in ignores.items():
        # 加上camera个数
        if CONFIG_KEY_OUT_QUEUE_SHAPE not in config[pid] or len(config[pid][CONFIG_KEY_OUT_QUEUE_SHAPE]) == 0:
            raise ValueError(f'{pid}.{CONFIG_KEY_OUT_QUEUE_SHAPE} must be set')
        else:
            config[pid][CONFIG_KEY_OUT_QUEUE_SHAPE] = (len(config[config[header_pid][CONFIG_KEY_CAMERA_GROUP]]), *config[pid][CONFIG_KEY_OUT_QUEUE_SHAPE])

        # str转np.dtype
        if CONFIG_KEY_OUT_QUEUE_DTYPE not in config[pid]:
            raise ValueError(f'{pid}.{CONFIG_KEY_OUT_QUEUE_DTYPE} must be set')
        else:
            config[pid][CONFIG_KEY_OUT_QUEUE_DTYPE] = np.dtype(config[pid][CONFIG_KEY_OUT_QUEUE_DTYPE])

    def search_and_set(pid, shape, dtype):
        if pid not in start_processors:
            return

        if pid not in ignores:
            if CONFIG_KEY_OUT_QUEUE_SHAPE in config[pid]:
                if shape != config[pid][CONFIG_KEY_OUT_QUEUE_SHAPE]:
                    raise ValueError(f'{pid}.{CONFIG_KEY_OUT_QUEUE_SHAPE} have multiple shapes: {shape} and {config[pid][CONFIG_KEY_OUT_QUEUE_SHAPE]}')
            else:
                config[pid][CONFIG_KEY_OUT_QUEUE_SHAPE] = shape

            if CONFIG_KEY_OUT_QUEUE_DTYPE in config[pid]:
                if dtype != config[pid][CONFIG_KEY_OUT_QUEUE_DTYPE]:
                    raise ValueError(f'{pid}.{CONFIG_KEY_OUT_QUEUE_DTYPE} have multiple dtypes: {dtype} and {config[pid][CONFIG_KEY_OUT_QUEUE_DTYPE]}')
            else:
                config[pid][CONFIG_KEY_OUT_QUEUE_DTYPE] = dtype

        if pid in producer_consumers_dict:
            for consumer_pid in producer_consumers_dict[pid]:
                search_and_set(consumer_pid, shape, dtype)

    for header_pid in headers:
        valid_camera_list = config[CONFIG_KEY_VALID_CAMERAS]
        firset_valid_camera = valid_camera_list[0]
        shape = (len(valid_camera_list), len(firset_valid_camera[2]), *firset_valid_camera[2][0])
        dtype = np.dtype('uint8')
        search_and_set(header_pid, shape, dtype)
    
    def modify_dtype(pid):
        if CONFIG_KEY_OUT_QUEUE_DTYPE in config[pid] and isinstance(config[pid][CONFIG_KEY_OUT_QUEUE_DTYPE], str):
            config[pid][CONFIG_KEY_OUT_QUEUE_DTYPE] = np.dtype(config[pid][CONFIG_KEY_OUT_QUEUE_DTYPE])
    
    for pid in start_processors:
        modify_dtype(pid)


def get_processors_with_queue(start_processors=None):
    freeze_support()

    if start_processors is None:
        start_processors = set()

    # processor视角的in/out
    processor_in_out_dict = dict()
    # queue视角的producer/consumer
    producer_consumers_dict = dict()
    # queue视角的(producer, consumer): get_array_from_high_performance_queue
    consumer_get_array_from_producer_dict = dict()
    for processor_id, config_dict in config.items():
        if CONFIG_KEY_IN_QUEUE in config_dict and processor_id in start_processors:
            if processor_id in processor_in_out_dict:
                in_set, out_set = processor_in_out_dict[processor_id]
            else:
                in_set = set()
                out_set = set()
                in_out_list = (in_set, out_set)
                processor_in_out_dict[processor_id] = in_out_list

            for in_pid_info in config_dict[CONFIG_KEY_IN_QUEUE]:
                in_pid, is_get_array = config_formatter.parse_in_queue(in_pid_info)
                in_set.add(in_pid)
                consumer_get_array_from_producer_dict[(in_pid, processor_id)] = is_get_array

                if in_pid in producer_consumers_dict:
                    consumers_list = producer_consumers_dict[in_pid]
                else:
                    consumers_list = []
                    producer_consumers_dict[in_pid] = consumers_list
                consumers_list.append(processor_id)

            for in_pid in in_set:
                if in_pid in processor_in_out_dict:
                    processor_in_out_dict[in_pid][1].add(processor_id)
                else:
                    in_set = set()
                    out_set = set()
                    out_set.add(processor_id)
                    in_out_list = (in_set, out_set)
                    processor_in_out_dict[in_pid] = in_out_list

    set_config_by_topology(producer_consumers_dict, start_processors)

    producer_queue_info_dict = dict()
    consumer_in_out_queue_dict = dict()
    for producer_pid, consumers_list in producer_consumers_dict.items():
        if producer_pid in start_processors:
            all_consumer_ignore_array = True
            for consumer_pid in consumers_list:
                if consumer_get_array_from_producer_dict[(producer_pid, consumer_pid)]:
                    all_consumer_ignore_array = False
                    break

            verbose = 0
            if config[producer_pid][CONFIG_KEY_OUT_QUEUE_SHAPE][0] == 0:
                raise ValueError(f'processor: {producer_pid} has a zero sized output')
            hp_queue = high_performance_queue.One2ManyQueue(config[producer_pid][CONFIG_KEY_OUT_QUEUE_SHAPE], config[producer_pid][CONFIG_KEY_OUT_QUEUE_DTYPE], len(consumers_list), broadcast=True, ignore_array=all_consumer_ignore_array, verbose=verbose)
            out_pid2consumer_idx_dict = dict()
            for idx, (consumer_pid, consumer_queue) in enumerate(zip(consumers_list, hp_queue.get_consumers())):
                consumer_in_out_queue_dict[(producer_pid, consumer_pid)] = consumer_queue
                out_pid2consumer_idx_dict[consumer_pid] = idx
            producer_queue_info_dict[producer_pid] = (out_pid2consumer_idx_dict, hp_queue)

    pid_processor_dict = gen_pid_processor_dict()

    pid_without_queue_dict = {pid: processor for pid, processor in pid_processor_dict.items() if pid not in processor_in_out_dict}

    process_list = list()
    for key, (in_set, out_set) in processor_in_out_dict.items():
        if key in start_processors:
            in_queue_list = []
            for pid in in_set:
                if pid in start_processors:
                    in_queue_list.append((pid, consumer_in_out_queue_dict[(pid, key)]))
            out_queue_info = producer_queue_info_dict.get(key, None)

            p = pid_processor_dict[key](in_queue_list=in_queue_list, out_queue_info=out_queue_info)
            process_list.append(p)

    for pid, processor in pid_without_queue_dict.items():
        if pid in start_processors:
            p = processor()
            process_list.append(p)

    return process_list


if __name__ == '__main__':
    process_list = get_processors_with_queue()

    for t in process_list:
        t.start()

    for t in process_list:
        t.join()
