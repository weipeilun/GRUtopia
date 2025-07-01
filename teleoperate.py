# encoding: utf-8

import logging
from system.tools import starter
from system.constants import CONFIG_KEY_ISAAC_SIM, CONFIG_KEY_VIDEO_SERVER, CONFIG_KEY_TEST, CONFIG_KEY_MOCK
from system.utils.path_util import get_relative_path
import yaml

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    all_xbq_processors = {
        CONFIG_KEY_ISAAC_SIM,
        CONFIG_KEY_VIDEO_SERVER,
        CONFIG_KEY_TEST,
        CONFIG_KEY_MOCK
    }

    with open(get_relative_path('system/config/start_teleoperate.yml'), 'r') as f:
        processor_config = yaml.safe_load(f)

    processor_to_start = set()
    for pid in processor_config['processors']:
        if pid in all_xbq_processors:
            processor_to_start.add(pid)
        else:
            logging.info(f'Processor {pid} not supported.')

    thread_list = starter.get_processors_with_queue(processor_to_start)

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()
