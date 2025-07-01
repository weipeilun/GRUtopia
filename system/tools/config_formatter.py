

def parse_in_source(in_source, do_validate):
    pid = in_source
    ip = 'localhost'
    validate = do_validate
    if ':' in in_source:
        config_list = in_source.split(':')
        pid = config_list[0].strip()

        if len(config_list) >= 2:
            ip = config_list[1].strip()

        if len(config_list) >= 3:
            validate = config_list[2].strip().lower() == 'true'
    return pid, ip, validate


def parse_in_queue(in_queue):
    pid = in_queue
    get_array_from_high_performance_queue = True
    if ':' in in_queue:
        config_list = in_queue.split(':')
        pid = config_list[0].strip()

        if len(config_list) >= 2:
            get_array_from_high_performance_queue = not (config_list[1].strip().lower() == 'false')
    return pid, get_array_from_high_performance_queue
