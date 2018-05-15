from seq2seq.util.log import LogCollection

def name_parser(filename):
    return filename.split('/')[-2]

log = LogCollection()
log.add_log_from_folder('dumps', ext='.dump', name_parser=name_parser)

print(log.log_names)

def func(input_str):
    if 'full_focus' in input_str and 'hard' not in input_str and 'baseline' not in input_str:
        return True
    return False

def f128_256(input_str):
    if 'E64xH128' in input_str:
        return True
    return False

def data_name_parser(input_str):
    return ''


def func2(input_str):
    if 'new_compositions' in input_str:
        return True
    return False

def color_group(input_str):
    if 'baseline' in input_str:
        c = 'b'
    elif 'hard' in input_str:
        c = 'm'
    else:
        c = 'g'

    if 'pre_rnn' in input_str:
        l = ':'
    elif 'full_focus' in input_str:
        l = '-'
    elif 'post_rnn' in input_str:
        l = '--'

    return c+l

log.plot_metric('acc', restrict_model=f128_256, restrict_data=func2, data_name_parser=data_name_parser, color_group=color_group)
