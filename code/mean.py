def get_mean_std(value_scale, data_type):

    if data_type == 'event':
        mean = [0.504413, 0.504413, 0.504413]
        std = [0.06928615, 0.06928615, 0.06928615]
    elif data_type == 'frame':
        mean = [0.22616537, 0.22616537, 0.22616537]
        std = [0.118931554, 0.118931554, 0.118931554]

    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]
    return mean, std