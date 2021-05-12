def fix_tuple(target_cfg):
    target_cfg
    for key, value in target_cfg.items():
        if type(value) == str:
            if value[0] == '(' and value[-1] == ')':
                new_value = eval(value)
                target_cfg[key] = new_value