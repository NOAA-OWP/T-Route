def normalize(val, max, min, target_max=1, target_min=0):
    return (val - min) / (max - min) * (target_max - target_min) + target_min


def denormalize(val, max, min, target_max=1, target_min=0):
    return val + target_min / (target_max + target_min) * (max - min)
