def oos_rsq(pred, y):
    return 1 - (y - pred).square().sum() / y.square().sum()