import numpy as np
import matplotlib.pyplot as plt


def p_hat(p, y):
    # p - предсказанная моделью вероятность класса 1
    # y - реальный класс (0 или 1)

    # функция должна вернуть вероятность класса y, при предсказании
    # модели p
    if y==1:
        return p
    else:return (1-p)

def log_p_hat(p, y):
    return np.log(p_hat(p, y))

def likelihood(ps, ys):
    # ps - предсказанные вероятности класса 1 моделю для N объектов
    # ys - реальные классы N объектов

    # функция должна использовать p_hat и возвращать правдоподобие

    likelihood_ = None
    # < YOUR CODE STARTS HERE >
    probs = [p_hat(p,y) for (p,y) in zip(ps, ys)]
    likelihood_ = np.prod(probs)
    # < YOUR CODE ENDS HERE >
    return likelihood_


def loglikelihood(ps, ys):
    # ps - предсказанные вероятности класса 1 моделю для N объектов
    # ys - реальные классы N объектов
    # функция должна использовать log_p_hat и возвращать логарифм правдоподобия
    # (на количество делить не нужно)
    p = np.clip(ps, a_min=1e-6, a_max=1 - 1e-6)
    log_probs = [log_p_hat(p, y) for (p, y) in zip(ps, ys)]
    loglikelihood_ = None
    # < YOUR CODE STARTS HERE >
    loglikelihood_ = np.sum(log_probs)
    # < YOUR CODE ENDS HERE >
    return loglikelihood_

test_ps = [0.1, 0.2, 0.3, 0.4]
test_ys = [0, 1, 0, 1]
assert likelihood(test_ps, test_ys) == 0.0504
assert np.allclose(np.log(likelihood(test_ps, test_ys)), loglikelihood(test_ps, test_ys))
print("Tests passed!")