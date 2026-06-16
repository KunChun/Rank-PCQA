import scipy
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):

    # inx = np.negative(np.divide(X - bayta3, np.abs(bayta4)))
    #
    # if inx >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
    #     val = 1.0 / (1 + np.exp(-inx))
    # else:
    #     val = np.exp(inx) / (1 + np.exp(inx))

    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic

def cal_metrics(y_test, y_output):
    y_output_logistic = fit_function(y_test, y_output)
    test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
    test_SROCC = stats.spearmanr(y_output, y_test)[0]
    test_RMSE = np.sqrt(((y_output_logistic - y_test) ** 2).mean())
    test_KROCC = scipy.stats.kendalltau(y_output, y_test)[0]

    return test_PLCC, test_SROCC, test_KROCC, test_RMSE