"""
Utility module for deep learning
"""
import math


def average(num: list):
    """
    calculates the average of numbers in a list
    :param num: list of numbers
    :return: average of numbers
    """
    if isinstance(num, list):
        avg = sum(num) / len(num)
        return avg
    else:
        return num


def variance(num: list) -> float:
    """
    calculates variance of numbers in a list
    :param num: list of numbers
    :return: variance of numbers
    """
    if isinstance(num, list):
        n = len(num)
        var = sum((x - average(num)) ** 2 for x in num) / n
        return var
    else:
        return 0


def stddev(num: list) -> float:
    """
    calculates the standard deviation of numbers in a list
    :param num: list of numbers
    :return: standard deviation of numbers
    """
    if isinstance(num, list):
        var = variance(num)
        std_dev = math.sqrt(var)
        return std_dev
    else:
        return 0


def position(long, short) -> int:
    """
    method to interpret results predicted from the trained neural network to stockmarket position
    :param long: predicted result for long position
    :param short: predicted result for short position
    :return: 1/0/-1 for LONG/NO-POSITION/SHORT
    """
    if long == 1:
        return 1
    elif short == 1:
        return -1
    elif long == 0 and short == 0:
        return 0
