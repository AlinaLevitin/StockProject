import math


def average(num):
    avg = sum(num) / len(num)
    return avg


def variance(num):
    n = len(num)
    var = sum((x - average(num)) ** 2 for x in num) / n
    return var


def stddev(num):
    var = variance(num)
    std_dev = math.sqrt(var)
    return std_dev
