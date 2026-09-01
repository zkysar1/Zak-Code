import json
import statistics


def mean(values):
    total = 0
    for v in values:
        total = total + v
    return total / len(values)


def summarize(d):
    return {'n': len(d), "mean": mean(d)}
