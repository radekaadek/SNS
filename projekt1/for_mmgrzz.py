import time
import numpy as np
from numba import jit
from datetime import date
import math

@jit
def julday(y,m,d,h=0):
    '''
    Simplified Julian Date generator, valid only between
    1 March 1900 to 28 February 2100
    '''
    if m <= 2:
        y = y - 1
        m = m + 12
    jd = np.floor(365.25*(y+4716))+np.floor(30.6001*(m+1))+d+h/24-1537.5
    return jd

a = julday(2020, 1, 1)

def math_julday(y,m,d,h=0):
    '''
    Simplified Julian Date generator, valid only between
    1 March 1900 to 28 February 2100
    '''
    if m <= 2:
        y = y - 1
        m = m + 12
    jd = math.floor(365.25*(y+4716))+math.floor(30.6001*(m+1))+d+h/24-1537.5
    return jd

b = math_julday(2020, 1, 1)

def calculate_difference_julday(y1, m1, d1, y2, m2, d2):
    julian1 = julday(y1, m1, d1)
    julian2 = julday(y2, m2, d2)
    return abs(julian1 - julian2)

def calculate_difference_standard(y1, m1, d1, y2, m2, d2):
    date1 = date(y1, m1, d1)
    date2 = date(y2, m2, d2)
    return abs((date2 - date1).days)

def calculate_difference_math(y1, m1, d1, y2, m2, d2):
    julian1 = math_julday(y1, m1, d1)
    julian2 = math_julday(y2, m2, d2)
    return abs(julian1 - julian2)

# Generate 1000 different dates
dates = []
for i in range(1000):
    year1 = np.random.randint(1901, 2099)
    month1 = np.random.randint(1, 13)
    day1 = np.random.randint(1, 28)
    dates.append((year1, month1, day1))
diffs1 = []
diffs2 = []
diffs3 = []
# Measure time for julday function
start_time = time.time()
for year1, month1, day1 in dates:
    for year2, month2, day2 in dates:
        difference_julday = calculate_difference_julday(year1, month1, day1, year2, month2, day2)
        diffs1.append(difference_julday)
julday_time = time.time() - start_time

# Measure time for standard library function
start_time = time.time()
for year1, month1, day1 in dates:
    for year2, month2, day2 in dates:
        difference_standard = calculate_difference_standard(year1, month1, day1, year2, month2, day2)
        diffs2.append(difference_standard)
standard_time = time.time() - start_time

# Measure time for math function
start_time = time.time()
for year1, month1, day1 in dates:
    for year2, month2, day2 in dates:
        difference_math = calculate_difference_math(year1, month1, day1, year2, month2, day2)
        diffs3.append(difference_math)
math_time = time.time() - start_time


print(f"diffs are equal: {diffs1==diffs2==diffs3}")
print("Time taken for julday function:", julday_time)
print("Time taken for standard library function:", standard_time)
print("Time taken for math function:", math_time)

