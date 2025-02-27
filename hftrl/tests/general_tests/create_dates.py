from hftrl.env.utils.dategeneration import get_random_date


# {1: {'max_p': 26.0, 'min_p': 78.0},
#  2: {'max_p': 14.0, 'min_p': 65.0},
#  3: {'max_p': 2.0, 'min_p': 50.0},
#  4: {'max_p': 47.0, 'min_p': 6.0},
#  5: {'max_p': 5.0, 'min_p': 56.0},
#  8: {'max_p': 17.0, 'min_p': 26.0},
#  9: {'max_p': 35.0, 'min_p': 4.0},
#  10: {'max_p': 6.0, 'min_p': 105.0},
#  11: {'max_p': 9.0, 'min_p': 38.0},
#  12: {'max_p': 54.0, 'min_p': 1.0},
#  15: {'max_p': 4.0, 'min_p': 62.0},
#  16: {'max_p': 3.0, 'min_p': 53.0},
#  17: {'max_p': 43.0, 'min_p': 7.0},
#  18: {'max_p': 6.0, 'min_p': 39.0}}

trading_days = {
    "year": 2024,
    "month_day":{4:[1,2,3,4,5,8,9,10,11,12,15,16,17,18]},
    "preload_hour":4,
    "start_hour" : 12,
    "preload_minute": 1,
    "end_hour" : 16,
    "minutes_to_add" : 30}

pre_d, start_d, end_d = get_random_date(**trading_days)

print(pre_d)
print(start_d)
print(end_d)