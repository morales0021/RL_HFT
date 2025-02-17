import random
import pytz
from datetime import datetime, timedelta

def get_random_date(year: int, month_day: dict, preload_hour: int, preload_minute: int,
                    start_hour: int, end_hour: int, minutes_to_add: int
                    ):
    
    m = random.choice(list(month_day.keys()))
    d = random.choice(month_day[m])
    rand_hour = random.choice(list(range(start_hour, end_hour)))
    rand_min = random.choice(list(range(0,59)))

    time_delta = timedelta(minutes=minutes_to_add)

    pre_date = datetime(
        year = year, month = m, day = d,
        hour = preload_hour, minute = preload_minute, tzinfo = pytz.utc)
    
    start_date = datetime(
        year = year, month = m, day = d,
        hour = rand_hour, minute = rand_min, tzinfo = pytz.utc)
    
    end_date = start_date + time_delta

    return pre_date, start_date, end_date