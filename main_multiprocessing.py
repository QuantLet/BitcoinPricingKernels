import multiprocessing 
import os
from main import run
from src.brc import BRC
import datetime

if __name__ == '__main__':

    # Blacklist existing output
    existing_files = os.listdir('pricingkernel/plots')

    brc = BRC()
    run_dates = [brc.first_day]
    curr_date = brc.first_day
    while curr_date < brc.last_day:
        curr_date += datetime.timedelta(1)
        out = next((s for s in existing_files if curr_date.strftime('%Y-%m-%d') in s), None) 
        print(out)
        if out is None:
            run_dates.append(curr_date)

    multiprocessing.set_start_method('spawn')
    n_cpu = min(multiprocessing.cpu_count(), 4)
    p = multiprocessing.Pool(processes = n_cpu)
    p.map(run, run_dates)
