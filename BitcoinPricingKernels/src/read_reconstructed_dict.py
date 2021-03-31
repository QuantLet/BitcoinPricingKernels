#with open('mycsvfile.csv', 'wb') as f:  # You will need 'wb' mode in Python 2.x
#    w = csv.DictWriter(f, my_dict.keys())
#    w.writeheader()
#    w.writerow(my_dict)
import csv

out = {}
# keynames in w.fieldnames
with open('mycsvfile.csv', 'r') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictReader(f)
    for row in w:
        print(row)
    


def save_dict(moneyness, gdom, gstar, spd, curr_day, tau, maturitydate_char, fname):
    """
    gdom is spd domain
    gstar is estimated hd
    curr_day is current day
    tau is time until maturity
    """

    if os.path.isfile(fname):
        _dict = load_dict(fname)
    else:
        _dict = {}

    curr_key = curr_day + '_' + str(tau) + '_' + str(maturitydate_char)
    _dict[curr_key] =  (moneyness, gdom, gstar, spd)
    with open(fname, 'wb') as f:
        pickle.dump(_dict, f)
    return _dict

def load_dict(fname):
    with open(fname, 'rb') as f:
        _dict = pickle.load(f)
    return _dict


def save_a(a):
    out = {}
    for ele in a.items()



# This worked below!

# Save
import pickle
with open('test.txt', 'wb') as f:
    pickle.dump(a, f)

# load
import pickle
# This is the fucking right imported one!!
with open('test.txt', 'rb') as f:
    d = pickle.load(f)
    print(d.keys())
