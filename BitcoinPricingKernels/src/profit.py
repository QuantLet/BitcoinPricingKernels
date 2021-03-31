import pandas as pd
import pickle
import matplotlib.pyplot as plt

"""
Todo for the Profits and Strategies:
IMPORTANT: Check if profit function is always below 0 for some days!!


"""
print('check todo')

file = open("out/profit.txt",'rb')
profit = pickle.load(file)
plt.plot(profit)
plt.title('Trading Profit')
plt.xlabel('Trade #')
plt.savefig('pricingkernel/plots/profit.png')

df = pd.DataFrame(profit)
df.to_csv('out/profit.csv')


file.close()