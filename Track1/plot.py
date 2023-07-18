import numpy as np
from matplotlib import pyplot as plt

ws_list = list(range(1,13,2))
ws_list = [0] + ws_list

polas = [0.8,0.823,0.82,0.818,0.819,0.823,0.819]
itsts = [None, 0.793,0.782,0.778,0.78,0.778,0.771]
emps = [None, 0.718,0.729,0.737,0.734,0.733,0.734]

plt.plot(ws_list, polas, c='red', label='Polarity')
plt.plot(ws_list, itsts, c='green', label='Intensity')
plt.plot(ws_list, emps, c='blue', label='Empathy')
plt.scatter(ws_list, polas, c='red')
plt.scatter(ws_list, itsts, c='green')
plt.scatter(ws_list, emps, c='blue')
plt.xlabel('Window Size')
plt.ylabel('Pearson Coefficient')
plt.yticks(np.arange(0.7,0.84,0.02))
plt.legend(loc='best')

plt.show()