import numpy as np
import math

import matplotlib.pyplot as plt
from IPython import display

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("etfs/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])

def plot_custom(data, actions, cat, save, index):
    
	print(len(data))
	print(len(actions))
	print(len(cat))

	plt.clf()
	plt.title('Training...')
	plt.xlabel('Number of Trade Days')
	plt.ylabel('Stock Value')

	plt.plot(data)
	#plt.plot(mean_scores)
	# use colormap
	colormap = np.array(['b', 'r', 'g'])
	# depict illustration
	plt.scatter(range(len(data)), data, c=colormap[cat])

	plt.ylim(ymin=0)
	#plt.text(len(scores)-1, scores[-1], str(scores[-1]))
	#plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
	plt.show(block=False)
	
	if save:
		plt.savefig(f'figures/model_CEF_sbt_{str(index)}.png')
	
	plt.pause(.1)
	
	
def plot_progress(scores, mean_scores, save, index):
	plt.ion()
	display.clear_output(wait=True)
	display.display(plt.gcf())
	plt.clf()
	plt.title('Training...')
	plt.xlabel('Number of Games')
	plt.ylabel('Score')


	plt.plot(scores)
	plt.plot(mean_scores)

	plt.ylim(ymin=0)
	plt.text(len(scores)-1, scores[-1], str(scores[-1]))
	plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
	plt.show(block=False)

	if save:
		plt.savefig(f'figures/model_CEF_epc_{str(index)}.png')

	plt.pause(.1)