from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def scatterplot3D(x, y, z):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x, y, z, c='skyblue', s=60)
	ax.view_init(30, 185)
	plt.show()


def surface_plot(x, y, z):
	# Make the plot
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis, linewidth=0.2)
	plt.show()


if __name__ == "__main__":

	# get data
	polygraphy_file = "/Users/wassilyminkow/Data/Hamster_Data/hamster_without_torpor/hamster_without_torpor.csv"
	data = pd.read_csv(polygraphy_file)
	gap = 6*12
	temperature = np.asanyarray(((data["dG"]).rolling(gap).mean().dropna()))
	acceleration = np.asanyarray(((data["T"]).rolling(gap).mean().dropna()))

	# prepare data
	gap = 6*24*2
	drop = temperature.shape[0]%gap
	n = (temperature.shape[0]-drop)/gap
	z = np.concatenate([np.repeat(np.arange(n), gap), np.repeat(n, drop)])
	df = pd.DataFrame({'X': temperature, 
    	               'Y': acceleration, 
        	           'Z': z})

	# test surface plot
	scatterplot3D(df.iloc[:, 2], df.iloc[:, 0], df.iloc[:, 1])
