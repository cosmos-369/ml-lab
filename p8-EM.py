import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

data = pd.read_csv('./data/em.csv')
print("input data and shape")
print(data.shape)
data.head()

f1=data['v1'].values
f2=data['v2'].values
x=np.array(list(zip(f1,f2)))

print("X",x)
print('Graph for whole dataset')
plt.scatter(f1,f2,c='black',s=7)
plt.show()

kmeans=KMeans(20,random_state=0)
labels=kmeans.fit(x).predict(x)
Centroids=kmeans.cluster_centers_

print("labels  ",labels)
print("Centroids  ",Centroids)
print("Graph using KMeans algorithm")

plt.scatter(x[:,0],x[:,1],c=labels,s=40,cmap='viridis')
plt.scatter(Centroids[:,0],Centroids[:,1],marker='*',s=200,c='#050505')
plt.show()


gmm=GaussianMixture(n_components=3).fit(x)
labels=gmm.predict(x)
probs=gmm.predict_proba(x)
size=10*probs.max(1)**3
print('Graph using EM algorithm')
plt.scatter(x[:,0],x[:,1],c=labels,s=size,cmap='viridis')
plt.show()