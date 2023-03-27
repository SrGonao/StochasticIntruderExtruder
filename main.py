# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:10:11 2023

@author: g-spa
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pickle
import time
pio.renderers.default = "browser"
 
import poreNetwork
import importlib
importlib.reload(poreNetwork) 

    
faujasite={"n_max":{0:4,1:4},"type_accept":{1:0,0:1},"n_directions":{0:4,1:4},"directions":{0:[1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)],
                                                                                            1:[1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)],
                                                                                            2:[-1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],
                                                                                            3:[-1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3)]},"rotation":{0:1,1:-1},"reverse":{0:0,1:1,2:2,3:3,4:4}}
zif8={"n_max":{0:8},"type_accept":{0:0},"n_directions":{0:8},"directions":{0:[1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],
 1:[1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)],
 2:[1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)],
 3:[-1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],
 4:[1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3)],
 5:[-1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)],  
 6:[-1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)],
 7:[-1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3)]},"rotation":{0:1},"reverse":{0:7,7:0,1:5,5:1,2:6,6:2,3:4,4:3}}

zsm22={"n_max":{0:2},"type_accept":{0:0},"n_directions":{0:2},"directions":{0:[0,0,1],
 1:[0,0,-1]},"rotation":{0:1},"reverse":{0:1}}

mfi={"n_max":{0:4,1:4},"type_accept":{0:1,1:0},"n_directions":{0:4,1:4},
     "directions":{0:[0,0,1],
                 1:[0,0,-1],
                 2:[0,1/np.sqrt(2),0],
                 3:[1/np.sqrt(2),0,0]},"rotation":{0:1,1:-1},"reverse":{0:0,1:1,2:2,3:3}}



topology = zif8
first_node = poreNetwork.Node(0,[0,0,0],topology)
network = poreNetwork.Network(topology)


#Time how long it takes to grow the network
start = time.time()
network.grow(10)
end = time.time()
print("Time to grow: "+str(end-start))  

for node in network.nodes.values():
    print("Node: "+str(node.nid)+" has these neighbours:")
    print(len(node.n_list))
dataobjects = []    
color = ["black","red"]
surf = network.surface
for node in network.nodes.values():
    if node in surf:
        color = "black"
    else:
        color = "red"
    dataobjects.append(go.Scatter3d(x=[node.position[0]],y=[node.position[1]],z=[node.position[2]],marker=dict(size=12,
                              color=color)))
    #for n in node.n_list:
    #    dataobjects.append(go.Scatter3d(x=[node.position[0],n.position[0]],y=[node.position[1],n.position[1]],z=[node.position[2],n.position[2]],line=dict(color=color)))
fig = go.Figure(data=dataobjects)
fig.write_html("plot.html", auto_open=True) 

#Ising model



for n in [2,3,4,5,6,7,8,9,10,15,20,25]:
    topology = zif8
    first_node = poreNetwork.Node(0,[0,0,0],topology)
    network = poreNetwork.Network(topology)
    network.grow(n)

    for p in np.linspace(-7,3,num=50):
        ising = poreNetwork.Ising(network)
        nsteps = 700000
        pressures = np.ones(nsteps)*p
        energies, fillings = ising.run(nsteps,pressures)
        plt.scatter(p,np.mean(fillings[:-100])/len(network.nodes),color="black")
    plt.title(f"Filling fraction vs pressure, growth steps = {n}")
