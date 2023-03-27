#This is the network class that will be used to create the network
#The network will be import as a module in the main program
import json
import numpy as np
from scipy.spatial import distance
import pickle
from numba import jit

class Network:
    def __init__(self,topology): 
        self.topology = topology
        first_node = Node(0,[0.0,0.0,0.0],topology)
        self.surface = [first_node]
        self.nodes = {first_node.hash:first_node}
        

    def add_node(self,node):
        node.nid = len(self.nodes)
        hash = node.hash
        self.nodes[hash] = node
        
        
    

    def grow(self, n):
        for i in range(n):
            nodes = list(self.nodes.values())
            for node in nodes:
                for direction in node.free_directions:
                    ntype = self.topology["type_accept"][node.ntype]
                    nposition = node.position + self.topology["directions"][direction]*self.topology["rotation"][ntype]
                    new_node = Node(ntype,nposition,self.topology)
                    new_node.nid = len(self.nodes)
                    node.add_neighbour(new_node)
                    if new_node.hash not in self.nodes:                 
                        self.add_node(new_node)                       
                    else:
                        to_merge = self.nodes[new_node.hash]
                        to_merge.merge(new_node)
        self.update_surface()
        self.update_nodes()
        

    def update_surface(self):
        surface = []
        for node in self.nodes.values():
            if node.n_number < self.topology["n_directions"][node.ntype]:
                    surface.append(node)
         
        self.surface = surface

    def update_nodes(self):
        for node in self.nodes.values():
            node.surface = False
            if node in self.surface:
                node.surface = True
            
    def condition_network(self,condition):
        #Remove from the newtork nodes that don't match a condition
        for node in self.nodes.values():
            if not condition(node):
                self.remove_node(node)

    def remove_node(self,node):
        #Remove a node from the network
        for nnode in node.n_list:
            nnode.remove_neigh(node)
        del self.nodes[node.nid]

    
       
    def save_network(self,filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_network(self,filename):
        with open(filename, 'rb') as f:
            self = pickle.load(f)
        return self
    
    def save_network_json(self,filename):
        #Save the nodes in the network and the position of the neighbours
        nodes = self.nodes
        nodesjson = {node.hash:node.NodetoJson() for node in nodes.values()}
        
        with open(filename, 'w') as f:
            json.dump(nodesjson,f, indent=4,separators=(',', ': '))

class Node:
  
    def __init__(self,ntype,position,topology):
        
        self.ntype = ntype
        self.position = np.array(position)
        self.free_directions = np.arange(topology["n_directions"][ntype])
        self.topology=topology
        self.n_number=0
        self.n_list=[]
        self.surface=False
        self.state = 0  
        #Hash is used to identify the node, its a string of the position with 3 decimals
        self.hash = str(np.round(self.position,3))
        self.nid = 0 
    def NodetoJson(self):
        #Returns a json serializable dictionary with the position of the node and the position of the neighbours
        nlist = [n.position.tolist() for n in self.n_list]
        return {"position":self.position.tolist(),"n_list":nlist}

    def add_neighbour(self,node):
        #Add a neighbour to the node
        if node not in self.n_list:
            self.n_list.append(node)
            node.n_list.append(self)
            self.n_number=self.n_number+1
            node.n_number=node.n_number+1
            direction = self.neighbour_direction(node)
            self.free_directions = np.setdiff1d(self.free_directions,direction)
            
            reversed=self.topology["reverse"][direction]
            node.free_directions = np.setdiff1d(node.free_directions,reversed)
            
            
        
    def merge(self,node):
        #Merge two nodes

        nlist = node.n_list
        freedirections = node.free_directions
        for nnode in nlist:
            self.n_list.append(nnode)
            direction = self.neighbour_direction(nnode)
            self.free_directions = np.setdiff1d(self.free_directions,direction)    
        self.n_number = self.n_number+node.n_number

    def remove_neigh(self,node):
        if node in self.n_list:
            self.n_list.remove(node)
            self.n_number=self.n_number-1
            direction = self.neighbour_direction(self,node)
            self.free_directions = np.append(self.free_directions,direction)

    def neighbour_direction(self,node):
        distance = np.round(node.position-self.position,3)
        distances = self.topology["directions"].values()
        for d in distances:
            d[0] = round(d[0],3)
            d[1] = round(d[1],3)
            d[2] = round(d[2],3)
        distances = list(distances)
        #what index is the list distance in the list of distances lists
        direction = distances.index(distance.tolist())
        return direction

    
    

class Ising():

    #Use the network to simulate an Ising model
    def __init__(self,network):
        self.network = network
        self.nodes = network.nodes
        self.surface = network.surface
        self.max_neighbours = network.topology["n_max"][0] #for now only one type of node
        #TO DO: make it work for more than one type of node on the network
        #create a connectivity matrix
        self.connectivity = np.zeros((len(self.nodes),len(self.nodes)))
        for nid,node in enumerate(self.nodes.values()):
            for nnode in node.n_list:
                self.connectivity[node.nid,nnode.nid]=1
                self.connectivity[nnode.nid,node.nid]=1
        self.connectivity = self.connectivity.astype(int)
        #create a matrix with the states of the nodes
        self.states = np.zeros(len(self.nodes))
        #create a index to identify the nodes
        self.index = np.arange(len(self.nodes))
        #create a index of the surface nodes
        self.surface_index = np.array([node.nid for node in self.surface])
        #create a index of the bulk nodes
        self.bulk_index = np.setdiff1d(self.index,self.surface_index)
        #set the states of the surface nodes as 1
        self.states[self.surface_index]=1
        self.wet_neighbours = self.get_number_wet_of_neighbours()
    #use numba to speed up the code
    #@jit(nopython=True)
    def get_number_wet_of_neighbours(self):
        #get the number of wet neighbours of each node
        return np.dot(self.states,self.connectivity)

    def run(self,nsteps,pressures,T=1):
        n_nodes = len(self.nodes)
        #create a random number for each step
        random_number = np.random.rand(nsteps)
        random_idx = np.random.choice(self.bulk_index,nsteps)
        #get the energy of the network
        interaction_energy = self.get_networkEnergy()
        field_energy = self.get_pressureEnergy(pressures[0])
        energy = interaction_energy + field_energy
        self.energy = energy
        energies = [energy]
        filled = self.get_networkFilled()
        sum_states = [filled]
        # print(self.get_networkEnergy(pressure))
        self.T = T
        for n in range(nsteps):
            #update the states of the nodes
            changedStates,deltaE = self.updateStates(random_number[n],random_idx[n],pressures[n])
           
            #update the energy of the network
            energy = self.energy + deltaE
            self.energy=energy
            #print(energy,changedStates,self.get_networkEnergy()+self.get_pressureEnergy(pressures[n]))
            energies.append(self.energy)
            filled = sum_states[-1] + changedStates
            sum_states.append(filled)
            # print(self.get_networkEnergy(pressure))
        return energies,sum_states
    
    #@jit(nopython=True)
    def updateStates(self,random,idx,pressure):
        dE = self.deltaE(idx,pressure)
        T = self.T
       # print(dE)
        if dE < 0 or random < np.exp(-dE/T):
            change = -2*self.states[idx]+1
            self.states[idx] = 1-self.states[idx]
          #  print("Changed",idx,self.states[idx],change)
            neighbour_idxs = self.connectivity[idx,:] 
            neighbours = self.index[neighbour_idxs==1]
          #  print(neighbours)
            for ni in neighbours:
                self.wet_neighbours[ni] = self.wet_neighbours[ni] + change
        else:
            change = 0
            dE = 0
        return change,dE

    #@jit(nopython=True)
    def deltaE(self,idx,pressure):
        current_energy = self.energy
        #get the state of the node
        current_state = self.states[idx]
        new_state = 1-current_state
        #get the number of wet neighbours of the node
        current_wet_neighbours = self.wet_neighbours[idx]
        dE = -(2*new_state-1)*(2*current_wet_neighbours-self.max_neighbours)-(2*new_state-1)*pressure
        return 2*dE

    
    #@jit(nopython=True)
    def get_networkEnergy(self):
        #get the energy of the network
        energy = np.dot(-(2*self.states-1),(2*self.wet_neighbours-self.max_neighbours))
        return energy   

    #@jit(nopython=True)
    def get_pressureEnergy(self,pressure):
        #get the energy of the network
        energy = np.sum(-(2*self.states-1)*pressure)
        return energy 
    
    #@jit(nopython=True)
    def get_networkFilled(self):
        filled = np.sum(self.states)
        return filled
    
    
    
            
    

