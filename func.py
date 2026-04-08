# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree

## definitions
dd = 1/np.sqrt(3) + 0.01  ## threshold for connections
eta= 1E-5

# Functions

def blank_rect_lattice(size_zz, size_ac, origin):
    '''Generates an initial graphene segment, and returns coordinates, sublattices, and distance matrix'''
    #size_(zz/ac) = length of the blank lattice in the zigzag and armchair directions, ie size*size square
    #origin = starting point for generation, a vector so a 1*2 matrix. Usually (0,0)
    
    
    cellsubs=np.array((1, -1, 1, -1))
    offsets = np.array( [  [0.5, -1/(2*np.sqrt(3))],[0, 0], [0, 1/np.sqrt(3)], [0.5, np.sqrt(3)/2] ])
    
    vectorzz=np.array([1,0])
    vectorac=np.array([0, np.sqrt(3)])

    
    ilist, jlist = np.arange(size_zz), np.arange(size_ac)
    isize, jsize=len(ilist), len(jlist)
    ilist, jlist = np.meshgrid(ilist, jlist)
    ilist, jlist = ilist.flatten(), jlist.flatten()

    allcoords = np.empty([len(offsets)*len(ilist), 2])
    
    subs = np.empty(len(offsets)*len(ilist))
        
    for a, (i, j) in enumerate(zip(ilist, jlist)):
        for b, off in enumerate(offsets):
            allcoords[len(offsets)*a+b] = np.array(origin) + i*np.array(vectorzz) + j*np.array(vectorac) + off
            subs[len(offsets)*a+b] = cellsubs[b]
        
    dmat = distance_matrix(allcoords, allcoords)

    return allcoords, subs, dmat   


# Function for calculating the surface GFs from the leads (left and right)
# Not Rubio-Sancho, but should be equivalent..
# Doubles the lead each time, and gets both lead SGFs at the same time
def surface(H_00, V1, V2, epsilon, energy, eta):
    ii = np.identity(len(H_00), dtype = 'complex128')
    g00 = np.linalg.inv(np.dot(ii, energy + eta*1j) - H_00)
    g11old, gLLold, g1Lold, gL1old = g00, g00, g00, g00
    errL, errR = 1, 1
    
    while ((errL > epsilon) | (errR > epsilon)):
        t1 = np.dot(g11old, V2)
        t2 = np.dot(gLLold, V1)
        A = np.linalg.multi_dot([np.linalg.inv(ii - np.dot(t2, t1)), t2, g1Lold])
        B = np.linalg.multi_dot([np.linalg.inv(ii - np.dot(t1, t2)), t1, gL1old])
        gLLnew = gLLold + np.linalg.multi_dot([gL1old, V2, A])
        g11new = g11old + np.linalg.multi_dot([g1Lold, V1, B])
        g1Lnew = np.linalg.multi_dot([g1Lold, V1, g1Lold + np.dot(t1, A)])
        gL1new = np.linalg.multi_dot([gL1old, V2, gL1old + np.dot(t2, B)])
        # Biggest error on either side
        errL = np.abs((g11new - g11old)).flatten().max()
        errR = np.abs((gLLnew - gLLold)).flatten().max()
        # Update GFs
        g11old, gLLold, gL1old, g1Lold = g11new, gLLnew, gL1new, g1Lnew
    return gLLold, g11old


# Splits a coordinate list into cells, given the indices of a "final cell" (starting_set here)
def SplitLattice(coords, starting_set, dist):
    
    cells_other = []
    
    tree = cKDTree(coords)
    neighbours = tree.query_ball_tree(tree, dist) 
    
    unassigned = list(np.arange(len(coords)))
    
    cells_other = [starting_set]
    
    while len(unassigned) > 0:
        unassigned = list( set(unassigned) - set(cells_other[-1]) )
        temp = [i for j in cells_other[-1] for i in neighbours[j]]
        new = list(set(unassigned) & set(temp)) 
        if (len(new) > 0):
            cells_other.append(new) 
            
    cells_other = cells_other[::-1]
    return cells_other


#functions to get the corresponding hamiltonian indices using KDtrees
def get_full_pairs(tree, dist):
    pairs = list(tree.query_pairs(dist)) 
    all_pairs=[]
    for pair in pairs:
        all_pairs.append(pair[::-1])
    all_pairs = np.array(pairs + all_pairs)
    return all_pairs

def get_full_pairs2(tree, tree2, dist):
    data = list(tree.query_ball_tree(tree2, dist)) 
    # convert to lists of pairs
    pairs = [(row_idx, item) for row_idx, row in enumerate(data) for item in row]
    
    if len(pairs) == 0:
        pairs = np.empty([0,2])
    return  np.array(pairs )

# %%

# %%
