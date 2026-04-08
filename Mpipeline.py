# 5- import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial import distance_matrix
# from scipy.spatial import cKDTree
# import datetime

from func import blank_rect_lattice, surface, get_full_pairs, get_full_pairs2, SplitLattice,surface
from scipy.spatial import cKDTree
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
from matplotlib import cm
import glob
import json
import model_mlene 
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 



# set directories, descriptor and load trained network
# model_dir='/home/powersr/Dropbox/ml_edge_paper/mlene/ml-ene/bestmodel/'     # location of trained network(s)
# custom_dir='custom/'       # location of custom geometry files

desc = model_mlene.con_all_desc   # load model from paper
numzz, numcor, numac, desc_type, descriptor, model_name, desc_fn = desc
model = tf.keras.models.load_model(descriptor+'.'+model_name+'_best.h5',compile=False)

## NN distances
# dd = 1/np.sqrt(3) + 0.01
# eta= 1E-5
a = 1.0 
acc = a / np.sqrt(3) 
dd = 1 / np.sqrt(3) + 0.01 # threshold for connections 
eta = 1e-5  
hubU = 1.33 # Hubbard U parameter for moments


def blank_rect_lattice(size_zz, size_ac, origin):
    '''Generates the initial graphene sheet from which to etch the structure'''
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

def count_neighbours(site_list, dmat, neigh_dist):
    '''counts the number of neighbours of each site'''
    neighs = [np.isclose(dd,neigh_dist).sum() for dd in dmat[:, site_list][site_list, :]]
    return np.array(neighs)

def count_and_track_neighbours(site_list, dmat, neigh_dist):
    '''counts and keeps track of the number of neighbours of each site'''
    outlist=[]
    neighs=[]
    for dd in dmat[np.ix_(site_list, site_list)]:
        inlist =np.isclose(dd,neigh_dist)
        outlist.append(list(np.array(np.where(inlist==True)).flatten()))
        neighs.append(inlist.sum())
    return np.array(neighs), outlist

def classify_sites(sysneigh, neighlists):
    '''determines whether sites are type 0 (bulk), 1 (zigzag), 2 (corner) or 3(armchair)'''
    classification = np.zeros_like(sysneigh)
    
    #for each site, keep track of how many neighbours its neighbours have
    for i, neigh in enumerate(sysneigh):
        if neigh == 2:
            sumneigh = sysneigh[neighlists[i]].sum()
            if(sumneigh==6):
                classification[i] =1
            elif (sumneigh==4):
                classification[i] =2
            elif (sumneigh==5):
                classification[i] =3
    return classification

# Function for calculating the surface GFs from the leads (left and right)
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

def SplitLattice(coords, starting_set, dist):
    cells_other = []
    tree = cKDTree(coords)
    neighbours = tree.query_ball_tree(tree, dist) 
    unassigned = list(np.arange(len(coords)))
    cells_other = [starting_set]
    while len(unassigned) > 0:
        unassigned = list(set(unassigned) - set(cells_other[-1]))
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

def get_trans(es, h_dev, onsites, SLs, SRs, vdevl_lead, vdevr_lead, cells, llead_ind, rlead_ind, eta):
    trans=np.zeros_like(es)
    
    for bb, en in enumerate(es):
        ind = cells[0]

        H0 = h_dev[np.ix_(ind, ind)]
        H0 += np.diag(onsites[ind])
        ii = np.identity(len(H0), dtype = 'complex128')

        g_old = np.linalg.inv(np.dot(ii, en + eta*1j) - H0)
        
        # Build device GF recursively (Loop over cells) (WITHOUT ADDING LEADS)
        for i in range(1, len(cells)):
            
            #indices
            ind = cells[i]
            ind_m = cells[i - 1]


            # cell H and g
            H0 = h_dev[np.ix_(ind, ind)]

            H0 += np.diag(onsites[ind])
            ii = np.identity(len(H0), dtype = 'complex128')
            g_ii = np.linalg.inv(np.dot(ii, en + eta*1j) - H0)


            # connection matrices    (use transposes for vrl)
            # lr means i-1 to i, not necessarily left-to-right

            VLR = h_dev[np.ix_(ind_m, ind)]

            G_ii = np.dot(np.linalg.inv(ii - np.linalg.multi_dot([g_ii, np.conj(VLR.T), g_old, VLR])), g_ii)
            g_old = G_ii
            
        SL = SLs[bb]
        SR = SRs[bb]

        # self-energies for each lead
        self_l = np.linalg.multi_dot([vdevl_lead, SL, np.conj(vdevl_lead.T)])
        self_r = np.linalg.multi_dot([vdevr_lead, SR, np.conj(vdevr_lead.T)])

       # gammas from each lead        
        Gamma_L = 1j*(self_l - np.conj(self_l.T))
        Gamma_R = 1j*(self_r - np.conj(self_r.T)) 

        # add self energies using dyson
        full_self = np.zeros_like(g_old)
        full_self[np.ix_(llead_ind, llead_ind)] = self_l
        full_self[np.ix_(rlead_ind, rlead_ind)] = self_r

        ii = np.identity(len(g_old), dtype = 'complex128')
        full_g =  np.linalg.inv( ii - np.dot(g_old, full_self)).dot(g_old)

        GLR = full_g[np.ix_(llead_ind, rlead_ind)]

        trans[bb] = np.real(np.trace(np.linalg.multi_dot([np.conj(GLR.T), Gamma_L, GLR, Gamma_R])))
            
          
    return trans


# generate a disordered GNR: takes in pristine geometry,
# returns disordered info

def generate_disordered_GNR(a, subs, dmat, sweeps, probs):
    newinside = np.arange(len(subs)).flatten()
    newlist=count_neighbours(newinside, dmat, 1/np.sqrt(3) )
    
    
    for sweep in np.arange(sweeps):
        isInside=[]
        for i, point in enumerate(newinside):
            if (newlist[i] == 3) or (a[point,0] < dis_x1) or (a[point,0] >= dis_x2) :
                isInside.append(point)
            else:
                if (np.random.rand() > probs[sweep]):
                    isInside.append(point)
                
        newinside = np.array(isInside)
        newlist=count_neighbours(newinside, dmat, 1/np.sqrt(3) )

    while len(newinside[newlist<2]) > 0:
        newinside=newinside[newlist>1]
        newlist=count_neighbours(newinside, dmat, 1/np.sqrt(3) )
        
        
    return newinside



from pathlib import Path
from datetime import datetime
import numpy as np

if __name__ == "__main__":

    # -------- system and calculation params --------
    e_start, e_final, e_pts = 0.0, 1.2, 101
    es = np.linspace(e_start, e_final, e_pts)

    z_index = 20        # must be even
    d_length = 30
    buffer = 2

    sweeps = 0
    P0 = 0
    probs = P0 * (1 - np.arange(sweeps) / sweeps) ** 2

    num_confs = 1

    f_length = d_length + 4 * buffer

    dis_x1 = buffer * 2.0
    dis_x2 = (f_length - 2 * buffer) * 1.0 - 0.001

    trans_x1 = buffer * 1.0
    trans_x2 = (f_length - 1 * buffer) * 1.0 - 0.001

    # output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("msystems") / f"ZGNR_{z_index}_L{d_length}_p{P0}_n{sweeps}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    # -------- base system --------
    a, subs, dmat = blank_rect_lattice(f_length, z_index // 2, (0.0, 0.0))

    yc = a[:, 1].mean()
    topsites = len(np.where(a[:, 1] > yc)[0])
    botsites = len(np.where(a[:, 1] < yc)[0])

    # -------- leads --------
    t_a, t_subs, t_dmat = blank_rect_lattice(1, z_index // 2, (0.0, 0.0))
    xl = t_a[:, 0] - 1.0
    xr = t_a[:, 0] + (d_length + 2 * buffer) * 1.0
    yl = t_a[:, 1]

    ml = np.loadtxt('correctz20_chain60.dat', usecols=(3,))

    ltree = cKDTree(np.array((xl, yl)).T)
    rtree = cKDTree(np.array((xr, yl)).T)

    h_lead = np.isclose(
        distance_matrix(np.array((xl, yl)).T, np.array((xl, yl)).T),
        1 / np.sqrt(3)
    ) * (-1.0)

    vlr_lead = np.isclose(
        distance_matrix(np.array((xl, yl)).T, np.array((xl + 1, yl)).T),
        1 / np.sqrt(3)
    ) * (-1.0)

    # -------- load self-energies --------
    se_dir = Path("msystems")
    # SLs, SRs = [], []
    # SLs_up, SRs_up = [], []
    # SLs_down, SRs_down = [], []
    # for en in es:
    #     SL, SR = surface(h_lead, vlr_lead, vlr_lead.T, 1E-7, en, eta)
    #     SLs.append(SL)
    #     SRs.append(SR)
    #     SL, SR = surface(h_lead - np.diag(ml)*1.33/2, vlr_lead, vlr_lead.T, 1E-7, en, eta)
    #     SLs_up.append(SL)
    #     SRs_up.append(SR)
    #     SL, SR = surface(h_lead + np.diag(ml)*1.33/2, vlr_lead, vlr_lead.T, 1E-7, en, eta)
    #     SLs_down.append(SL)
    #     SRs_down.append(SR)
    # np.save(str(int(z_index/2))+'_ZGNR_SL_'+str(es[0])+'_'+str(es[-1])+'_'+str(eta)+'.npy', SLs)
    # np.save(str(int(z_index/2))+'_ZGNR_SLup_'+str(es[0])+'_'+str(es[-1])+'_'+str(eta)+'.npy', SLs_up)
    # np.save(str(int(z_index/2))+'_ZGNR_SLdown_'+str(es[0])+'_'+str(es[-1])+'_'+str(eta)+'.npy', SLs_down)
    # np.save(str(int(z_index/2))+'_ZGNR_SR_'+str(es[0])+'_'+str(es[-1])+'_'+str(eta)+'.npy', SRs)
    # np.save(str(int(z_index/2))+'_ZGNR_SRup_'+str(es[0])+'_'+str(es[-1])+'_'+str(eta)+'.npy', SRs_up)
    # np.save(str(int(z_index/2))+'_ZGNR_SRdown_'+str(es[0])+'_'+str(es[-1])+'_'+str(eta)+'.npy', SRs_down)


    SLs = np.load(se_dir / f"{int(z_index/2)}_ZGNR_SL_{es[0]}_{es[-1]}_{eta}.npy", allow_pickle=True)
    SRs = np.load(se_dir / f"{int(z_index/2)}_ZGNR_SR_{es[0]}_{es[-1]}_{eta}.npy", allow_pickle=True)

    SLs_up = np.load(se_dir / f"{int(z_index/2)}_ZGNR_SLup_{es[0]}_{es[-1]}_{eta}.npy", allow_pickle=True)
    SRs_up = np.load(se_dir / f"{int(z_index/2)}_ZGNR_SRup_{es[0]}_{es[-1]}_{eta}.npy", allow_pickle=True)

    SLs_down = np.load(se_dir / f"{int(z_index/2)}_ZGNR_SLdown_{es[0]}_{es[-1]}_{eta}.npy", allow_pickle=True)
    SRs_down = np.load(se_dir / f"{int(z_index/2)}_ZGNR_SRdown_{es[0]}_{es[-1]}_{eta}.npy", allow_pickle=True)

    # SLs = np.load(str(int(z_index / 2)) + '_ZGNR_SL_' + str(es[0]) + '_' + str(es[-1]) + '_' + str(eta) + '.npy')
    # SRs = np.load(str(int(z_index / 2)) + '_ZGNR_SR_' + str(es[0]) + '_' + str(es[-1]) + '_' + str(eta) + '.npy')
    # SLs_up = np.load(str(int(z_index / 2)) + '_ZGNR_SLup_' + str(es[0]) + '_' + str(es[-1]) + '_' + str(eta) + '.npy')
    # SRs_up = np.load(str(int(z_index / 2)) + '_ZGNR_SRup_' + str(es[0]) + '_' + str(es[-1]) + '_' + str(eta) + '.npy')
    # SLs_down = np.load(str(int(z_index / 2)) + '_ZGNR_SLdown_' + str(es[0]) + '_' + str(es[-1]) + '_' + str(eta) + '.npy')
    # SRs_down = np.load(str(int(z_index / 2)) + '_ZGNR_SRdown_' + str(es[0]) + '_' + str(es[-1]) + '_' + str(eta) + '.npy')
    
    # optional: copy/save these in outdir too
    np.save(outdir / f"{int(z_index/2)}_ZGNR_SL_{es[0]}_{es[-1]}_{eta}.npy", SLs, allow_pickle=True)
    np.save(outdir / f"{int(z_index/2)}_ZGNR_SR_{es[0]}_{es[-1]}_{eta}.npy", SRs, allow_pickle=True)
    np.save(outdir / f"{int(z_index/2)}_ZGNR_SLup_{es[0]}_{es[-1]}_{eta}.npy", SLs_up, allow_pickle=True)
    np.save(outdir / f"{int(z_index/2)}_ZGNR_SRup_{es[0]}_{es[-1]}_{eta}.npy", SRs_up, allow_pickle=True)
    np.save(outdir / f"{int(z_index/2)}_ZGNR_SLdown_{es[0]}_{es[-1]}_{eta}.npy", SLs_down, allow_pickle=True)
    np.save(outdir / f"{int(z_index/2)}_ZGNR_SRdown_{es[0]}_{es[-1]}_{eta}.npy", SRs_down, allow_pickle=True)

   
    # -------- loop over disorder configurations --------
    for conf_id in range(1, num_confs + 1):
        tag = f"conf_{conf_id:04d}"
        print(f"{datetime.now()} -- {tag} started")

        # disorder
        newinside = generate_disordered_GNR(a, subs, dmat, sweeps, probs)

        # system attributes
        sysneigh, neighlists = count_and_track_neighbours(newinside, dmat, 1 / np.sqrt(3))
        syscoords = a[newinside]
        syssubs = subs[newinside]
        sysedges = classify_sites(sysneigh, neighlists)

        systopsites = len(np.where(syscoords[:, 1] > yc)[0])
        sysbotsites = len(np.where(syscoords[:, 1] < yc)[0])

        t_vacs = topsites - systopsites
        b_vacs = botsites - sysbotsites
        tot_vacs = t_vacs + b_vacs

        print(f"{datetime.now()} -- {tag} vacancies: top={t_vacs}, bottom={b_vacs}, total={tot_vacs}")

        # temporary geometry file for descriptor generation
        temp_geom = outdir / f"{tag}_geometry.gz"
        final = np.zeros_like(syscoords[:, 0])

        np.savetxt(
            temp_geom,
            np.c_[syscoords[:, 0], syscoords[:, 1], syssubs, sysedges, final],
            delimiter=','
        )

        # descriptors + moments
        site_desc = desc_fn(str(temp_geom))
        X = site_desc[:, :-3]
        X = np.expand_dims(X, axis=1)
        pred_mom = model.predict(X).flatten()

        # overwrite temp file with predicted moments if you want
        np.savetxt(
            temp_geom,
            np.c_[syscoords[:, 0], syscoords[:, 1], syssubs, sysedges, pred_mom],
            delimiter=','
        )

        # remove hubbard buffer
        trans_ind = np.where(
            (syscoords[:, 0] > (trans_x1 - 0.001)) &
            (syscoords[:, 0] < trans_x2)
        )[0]

        # transport geometry
        x1 = syscoords[trans_ind, 0] - trans_x1
        y1 = syscoords[trans_ind, 1]
        subs1 = syssubs[trans_ind]
        m1 = (subs1 * pred_mom[trans_ind])

        onsites = np.zeros_like(x1)
        onsites_up = -m1 * 1.33 / 2
        onsites_down = m1 * 1.33 / 2

        # connectivity
        dtree = cKDTree(np.array((x1, y1)).T)
        dev_l_pairs = get_full_pairs2(dtree, ltree, dd)
        dev_r_pairs = get_full_pairs2(dtree, rtree, dd)

        dev_sites_l = dev_l_pairs[:, 0].astype(int) if dev_l_pairs.size else np.array([], dtype=int)
        dev_sites_r = dev_r_pairs[:, 0].astype(int) if dev_r_pairs.size else np.array([], dtype=int)

        if len(dev_sites_l) == 0 or len(dev_sites_r) == 0:
            print(f"{datetime.now()} -- {tag} skipped: disconnected from lead")
            continue

        llead_ind = np.arange(len(dev_sites_l))
        rlead_ind = np.arange(len(dev_sites_r)) + len(dev_sites_l)

        starting_cell = np.concatenate([dev_sites_l, dev_sites_r])
        cells = SplitLattice(np.array((x1, y1)).T, starting_cell, dd)

        h_dev = np.isclose(
            distance_matrix(np.array((x1, y1)).T, np.array((x1, y1)).T),
            1 / np.sqrt(3)
        ) * (-1.0)

        vdevl_lead = np.isclose(
            distance_matrix(np.array((x1[dev_sites_l], y1[dev_sites_l])).T, np.array((xl, yl)).T),
            1 / np.sqrt(3)
        ) * (-1.0)

        vdevr_lead = np.isclose(
            distance_matrix(np.array((x1[dev_sites_r], y1[dev_sites_r])).T, np.array((xr, yl)).T),
            1 / np.sqrt(3)
        ) * (-1.0)

        # -------- save structure data --------
        np.savez(
            outdir / f"{tag}_structure_data.npz",
            conf_id=np.array(conf_id),
            z_index=np.array(z_index),
            d_length=np.array(d_length),
            buffer=np.array(buffer),
            sweeps=np.array(sweeps),
            P0=np.array(P0),
            syscoords=syscoords,
            syssubs=syssubs,
            sysedges=sysedges,
            pred_mom=pred_mom,
            trans_ind=trans_ind,
            x1=x1,
            y1=y1,
            subs1=subs1,
            onsites=onsites,
            onsites_up=onsites_up,
            onsites_down=onsites_down,
            h_dev=h_dev,
            cells=np.array(cells, dtype=object),
            dev_sites_l=dev_sites_l,
            dev_sites_r=dev_sites_r,
            llead_ind=llead_ind,
            rlead_ind=rlead_ind,
            vdevl_lead=vdevl_lead,
            vdevr_lead=vdevr_lead,
            t_vacs=np.array(t_vacs),
            b_vacs=np.array(b_vacs),
            tot_vacs=np.array(t_vacs + b_vacs)
        )

        # -------- transport --------
        print(f"{datetime.now()} -- {tag} transmission started")

        trans = get_trans(es, h_dev, onsites, SLs, SRs, vdevl_lead, vdevr_lead, cells, llead_ind, rlead_ind, eta)
        trans_up = get_trans(es, h_dev, onsites_up, SLs_up, SRs_up, vdevl_lead, vdevr_lead, cells, llead_ind, rlead_ind, eta)
        trans_down = get_trans(es, h_dev, onsites_down, SLs_down, SRs_down, vdevl_lead, vdevr_lead, cells, llead_ind, rlead_ind, eta)

        # -------- save transport --------
        np.savez_compressed(
            outdir / f"{tag}_transport_results.npz",
            conf_id=np.array(conf_id),
            energies=es,
            trans=trans,
            trans_up=trans_up,
            trans_down=trans_down,
            z_index=np.array(z_index),
            d_length=np.array(d_length),
            buffer=np.array(buffer),
            sweeps=np.array(sweeps),
            P0=np.array(P0)
        )

        print(f"{datetime.now()} -- Finished {tag}")

    print("All saved in:", outdir)

