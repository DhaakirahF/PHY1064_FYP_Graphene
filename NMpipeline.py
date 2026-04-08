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


# Constanst 
a = 1.0
acc = a/ np.sqrt(3)
dd = 1/np.sqrt(3) + 0.01    # threshold for connections
eta = 1e-5

#System generation  clean pristine sytem 
def build_pristine_system(d_length: int, z_index: int):## must be Z even!

    a, subs, dmat = blank_rect_lattice(d_length, z_index/2, (0.0, 0.0))
    h_dev = np.isclose(distance_matrix(a, a), acc) * (-1.0)
    onsites = np.zeros_like(subs)

    # leads
    t_a, _, _ = blank_rect_lattice(1, z_index/2, (0.0, 0.0))
    xl = t_a[:, 0] - 1.0
    xr = t_a[:, 0] + (d_length) * 1.0
    yl = t_a[:, 1]

    h_lead = np.isclose(
        distance_matrix(np.column_stack((xl, yl)), np.column_stack((xl, yl))),
        acc
    ) * (-1.0)

    vlr_lead = np.isclose(
        distance_matrix(np.array((xl, yl)).T, np.array((xl + 1, yl)).T),
        acc
    ) * (-1.0)

    ltree = cKDTree(np.array((xl, yl)).T)
    rtree = cKDTree(np.array((xr, yl)).T)
    dtree = cKDTree(a)

    _d_pairs = get_full_pairs(dtree, dd)

    dev_l_pairs = get_full_pairs2(dtree, ltree, dd)
    dev_r_pairs = get_full_pairs2(dtree, rtree, dd)

    dev_sites_l = dev_l_pairs[:, 0].astype(int) if dev_l_pairs.size else np.array([], dtype=int)
    dev_sites_r = dev_r_pairs[:, 0].astype(int) if dev_r_pairs.size else np.array([], dtype=int)

    starting_cell = list(np.concatenate([dev_sites_l, dev_sites_r])) if (dev_sites_l.size + dev_sites_r.size) else []
    cells = SplitLattice(a, starting_cell, dd) if starting_cell else []
    cells_obj = np.array(cells, dtype=object)

    llead_ind = np.arange(len(dev_sites_l))
    rlead_ind = np.arange(len(dev_sites_r)) + len(dev_sites_l)

    vdevl_lead = (
        np.isclose(distance_matrix(a[dev_sites_l], np.array((xl, yl)).T), acc) * (-1.0)
        if dev_sites_l.size else np.zeros((0, len(xl)))
    )
    vdevr_lead = (
        np.isclose(distance_matrix(a[dev_sites_r], np.array((xr, yl)).T), acc) * (-1.0)
        if dev_sites_r.size else np.zeros((0, len(xr)))
    )

    return dict(
        a=a, subs=subs, dmat=dmat, cells=cells_obj,
        h_dev=h_dev, h_lead=h_lead, vlr_lead=vlr_lead,
        onsites=onsites,
        dev_sites_l=dev_sites_l, dev_sites_r=dev_sites_r,
        llead_ind=llead_ind, rlead_ind=rlead_ind,
        vdevl_lead=vdevl_lead, vdevr_lead=vdevr_lead
    )

#Padding 
def base_for_disorder(disorder_length: int, padding: int, z_index: int):
  
    xdim = disorder_length + 2*padding
    a, subs, dmat = blank_rect_lattice(xdim, z_index/2, (0.0, 0.0))

    startx = padding * 1.0
    endx   = xdim * 1.0 - padding * 1.0
    startx2 = 1.0
    endx2   = xdim * 1.0 - 1.0

    return dict(
        xdim=xdim,
        a=a, subs=subs, dmat=dmat,
        startx=startx, endx=endx,
        startx2=startx2, endx2=endx2
    )

#Disorder bit sytem (functions) to be placed for adding diff disorder without ghanogin whole script 

#edge etching,etches and dangling atoms removed .. returns bolean mask whats kept and removed 

def edge_etch_keep_mask(base, num_etchs: int, edge_prob: float):

    a = base["a"]
    dmat = base["dmat"]
    xcoords = a[:, 0]

    startx, endx = base["startx"], base["endx"]
    startx2, endx2 = base["startx2"], base["endx2"]

    H = -1.0 * np.isclose(dmat, acc)
    num_neighbours = np.array([np.count_nonzero(row == -1.0) for row in H])

    keep = np.ones(len(a), dtype=np.int8)

    # edge etching 
    for _ in range(num_etchs):
        still_here = np.where(keep == 1)[0]

        for i in still_here:
            if (xcoords[i] < endx) and (xcoords[i] >= startx):
                if num_neighbours[i] < 3:
                    if np.random.rand() < edge_prob:
                        keep[i] = 0
                        H[:, i] = 0.0   # ONLY zero when atom is removed

        num_neighbours = np.array(
            [np.count_nonzero(row == -1.0) for row in H]
        )

    # dangling atom cleanup
    still_here = np.where(keep == 1)[0]

    while len(np.intersect1d(np.where(num_neighbours < 2)[0], still_here)) > 2:
        for i in still_here:
            if (num_neighbours[i] < 2) and (xcoords[i] < endx2) and (xcoords[i] >= startx2):
                keep[i] = 0
                H[:, i] = 0.0

        num_neighbours = np.array(
            [np.count_nonzero(row == -1.0) for row in H]
        )
        still_here = np.where(keep == 1)[0]

    return keep == 1


#building disorder device :Compress base lattice by keep_mask build h_dev/couplings/cells,Leads are built using d_length = base["xdim"]
#mask ,atoms to keep ,keeping atoms not removed ,this removes them from the coordinate array fot the final structure
#returns new geometry info 
def build_device_disordered(base, keep_mask, z_index: int):
    xdim = base["xdim"]
    d_length = xdim

    a = base["a"][keep_mask]
    subs = base["subs"][keep_mask]
    dmat = distance_matrix(a, a)

    h_dev = np.isclose(dmat, acc) * (-1.0)
    onsites = np.zeros_like(subs)

    # leads
    t_a, _, _ = blank_rect_lattice(1, z_index/2, (0.0, 0.0))
    xl = t_a[:, 0] - 1.0
    xr = t_a[:, 0] + d_length * 1.0
    yl = t_a[:, 1]

    h_lead = np.isclose(
        distance_matrix(np.column_stack((xl, yl)), np.column_stack((xl, yl))),
        acc
    ) * (-1.0)

    vlr_lead = np.isclose(
        distance_matrix(np.array((xl, yl)).T, np.array((xl + 1, yl)).T),
        acc
    ) * (-1.0)

    ltree = cKDTree(np.array((xl, yl)).T)
    rtree = cKDTree(np.array((xr, yl)).T)
    dtree = cKDTree(a)

    _d_pairs = get_full_pairs(dtree, dd)

    dev_l_pairs = get_full_pairs2(dtree, ltree, dd)
    dev_r_pairs = get_full_pairs2(dtree, rtree, dd)

    dev_sites_l = dev_l_pairs[:, 0].astype(int) if dev_l_pairs.size else np.array([], dtype=int)
    dev_sites_r = dev_r_pairs[:, 0].astype(int) if dev_r_pairs.size else np.array([], dtype=int)

    starting_cell = list(np.concatenate([dev_sites_l, dev_sites_r])) if (dev_sites_l.size + dev_sites_r.size) else []
    # cells = SplitLattice(a, starting_cell, dd) if starting_cell else []
    # cells_obj = np.array(cells, dtype=object)
    # cells = SplitLattice(a, starting_cell, dd) if starting_cell else []
    # cells = [np.array(cell, dtype=int) for cell in cells]
    cells = SplitLattice(a, starting_cell, dd) if starting_cell else []
    cells_obj = [np.array(cell, dtype=int) for cell in cells]

    llead_ind = np.arange(len(dev_sites_l))
    rlead_ind = np.arange(len(dev_sites_r)) + len(dev_sites_l)

    vdevl_lead = (
        np.isclose(distance_matrix(a[dev_sites_l], np.array((xl, yl)).T), acc) * (-1.0)
        if dev_sites_l.size else np.zeros((0, len(xl)))
    )
    vdevr_lead = (
        np.isclose(distance_matrix(a[dev_sites_r], np.array((xr, yl)).T), acc) * (-1.0)
        if dev_sites_r.size else np.zeros((0, len(xr)))
    )
    
  
    return dict(
        a=a, subs=subs, dmat=dmat, cells=cells_obj,
        h_dev=h_dev, h_lead=h_lead, vlr_lead=vlr_lead,
        onsites=onsites,
        dev_sites_l=dev_sites_l, dev_sites_r=dev_sites_r,
        llead_ind=llead_ind, rlead_ind=rlead_ind,
        vdevl_lead=vdevl_lead, vdevr_lead=vdevr_lead,
    )


    
#Getting the transmission
def transmission_energies (es, h_dev, onsites, SLs, SRs, vdevl_lead, vdevr_lead, cells, llead_ind, rlead_ind, eta):
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


# def save_device_plot(folder: Path, tag: str, coords):
#     fig, ax = plt.subplots()
#     ax.plot(coords[:, 0], coords[:, 1], "o", ms=3)
#     ax.set_aspect("equal")
#     fig.savefig(folder / f"{tag}_device.png", dpi=300)
#     plt.close(fig)

# def save_transport_plot(folder: Path, tag: str, es, trans):
#     fig, ax = plt.subplots()
#     ax.plot(es, trans)
#     ax.set_xlabel("Energy (|t|)")
#     ax.set_ylabel("Transmission")
#     fig.savefig(folder / f"{tag}_transport.png", dpi=300)
#     plt.close(fig)


# #Get and save SE once and save 
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

# to load SEs from file instead
# SLs = np.load(str(int(z_index/2))+'_ZGNR_SL_'+str(es[0])+'_'+str(es[-1])+'_'+str(eta)+'.npy')
# SRs = np.load(str(int(z_index/2))+'_ZGNR_SR_'+str(es[0])+'_'+str(es[-1])+'_'+str(eta)+'.npy')
# SLs_up = np.load(str(int(z_index/2))+'_ZGNR_SLup_'+str(es[0])+'_'+str(es[-1])+'_'+str(eta)+'.npy')
# SRs_up = np.load(str(int(z_index/2))+'_ZGNR_SRup_'+str(es[0])+'_'+str(es[-1])+'_'+str(eta)+'.npy')
# SLs_down = np.load(str(int(z_index/2))+'_ZGNR_SLdown_'+str(es[0])+'_'+str(es[-1])+'_'+str(eta)+'.npy')
# SRs_down = np.load(str(int(z_index/2))+'_ZGNR_SRdown_'+str(es[0])+'_'+str(es[-1])+'_'+str(eta)+'.npy')



if __name__ == "__main__":

 #--------system and calculation params------------# 
    e_start, e_final, e_pts = 0.0, 1.2, 101
    es = np.linspace(e_start, e_final, e_pts)

    z_index = 20# must be even
    disorder_length = 30
    padding = 2

    num_etchs = 0     
    edge_prob = 0

    num_confs = 1  

    # output folder 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("nmsystems") / f"ZGNR_{z_index}_L{disorder_length}_p{edge_prob}_n{num_etchs}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    # padded pristine lattice
    base = base_for_disorder(disorder_length, padding, z_index)

    # ---- build one "padded clean" device to get lead matrices for SE calc ----
    keep_all = np.ones(len(base["a"]), dtype=bool)
    padded_clean_dev = build_device_disordered(base, keep_all, z_index)

    # # Save padded clean reference 
    # np.savez(
    #     outdir / "run00_padded_clean_structure_data.npz",
    #     **padded_clean_dev,
    #     is_clean=np.array(1),
    #     disorder_length=np.array(disorder_length),
    #     padding=np.array(padding),
    #     z_index=np.array(z_index),
    # )
    # save_device_plot(outdir, "run00_padded_clean", padded_clean_dev["a"])

    #  precompute self-energies 
    SLs, SRs = [], []
    for en in es:
        SL, SR = surface(padded_clean_dev["h_lead"], padded_clean_dev["vlr_lead"], padded_clean_dev["vlr_lead"].T, 1e-7, en, eta)
        SLs.append(SL)
        SRs.append(SR)
    SLs = np.array(SLs, dtype=object)
    SRs = np.array(SRs, dtype=object)

    np.save(outdir / f"{int(z_index/2)}_ZGNR_SL_{es[0]}_{es[-1]}_{eta}.npy", SLs, allow_pickle=True)
    np.save(outdir / f"{int(z_index/2)}_ZGNR_SR_{es[0]}_{es[-1]}_{eta}.npy", SRs, allow_pickle=True)

    # loop over disorder configurations 
    for conf_id in range(1, num_confs + 1):
        tag = f"conf_{conf_id:04d}"

        keep_mask = edge_etch_keep_mask(base, num_etchs=num_etchs, edge_prob=edge_prob)
        dev = build_device_disordered(base, keep_mask, z_index)

        # save structure (includes conf_id + params)
        np.savez(
            outdir / f"{tag}_structure_data.npz",
            **dev,
            conf_id=np.array(conf_id),
            edge_prob=np.array(edge_prob),
            num_etchs=np.array(num_etchs),
            disorder_length=np.array(disorder_length),
            padding=np.array(padding),
            z_index=np.array(z_index),
        )
        # save_device_plot(outdir, tag, dev["a"])

        # transport
        trans = transmission_energies(
            es,
            dev["h_dev"], dev["onsites"],
            SLs, SRs,
            dev["vdevl_lead"], dev["vdevr_lead"],
            dev["cells"], dev["llead_ind"], dev["rlead_ind"],
            eta
        )

        # save results 
        np.savez_compressed(
            outdir / f"{tag}_transport_results.npz",
            conf_id=np.array(conf_id),
            energies=es,
            transmission=trans,
            edge_prob=np.array(edge_prob),
            num_etchs=np.array(num_etchs),
            disorder_length=np.array(disorder_length),
            padding=np.array(padding),
            z_index=np.array(z_index),
        )
        # save_transport_plot(outdir, tag, es, trans)

        print("Finished", tag)

    print("All saved in:", outdir)
