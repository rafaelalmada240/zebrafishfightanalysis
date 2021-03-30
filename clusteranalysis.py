"""
@author: Rafael Almada
"""

import numpy as np

def cluster_bev(embedding, blobs):
    resolution = blobs.shape[0]
    n_grid = resolution*1j

    q1,q2 = np.mgrid[embedding[:,0].min():embedding[:,0].max():n_grid, 
                      embedding[:,1].min():embedding[:,1].max():n_grid]
    thr1 = (embedding[:,0].max()-embedding[:,0].min())/resolution
    thr2 = (embedding[:,1].max()-embedding[:,1].min())/resolution
    thr = [thr1,thr2]
    #ones_im = np.where(bin_im!1)
    b1,b2 = get_clusters_labels(blobs)
    res = cluster_time_labels(thr,b1,b2,q1[:,0],q2[0],embedding)
    
    return b1,b2,res,thr

def get_clusters_labels(blobs):
    b1 = []
    b2 = []
    for i in range(1,np.max(blobs)+1):
        b1.append(np.where(blobs==i)[1])
        b2.append(np.where(blobs==i)[0])
    return b1, b2
        
def cluster_time_labels(thr,b1,b2,q1,q2,embedding):
    res = []
    len_b = len(b1)

    for i in range(len_b):
        r = []
        for j in range(len(b1[i])):
            a = (embedding[:,0]<=(q1[b2[i][j]]+thr[0])) & (embedding[:,0]>=(q1[b2[i][j]]))
            b = (embedding[:,1]<=(q2[b1[i][j]]+thr[1])) & (embedding[:,1]>=(q2[b1[i][j]]))
            r.append(np.array(np.where(a*b==True))[0].flatten())
        res.append(np.concatenate(r,axis=0))
    return res

def cluster_bev_avg(Dataset,Bevmat):
    len_var = Dataset.shape[0]
    len_cl = Bevmat.shape[1]
    
    avg_var = np.zeros((len_var,len_cl))
    
    for i in range(len_var):
        for j in range(len_cl):
            if (i == 1) or (i == 2):
                if i == 1:
                    X_i = np.mean(np.abs(np.cos(np.pi*X[i,np.where(bevmat_conv[:,j]==1)[0]])))
                if i == 2:
                    X_i = np.median(np.cos(np.pi*X[i,np.where(bevmat_conv[:,j]==1)[0]]))
            else:
                X_i = np.mean(X[i,np.where(bevmat_conv[:,j]==1)[0]])
            avg_var[i,j] = X_i
            
    return avg_var

def not_in_array(len_d,array):
    loc_list = []
    for i in range(len_d):
        if i not in array:
            loc_list.append(i)
    return np.array(loc_list)

def cluster_label_behaviors(Dataset, Bevmat):
    avg_var = cluster_bev_avg(Dataset, Bevmat)
    
    # Distinguish fight from not fight
    fight_loc1 = np.where(avg_var[4]>np.median(avg_var[4]))[0]
    fight_loc2 = np.where(avg_var[0]<np.median(avg_var[0]))[0]
    fight = np.array(list(set(fight_loc1).intersection(fight_loc2)))
    
    # Distinguish symmetric from assymetric fight 
    sym_fight = fight[np.where(avg_var[5][fight]<np.median(avg_var[1][fight]))[0]]
    asym_fight = fight[np.where(avg_var[5][fight]>=np.median(avg_var[1][fight]))[0]]

    not_fight = not_in_array(N,fight)

    # Distinguish freeze from not freeze
    freeze = not_fight[np.where(avg_var[4][not_fight]<0.6*np.mean(avg_var[4]))[0]]
    n_freeze = not_fight[np.where(avg_var[4][not_fight]>=0.6*np.mean(avg_var[4][not_fight]))[0]]

    # Distinguish displays from not displays
    n_display = n_freeze[np.where(avg_var[1][n_freeze]<0.3)[0]]
    display = n_freeze[np.where(avg_var[1][n_freeze]>=0.3)[0]]
    
    #Distinguish between passive and active displays
    act_d = display[np.where(avg_var[4][display]>np.mean(avg_var[4][display]))]
    p_disp = display[np.where(avg_var[4][display]<=np.mean(avg_var[4][display]))]
    
    return [p_disp, act_d, asym_fight, sym_fight, freeze, n_display]
    
def reorder_label_bevmat(behav_label, Bevmat):
    bevmat2 = np.zeros(Bevmat.shape)
    pas_disp = behav_label[0]
    act_disp = behav_label[1]
    sym_fight = behav_label[3]
    asym_fight = behav_label[2]
    freeze = behav_label[4]
    other = behav_label[5]
    
    ld1 = len(pas_disp)
    ld2 = len(pas_disp)+len(act_disp)
    la2 = ld2 + len(asym_fight)
    la1 = la2 + len(sym_fight)
    lfr = la1 + len(freeze)
    lres = lfr+len(other)
    bevmat2[:,:ld1] = np.array(Bevmat[:,pas_disp])
    bevmat2[:,ld1:ld2] = np.array(Bevmat[:,act_disp])
    bevmat2[:,ld2:la2] = np.array(Bevmat[:,asym_fight])
    bevmat2[:,la2:la1] = np.array(Bevmat[:,sym_fight])
    bevmat2[:,la1:lfr] = np.array(Bevmat[:,freeze])
    bevmat2[:,lfr:lres] = np.array(Bevmat[:,other])
    
    return bevmat2

def symbol_seq(bevmat):
    bevmat1 = np.zeros(bevmat.shape)
    for i in range(bevmat.shape[1]):
        bevmat1[:,i] = bevmat[:,i]*(i+1)

    symbseq = np.sum(bevmat1,axis=1)%(bevmat.shape[1]+1)
    return symbseq


def compress_seq(symbseq):
    symb_loc = []
    t_dwell = []
    s = 0
    for i in range(len(symbseq)-2):
        s += 1
        if symbseq[i+1] != symbseq[i]:
            symb_loc.append(i)
            t_dwell.append([symbseq[i],s])
            s = 0
    
    t_dwell_arr = np.array(t_dwell)
    comp_seq = symbseq[symb_loc]
    return comp_seq, t_dwell_arr

def t_dwell_dist(t_dwell_arr,len_bevmat,dt):
    avg_td = np.zeros(len_bevmat)
    std_td = np.zeros(len_bevmat)
    for i in range(len_bevmat):
        avg_td[i] = np.mean(t_dwell_arr[np.where(t_dwell_arr[:,0]==i+1)][:,1])*dt
        std_td[i] = np.std(t_dwell_arr[np.where(t_dwell_arr[:,0]==i+1)][:,1])*dt
    return avg_td, std_td

def transition_matrix(seq):
    ## Credits in https://stackoverflow.com/questions/46657221
    ## to users/4996248/john-coleman
    n = int(1+np.max(seq))
    M = [[0]*n for _ in range(n)]
    
    for (i,j) in zip(seq,seq[1:]):
        M[int(i)][int(j)] += 1
    
    M1 = np.array(M)
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M,M1

def transition_matrix_tau(seq,tau):
    ## Credits in https://stackoverflow.com/questions/46657221
    ## to users/4996248/john-coleman
    n = int(1+np.max(seq))
    M = [[0]*n for _ in range(n)]
    
    for (i,j) in zip(seq,seq[tau:]):
        M[int(i)][int(j)] += 1
    
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

def prop_trans_mat(M):
    n_elem = np.array(M).shape[0]-1
    M1 = np.abs(np.array(M)[1:,1:]-0*np.array(M)[1:,1:]*np.identity(n_elem))
    M00 = (np.sum(M1, axis=1))
    m1_2 = np.zeros(M1.shape)
    for i in range(len(M00)):
        m1_2[i] = M1[i]/M00[i]
    return m1_2


def entropy_rate(m1_2,comp_seq,len_bevmat):
    hist_seq, seq_n = np.histogram(comp_seq,bins=len_bevmat,density=True)
    ent_1 = []
    for i in range(len_bevmat):
        ent_1.append(np.nansum(hist_seq[i]*m1_2[i]*np.log(m1_2[i]+0.01)))
    ent2 = np.sum(np.array(ent_1))
    return ent2

def graph_entropy(m1_2):
    k = np.sum(m1_2>0,axis=1)
    E = np.sum(np.triu(m1_2>0))
    Hent = np.sum(k*np.log2(k))/(2*E)
    return Hent

def process_analysis(comp_seq, len_bevmat):
    N1 = len_bevmat
    N = int(len(comp_seq)/2)-2
    eig_vec  = np.zeros((N,N1))
    Hent_vec = np.zeros((N,))
    ent2_vec = np.zeros((N,))
    for i in range(1,N+1):
        M1 = transition_matrix_tau(comp_seq,i)
        m1 = prop_trans_mat(M1)
        Hent_vec[i-1] = graph_entropy(m1)
        eig_vec[i-1] = np.sort(np.abs(np.linalg.eig(np.nan_to_num(m1))[0]))
       
        ent2_vec[i-1] = entropy_rate(m1,comp_seq,len_bevmat)
    return eig_vec, ent2_vec,Hent_vec
