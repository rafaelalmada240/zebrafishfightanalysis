"""
@author: Rafael Almada
"""

import numpy as np
import userfunctions as uf

def kinematic_analysis(positions,dt):
    # Distance
    
    Lbody = np.mean(np.ma.sum(np.ma.sum(np.diff(np.ma.masked_invalid(positions),axis=2)**2,axis=3)**0.5,axis=2))
    
    pos_shape = positions.shape
    dist_head = np.zeros((pos_shape[0],pos_shape[1]))
    dist_var = np.zeros((pos_shape))
    for f1 in range(pos_shape[1]):
        if f1 == 0:
            f2 = 1
        else:
            f2 = 0
        for bpt in range(pos_shape[2]):
            dist_var[:,f1,bpt] = positions[:,f1,0] - positions[:,f2,bpt]
            
    dist_head = np.ma.min(np.sqrt(np.ma.sum(np.ma.masked_invalid(dist_var)**2,axis = 3)),axis=2)/Lbody
    distance = np.ma.mean(dist_head,axis=1)
    
    #Speed
    
    normed_pos =  np.ma.masked_invalid(positions)/Lbody
    speed = np.mean(np.sqrt(np.ma.sum(np.diff(normed_pos,axis=0)**2,axis=2)),axis=2)*(1/dt)
    
    #Angular variables
    
    pos_cent = np.zeros(pos_shape)
    for bpt in range(pos_shape[2]):
        pos_cent[:,:,bpt] = positions[:,:,bpt] - positions[:,:,1]
        
    ## hp vec
    
    hp_vec = pos_cent[:,:,0]
    hp_n = np.zeros(hp_vec.shape)
    for fish in range(pos_shape[1]):
        hp_n[:,fish] = uf.normalize_vec(hp_vec[:,fish]) 
        
        
    heading = np.zeros(hp_n.shape)
    for fish in range(pos_shape[1]):
        for coord in range(pos_shape[3]):
            heading[:,fish,coord] = np.ma.convolve(hp_n[:,fish,coord],uf.gauss_kernel95(10),propagate_mask = False)[19:-20]
            
    alignment = (2 - (np.ma.sum(np.diff(heading,axis=1)**2,axis=2)[:,0]))/2
    
    d_head = np.diff(heading,axis=0)/dt
    dh_n = np.zeros(d_head.shape)
    for fish in range(pos_shape[1]): 
        dh_n[:,fish] = uf.normalize_vec(d_head[:,fish])
     
    acc_align = (2-np.sum(np.diff(dh_n,axis=1)**2,axis=2)[:,0])/2
    
    ## ht vec
    
    ht_vec = pos_cent[:,:,2]
    ht_n = np.zeros(ht_vec.shape)
    for fish in range(pos_shape[1]):
        ht_n[:,fish] = uf.normalize_vec(ht_vec[:,fish]) 
    
    dntc = np.diff(ht_n,axis=0)/dt
    sdntc = np.sum(dntc**2,axis=2)**0.5
    
    z_conv = np.zeros(sdntc.shape)
    z_conv[:,0] = np.convolve(sdntc[:,0],uf.rect_kernel(100),'same')
    z_conv[:,1] = np.convolve(sdntc[:,1],uf.rect_kernel(100),'same')
    
    # State variables 
    Single_state_var = {}
    Single_state_var['distance'] = dist_head[:-1]
    Single_state_var['speed'] = speed
    Single_state_var['heading'] = heading[:-1]
    Single_state_var['tail beat rate'] = z_conv[:-1]
    Single_state_var['z position'] = np.mean(positions[:-1,:,:,2],axis=2)/Lbody
    Single_state_var['tail z bend'] = ht_n[:,:,2]
    
    Comp_state_var = {}
    Comp_state_var['average distance'] = np.convolve(distance[:-1],uf.rect_kernel(75),'same')
    Comp_state_var['mean speed'] = np.convolve(np.mean(speed[:]%30,axis=1),uf.gauss_kernel95(10),'same')
    Comp_state_var['alignment'] = np.convolve(alignment[:-1],uf.gauss_kernel95(10),'same')
    Comp_state_var['acc alignment'] = acc_align
    Comp_state_var['mean tail beating rate'] = np.mean(z_conv,axis=1)[:-1]
    Comp_state_var['var tail beating rate'] = np.var(z_conv,axis=1)[:-1]
    
    return Single_state_var, Comp_state_var

def fight_analysis(s_dict,c_dict):
    
    d_head = s_dict['distance']
    speed = s_dict['speed']
    z_pos = s_dict['z position']+0.5
    
    dist_conv = c_dict['average distance']
    speed_m = c_dict['mean speed']
    dmag_conv = c_dict['alignment']
    s_rot_conv = c_dict['acc alignment']
    
    
    # Circling
    circ1 = np.where(s_rot_conv<-0.5)[0]
    circ2 = np.where(dmag_conv <-0.4)[0]
    circ3 = np.where(dist_conv < 2.5)[0]

    circ_t = set(set(circ1).intersection(circ2)).intersection(circ3)

    circv = np.zeros(len(d_head))
    circv[list(circ_t)]=1

    circ_v = np.convolve(circv,uf.rect_kernel(100),'same')>0.40
    
    # Passive Display
    disp1 = np.where(np.abs(dmag_conv)>0.7)[0]
    disp2 = np.where(dist_conv<2.5)[0]
    disp4 = np.where(speed_m <3)[0]

    disp_r = list(((set(disp1).intersection(disp2)).intersection(disp4)))

    disp_v = np.zeros(len(d_head))
    disp_v[disp_r]=1

    disp_v = (np.convolve(disp_v,uf.rect_kernel(100),'same')>0.5)
    
    #Contest
    loc_close0 = np.where(np.abs(d_head[:,0])<0.25)[0]
    loc_close1 = np.where(d_head[:,1]<0.25)[0]

    loc_sp0 = np.where(speed[:,0]>3)[0]
    loc_sp1 = np.where(speed[:,1]>3)[0]

    loc_att0 = list(set(loc_close0).intersection(loc_sp0))
    loc_att1 = list(set(loc_close1).intersection(loc_sp1))

    att_ev = np.zeros((len(d_head),2))
    att_ev[loc_att0,0] = 1
    att_ev[loc_att1,1] = 1
    
    att_conv = np.zeros(att_ev.shape)
    for i in range(2):
        att_conv[:,i] = np.convolve(att_ev[:,i],uf.rect_kernel(100),'same')>0.05
        
        
    #Freezing
    fr11 = np.where((z_pos)<2.5)[0]
    fr12 = np.where(speed[:,0]<1)[0]

    fr21 = np.where((z_pos)<2.5)[0]
    fr22 = np.where(speed[:,1]<1)[0]
    
    fr3 = np.where(dist_conv>2)[0]

    freeze1 = list((set(fr11).intersection(fr12)).intersection(fr3))
    freeze2 = list((set(fr21).intersection(fr22)).intersection(fr3))

    freeze_v = np.zeros((len(d_head),2))
    freeze_v[freeze1,0] = 1
    freeze_v[freeze2,1] = 1

    for i in range(2):
        freeze_v[:,i] = np.convolve(freeze_v[:,i],uf.rect_kernel(100),'same') > 0.5
        
    State_dict = {}
    State_dict['passive display'] = disp_v
    State_dict['circling'] = circ_v
    State_dict['aggressive state'] = att_conv
    State_dict['freezing'] = freeze_v
    
    return State_dict

def compute_etho_array(State_array, scale):
    ethogram_array = np.zeros(State_array.shape)

    n_win = scale 
    len_p = State_array.shape[1]
    L_run = int(len_p/n_win)
    for i in range(L_run):
        state_sum = np.sum(State_array[:,i*n_win:(i+1)*n_win],axis=1)
        if np.sum(state_sum) == 0:
            ethogram_array[5,i*n_win:(i+1)*n_win] = 1
        if np.sum(state_sum) != 0:
            n_st = np.argmax(state_sum)
            if (n_st == 0) or (n_st == 1):# Display states
                ethogram_array[i,i*n_win:(i+1)*n_win] = 1
            if (n_st == 4) or (n_st == 5):#Freezing state
                ethogram_array[4,i*n_win:(i+1)*n_win] = 1
            if (n_st == 2) or (n_st == 3):#Aggressive states
                if np.abs(state_sum[2]-state_sum[3])/100 > 0.8:
                    ethogram_array[3,i*n_win:(i+1)*n_win] = 1#Asymmetric
                if np.abs(state_sum[2]-state_sum[3])/100 < 0.8:
                    ethogram_array[2,i*n_win:(i+1)*n_win] = 1#Symmetric
    return ethogram_array