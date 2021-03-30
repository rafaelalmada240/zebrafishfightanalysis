"""
@author: Rafael Almada
"""


import numpy as np
from scipy import interpolate as interp

def rect_kernel(w):
    h0 = np.array([1/w for i in range(w)])
    return h0/np.sum(h0)

def gauss_kernel95(sigma):
    h = 1/(np.sqrt(2*np.pi))*np.exp(-0.5*np.array([i/sigma for i in range(-3*sigma,3*sigma)])**2)/sigma
    return h

def windowed_average(data, window):
    t = time.time()
    win_data = np.convolve(data,rect_kernel(window),'same')
    return win_data

def windowed_std(data, window):
    t = time.time()
    mean_data = windowed_average(data,window)
    win_data = np.convolve((data - mean_data)**2,rect_kernel(window),'same')**0.5
    return win_data

def interpolate_tracks(data):
    frame_vec = np.where(np.ma.masked_invalid(data).mask==0)[0]
    data_interp = data[np.where(np.ma.masked_invalid(data).mask==0)[0]]
    N_nan = np.sum(np.ma.masked_invalid(data).mask)
    if frame_vec[-1]==len(data):
        frame_vector = np.arange(len(data))
    else:
        frame_vector = np.arange(frame_vec[-1]+1)
    data_array = np.array(data_interp)
    frame_array = np.array(frame_vec)  
    pt_max = interp.interp1d(frame_array,data_array,kind='cubic', fill_value="extrapolate")
    interp_data = pt_max(frame_vector[:])
    return interp_data, N_nan

def filt_trajectories(positions):
    positions_filt = np.zeros(positions.shape)
    N_NaN = np.zeros((positions.shape[1:4]))
    
    pos_shape = positions.shape
    for i in range(pos_shape[1]):
        for j in range(pos_shape[2]):
            for k in range(pos_shape[3]):
                pos_interp, N_NaN[i,j,k] = interpolate_tracks(positions[:,i,j,k])
                positions_filt[:,i,j,k] = np.ma.convolve(pos_interp,gauss_kernel95(2),propagate_mask=False)[3:-4]
    return positions_filt,N_NaN


def normalize_vec(vec):
    """ For vec of shape N x M, with N being the number of elements and M the number of coordinates"""
    vec_sum = np.sqrt(np.sum(vec**2,axis=1))
    vec_n = np.zeros(vec.shape)
    for i in range(vec.shape[1]):
        vec_n[:,i] = vec[:,i]/vec_sum
        
    return vec_n

def bin_xy(x_vec,y_vec,n_bins):
    min_x = np.nanmin(x_vec)
    max_x = np.nanmax(x_vec)#int(np.sqrt(len(x_vec)))
    if max_x == np.nan:
        max_x = np.nanquantile(x_vec,0.995)
        
    bin_size = (max_x - min_x)/n_bins

    bins = np.arange(min_x,max_x,bin_size)
    y_bin = np.zeros((bins.shape))
    for i in range(n_bins-1):
        bin_vec = np.where((x_vec>= bins[i])== (x_vec < bins[i+1]))
        y_bin[i] = np.nanmedian(y_vec[bin_vec])
    y_bin[-1] = y_bin[-2]
    return y_bin,bins

def bin_xyz(x_vec,y_vec,z_vec,n_bins):
    min_x = np.ma.min(x_vec)
    max_x = np.ma.max(x_vec)#int(np.sqrt(len(x_vec)))
        
        
    min_y = np.ma.min(y_vec)
    max_y = np.ma.max(y_vec)
        
    bin_sizex = (max_x - min_x)/n_bins
    bin_sizey = (max_y - min_y)/n_bins

    binsx = np.arange(min_x,max_x,bin_sizex)
    binsy = np.arange(min_y,max_y,bin_sizey)
    z_bin = np.zeros((binsx.shape[0],binsy.shape[0]))
    
    for i in range(n_bins-1):
        bin_vecx = list(np.where((x_vec>= binsx[i])== (x_vec < binsx[i+1]))[0])
        for j in range(n_bins-1):
            bin_vecy = list(np.where((y_vec>= binsy[j])== (y_vec < binsy[j+1]))[0])
            bin_vec = list(set(bin_vecx).intersection(bin_vecy))
            if len(bin_vec) < 1:
                z_bin[j,i] = 0
            else:
                z_bin[j,i] = np.nanmean(z_vec[bin_vec])
    return z_bin,binsx,binsy


def gen_linear_mod_fit(x,y):
    #g(x) = a + bx, y = e^g(x)/(1+e^g(x))
    
    g_x = -np.log(1/y - 1)
    sx = np.std(x)
    sg = np.std(g_x)
    rgx = np.corrcoef(x,g_x)[0,1]
    
    beta = rgx*sg/sx
    alpha = np.mean(g_x)-beta*np.mean(x)
    return beta, alpha

def LinModv1 (indep_var, depen_var, w):
    ''' indep_var should be of shape n_totalxn_elem and the same for dep_var'''
    t = time.time()
    ind_var = np.zeros((indep_var.shape[0],indep_var.shape[1]+1))
    ind_var[:,0] = np.ones((indep_var.shape[0],))
    ind_var[:,1:] = indep_var
        
    dep_var = np.zeros((depen_var.shape[0],depen_var.shape[1]+1))
    dep_var[:,1:] = depen_var
    n_elem = ind_var.shape[1]
    n_total = ind_var.shape[0]
    n_bins = int(n_total/w)
    
    B_vec = np.zeros((n_bins,n_elem,n_elem))
    Corr_vec = np.zeros((n_bins,n_elem,n_elem))
    Corr_est = np.zeros((n_bins,n_elem,n_elem))
    r_vec = np.zeros((n_bins,n_elem))
    j = 0
    for i in range(n_bins):
        Cxx = np.dot(ind_var[i*w:(i+1)*w].T,ind_var[i*w:(i+1)*w])
        Cyx = np.dot(dep_var[i*w:(i+1)*w].T,ind_var[i*w:(i+1)*w])
        Corr_vec[i] = np.cov(dep_var[i*w:(i+1)*w].T)
        if np.linalg.det(Cxx) > sys.float_info.epsilon:
            iCxx = np.linalg.inv(np.nan_to_num(Cxx))
            B_vec[i] = np.dot(Cyx,iCxx)
            dep_est = np.dot(B_vec[i],ind_var[i*w:(i+1)*w].T).T
            Corr_est[i] = np.diag(np.cov(dep_est.T,dep_var[i*w:(i+1)*w].T)[:n_elem,n_elem:])
            r_vec[i,:] = np.diag(np.corrcoef(dep_est.T,dep_var[i*w:(i+1)*w].T)[:n_elem,n_elem:])
        else:
            j += 1
            continue
    print('Number of invalid windows: ', j)
    return B_vec, r_vec, Corr_vec, Corr_est

def LinModv2 (indep_var, depen_var, w,thresh):
    ''' indep_var should be of shape n_totalxn_elem and the same for dep_var'''
    t = time.time()
    ind_var = np.zeros((indep_var.shape[0],indep_var.shape[1]+1))
    ind_var[:,0] = np.ones((indep_var.shape[0],))
    ind_var[:,1:] = indep_var
        
    dep_var = np.zeros((depen_var.shape[0],depen_var.shape[1]+1))
    dep_var[:,1:] = depen_var
    n_elem = ind_var.shape[1]
    n_total = ind_var.shape[0]
    B_vec = []
    r_vec = []
    w_vec = []
    i = 0
    while i+w < n_total:
    
    #B_vec = np.zeros((n_bins,n_elem,n_elem))
    #r_vec = np.zeros((n_bins,n_elem))
   
        j = 0
        Cxx = np.dot(ind_var[i:i+w].T,ind_var[i:i+w])
        Cyx = np.dot(dep_var[i:i+w].T,ind_var[i:i+w])
        if np.linalg.det(Cxx) > sys.float_info.epsilon:
            iCxx = np.linalg.inv(np.nan_to_num(Cxx))
            B_ = np.dot(Cyx,iCxx)
            dep_est = np.dot(B_,ind_var[i:i+w].T).T
            r_ = np.diag(np.corrcoef(dep_est.T,dep_var[i:i+w].T)[:n_elem,n_elem:])
            if np.nanmedian(r_)<thresh and w < 100:
                w += 1
            else:
                i += w
                B_vec.append(B_)
                r_vec.append(r_)
                w_vec.append(w)
        else:
            j += 1
            w += 1
            continue
    print('Number of invalid windows: ', j)
    print('Number of points evaluated: ', i)
    B_vec = np.array(B_vec)
    r_vec = np.array(r_vec)
    w_vec = np.array(w_vec)
    return B_vec, r_vec, w_vec
    
    

def min_dist_vec(X,len_vec):
    n_real = len(X.T)/len_vec
    int_win = int(n_real)
    if n_real - int_win < 0.5:
        n_win = int_win
    else:
        n_win = int_win + 1
    sample_vec = np.zeros(len_vec)
    i_stop = 0
    for i in range(len_vec):
        mean_x = np.mean(X[:,i*n_win:(i+1)*n_win],axis=1)
        dist_x = np.sum((X[:,i*n_win:(i+1)*n_win].T-mean_x)**2,axis=1)
        if len(dist_x)==0:
            i_stop = i
            break
        else:
            sample_vec[i] = np.arange(i*n_win,(i+1)*n_win)[np.argmin(dist_x)]
            i_stop += 1
    return np.array(sample_vec[:i_stop],'int32')

def diff_2d(X):
    diff_x = np.zeros(X.shape)
    for i in range(X.shape[0]):
        diff_x[i,:-1] = np.diff(X[i])
    diff_y = np.zeros(X.shape)
    for j in range(X.shape[1]):
        diff_y[:-1,j] = np.diff(X[:,j])
        
    return diff_x, diff_y

def peaks_2D(X):
    
    #Gradient
    Dby,Dbx = diff_2d(X)
    
    #Hessian
    Dbxy, Dbxx = diff_2d(Dbx)
    Dbyy, Dbyx = diff_2d(Dby)
    
    Det = Dbxx*Dbyy-Dbxy*Dbyx
    Lap = Dbxx+Dbyy #Laplacian = trace of hessian
    
    Maxima2 = Det>0
    Maxima3 = Lap<0
    
    Maxima =Maxima2*Maxima3
    return Maxima*X

def gen_rand_seq(seq):
    new_ord = np.random.permutation(len(seq))
    new_seq = seq[new_ord]
    return new_seq