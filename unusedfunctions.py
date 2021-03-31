"""
@author: Rafael Almada
"""

import numpy as np
import time
from scipy import signal, stats,special
from scipy import interpolate as interp
import math as math
import sys as sys
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage

def windowed_correlation(data0, data1, window):
    t = time.time()
    N_rows=int(data0.shape[0]/window)
    md0 = windowed_average(data0,window)
    s0 = windowed_std(data0,window)
    md1 = windowed_average(data1,window)
    s1 = windowed_std(data1,window)
    windowed_data = np.convolve((data0-md0)*(data1-md1),rect_kernel(window),'same')/(s0*s1)
    return windowed_data


def auto_correlation(data, window):
    t = time.time()
    N_rows=int(data0.shape[0]/window)
    data_array = np.zeros((N_rows,window))
    for i in range(N_rows):
        data_array[i,:]=data[i*window:i*window+window]
    windowed_data = np.diag(np.ma.corrcoef(data_array)[1:,:-1])
    return windowed_data

def angle_xcorr(data1,data2):
    t = time.time()
    N = len(data1)
    a_corr = np.zeros((2*N-1,))
    for i in range(1,2*N-1):
        #a_corr.append(np.mean(np.cos((data1[:-i]-data2[i:]))))
        if i > N:
            xn = data1[:N-i]
            yn = data2[i-N:]
            a_corr[i] = (np.mean(np.cos((xn-yn))))
            
        if i <= N:
            xn = data1[N-i:]
            yn = data2[:i]            
            a_corr[i] = (np.mean(np.cos((xn-yn))))

    return a_corr

##Derivatives

def der(data,dt):
    return np.diff(data)/dt

##Signal analysis

def max_Freq_time(t, f, Sxx):
    sizeSxx = Sxx.shape[1]
    max_index = np.zeros((sizeSxx,))
    f_max = np.zeros((sizeSxx,))
    for p in range(sizeSxx):
        Sxxp = list(Sxx[:,p])
        max_index[p] = int(Sxxp.index(max(Sxxp)))
        f_max[p] = f[int(max_index[p])]
    return f_max


# For EMD import EMD from PyEMD, for hilbert use signal.hilbert()

#Exploring lagged variables


def mse_time(data):
    mse_d = []
    for i in range(1,len(data)):
        mse_d.append(np.median(np.abs((data[:-i]-data[i:]))**2))
    mse_d = np.array(mse_d)
    return mse_d

def entropy(data,lower,upper):
    data_size = len(data)
    num_bins = int(np.sqrt(data_size))
    s_dist = stats.relfreq(data,num_bins,(lower, upper))
    H = 0
    for i in range(num_bins):
        H += -s_dist.frequency[i]*np.log(max(s_dist.frequency[i],1/data_size))
    return H

def mutual_entropy(data1,data2,k,std):
    # k delays 
    data_size = min(len(data1),len(data2))
    num_bins = int(np.sqrt(data_size))
    
    bs1 = (np.max(data1)-np.min(data1))/num_bins
    bs2 = (np.max(data2)-np.min(data2))/num_bins
    
    
    if k != 0:
        C_ij = np.histogram2d(data1[:data_size][:-k],data2[:data_size][k:],num_bins,density=True)[0]
    else:
        C_ij = np.histogram2d(data1[:data_size],data2[:data_size],num_bins,density=True)[0]
    
    C_ij = ndimage.gaussian_filter(C_ij,std)
    C_ij = C_ij/np.sum(C_ij)
    C_i = np.sum(C_ij,axis=1)
    C_j = np.sum(C_ij,axis=0)

    I = 0
    for i in range(len(C_i)):
        for j in range(len(C_j)):
            if C_ij[i,j] != 0 and ((C_i[i] != 0) and (C_j[j] != 0)):
                I -= C_ij[i,j]*np.log2(C_ij[i,j]/(C_i[i]*C_j[j]))

    return I, bs1, bs2

def windowed_entropy(data,window,lower,upper,steps):
    t = time.time()
    len_data = len(data)
    win_en = np.zeros((len_data-window))
    for i in range(0,len_data-window-steps,steps):
        win_en[i] = entropy(data[i:i+window],lower,upper)
        win_en[i:i+steps] = win_en[i]*np.ones((steps,))
    print("elapsed time for windowed entropy is: ", time.time()-t)
    return win_en


def windowed_mutual_entropy(data1, data2,window,lower,upper,steps):
    t = time.time()
    len_data = min(len(data1),len(data2))
    win_en = np.zeros((len_data-window))
    for i in range(0,len_data-window-steps,steps):
        win_en[i] = mutual_entropy(data1[i:i+window],data2[i:i+window],lower,upper)
        win_en[i:i+steps] = win_en[i]*np.ones((steps,))
    print("elapsed time for windowed entropy is: ", time.time()-t)
    return win_en

def ang_corr_mat(dataarray):
    N = dataarray.shape[0]
    corr = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            corr[i,j] = np.median(np.cos(dataarray[i,:]-dataarray[j,:]))
    return corr
            
def probwin(data, window, nn, overlap, l_low, l_high): 

    """
       Creates an array displaying the histogram of a variable over time
       Input: data - a 1D array
       window -  int corresponding to size of window
       nn - number of bins in histogram, determines histogram's resolution
       overlap - number of overlapping frames between each window
    """

    tt = time.time()
    
    N = data.shape[0]
    pdf_array=np.zeros((int(N/window),nn))
    vec_var = np.zeros((nn,))
    time_var = np.array([i*window for i in range(int(N/window))])
    max_vec = np.zeros(int(N/window))
    skew_vec = np.zeros(int(N/window))
    min_lim = l_low
    max_lim = l_high    
    s = 1/2*(max_lim-min_lim)/int(N)
    for i in range(0,int(N/window)-1):
        
        pdf1,a,b,c = stats.relfreq(data[i*window:(i+1)*window],nn,(min_lim-s,max_lim+s))
        pdf2,low_lim,bin_s,c = stats.relfreq(data[(i+1)*window-overlap:(i+2)*window-overlap],nn,(min_lim-s,max_lim+s))
        
        pdf_array[i,:] = pdf1
    
        skew_vec[i] = np.nansum((data[i*window:(i+1)*window]-np.nanmean(data[i*window:(i+1)*window]))**3)/window
        max_vec[i] = bin_s*np.argmax(pdf1)+low_lim
    
    vec_var = np.array([i*bin_s+low_lim for i in range(nn)])
    print(time.time()-tt)
    
    return skew_vec, max_vec, vec_var, time_var, pdf_array

def intrinsic_coord(pos):
    """Coordinate transformation to intrinsic fish centric coordinates
    Returns the fish centric trajectories and the 3D rotation matrix of the HP vector
    """
    
    np.seterr(divide='ignore', invalid='ignore')
    
    N_total = pos.shape[0]

    pos_transl = np.zeros((pos.shape))

    #translation of points
    for i in range(pos.shape[2]):
        pos_transl[:,:,i,:] = pos[:,:,i,:]-pos[:,:,1,:]
    
    #rotation of points
    #Rotation matrix
    r_1 = np.sum(pos_transl[:,:,0,:]**2,axis=2)**0.5
    vec_1 = np.zeros((N_total,2,3))
    for i in range(pos.shape[3]):
        vec_1[:,:,i] = pos_transl[:,:,0,i]/r_1
    
    #Compute the orthogonal complement to v_1
    vec_2 = np.zeros((vec_1.shape))
    vec_2[:,:,0]= np.ones((N_total,2))
    vec_2[:,:,2]= -vec_1[:,:,0]/vec_1[:,:,2]
    
    r_2 = np.sum(vec_2[:,:,:]**2,axis=2)**0.5
    for i in range(pos.shape[3]):
        vec_2[:,:,i] = vec_2[:,:,i]/r_2
    
    vec_3 = np.zeros((vec_1.shape))
    vec_3[:,:,1]= np.ones((N_total,2))
    vec_3[:,:,2]= -vec_1[:,:,1]/vec_1[:,:,2]

    #Gram-Schmidt process to create a orthogonal basis

    #since the vector contains NaN elements we need to manually define a dot product
    dot_prod = np.zeros((vec_2.shape[0],vec_2.shape[1]))
    for i in range(pos.shape[0]):
        for f in range(pos.shape[1]):
            dot_prod[i,f] =  np.nansum(vec_2[i,f,:]*vec_3[i,f,:])


    for k in range(pos.shape[3]):
        for f in range(pos.shape[1]):
            vec_3[:,f,k] = vec_3[:,f,k] -dot_prod[:,f]*vec_2[:,f,k]
        
    r_3 = np.sum(vec_3[:,:,:]**2,axis=2)**0.5
    for i in range(pos.shape[3]):
        vec_3[:,:,i] = vec_3[:,:,i]/r_3
    
    Rot = np.zeros((pos.shape))
    Rot[:,:,0,:] = vec_1
    Rot[:,:,1,:] = vec_3
    Rot[:,:,2,:] = vec_2

    
    pos_intrinsic = np.zeros((pos.shape))

    #rotation of points
    for i in range(pos.shape[2]):
        for j in range(pos.shape[3]):
            pos_intrinsic[:,:,i,j] = np.sum(pos_transl[:,:,i,:]*Rot[:,:,j,:],axis=2)
    return pos_intrinsic, Rot

def kstest_emp(kde1,kde2,alpha,N,M):
    """
    This function does the Kolmogorov-Smirnov test to compare two empirical distributions
    Inputs:
    kde1, kde2 - histograms with same number of bins
    alpha - significance level
    N, M - the number of samples in both datasets used
    
    Returns:
    A boolean that is equal to True if the distributions are different and False otherwise
    """
    cdf1 = np.cumsum(kde1)
    cdf2 = np.cumsum(kde2)
    Dn = np.max(np.abs(cdf1-cdf2)) #Test statistic
    if N==M:
        Da = np.sqrt(-np.log(alpha/2)*0.5)*np.sqrt(2/N)
    else:
        Da = np.sqrt(-np.log(alpha/2)*0.5)*np.sqrt((N+M)/(N*M))
    
    pv = 2*np.exp(-(N*Dn**2/2))
    print('p-value is : ',pv)
    print('Test statistic : ',Dn)
    print('$D_\\alpha$ for $\\alpha$ ',alpha, ' is ',Da)
    
        
    return Dn>Da

def nancorrcoef_eqv2(x,y,N):
    cxy = np.zeros((2*N,))
    L = 0
    if N != len(x):
        L = len(x)
    else:
        L = N
    l_vec = []
    for i in range(2*N):
        
        if i > N:
            xn = x[:L+N-i]
            yn = y[i-N:]
            
            x_nan1 = []
            x_nan2 = []
            for j in range(L+N-i):
                if (math.isnan(xn[j])== 0) and (math.isnan(yn[j])== 0) :
                    x_nan1.append(xn[j])
                    x_nan2.append(yn[j]) 
            x_nan1 = np.array(x_nan1)
            z_1 = (x_nan1-np.mean(x_nan1))/np.std(x_nan1)
            x_nan2 = np.array(x_nan2)
            z_2 = (x_nan2-np.mean(x_nan2))/np.std(x_nan2)
            cxy[i] = np.sum(z_1*z_2)
            
        if i <= N:
            xn = x[N-i:]
            yn = y[:L-N+i]
            x_nan1 = []
            x_nan2 = []
            for j in range(len(xn)):
                if (math.isnan(xn[j])== 0) and (math.isnan(yn[j])== 0) :
                    x_nan1.append(xn[j])
                    x_nan2.append(yn[j])
            x_nan1 = np.array(x_nan1)
            z_1 = (x_nan1-np.mean(x_nan1))/np.std(x_nan1)
            x_nan2 = np.array(x_nan2)
            z_2 = (x_nan2-np.mean(x_nan2))/np.std(x_nan2)
            cxy[i] = np.sum(z_1*z_2)
        
        l_vec.append(i-N)
    return l_vec, cxy

def nancorrcoef_eq(x,y):
    N = len(x)
    cxy = np.zeros((2*N,))
    l_vec = []
    for i in range(2*N):
        
        if i > N:
            xn = x[:N-i]
            yn = y[i-N:]
            x_nan1 = []
            x_nan2 = []           
            for j in range(len(xn)):
                if (math.isnan(xn[j])== 0) and (math.isnan(yn[j])== 0) :
                    x_nan1.append(xn[j])
                    x_nan2.append(yn[j]) 
            x_nan1 = np.array(x_nan1)
            z_1 = (x_nan1-np.mean(x_nan1))/np.std(x_nan1)
            x_nan2 = np.array(x_nan2)
            z_2 = (x_nan2-np.mean(x_nan2))/np.std(x_nan2)
            cxy[i] = np.sum(z_1*z_2)
            
        if i <= N:
            xn = x[N-i:]
            yn = y[:i]
            x_nan1 = []
            x_nan2 = []
            for j in range(len(xn)):
                if (math.isnan(xn[j])== 0) and (math.isnan(yn[j])== 0) :
                    x_nan1.append(xn[j])
                    x_nan2.append(yn[j])
            x_nan1 = np.array(x_nan1)
            z_1 = (x_nan1-np.mean(x_nan1))/np.std(x_nan1)
            x_nan2 = np.array(x_nan2)
            z_2 = (x_nan2-np.mean(x_nan2))/np.std(x_nan2)
            cxy[i] = np.sum(z_1*z_2)
        
        l_vec.append(i-2*N)
    return l_vec, cxy

def nancorrcoef(x,y):
    #x_nan = []
    #y_nan = []
    #for j in range(len(x)):
    #    if (math.isnan(x[j])== 0) and (math.isnan(y[j])== 0) :
    #        x_nan.append(x[j])
    #        y_nan.append(y[j])
    #x_nan = np.array(x_nan)
    #y_nan = np.array(y_nan)
    a = np.ma.masked_invalid(x)
    b = np.ma.masked_invalid(y)
    msk = (~a.mask&~b.mask)
    x_nan = x[msk]
    y_nan = y[msk]
    cxy = np.ma.corrcoef(x_nan,y_nan)
    return cxy

def nancov(x,y):
    #x_nan = []
    #y_nan = []
    #for j in range(len(x)):
    #    if (math.isnan(x[j])== 0) and (math.isnan(y[j])== 0) :
    #        x_nan.append(x[j])
    #        y_nan.append(y[j])
    #x_nan = np.array(x_nan)
    #y_nan = np.array(y_nan)
    a = np.ma.masked_invalid(x)
    b = np.ma.masked_invalid(y)
    msk = (~a.mask&~b.mask)
    x_nan = x[msk]
    y_nan = y[msk]
    cxy = np.ma.cov(x_nan-np.mean(x_nan),y_nan-np.mean(y_nan))
    return cxy

def Correlate_lag_win(data1,data2,win):
    N = data1.shape[0]
    n_bins = int(N/win)
    Corr_mat = np.zeros((n_bins,win*2-1))
    

    for i in range(n_bins):
        f1 = data1[i*win:(i+1)*win]
        f2 = data2[i*win:(i+1)*win]
        x1 =  np.ma.masked_invalid(f1)
        x2 = np.ma.masked_invalid(f2)
        m1 =x1-np.ma.mean(x1)
        m2 = x2-np.ma.mean(x2)
        zx1 = m1/np.ma.std(x1)
        zx2 = m2/np.ma.std(x2)
        lab = (~x1.mask & ~x2.mask)
        
        x_nan1 = zx1[lab]
        x_nan2 = zx2[lab]
        
        v = np.arange(0,win)[lab]
        
        lv = np.array([(win-1-np.sort(v)[::-1]),win-1+v]).flatten()
        

        if (len(x_nan1)<win) & (len(x_nan1) > 0) :
            ll = win-len(x_nan1)
            lu = win+len(x_nan1)-1
        else: 
            ll = 0
            lu = 2*win-1
        if (len(x_nan1)>0)&(len(lv)>2):
            Corr_mat[i,lv[1:]] = np.ma.correlate(x_nan1,x_nan2,mode='full',propagate_mask=False)/(win-1)
    return Corr_mat, np.arange(-win,win),np.arange(0,n_bins)


def CorrAng_lag_win(data1,data2,win):
    N = data1.shape[0]
    n_bins = int(N/win)
    Corr_mat = np.zeros((n_bins,win*2-1))
    

    for i in range(n_bins):
        f1 = data1[i*win:(i+1)*win]
        f2 = data2[i*win:(i+1)*win]
        x1 =  np.ma.masked_invalid(f1)
        x2 = np.ma.masked_invalid(f2)
        lab = (~x1.mask & ~x2.mask)
        
        x_nan1 = np.exp(1j*x1[lab])
        x_nan2 = np.exp(1j*x2[lab])
        
        v = np.arange(0,win)[lab]
        
        lv = np.array([(win-1-np.sort(v)[::-1]),win-1+v]).flatten()

        if (len(x_nan1)<win) & (len(x_nan1) > 0) :
            ll = win-len(x_nan1)
            lu = win+len(x_nan1)-1
        else: 
            ll = 0
            lu = 2*win-1
        if len(x_nan1)>0:
            
            Corr_mat[i,lv] = np.ma.correlate(x_nan1,np.conj(x_nan2),mode='full',propagate_mask=False)/(win-1)
    return Corr_mat, np.arange(-win,win),np.arange(0,n_bins)

def plot_w_error(data,xl,yl,h,n):
    conv_tracks = np.ma.convolve(data,h,'full',propagate_mask=False)
    std_tracks = np.ma.convolve((data-conv_tracks[:len(data)])**2,h,'full',propagate_mask=False)**0.5
    x = np.array([i for i in range(std_tracks.shape[0])])
    f1 = conv_tracks+std_tracks
    f2 = conv_tracks-std_tracks

    ax = plt.subplot(n)
    ax.fill_between(x,f2,f1,alpha=0.35)
    ax.plot(conv_tracks)
    plt.xlabel(xl,fontsize=18)
    plt.ylabel(yl,fontsize=18)
    plt.grid(True)

    
def on_top(data,threshold):
    on_top_vec = np.zeros(len(data))
    pos_val = np.where(data > threshold)[0]
    neg_val = np.where(data < -threshold)[0]
    on_top_vec[pos_val] = 1
    on_top_vec[neg_val] = -1
    return on_top_vec
    
#Original code from https://www.kaggle.com/tigurius/introduction-to-taken-s-embedding

def takensEmbedding (data, delay, dimension):
    "This function returns the Takens embedding of data with delay into dimension, delay*dimension must be < len(data)"
    if delay*dimension > len(data):
        raise NameError('Delay times dimension exceed length of data!')    
    embeddedData = np.array([data[0:len(data)-delay*dimension]])
    for i in range(1, dimension):
        embeddedData = np.append(embeddedData, [data[i*delay:len(data) - delay*(dimension - i)]], axis=0)
    return embeddedData;



def mutualInformation(data, delay, nBins):
    "This function calculates the mutual information given the delay"
    I = 0;
    xmax = max(data);
    xmin = min(data);
    delayData = data[delay:len(data)];
    shortData = data[0:len(data)-delay];
    sizeBin = abs(xmax - xmin) / nBins;
    #the use of dictionaries makes the process a bit faster
    probInBin = {};
    conditionBin = {};
    conditionDelayBin = {};
    for h in range(0,nBins):
        if h not in probInBin:
            conditionBin.update({h : (shortData >= (xmin + h*sizeBin)) & (shortData < (xmin + (h+1)*sizeBin))})
            probInBin.update({h : len(shortData[conditionBin[h]]) / len(shortData)});
        for k in range(0,nBins):
            if k not in probInBin:
                conditionBin.update({k : (shortData >= (xmin + k*sizeBin)) & (shortData < (xmin + (k+1)*sizeBin))});
                probInBin.update({k : len(shortData[conditionBin[k]]) / len(shortData)});
            if k not in conditionDelayBin:
                conditionDelayBin.update({k : (delayData >= (xmin + k*sizeBin)) & (delayData < (xmin + (k+1)*sizeBin))});
            Phk = len(shortData[conditionBin[h] & conditionDelayBin[k]]) / len(shortData);
            if Phk != 0 and probInBin[h] != 0 and probInBin[k] != 0:
                I -= Phk * math.log( Phk / (probInBin[h] * probInBin[k]));
    return I;

def false_nearest_neighours(data,delay,embeddingDimension):
    "Calculates the number of false nearest neighbours of embedding dimension"    
    embeddedData = takensEmbedding(data,delay,embeddingDimension);
    #the first nearest neighbour is the data point itself, so we choose the second one
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(embeddedData.transpose())
    distances, indices = nbrs.kneighbors(embeddedData.transpose())
    #two data points are nearest neighbours if their distance is smaller than the standard deviation
    epsilon = np.std(distances.flatten())
    nFalseNN = 0
    for i in range(0, len(data)-delay*(embeddingDimension+1)):
        if (0 < distances[i,1]) and (distances[i,1] < epsilon) and ( (abs(data[i+embeddingDimension*delay] - data[indices[i,1]+embeddingDimension*delay]) / distances[i,1]) > 10):
            nFalseNN += 1;
    return nFalseNN


