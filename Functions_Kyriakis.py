## -*- coding: utf-8 -*-
#Created on Fri Apr 2017
#@author: Kyriakis Dimitrios


########################### Libraries  ########################################
import numpy as np
import sys
#=============================================================================#


#=============================================================================#
##########################  NORMALIZARION #####################################
#=============================================================================#
def Normalization(X):    
    D = X.shape[0]
    N = X.shape[1]
    maxi =np.amax(X,axis=1)
    mini = np.amin(X,axis=1)
    mean = np.mean(X,axis=1)
    mean_matrix = np.repeat((mean).reshape(D,1),N,axis =1) 
    max_min = np.repeat((maxi-mini).reshape(D,1),N,axis =1)
    X = (X-mean_matrix)/max_min
    return X,mean_matrix,max_min

#=============================================================================#


#=============================================================================#
################################# K-MEANS  ####################################
#=============================================================================#

def kmeans(X,k,Distance,maxiters,reps):
    import scipy.spatial.distance
    ################  NORMALIZARION ####################
    X, mean_matrix, max_min = Normalization(X)
    D = X.shape[0]
    N = X.shape[1]
    S = np.cov(X)
    names = ["C"+str(x) for x in range(k)]
    e = 1e-10
    ################  Distance Measures ######################
    Eucl_func =  lambda x,y: np.linalg.norm(x.reshape(D,1)-y.reshape(D,1))
    Manha_func = lambda x,y: abs(np.sum(x.reshape(D,1)-y.reshape(D,1)))
    Mahala_func =  lambda x,y: np.sqrt((x-y).reshape(1,D).dot(np.linalg.inv(S)).dot((x.reshape(D,1)-y.reshape(D,1)).reshape(D,1)))[0][0]
    if Distance =="Euclidean":
        Dist_func = Eucl_func
    elif Distance =="Manhatan":
        Dist_func = Manha_func
    elif Distance =="Mahalanobis":
        S = np.cov(X)
        Dist_func = Mahala_func
    
    ##########################   Repetition  #################################

    for r in range(reps):        
        ####### RANDOM Centrioids #######
        indx = np.random.randint(X.shape[1],size=k)
        Centr_old = X[:,indx]
        diff = 10
        max_iters = maxiters
        
        #######################  Convergence ##################################
        while diff > e:
            #################### INITIALIZE ###################################
            max_iters -= 1
            Cluster_Dic = dict.fromkeys(names,np.zeros((D,1)))
            Cluster_Num = dict.fromkeys(names,0)
            Centr_new = np.zeros((D,k))
            labels =[]
            ##################  PARSING VECTROS ###############################
            for i in range(N):
                ################### DISTANCE FROM CENTROIDS ###################
                dist=[]
                for c in range(Centr_old.shape[1]):
                    Metric_distance = Dist_func(Centr_old[:,c],X[:,i])
                    dist.append(Metric_distance)   
                clash = "C"+str(dist.index(min(dist)))
                clash_num = dist.index(min(dist))
                labels.append(clash_num)
                ################### ADD VECTROR TO CLUSTER ####################
                if Cluster_Num[clash] == 0:
                    Cluster_Num[clash] += 1
                    Cluster_Dic[clash] = X[:,i].reshape(D,1)
                else:
                    Cluster_Dic[clash] = np.concatenate((Cluster_Dic[clash],X[:,i].reshape(D,1)),axis=1)
                #################E  RE CALCULATE CENTROID #####################
                new = np.average(Cluster_Dic[clash],axis=1)
                Centr_new[:,clash_num] = new                
            ##########  DISTANCE BETWEEN OLD AND NEW CENTROIDS ################
            diff = 0
            for cluster in range(k):
                diff += Dist_func(Centr_new[:,cluster],Centr_old[:,cluster])
            sys.stdout.write("\rDIFF_W: {:.2e}\tRepetition: {}".format(diff,r+1))
            sys.stdout.flush()
            Centr_old = np.copy(Centr_new)
            if max_iters == 0:
                break
        ##################### DISTANCE OLD VS NEW CLUSTER #####################
        ################ KEEP CLUSTER WITH THE SMALLER DISTANCE ###############
        dist_C_new = 0
        for cl in Cluster_Dic.keys():
            dist_metric = {"Euclidean":'euclidean',"Manhatan":'cityblock', "Mahalanobis":'mahalanobis'}
            if Distance == "Mahalanobis":
                dist_C_new += scipy.spatial.distance.pdist(Cluster_Dic[cl],dist_metric[Distance], VI= np.linalg.inv(S))[0]
            else:
                dist_C_new += scipy.spatial.distance.pdist(Cluster_Dic[cl],dist_metric[Distance])[0]
        if r == 0:
             Centroids = np.copy(Centr_new)
             dist_C_old = dist_C_new
             cl_labels = labels
        else:
            if dist_C_new < dist_C_old:
                Centroids = np.copy(Centr_new)
                dist_C_old = dist_C_new
                cl_labels = labels
    print("\nCluster Distance {:.5}".format(dist_C_new))  
#    print(Centroids.shape)
    Centroids  = (Centroids*max_min[:,:k])+mean_matrix[:,:k]      
    return(Centroids,cl_labels)

#/////////////////////////////////////////////////////////////////////////////#




###############################################################################
""" 
    # 3.  Kernel PCA
       VC (Vapnik-Chervonenkis) theory tells us that often mappings 
       which take us into a higher dimensional space than the dimension of the 
       input space provide us with greater classification power
"""
###############################################################################

def Kernel(X,M):
    #### PRE-PROCESS
    D = X.shape[0]
    N = len(X[0])
    mean = X.mean(axis=1)
    mean_matrix = np.array([mean]*N)
    X = (X-mean_matrix.T)
    # KERNELS
    Kernel_choice = input("Choose kernel:\n\t1.Gaussian = exp(-γ||xi-xj||**2)\n\t2.Polynomial  = (1+(xixj))**p\n\t3.Hyperbolic tangent = tanh(xixj+δ)\nKernel (1/2/3)= ")
    Kernel_dic = {"1":"γ","2":"p","3":"δ"}
    Kernel_parameter = input("Choose value for {}:\t".format(Kernel_dic[Kernel_choice]))
    b=p=d = float(Kernel_parameter)
    Gaussian = lambda xi,xj : np.exp((-b*(np.linalg.norm(xi-xj)**2)))		
    Polynomial = lambda xi,xj : (1+(np.inner(xi,xj)))**p
    Hyperbolic_tangent = lambda xi,xj : np.tanh(np.inner(xi,xj)+d)
    count =0
    lista=[]
    for i in range(len(X[0])):
        if len(lista) !=0:
            if count == 1:
                l = np.array(lista)
                K = np.array([l])
            else:
                l = np.array([lista])
                K = np.concatenate((K,l),axis=0)
        lista=[]
        count +=1
        for j in range(len(X[0])):
            if Kernel_choice == "1": 
                z = Gaussian(X[:,i],X[:,j])
            elif Kernel_choice == "2":
                z = Polynomial(X[:,i],X[:,j])
            elif Kernel_choice == "3":
                z = Hyperbolic_tangent(X[:,i],X[:,j])
            lista.append(z)
    l = np.array([lista])
    K = np.concatenate((K,l),axis=0)
    print("K.shape = {}".format(K.shape))        
    N_1 = np.ones((N,N))*(1/N)
    Kbar = K - N_1.dot(K) - K.dot(N_1) +N_1.dot(K).dot(N_1)
    eigen_Vals,eigen_Vecs = np.linalg.eig(Kbar)
    idx = eigen_Vals.argsort()[::-1]   
    U = eigen_Vecs[:,idx[:M]]
    Y = U.T.dot(Kbar)
    return Y




#=============================================================================#
################################## PLOT #######################################
#=============================================================================#
####THEORETICAL PLOTS ####
def Cluster_plot_th(X,k,Centroids,cl_labels,Title):
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score
    from matplotlib import cm as cm
    silhouette_avg = silhouette_score(X.T, cl_labels,metric='euclidean')
    color_list =cm.brg(np.linspace(0,1,k))
    counter = [0]*k
    if X.shape[0]==2:      
        fig2, bx = plt.subplots(figsize=(10, 6))
        bx = plt.subplot(111)
        bx.grid()
        for i,f in zip(range(X.shape[1]),cl_labels):
            x = X[0,i]
            y = X[1,i]    
            if counter[f] == 0:
                bx.scatter(x, y,marker=".",s=80,c=color_list[f],label="Class{} = {}".format(f+1,cl_labels.count(f)))
                counter[f] += 1
            else:
                bx.scatter(x, y,marker=".",s=80,c=color_list[f])
        x = Centroids[0,:]
        y = Centroids[1,:]
        bx.scatter(x, y,marker="x",s=150,c="black",label="Centroids")
        bx.scatter(x, y,marker=" ", label='Silhoutte Score = {}'.format(round(silhouette_avg,2)))
        bx.set_xlabel("X Axis")
        bx.set_ylabel("Y Axis")
        chartBox = bx.get_position()
        bx.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
        bx.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=2)
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        bx = fig.add_subplot(111, projection='3d')
        bx.grid()
        for i,f in zip(range(X.shape[1]),cl_labels):
            x = X[0,i]
            y = X[1,i]   
            z = X[2,i]
            if counter[f] == 0:
                bx.scatter(x, y,z,marker=".",c=color_list[f],label="Class{} = {}".format(f+1,cl_labels.count(f)))
                counter[f] += 1
            else:
                bx.scatter(x, y,z,marker=".",c=color_list[f])
        x = Centroids[0,:]
        y = Centroids[1,:]
        z = Centroids[2,:]
        bx.scatter(x, y,z,marker="x",s=150,c="black",label="Centroids")
        bx.scatter(x, y,z,marker=' ', label='Silhoutte Score = {}'.format(round(silhouette_avg,2)))
        bx.set_xlabel("X Axis")
        bx.set_ylabel("Y Axis")
        bx.set_zlabel("Z Axis")
        chartBox = bx.get_position()
        bx.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
        bx.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=2)
    plt.title(Title)
    plt.show()
    return

########### PRACTICAL ###############
def Cluster_plot(X,k,Centroids,cl_labels,Title):
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score
    from matplotlib import cm as cm
    silhouette_avg = silhouette_score(X.T, cl_labels,metric='euclidean')
    color_list = cm.brg(np.linspace(0,1,k))
    counter = [0]*k
    if X.shape[0]==2:      
        fig2, bx = plt.subplots(figsize=(10, 6))
        bx = plt.subplot(111)
        bx.grid()
        controls =[0,0,0]
        normal_e = [1]*18
        normal = [2]*6
        high_e = [3]*6
        high = [4]*18
        original = controls+normal_e+normal+high_e+high
        list_mark = ('o','x','*', 'v','s', 'p', '*', 'h', 'H', 'D', 'd', 'P')
        for i,f,ori in zip(range(X.shape[1]),cl_labels,original):
            x = X[0,i]
            y = X[1,i]    
            if counter[f] == 0:
                bx.scatter(x, y,marker=list_mark[ori],c=color_list[f],label="Class{} = {}".format(f+1,cl_labels.count(f)))
                counter[f] += 1
            else:
                bx.scatter(x, y,marker=list_mark[ori],c=color_list[f])
        x = Centroids[0,:]
        y = Centroids[1,:]
        bx.scatter(x, y,marker="x",s=150,c="black",label="Centroids")
        bx.scatter(x, y,marker=" ", label='Silhoutte Score = {}'.format(round(silhouette_avg,2)))
        bx.set_xlim([-10,10])
        bx.set_ylim([-7.5,7.5])
        bx.set_xlabel("X Axis")
        bx.set_ylabel("Y Axis")
        chartBox = bx.get_position()
        bx.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
        bx.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=2)

    else:
        ##### 3D  ###################
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        bx = fig.add_subplot(111, projection='3d')
        bx.grid()
        controls =[0,0,0]
        normal_e = [1]*18
        normal = [2]*6
        high_e = [3]*6
        high = [4]*18
        original = controls+normal_e+normal+high_e+high
        list_mark = ('o','x', 'v','*','s', 'p', '*', 'h', 'H', 'D', 'd', 'P')
        for i,f,ori in zip(range(X.shape[1]),cl_labels,original):
            x = X[0,i]
            y = X[1,i]   
            z = X[2,i]
            if counter[f] == 0:
                bx.scatter(x, y,z,marker=list_mark[ori],c=color_list[f],label="Class{} = {}".format(f+1,cl_labels.count(f)))
                counter[f] += 1
            else:
                bx.scatter(x, y,z,marker=list_mark[ori],c=color_list[f])
        x = Centroids[0,:]
        y = Centroids[1,:]
        z = Centroids[2,:]
        bx.scatter(x, y,z,marker="x",s=150,c="black",label="Centroids")
        bx.scatter(x, y,z,marker=' ', label='Silhoutte Score = {}'.format(round(silhouette_avg,2)))
        bx.set_xlabel("X Axis")
        bx.set_ylabel("Y Axis")
        bx.set_zlabel("Z Axis")
        chartBox = bx.get_position()
        bx.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
        bx.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=2)
    plt.title(Title)
    plt.show()
    return

#//////////////////////////////////////////////////////////////////////////////#





#====================================================================================#
##  http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals#
#====================================================================================#
def plot_cov_ellipse(cov, pos,nstd=2 , ax=None):#nstd=2
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    for i in range(nstd):
        s = i+1
        width, height = 2 * s * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,ec="red",fc="none",lw =2,ls="dashed")
        ax.add_artist(ellip)
    return ellip



