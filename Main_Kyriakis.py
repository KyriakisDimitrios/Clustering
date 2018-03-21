# -*- coding: utf-8 -*-
"""
Created on Apr 2 2017
@author: Kyriakis Dimitrios
"""

import numpy as np
from Functions_Kyriakis import*
import time


EXERCISE = input("Choose exercise\n\t1.Theoretical\n\t2.Practical\n(1/2): ")

#=============================================================================#
########################## THEORETICAL ########################################
#=============================================================================#

if EXERCISE == "1":
    ################# CREATE DATASET #########################
    Size_1 = 220
    Size_2 = 280
    D =2 
    mean_1 =[1,1]
    cov_1 = 0.5*np.eye(D)
    mean_2 = [-1,-1]
    cov_2 = 0.75*np.eye(D)
    X1 = np.random.multivariate_normal(mean_1, cov_1, size=Size_1).T
    X2 = np.random.multivariate_normal(mean_2, cov_2, size=Size_2).T
    X = np.concatenate((X1,X2),axis=1)
    
    ################## NUM OF CLUSTERS ######################
    k = int(input("Number of Clusters = "))
    reps = int(input("Number of Repeats = "))
    maxiters = int(input("Max iterations in while = "))
    ############## CHOOSE CLUSTERING METHOD #################
    Clustering_type =input("Choose Clustering Method:\n\t1.K-means\n\t2.Mixure of Gaussians\n(1/2):")
    
    ####### K-MEANS #########
    if Clustering_type =="1":
        Dist_ch = input("Metric Distance:\n\t1.Euclidean\n\t2.Manhatan\n\t3.Mahalanopis\n (1/2/3): ")
        Dist_Dic ={"1":"Euclidean","2":"Manhatan","3":"Mahalanobis"}
        print("\n")
        Distance = Dist_Dic[Dist_ch]
        t1 = time.time()
        Centroids, cl_labels = kmeans(X,k,Distance,maxiters,reps)
        t2 = time.time()
        print("\n")
        print('LD Time : {m} min and {s} sec'.format(m=(t2 - t1) // 60, s=(t2 - t1) % 60))
        Title = "K-Means"
        Cluster_plot_th(X,k,Centroids,cl_labels,Title)
    
    ########## MOG ###########
    elif Clustering_type =="2":
        from MOG import*
        t1 = time.process_time()
        Centroids, z,S,Gamma = MOG(X,k,reps,"",maxiters)
        t2 = time.process_time()
        print("\n")
        print('LD Time : {m} min and {s} sec'.format(m=(t2 - t1) // 60, s=(t2 - t1) % 60))
        Title = "Mixture of Gausians"
        Cluster_plot_th(X,k,Centroids,z,Title)
        #====================================================================================#
        ##  http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals#
        #====================================================================================#
        for cluster in range(k):
            name = "C{}".format(cluster)
            plot_cov_ellipse(S[name],Centroids[:,cluster],2)

#========================================================================================#
################################## PRACTICALL  ###########################################
#========================================================================================#

elif EXERCISE=="2":
    ##DOWNLOAD DATA
#    import os
#    os.system("wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS6nnn/GDS6248/soft/GDS6248_full.soft.gz")
#    os.system('grep -v "!" GDS6248_full.soft | grep -v "#" |grep -v "\^"|grep -v "ID_REF" | awk -v f=3 -v t=53 "{for(i=f;i<=t;i++) printf("%s%s",$i,(i==t)?"\n":OFS)}" > Data')
    
    Clustering_type =input("Choose Clustering Method:\n\t1.K-means\n\t2.Mixure of Gaussians\n(1/2):")
    
    #####   LOAD YOUR DATA   #####
    X = np.loadtxt('Data') # Convert data to matrix
    
    ##### SETTING PARAMETERS #####
    k = int(input("Number of Clusters = "))
    reps = int(input("Number of Repeats = "))
    maxiters = int(input("Max iterations in while = "))
    M = int(input("How Many Dimension:\t"))

    ################ REDUCE DIMENSIONS #######################
    Y = Kernel(X,M) # Hyperbolic with d = 4
    #=======================================================#
    ##################### K-MEANS ###########################
    #=======================================================#
    if Clustering_type =="1":
        Dist_ch = input("K-means Distance Metric:\n\t1.Euclidean\n\t2.Manhatan\n\t3.Mahalanobis\n(1/2/3)")
        Dist_Dic ={"1":"Euclidean","2":"Manhatan","3":"Mahalanobis"}
        Distance = Dist_Dic[Dist_ch]
        Centroids, cl_labels = kmeans(Y,k,Distance,maxiters,reps)
        Title = "K-Means" 
        Cluster_plot(Y,k,Centroids,cl_labels,Title)
        
    #=======================================================#
    ################ MIXTURE OF GAUSIANS ####################
    #=======================================================#
    elif Clustering_type =="2":
        k_means_ch = input("Run kmeans before MOG?\n(y/n)")
        
        ############### K-MEANS BEFORE MOG ##################
        if k_means_ch == "y":
            ## C
            Dist_ch = input("K-means Distance Metric:\n\t1.Euclidean\n\t2.Manhatan\n\t3.Mahalanobis\n(1/2/3)")
            Dist_Dic ={"1":"Euclidean","2":"Manhatan","3":"Mahalanobis"}
            print("\n")
            Distance = Dist_Dic[Dist_ch]
            from MOG import*
            MEANS, cl_labels = kmeans(Y,k,Distance,maxiters,reps)
            Title = " K-Means "
            Cluster_plot(Y,k,MEANS,cl_labels,Title)
            Centroids, z,S,Gamma = MOG(Y,k,reps,MEANS,maxiters)
            
        
        ##############  MOG without K-MEANS #################        
        elif k_means_ch =="n" or k_means_ch=="":
            from MOG import*
            Centroids, z,S,Gamma = MOG(Y,k,reps,"",maxiters)

        ##############  PLOT CLUSTERS #######################
        Title = " Mixture of Gausians "
        Cluster_plot(Y,k,Centroids,z,Title)
        #====================================================================================#
        ##  http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals#
        #====================================================================================#
        if M==2:
            for cluster in range(k):
                name = "C{}".format(cluster)
                plot_cov_ellipse(S[name],Centroids[:,cluster],2)

    if M >3:
        Title = "PC1 vs PC2"
        Cluster_plot(Y[:2,:],k,Centroids,z,Title)
        Title = "PC1 vs PC3"
        Cluster_plot(Y[[0,2],:],k,Centroids,z,Title)
        Title = "PC2 vs PC3"
        Cluster_plot(Y[[1,2],:],k,Centroids,z,Title)
            
        


