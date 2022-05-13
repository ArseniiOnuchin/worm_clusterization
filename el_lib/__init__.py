import numpy as np
import pandas as pd
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigs
import math
from random import shuffle
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

class Alpha():
    
    def __init__(self, graph):
        self.graph = graph
    
    def meta_clusters(self, meta_param, clust=True):

        val_map = dict(self.graph.nodes(data=meta_param))

        values = [val_map.get(node) for node in self.graph.nodes()]

        if clust==True:
            d = {ni: indi for indi, ni in enumerate(set(values))}
            meta_clust = [d[ni] for ni in values]
            return meta_clust

        else:
            return values

        

    def adjacency_mat(self):
        adj = np.array(nx.adjacency_matrix(self.graph).todense())
        return adj

    #function for symmetrization of initial connectome matrix
    def preprocessing_matrix(self, adj):
        #adjacency matrix without multiedges
        adj_di = adj.copy()
        adj_di[adj_di > 1] = 1

        #symmetric matrix
        adj_sym = adj_di + adj_di.T
        adj_sym[adj_sym > 1] = 1

        #delete 1 from diagonal
        x = np.zeros((279,279))
        np.fill_diagonal(x, np.diag(adj_sym))
        adj_sym = adj_sym - x

        return adj_di, adj_sym

    #edges extracting on purpose of backprop matrix construction
    def edges_extracting(self, adj_sym):
        # find all edges 
        graph_sym = nx.from_numpy_matrix(adj_sym)
        edges_gr_sym = np.array(graph_sym.edges)
        edges_gr_sym_i = edges_gr_sym[::,::-1]
        all_edg_sym = np.concatenate((edges_gr_sym, edges_gr_sym_i))
        return all_edg_sym

    #Nonbacktracking matrix construction function
    def nonback(self, all_edg_sym):
        nonback = np.zeros((np.shape(all_edg_sym)[0], np.shape(all_edg_sym)[0]))
        for i in range(np.shape(all_edg_sym)[0]): 
            for j in range(np.shape(all_edg_sym)[0]): 
                if all_edg_sym[i,1] == all_edg_sym[j,0] and all_edg_sym[i,0] != all_edg_sym[j,1]: 
                    nonback[i,j] = 1
        return nonback

    #Flowmatrix construction function (from adjacency matrix, nonbacktracking matrix and list of edges)
    def flowmat(self, adj, nonback, all_edg_sym):
        degrees = np.sum(adj, axis=1)
        flow_matrix = np.copy(nonback)
        for i in range(np.shape(all_edg_sym)[0]): 
            for j in range(np.shape(all_edg_sym)[0]): 
                if nonback[i,j] != 0 and degrees[all_edg_sym[j][0]] > 1:
                    flow_matrix[i,j] = 1/(degrees[all_edg_sym[j][0]] - 1)
        return flow_matrix

    #Function for translation eigenvalues from flowmatrix to symmetric adjacency matrix of connectome
    def translation_eig_vec(self, eig_vals, eig_vecs, edges, adj, max_clust_num, tail=False):
        degrees = np.sum(adj, axis=1)
        cr_rad = np.sqrt(np.mean(np.array(degrees)/(np.array([x-1 if x>1 else x for x in degrees])))/(np.mean(degrees)))
        if tail == True: 
            eig_vals = eig_vals[eig_vals > cr_rad]
        order = np.argsort(-np.abs(np.array(eig_vals)))
    
        vecs = eig_vecs[:,order[1:max_clust_num]]
        vals = np.array(eig_vals)[order[1:max_clust_num]] 
    
        len_of_tail = np.shape(vals)[0]
        
        eig = np.zeros((np.shape(adj)[0],len(vals)))

        for i in range(len(edges)):  
            for k in range(len(vals)):
                eig[edges[i][0],k] += vecs[i,k]

        return vals, eig, cr_rad, len_of_tail

    def sorted_cluster(self, x, model=KMeans()):
        model = self.sorted_cluster_centers_(model, x)
        model = self.sorted_labels_(model, x)
        return model

    def sorted_cluster_centers_(self, model, x):
        model.fit(x)
        new_centroids = []
        magnitude = []
        for center in model.cluster_centers_:
            magnitude.append(np.sqrt(center.dot(center)))
        idx_argsort = np.argsort(magnitude)
        model.cluster_centers_ = model.cluster_centers_[idx_argsort]
        return model

    def sorted_labels_(self, sorted_model, x):
        sorted_model.labels_ = sorted_model.predict(x)
        return sorted_model

    # kmeans on first 9 eigenvectors (each cluster on all vectors)
    def fm_clusters_all(self, translated_eig_vec):
        colours = np.zeros((np.shape(translated_eig_vec)[1], np.shape(translated_eig_vec)[0]))

        for i in np.arange(1,np.shape(translated_eig_vec)[1]+1):
            if np.shape(translated_eig_vec)[1] != 0:
            
                cluster = KMeans(n_clusters=i+1, n_init=1000, max_iter = 20000)
                cluster.fit(translated_eig_vec)
                cluster = self.sorted_cluster(translated_eig_vec, cluster)
                colours[i-1,:] = cluster.labels_
             
            else:  
                colours[i-1,:] = [0] * np.shape(translated_eig_vec)[0]   
        return colours
    
    # kmeans on first 9 eigenvectors (each n clusters on n-1 vectors)
    def fm_clusters_pervec(self, translated_eig_vec):
        colours = np.zeros((np.shape(translated_eig_vec)[1], np.shape(translated_eig_vec)[0]))

        for i in np.arange(1,np.shape(translated_eig_vec)[1]+1):
            if np.shape(translated_eig_vec)[1] != 0:
            
                cluster = KMeans(n_clusters=i+1, n_init=1000, max_iter = 20000)
                cluster.fit(translated_eig_vec[:,:i+1])
                cluster = self.sorted_cluster(translated_eig_vec[:,:i+1], cluster)
                colours[i-1, :] = cluster.labels_
    
            else:  
                colours[i-1, :] = [0] * np.shape(translated_eig_vec)[0]   
        return colours

    #Ordering adjacency matrix by clusters
    def order_matrix(self, mat, labels):
        indices = np.argsort(labels)
        mat_sorted = np.copy(mat)
        for i in range(len(indices)):
            for j in range(len(indices)):
                if i == j:
                    mat_sorted[i, j] = 0
                else:
                    mat_sorted[i, j] = mat[indices[i], indices[j]]
        return mat_sorted


    def power(self, cl_sizes, adj_mat):
    #function to get adjancency matrix with clusters ordered by cluster power
    #cl_sizes -- sizes of the clusters
    #adj_mat -- adjancency matrix ordered by clusters

        powers = np.zeros(np.shape(cl_sizes))
        frams = self.fram(cl_sizes, len(cl_sizes))

        for i in range(0,np.shape(cl_sizes)[0]):
            if i == 0:
                cluster_edges = np.sum(adj_mat[0:int(frams[0]), 0:int(frams[0])])
                outside_cl_edges = np.sum(adj_mat[0:int(frams[0]), :])*2 #because matrix symmetric
                powers[i] = cluster_edges/(outside_cl_edges - cluster_edges)
            else:
                cluster_edges = np.sum(adj_mat[int(frams[i-1]):int(frams[i]), int(frams[i-1]):int(frams[i])])
                outside_cl_edges = np.sum(adj_mat[int(frams[i-1]):int(frams[i]), :])*2 #because matrix symmetric
                powers[i] = cluster_edges/(outside_cl_edges - cluster_edges)
        return powers

    def order_distances(self, distances, labels):
        indices = np.argsort(labels)
        dist_sorted = np.copy(distances)
        for i in range(len(indices)):
            dist_sorted[i] = distances[indices[i]]
        return dist_sorted 

    def colours_sbm(self, translated_eig_vec):
        if np.shape(translated_eig_vec)[1] != 0: 
            cluster = KMeans(n_clusters=np.shape(translated_eig_vec)[1]+1, n_init=400, max_iter=9000)
            cluster.fit(translated_eig_vec)
            clusters = self.sorted_cluster(translated_eig_vec, cluster)
            colours = np.array(clusters.labels_)            
        else:  
            colours = np.array([0] * np.shape(translated_eig_vec)[0])            
        return colours

    def clusters_laplac(self, v, num_of_clusters):
        D = np.diag(v.sum(axis=1))
        L = D-v
        vals, vecs = np.linalg.eig(L)
        vals = vals[np.argsort(vals)]
        vecs = vecs[:,np.argsort(vals)]
        colours = [[] for _ in range(num_of_clusters)]
        for i in range(1,num_of_clusters+1):
            y = vecs[:,1:i+1]
            clusters = KMeans(n_clusters=i+1)
            clusters.fit(y)
            clusters = self.sorted_cluster(y, clusters)
            colours[i-1] = clusters.labels_
        return y, colours
    
    def clusters_norm_laplac(self, adj_matrix, num_of_clusters):
        ones = np.linspace(1,1,np.shape(adj_matrix)[0])
        I = np.diag(ones)
        D = np.diag(adj_matrix.sum(axis=1))
        D_power = np.nan_to_num(np.power(D, -1/2), posinf=0.0)
        L_norm = I - D_power @  adj_matrix @ D_power

        vals, vecs = np.linalg.eig(L_norm)
        vals = vals[np.argsort(vals)]
        vecs = vecs[:,np.argsort(vals)]
        colours = [[] for _ in range(num_of_clusters)]

        for i in range(1,num_of_clusters+1):
            y = vecs[:,1:i+1]
  
            clusters = KMeans(n_clusters=i+1)
            clusters.fit(y)
            clusters = self.sorted_cluster(y, clusters)
            colours[i-1] = clusters.labels_
        return y, colours
    
    #little bit different way, but the results almost the same
    def clusters_norm_laplac_alisa(self, adj_matrix, num_of_clusters):
        I = np.identity(np.shape(adj_matrix)[0])
        D = np.diag(adj_matrix.sum(axis=1))
  
        D_power = fractional_matrix_power(D,-1/2)
        L_norm = I - D_power @  adj_matrix @ D_power
        vals, vecs = np.linalg.eig(L_norm)
        vals = vals[np.argsort(vals)]
        vecs = vecs[:,np.argsort(vals)]
        colours = [[] for _ in range(num_of_clusters)]
        for i in range(1,num_of_clusters+1):
            y = vecs[:,1:i+1]
            kmeans = KMeans(n_clusters=i+1)
            kmeans.fit(y)
            colours[i-1] = kmeans.labels_

        return y, colours

    def modularity_matrix(self, adjacency_matrix):
        B = np.zeros(np.shape(adjacency_matrix))
        L = len(self.edges_extracting(adjacency_matrix))

        for i in range(np.shape(adjacency_matrix)[0]):
            for j in range(np.shape(adjacency_matrix)[0]):
                B[i,j] = (adjacency_matrix[i,j] - (np.sum(adjacency_matrix[i,:]) * np.sum(adjacency_matrix[j,:]) / L) )
        return B

    def clusters_modularity_matrix(self, modular_m, num_of_clusters):

        vals, vecs = np.linalg.eig(modular_m)
        vals = vals[np.argsort(vals)]
        vecs = vecs[:,np.argsort(vals)]

        colours = [[] for _ in range(num_of_clusters)]

        for i in range(num_of_clusters):
            y = vecs[:,:i+1].real

            clusters = KMeans(n_clusters=i+2)
            clusters.fit(y)
            clusters = self.sorted_cluster(y, clusters)
            colours[i] = clusters.labels_
        return colours
    
    #functions for win and wout calculating from clusterized worm matrix
    def complete_graph_edges(self, n):
        return n*(n-1)//2

    #function which will take a massive of colours
    def subgraphs(self, colours): 
        indices = np.argsort(colours)
        number = len(np.unique(colours))
        x = colours[indices]
        subgraphs = [[] for i in range(number)]
        for i in range(number):
            for j in range(len(x)):
                if x[j] == i: 
                    subgraphs[i].append(indices[j])
        return subgraphs

    def emp_probabilities(self, subgraphs, adj_sym):
        graph_final = nx.from_numpy_array(adj_sym)
        edges_in = []
        edges_in_all = []
    
        for i in range(len(subgraphs)):
            sub = graph_final.subgraph(subgraphs[i])
            edges_in.append(len(sub.edges))
            edges_in_all.append(self.complete_graph_edges(len(subgraphs[i])))
        
        edges_out = (len(graph_final.edges) - np.sum(np.array(edges_in))) 
    
        probability_in = np.sum(np.array(edges_in)) / np.sum(np.array(edges_in_all))
        probability_out = edges_out  / (self.complete_graph_edges(adj_sym.shape[0]) - np.sum(np.array(edges_in_all)))
    
        return edges_in, edges_in_all, edges_out, probability_in, probability_out, graph_final

    #cluster borders
    def fram(self, cluster_sizes, number_of_clusters):
        frames=np.zeros(number_of_clusters)
        frames[0]=cluster_sizes[0]
    
        for i in range(1,number_of_clusters):
            frames[i]=frames[i-1]+cluster_sizes[i]
        
        return frames  

    #cluster detection
    def clcheck(self, a, cluster_sizes, number_of_clusters):
        if a>=0 and a < self.fram(cluster_sizes, number_of_clusters)[0]:
            return 0
        else:    
            for i in range(0,number_of_clusters):
                if a >= self.fram(cluster_sizes, number_of_clusters)[i] and a < self.fram(cluster_sizes, number_of_clusters)[i+1]:
                    return i+1   

    #SBM generation
    def gensbm(self, number_of_nodes, number_of_clusters, connection_probabilities, cluster_sizes):
        sbm=np.zeros((number_of_nodes,number_of_nodes))
        clusters = []
        for i in range(0,number_of_nodes):
            clusters.append(self.clcheck(i,cluster_sizes, number_of_clusters))
            for j in range(0,i):
                if self.clcheck(i,cluster_sizes, number_of_clusters)==self.clcheck(j,cluster_sizes, number_of_clusters):
                    sbm[i,j]=np.random.choice([0, 1], p=[1-connection_probabilities[1], connection_probabilities[1]])
                    sbm[j,i]=sbm[i,j]
                else: 
                    sbm[i,j]=np.random.choice([0, 1], p=[1-connection_probabilities[0], connection_probabilities[0]])
                    sbm[j,i]=sbm[i,j]
        return sbm, clusters   
    
    #Function for mutual information calculation 
    #Kronecker delta
    def dlt(self, a, b):
        if a==b:
            return 1
        else:
            return 0
    
    #mutual info
    def mutinf(self, clusters, colors):
        a=0  
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if i!=j:
                    x = self.dlt(colors[i],colors[j])
                    y = self.dlt(clusters[i],clusters[j])
                    a += self.dlt(x,y)
        return a/((len(colors))**2-len(colors))  

    #modularity coefficient
    def modularity(self, adjacency_matrix, clusters):
        L = len(self.edges_extracting(adjacency_matrix))
        x = 0

        for i in range(np.shape(adjacency_matrix)[0]):
            for j in range(np.shape(adjacency_matrix)[0]):
                x += (adjacency_matrix[i,j] - (np.sum(adjacency_matrix[i,:]) * np.sum(adjacency_matrix[j,:]) / L) ) * self.dlt(clusters[i],clusters[j])
        mod_coeff = x / L
        return mod_coeff
    
    def pipeline(self, adj_sym, number_of_clusters, tail_state):
        all_edg_sym = self.edges_extracting(adj_sym)
        nb_matrix = self.nonback(all_edg_sym)
        flow_matrix = self.flowmat(adj_sym, nb_matrix, all_edg_sym)
        vals_flow, vecs_flow = eigs(flow_matrix, k = number_of_clusters+3, which='LR')

        vals, eig_vec, cr_rad, len_of_tail = self.translation_eig_vec(vals_flow, vecs_flow, all_edg_sym, adj_sym, number_of_clusters, tail_state)
    
        coloursbm = self.colours_sbm(eig_vec)
    
        return cr_rad, vals_flow, vals, coloursbm, len_of_tail

    def sbm_generation(self, w_out, w_in, classes):
        col=[[] for i in range(len(w_in))]
        sbm = [[] for i in range(len(w_in))]
        clusters = [[] for i in range(len(w_in))]
    
        cr_rad=[[] for i in range(len(w_in))]
        vals_flow = [[] for i in range(len(w_in))]
        vals = [[] for i in range(len(w_in))]

        for i in range(len(w_in)):
            sbm[i], clusters[i] = self.gensbm(279,len(classes),[w_out[i],w_in[i]],classes)
            cr_rad[i], vals_flow[i], vals[i], col[i] = self.pipeline(sbm[i], len(classes))
  
        return sbm, col, clusters, cr_rad, vals_flow, vals
    
    def MI(self, w_in, clusters, col):

        mutual_info_skl = np.zeros(len(w_in))
        mutual_info_self = np.zeros(len(w_in))
        ars_flow = np.zeros(len(w_in))
    
        for i in range(len(w_in)):
            mutual_info_self[i] = self.mutinf(clusters[i],col[i])
            mutual_info_skl[i] = adjusted_mutual_info_score(col[i],clusters[i]) 
            ars_flow[i] = adjusted_rand_score(col[i],clusters[i]) 
        
        return mutual_info_self, mutual_info_skl, ars_flow

    def MI_meta(self, colours, meta_clust):

        adj_mutual_info = np.zeros(len(colours))
        mutual_info = np.zeros(len(colours))
        adj_rand_score = np.zeros(len(colours))
        norm_mi = np.zeros(len(colours))
    
        for i in range(len(colours)):
            adj_mutual_info[i] = adjusted_mutual_info_score(meta_clust, colours[i])
            mutual_info[i] = mutual_info_score(meta_clust, colours[i])
            adj_rand_score[i] = adjusted_rand_score(meta_clust, colours[i])
            norm_mi[i] = normalized_mutual_info_score(meta_clust, colours[i])
        return adj_mutual_info, mutual_info, adj_rand_score, norm_mi


    def mean_MI_ARS_scores(self, adj_matrix, meta_clust, translated_eig_vec, iterations=200):
    
        measure_of_sim_fm = np.zeros((4, translated_eig_vec.shape[1], iterations))
        measure_of_sim_nl = np.zeros((4, translated_eig_vec.shape[1], iterations))
        measure_of_sim_m = np.zeros((4, translated_eig_vec.shape[1], iterations))
        M = self.modularity_matrix(adj_matrix)
    
        for i in range(iterations):    
            fm_cl_all = self.fm_clusters_all(translated_eig_vec)
            vecs_laplac_norm, nl = self.clusters_norm_laplac_alisa(adj_matrix, translated_eig_vec.shape[1])
            clusters_modularity = self.clusters_modularity_matrix(M, translated_eig_vec.shape[1])
        
            measure_of_sim_fm[0,:,i], measure_of_sim_fm[1,:,i], measure_of_sim_fm[2,:,i], measure_of_sim_fm[3,:,i] = self.MI_meta(fm_cl_all, meta_clust)
            measure_of_sim_nl[0,:,i], measure_of_sim_nl[1,:,i], measure_of_sim_nl[2,:,i], measure_of_sim_nl[3,:,i] = self.MI_meta(nl, meta_clust)
            measure_of_sim_m[0,:,i], measure_of_sim_m[1,:,i], measure_of_sim_m[2,:,i], measure_of_sim_nl[3,:,i] = self.MI_meta(clusters_modularity, meta_clust)
        
        return measure_of_sim_fm, measure_of_sim_nl, measure_of_sim_m
    
    #clusters from sbm by laplacians and modularity 
    #ARS, AMI for them
    def clusters_sbm(self, sbm, clusters, number_of_clusters):
    
        mi_self = np.zeros((len(sbm), len(sbm[0]), 4))
        mi_skl = np.zeros((len(sbm), len(sbm[0]), 4))
        ars = np.zeros((len(sbm), len(sbm[0]), 4))
    
        for i in range(len(sbm)):
            for j in range(len(sbm[0])):
                vecs_laplac, colours_laplac = self.clusters_laplac(sbm[i][j], number_of_clusters)
                vecs_laplac_norm, colours_laplac_norm = self.clusters_norm_laplac(sbm[i][j], number_of_clusters)
                vecs_laplac_norm_alisa, colours_laplac_norm_alisa = self.clusters_norm_laplac_alisa(sbm[i][j], number_of_clusters)
                M = self.modularity_matrix(sbm[i][j])
                clusters_modularity = self.clusters_modularity_matrix(M, number_of_clusters)
            
        
                mi_self[i, j, 0] = self.mutinf(clusters[0][0], colours_laplac[number_of_clusters-1])
                mi_skl[i, j, 0] = adjusted_mutual_info_score(colours_laplac[number_of_clusters-1], clusters[0][0])
                ars[i, j, 0] = adjusted_rand_score(colours_laplac[number_of_clusters-1], clusters[0][0])
            
                mi_self[i, j, 1] = self.mutinf(clusters[0][0], colours_laplac_norm[number_of_clusters-1])
                mi_skl[i, j, 1] = adjusted_mutual_info_score(colours_laplac_norm[number_of_clusters-1], clusters[0][0])
                ars[i, j, 1] = adjusted_rand_score(colours_laplac_norm[number_of_clusters-1], clusters[0][0])
            
                mi_self[i, j, 2] = self.mutinf(clusters[0][0], colours_laplac_norm_alisa[number_of_clusters-1])
                mi_skl[i, j, 2] = adjusted_mutual_info_score(colours_laplac_norm_alisa[number_of_clusters-1], clusters[0][0])
                ars[i, j, 2] = adjusted_rand_score(colours_laplac_norm_alisa[number_of_clusters-1], clusters[0][0])
            
                mi_self[i, j, 3] = self.mutinf(clusters[0][0], clusters_modularity[number_of_clusters-1])
                mi_skl[i, j, 3] = adjusted_mutual_info_score(clusters_modularity[number_of_clusters-1], clusters[0][0])
                ars[i, j, 3] = adjusted_rand_score(clusters_modularity[number_of_clusters-1], clusters[0][0])
            
        return mi_self, mi_skl, ars

    def rand_cols(self, clst_sizes, number_of_clusters):
        a = range(0, sum(clst_sizes))
        shuffle(a)
        rand_colors = np.zeros(sum(clst_sizes))
        for i in range(sum(clst_sizes)):
            rand_colors[a[i]] = self.clcheck(i, clst_sizes, number_of_clusters)
        return rand_colors

    #
    def p_vals(self, clusters, clst_sizes, number_of_clusters, groups, title):
        clvec=[[0 for _ in range(sum(clst_sizes))] for _ in range(1000)]
        for i in range(1000):
            clvec[i] = self.rand_cols(clst_sizes, number_of_clusters)
        distvec=[[[] for i in range(len(clst_sizes))] for j in range(1000)]
        for i in range(1000):
            for j in range(sum(clst_sizes)):  
                distvec[i][int(clvec[i][j])].append(j)  
        overlap=[np.zeros((len(clst_sizes), max(groups)+1)) for i in range(1000)]
        for i in range(1000):
            for j in range(len(clst_sizes)):
                for l in range(len(distvec[i][j])):
                    for k in range(max(groups)+1):
                        if groups[distvec[i][j][l]] == k:
                            overlap[i][j,k]+=round(1/list(groups).count(k),5)  
        our_vrlp=np.zeros((len(clst_sizes), max(groups)+1))
        for i in range(sum(clst_sizes)):
            our_vrlp[clusters[i],groups[i]]+=round(1/list(groups).count(groups[i]),5)  
        vrlp_sample=[[[] for i in range(len(clst_sizes))] for j in range(max(groups)+1)]
        for k in range(max(groups)+1):
            for j in range(len(clst_sizes)):
                 for i in range(1000):
                    vrlp_sample[k][j].append(overlap[i][j,k])    
        vals=np.zeros((len(clst_sizes), max(groups)+1))
        for i in range(len(clst_sizes)):
            for j in range(max(groups)+1):
                c = np.array(vrlp_sample[j][i])
                vals[i,j] = round(len(c[c>=our_vrlp[i,j]])/1000,5)     
        vals_logged = np.zeros((len(clst_sizes), max(groups)+1))
        for i in range(len(clst_sizes)):
            for j in range(max(groups)+1):
                if vals[i][j] != 0:
                    vals_logged[i][j] = -(math.log(vals[i][j],10))
                else:
                    vals_logged[i][j] = 5       
        df_logged = pd.DataFrame(vals_logged)
        plt.imshow(df_logged)
        plt.colorbar()
        plt.title(title)
        plt.show()      

    def dens(self, mat, clst_vec, clst_sizes, number_of_clusters, title):
        ord_mat = self.order_matrix(mat, clst_vec, number_of_clusters)
        denst = np.zeros((len(clst_sizes),len(clst_sizes)))
        for i in range(len(clst_vec)):
            for j in range(len(clst_vec)):
                k=self.clcheck(i, clst_sizes)
                l=self.clcheck(j, clst_sizes)
            if k==l:
                denst[k,l]+=ord_mat[i,j]/(clst_sizes[k]**2-clst_sizes[k])
            else:
                denst[k,l]+=ord_mat[i,j]/(clst_sizes[k]*clst_sizes[l])
        plt.imshow(denst)  
        plt.title(title)  
        plt.colorbar()
        plt.show()

    
    def coords_diff_pics(self, gang, col, name_x='clusters', name_y='ganglions'):
        n = len(set(col))
        m = len(set(gang))
        a=np.zeros((n,m))
        for i in range(len(col)):
            #a[int(col[i]),gang[i]] += 1/(gang.count(gang[i]))
            a[int(col[i]),gang[i]] += 1/(col.count(col[i]))
        x = []
        for i in range(n):
            x += [i+1] * m
        y = list(range(1,m+1)) * n
    
    
        fig = plt.figure(figsize=(20, 20))
        z = [0] * (m*n)
        dx = [0.5] * (m*n)
        dy = [0.5] * (m*n)
        dz = np.array(a).flatten()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(name_x, fontsize=20)
        ax.set_ylabel(name_y, fontsize=20)
        ax.set_zlabel('overlap', fontsize=20)
        colors = ['limegreen' if u > 0.5 else 'crimson' for u in dz]
        numb = len(dz[dz > 0.5])
        ax.bar3d(x,y,z, dx, dy, dz, color=colors, alpha=0.5)
        ax.set_title(f'{n} {name_x}, the number of green bars is {numb}', fontsize=20)
        plt.show()
    
    def self_adj_rsc(self,cl1,cl2):
        k=len(cl1)
        avg = (k**2-2*k+2)/(k**2)
        return (self.mutinf(cl1, cl2) - avg)/(1-avg)

    def cluster_limit(self,colours, adj_sym):
        probability_in = np.zeros((1, len(colours)))
        probability_out = np.zeros((1, len(colours)))
        c_in = np.zeros((1, len(colours)))
        c_out = np.zeros((1, len(colours)))
        c = np.zeros((1, len(colours)))
        optimal_clusters = np.zeros((len(colours),3))
    
        for i in range(len(colours)):
            subgr = self.subgraphs(colours[i])
            edges_in, edges_in_all, edges_out, probability_in[:,i], probability_out[:,i], graph_final = self.emp_probabilities(subgr, adj_sym)
    
        c_in = probability_in*adj_sym.shape[0]
        c_out = probability_out*adj_sym.shape[0]
        c = (c_in + c_out)/2
    
        for i in range(np.shape(c_in)[1]):
            optimal_clusters[i,:] = [c_in[:,i]-c_out[:,i], (i+2)*math.sqrt(c[:,i]), i+2]
        
        return c_in, c_out, probability_in, probability_out, c, optimal_clusters
    
    #probability statistics (w_in and w_out) for clusterization
    def w_statistics(self, clusters, adj_sym):
        w = np.zeros((len(clusters), 2))
        for i in range(len(clusters)):
            subgr = self.subgraphs(clusters[i])
            edges_in, edges_in_all, edges_out, w[i,0], w[i,1], graph_final = self.emp_probabilities(subgr, adj_sym)
        return w
    
    def mean_optimal_cluster(self, translated_eig_vec, iterations, adj_sym, fm_cl_pervec, fm_cl_all):
        mean_optimal_cluster_pv = np.zeros((np.shape(translated_eig_vec)[1],3,iterations))
        mean_optimal_cluster_all = np.zeros((np.shape(translated_eig_vec)[1],3,iterations))
    
        w = np.zeros((np.shape(translated_eig_vec)[1], 2, 5, iterations))
        M = self.modularity_matrix(adj_sym)
    
        clusters_all = np.zeros((iterations,translated_eig_vec.shape[1], translated_eig_vec.shape[0]))
        clusters_pv = np.zeros((iterations,translated_eig_vec.shape[1], translated_eig_vec.shape[0]))
    
        for i in range(iterations):    
            clusters_all[i,:,:] = self.fm_clusters_pervec(translated_eig_vec)
            clusters_pv[i,:,:] = self.fm_clusters_all(translated_eig_vec)
            vl, laplace = self.clusters_laplac(adj_sym, np.shape(translated_eig_vec)[1])
            nvl, norm_laplac = self.clusters_norm_laplac(adj_sym, np.shape(translated_eig_vec)[1])
            modularity = self.clusters_modularity_matrix(M, np.shape(translated_eig_vec)[1])
        
            c_in, c_out, w_in, w_out, c, mean_optimal_cluster_pv[:,:,i] = self.cluster_limit(fm_cl_pervec, adj_sym)
            c_in, c_out, w_in, w_out, c, mean_optimal_cluster_all[:,:,i] = self.cluster_limit(fm_cl_all, adj_sym)
        
            w[:,:,0,i] = self.w_statistics(clusters_pv[i,:,:], adj_sym)
            w[:,:,1,i] = self.w_statistics(clusters_all[i,:,:], adj_sym)
            w[:,:,2,i] = self.w_statistics(laplace, adj_sym)
            w[:,:,3,i] = self.w_statistics(norm_laplac, adj_sym)
            w[:,:,4,i] = self.w_statistics(modularity, adj_sym)
        
        mean_partition_pv = np.mean(mean_optimal_cluster_pv, axis=2)
        mean_partition_all = np.mean(mean_optimal_cluster_all, axis=2)
    
        return mean_partition_pv, mean_partition_all, w, clusters_all, clusters_pv
    
    #number of contacts between all pairs of verteces degrees in connectome
    def correlation_deg(self, adj):
        deg_in = np.sum(adj, axis=0).astype('int64')
        deg_out = np.sum(adj, axis=1).astype('int64')
        degree = deg_in + deg_out
    
        if np.sum(deg_in - deg_out) == 0:
            degree = degree//2
        
        E = np.zeros((np.max(degree), np.max(degree)))
        for i in range(np.max(degree)):
            for j in range(np.max(degree)):
                if i in degree and j in degree:
                    indices1 = [k for k, x in enumerate(degree) if x == i]
                    indices2 = [l for l, x in enumerate(degree) if x == j]
                    for m in range(len(indices1)):
                        for n in range(len(indices2)):
                            E[i,j] += adj[m,n]    
        return E

    def corr_deg_bin(self, adj):
    
        deg_in = np.sum(adj, axis=0).astype('int64')
        deg_out = np.sum(adj, axis=1).astype('int64')
        degree = deg_in + deg_out
    
        if np.sum(deg_in - deg_out) == 0:
            degree = degree//2
    
        bins = np.array(np.logspace(0, np.log(np.max(degree)), num=7, base=np.e), dtype=int)
    
        E = np.zeros((len(bins), len(bins)))   
        ps = np.zeros(len(bins),)
    
        for i in range(len(bins)-1):
            i_bin = []
            for j in range(bins[:-1][i], bins[1:][i]):
                i_bin.append(np.sum(np.diagonal(adj, offset=j)))
            ps[i] = np.mean(np.array(i_bin))
        
        for i in range(len(bins)):
            for j in range(len(bins)):
                if i in degree and j in degree:
                    indices1 = [k for k, x in enumerate(degree) if x == i]
                    indices2 = [l for l, x in enumerate(degree) if x == j]
                    for m in range(len(indices1)):
                        for n in range(len(indices2)):
                            E[i,j] += adj[m,n]    
        return E
    
    #distant probability P(s)
    #sorted by physical distances matrix needed as an argument 
    def d_prob(self, s, mat): 
        a = 0 
        for i in range(mat.shape[0]-s): 
            a += mat[i,i+s] 
        a = a/(mat.shape[0]-s)   
        return a 

    def delta_x(self, s, mat,soma_positions):
        s_diag = np.diagonal(mat, offset=s)
        distances = np.zeros(len(s_diag),)
    
        for i in range(len(s_diag)):
            distances[i] = s_diag[i] * abs(soma_positions[i] - soma_positions[i+s])
        return np.mean(distances)    

    def ps_bins(self, mat):
        x = np.array(np.logspace(0, np.log(np.shape(mat)[0]), num=8, base=np.e), dtype=int)
        ps = np.zeros(len(x),)
    
        for i in range(len(x)-1):
            i_bin = []
            for j in range(x[:-1][i], x[1:][i]):
                i_bin.append(np.sum(np.diagonal(mat, offset=j)))
            ps[i] = np.mean(np.array(i_bin))
        return ps

    def delta_x_bins(self, mat, x_delta):
        x = np.array(np.logspace(0, np.log(np.shape(mat)[0]), num=8, base=np.e), dtype=int)
        delta_x_bin = np.zeros(len(x),)
    
        for i in range(len(x)-1):
            i_bin = []
            for j in range(x[:-1][i], x[1:][i]):
                i_bin.append(x_delta[j])
            delta_x_bin[i] = np.mean(np.array(i_bin))
        return delta_x_bin 

    def sums_md(self,mat,cols):
        sum_mat=[0,0] #first -- in
        sum_dist=[0,0]
        ps_clusters = np.zeros((2,len(np.unique(cols))))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i!=j:
                    if cols[i] == cols[j]:
                        sum_mat[0] += mat[i,j]
                        sum_dist[0] += self.d_prob(abs(i-j),mat)
                        #if matrix have been sorted by clusters it will save P(s) and W_in for each cluster
                        ps_clusters[0,int(cols[i])] += self.d_prob(abs(i-j),mat)
                        ps_clusters[1,int(cols[i])] += mat[i,j]
                    else:
                        sum_mat[1] += mat[i,j]
                        sum_dist[1] += self.d_prob(abs(i-j),mat)
                    
        rel_in = sum_mat[0]/sum_dist[0]  
        rel_out = sum_mat[1]/sum_dist[1]         
        return rel_in, rel_out, sum_dist, ps_clusters

    #generate SBM with scaling effect
    def sbm_w_scaling(self, number_of_nodes, number_of_clusters, connection_probabilities, cluster_sizes, mat, param):
        sbm=np.zeros((number_of_nodes,number_of_nodes))
        for i in range(0,number_of_nodes):
            for j in range(0,number_of_nodes):
                k=self.d_prob(abs(i-j),mat)
                if i==j:
                    sbm[i,j]=0
                elif self.clcheck(i,cluster_sizes, number_of_clusters) == self.clcheck(j,cluster_sizes, number_of_clusters):
                    sbm[i,j]=np.random.choice([0, 1], p=[1-(connection_probabilities[1])*k*param, (connection_probabilities[1])*k*param])
                else: 
                    sbm[i,j]=np.random.choice([0, 1], p=[1-(connection_probabilities[0])*k*param, (connection_probabilities[0])*k*param])
        return sbm  

    def cost_int(self, sorted_adj, zeta, alpha):
        C_int = 0
        for i in range(np.shape(sorted_adj)[0]):
            for j in range(np.shape(sorted_adj)[1]):
               C_int += sorted_adj[i,j] * (i - j)**zeta
        
        return alpha*C_int