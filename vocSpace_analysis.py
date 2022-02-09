#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 07 14:42:32 2021

@author: silviapagliarini
"""
import os
import scipy as sp
import numpy as np
import matplotlib.animation as animation
from matplotlib import rcParams, cm, colors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import librosa
import librosa.feature
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import librosa.display
import librosa.effects
import Song_functions
import umap
import umap.plot
import scipy.signal as signal
import pandas as pd
import csv
import arff
import matplotlib
import seaborn as sns

def multidim_all(classes, babies, age, args):
    """
    Function to compare multidimensional sounds across families.
    """
    # Define colormap
    classes_colors = ['darkblue', 'red', 'gold']
    classes_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("manual", classes_colors)
    plt.register_cmap("manual", classes_cmap)

    # Labels
    labels = []
    colors = []
    sum_mfcc_list = []
    average_mfcc_list = []
    basename = []
    babyname = []
    timename = []
    agegroup = []
    for b in range(0, len(babies)):
        print(babies[b])

        for c in range(0, len(classes)):
            # Load data
            data = glob2.glob(args.data_dir + '/' + babies[b] + '/' + classes[c] + '/' + '*.mfcc.csv')

            if args.portion == True:
                how_many = args.portion_size
                if how_many > len(data):
                    how_many = len(data)
            else:
                how_many = len(data)

            i = 0
            while i < how_many:
                babyname.append(babies[b])
                basename_aux = os.path.basename(data[i])
                # timename_aux = basename_aux[20:20+5]
                timename_aux = basename_aux[22:22 + 5]
                if timename_aux[-1] == '_':
                    timename_aux = timename_aux[0:-1]
                elif timename_aux[-2] == '_':
                    timename_aux = timename_aux[0:-2]
                elif timename_aux[-3] == '_':
                    timename_aux = timename_aux[0:-3]

                if len(timename_aux) == 1:
                    timename.append('0000' + timename_aux)
                elif len(timename_aux) == 2:
                    timename.append('000' + timename_aux)
                elif len(timename_aux) == 3:
                    timename.append('00' + timename_aux)
                elif len(timename_aux) == 4:
                    timename.append('0' + timename_aux)
                else:
                    timename.append(timename_aux)

                basename.append(basename_aux[0:-9])

                if age[b] < 180:
                    agegroup.append('3mo')
                elif (age[b] >= 180 and age[b] < 250):
                    agegroup.append('6mo')
                elif (age[b] >= 250 and age[b] < 500):
                    agegroup.append('9mo')
                else:
                    agegroup.append('18mo')

                labels.append(classes[c])
                colors.append(classes_colors[c])

                # Load mfcc table
                mfcc_table = pd.read_csv(data[i], sep=';')
                mfcc_table = mfcc_table.to_numpy()

                sum_mfcc = 0
                k = 0
                while k < np.shape(mfcc_table)[0]:
                    sum_mfcc = sum_mfcc + mfcc_table[k][2::]
                    k = k + 1

                sum_mfcc_list.append(sum_mfcc)
                average_mfcc_list.append(sum_mfcc / len(sum_mfcc))

                i = i + 1

    labels = np.asarray(labels)
    agegroup = np.asarray(agegroup)

    legend_elements = []
    for l in range(0, len(classes)):
        legend_elements.append(
            Line2D([], [], marker='o', color=classes_colors[l], markersize=10, label=classes[l]))

    # UMAP
    mapper_sum = umap.UMAP(random_state=args.seed, spread=args.spread, n_neighbors=args.n_neigh, min_dist=args.min_d,
                           n_components=args.n_comp).fit(np.array(sum_mfcc_list))

    umap.plot.points(mapper_sum, np.asarray(labels), color_key_cmap="manual") #, background='black')
    plt.savefig(
        args.data_dir + '/' + 'ALLbabies_LENAlabels_' + '_opensmile_day_UMAP_mfcc_sum_' + str(
            args.n_neigh) + '_' + str(int(args.portion_size)) + '.pdf')
    plt.close('all')

    mapper_avg = umap.UMAP(random_state=args.seed, spread=args.spread, n_neighbors=args.n_neigh, min_dist=args.min_d,
                       n_components=args.n_comp).fit(np.array(average_mfcc_list))
    umap.plot.points(mapper_avg, np.asarray(labels), color_key_cmap='manual', background='black')
    plt.savefig(
        args.data_dir + '/' + 'ALLbabies_LENAlabels_' + '_opensmile_day_UMAP_mfcc_avg_' + str(
            args.n_neigh) + '_' + str(int(args.portion_size)) + '.pdf')
    plt.close('all')

    # t-SNE
    tsne_result = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(np.array(sum_mfcc_list))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, s=0.1, alpha=0.8)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('y-SNE 2')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(handles=legend_elements)
    plt.savefig(args.data_dir + '/' + 'ALLbabies_LENAlabels_' + '_opensmile_day_TSNE_mfcc_sum_' + str(
        args.n_neigh) + '_' + str(int(args.portion_size)) + '.pdf')

    # PCA
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(np.array(sum_mfcc_list))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pca_result[:,0], pca_result[:,1], c=colors, s=0.1, alpha=0.8)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(handles=legend_elements)
    plt.savefig(args.data_dir + '/' + 'ALLbabies_LENAlabels_' + '_opensmile_day_PCA_mfcc_sum_' + str(
            args.n_neigh) + '_' + str(int(args.portion_size)) + '.pdf')

    # Create summary
    with open(args.data_dir + '/' + 'ALL_BABYADULT_summary_LENAlabels_' + '_opensmile_day_UMAP_mfcc.csv', 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(['umap ID', 'filename', 'baby_ID', 'time ID', 'label', 'labelID', 'umap sum x', 'umap sum y', 'umap avg x', 'umap avg y', 'age'])

        i = 0
        while i < len(basename):
            csvwriter.writerow([i, basename[i], babyname[i], timename[i], labels[i], np.where(np.asarray(classes)==labels[i])[0][0], mapper_sum.embedding_[i][0], mapper_sum.embedding_[i][1], mapper_avg.embedding_[i][0], mapper_avg.embedding_[i][1], agegroup[i]])
            i = i + 1

    # Load data and create a single csv for each baby (to work better)
    summary = pd.read_csv(args.data_dir + '/' + 'ALL_BABYADULT_summary_LENAlabels_' + '_opensmile_day_UMAP_mfcc.csv')
    summary = pd.DataFrame.to_numpy(summary)

    for b in range(0, len(babies)):
        basename_aux = summary[:,1][np.where(summary[:,2]==babies[b])]
        timename_aux = summary[:,3][np.where(summary[:,2]==babies[b])]
        labels_aux = summary[:,4][np.where(summary[:,2]==babies[b])]
        labelsID_aux = summary[:,5][np.where(summary[:,2]==babies[b])]
        umapXsum = summary[:,6][np.where(summary[:,2]==babies[b])]
        umapYsum = summary[:,7][np.where(summary[:,2]==babies[b])]
        umapXavg = summary[:,8][np.where(summary[:,2]==babies[b])]
        umapYavg = summary[:,9][np.where(summary[:,2]==babies[b])]
        with open(args.data_dir + '/' + 'ALL_BABYADULTsummary_' + babies[
            b] + '_summary_LENAlabels_' + '_opensmile_day_UMAP_mfcc.csv',
                  'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow(
                ['umap ID', 'filename', 'baby_ID', 'time ID', 'label', 'labelID', 'umap sum x', 'umap sum y',
                 'umap avg x', 'umap avg y', 'PCA 1', 'PCA 2', 'tSNE 1', 'tSNE 2'])

            i = 0
            while i < len(basename_aux):
                csvwriter.writerow([i, basename_aux[i], basename_aux[i][0:11], timename_aux[i], labels_aux[i],
                                   labelsID_aux[i], umapXsum[i], umapYsum[i], umapXavg[i], umapYavg[i], pca_result[i,0], pca_result[i,1],
                                    tsne_result[i,0], tsne_result[i,1]])
                i = i + 1

    plt.close('all')
    print('Done')

def stat(classes, baby_id, args):
    """
    Function to compute statistical properties.
    TODO: check this later and keep only what we really comput. Try to make it easier to change which class to include.
    """
    # Load data
    if args.all == True:
        summary = pd.read_csv(
            args.data_dir + '/' + 'ALL_BABYADULTsummary_' + baby_id + '_summary_LENAlabels__opensmile_day_UMAP_mfcc.csv')
    else:
        summary = pd.read_csv(args.data_dir + '/' + 'BABYADULTsummary_' + baby_id +'_LENAlabels__opensmile_day_UMAP_mfcc.csv')
    # Sort data frame
    summary = summary.sort_values("time ID", ignore_index=True)
    #MEMO ['umap ID', 'filename', 'time ID', 'label', 'umap sum x', 'umap sum y', 'umap avg x', 'umap avg y']
    # Check order
    #print(summary["time ID"])
    #input()

    summary = pd.DataFrame.to_numpy(summary)
    if args.all == True:
        time = summary[:, 2]
        labels = summary[:, 4]
        labelsID = summary[:, 5]
        umapX = summary[:, 6]
        umapY = summary[:, 7]
        pcaX = summary[:, 10]
        pcaY = summary[:, 11]
        tsneX = summary[:, 12]
        tsneY = summary[:, 13]
    else:
        time = summary[:,1]
        labels = summary[:,3]
        labelsID = summary[:,4]
        umapX = summary[:,5]
        umapY = summary[:,6]

    # All in an array
    all_umap = np.zeros((len(umapX),2))
    all_pca = np.zeros((len(pcaX),2))
    all_tsne = np.zeros((len(tsneX),2))
    all_umap[:,0] = umapX
    all_umap[:,1] = umapY
    all_pca[:,0] = pcaX
    all_pca[:,1] = pcaY
    all_tsne[:,0] = tsneX
    all_tsne[:,1] = tsneY

    # Group coordinates
    how_many = []
    all_umapX = []
    all_umapY = []
    all_umapXY = []
    all_pcaX = []
    all_pcaY = []
    all_pcaXY = []
    all_tsneX = []
    all_tsneY = []
    all_tsneXY = []
    all_filename = []
    ratio = []
    hist2d_UMAP = []
    hist2d_PCA = []
    hist2d_tSNE = []
    for i in range(0,len(classes)):
        # UMAP
        aux_X = umapX[np.where(labels==classes[i])[0]]
        aux_Y = umapY[np.where(labels==classes[i])[0]]
        how_many.append(len(aux_X))
        all_umapX.append(aux_X)
        all_umapY.append(aux_Y)
        all_XY_aux = np.zeros((len(aux_X),2))
        all_filename.append(time[np.where(labels==classes[i])[0]])
        for j in range(0, len(aux_X)):
            all_XY_aux[j,0] = aux_X[j]
            all_XY_aux[j,1] = aux_Y[j]
        ratio.append(len(aux_X)/len(umapX))
        all_umapXY.append(all_XY_aux)

        hist_aux, xedges, yedges = np.histogram2d(aux_X, aux_Y, bins=10)

        hist2d_UMAP.append(hist_aux)
        fig = plt.figure(figsize=(int(xedges[-1] - xedges[-0] + 3), int(yedges[-1] - yedges[-0] + 3)))
        plt.imshow(hist_aux, interpolation='nearest', origin='lower',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.savefig(args.data_dir + '/' + baby_id + '_' + classes[i] + '_hist_UMAP.pdf')
        plt.close('all')

        # PCA
        aux_X = pcaX[np.where(labels==classes[i])[0]]
        aux_Y = pcaY[np.where(labels==classes[i])[0]]
        all_pcaX.append(aux_X)
        all_pcaY.append(aux_Y)
        all_XY_aux = np.zeros((len(aux_X),2))
        for j in range(0, len(aux_X)):
            all_XY_aux[j,0] = aux_X[j]
            all_XY_aux[j,1] = aux_Y[j]
        all_pcaXY.append(all_XY_aux)

        # tSNE
        aux_X = tsneX[np.where(labels==classes[i])[0]]
        aux_Y = tsneY[np.where(labels==classes[i])[0]]
        all_tsneX.append(aux_X)
        all_tsneY.append(aux_Y)
        all_XY_aux = np.zeros((len(aux_X),2))
        for j in range(0, len(aux_X)):
            all_XY_aux[j,0] = aux_X[j]
            all_XY_aux[j,1] = aux_Y[j]
        all_tsneXY.append(all_XY_aux)

    hist2d_UMAP = np.asarray(hist2d_UMAP)
    how_many = np.asarray(how_many)

    # centroids and entropy analysis
    centroids = []
    centroids_min_dist = np.zeros((len(classes), len(classes)))
    centroids_mean_dist = np.zeros((len(classes), len(classes)))
    centroids_PCA = []
    centroids_min_dist_PCA = np.zeros((len(classes), len(classes)))
    centroids_mean_dist_PCA = np.zeros((len(classes), len(classes)))
    centroids_tSNE = []
    centroids_min_dist_tSNE = np.zeros((len(classes), len(classes)))
    centroids_mean_dist_tSNE = np.zeros((len(classes), len(classes)))
    L2norm = []
    L2norm_PCA = []
    L2norm_tSNE = []
    entropy = []
    for i in range(0, len(classes)):
        # centroid
        centroid_UMAP = (sum(all_umapX[i]) / len(all_umapX[i]), sum(all_umapY[i]) / len(all_umapY[i]))
        centroids.append(centroid_UMAP)

        centroid_PCA = (sum(all_pcaX[i]) / len(all_pcaX[i]), sum(all_pcaY[i]) / len(all_pcaY[i]))
        centroids_PCA.append(centroid_PCA)

        centroid_tSNE = (sum(all_tsneX[i]) / len(all_tsneX[i]), sum(all_tsneY[i]) / len(all_tsneY[i]))
        centroids_tSNE.append(centroid_tSNE)

        # distance from the centroid
        for k in range(0,len(classes)):
            dist_UMAP = []
            dist_PCA = []
            dist_tSNE = []
            for j in range(0, len(all_umapX[k])):
                aux = np.array([all_umapX[k][j], all_umapY[k][j]])
                dist_aux_UMAP = np.linalg.norm(aux - centroid_UMAP)
                if dist_aux_UMAP != 0:
                    dist_UMAP.append(dist_aux_UMAP)

                aux = np.array([all_pcaX[k][j], all_pcaY[k][j]])
                dist_aux_PCA = np.linalg.norm(aux - centroid_PCA)
                if dist_aux_PCA != 0:
                    dist_PCA.append(dist_aux_PCA)

                aux = np.array([all_pcaX[k][j], all_pcaY[k][j]])
                dist_aux_tSNE = np.linalg.norm(aux - centroid_tSNE)
                if dist_aux_tSNE != 0:
                    dist_tSNE.append(dist_aux_tSNE)

            L2norm.append(dist_aux_UMAP / len(all_umapX[k]))
            centroids_min_dist[i, k] = np.min(dist_UMAP)
            centroids_mean_dist[i, k] = np.mean(dist_UMAP)
            L2norm_PCA.append(dist_aux_PCA / len(all_pcaX[k]))
            centroids_min_dist_PCA[i, k] = np.min(dist_PCA)
            centroids_mean_dist_PCA[i, k] = np.mean(dist_PCA)
            L2norm_tSNE.append(dist_aux_tSNE / len(all_tsneX[k]))
            centroids_min_dist_tSNE[i, k] = np.min(dist_tSNE)
            centroids_mean_dist_tSNE[i, k] = np.mean(dist_tSNE)

        # entropy
        from math import log
        p = hist2d_UMAP[i].flatten()
        p = p/np.sum(p)
        H = 0
        for h in range(len(p)):
            if p[h] != 0:
                H = H + p[h] * log(p[h], 2)

        entropy.append(-H)

    # covariance based on baby production
    adult_FANpre = []
    adult_FANpost = []
    baby_CHNSP = []
    for i in range(0, len(labelsID)-1):
        if labelsID[i] == 0:
            if labelsID[i-1] == 1:
                if np.shape(all_umap)[0]>i-2:
                    baby_CHNSP.append(all_umap[i,:])
                    adult_FANpre.append(all_umap[i-1,:])
                    adult_FANpost.append(all_umap[i+1,:])

    cov_adultFANpre_babyCHNSP = np.cov(adult_FANpre, baby_CHNSP)
    cov_adultFANpost_babyCHNSP = np.cov(adult_FANpost, baby_CHNSP)
    cov_babyCHNSP_self = np.cov(baby_CHNSP, baby_CHNSP)
    cov_babyCHNSP_self_pre = np.cov(baby_CHNSP[1::], baby_CHNSP[::-1])
    corr_coeff_adultFANpre_babyCHNSP = np.corrcoef(adult_FANpre, baby_CHNSP)
    corr_coeff_adultFANpost_babyCHNSP = np.corrcoef(adult_FANpost, baby_CHNSP)
    corr_coeff_baby_CHNSP_self = np.corrcoef(all_umap[1,:], all_umap[1,:])

    # Baby sound (no cry) versus caregiver: distance between centroids
    # init just in case
    dist_CHNSP_MAN_tSNE = 0
    dist_CHNSP_MAN_PCA = 0
    dist_CHNSP_MAN = 0
    centroids = np.asarray(centroids)
    dist_CHNSP_FAN = np.linalg.norm(centroids[0]-centroids[1])
    centroids_PCA = np.asarray(centroids_PCA)
    dist_CHNSP_FAN_PCA = np.linalg.norm(centroids_PCA[0]-centroids_PCA[1])
    centroids_tSNE = np.asarray(centroids_tSNE)
    dist_CHNSP_FAN_tSNE = np.linalg.norm(centroids_tSNE[0] - centroids_tSNE[1])
    if len(classes)>2:
        dist_CHNSP_MAN_tSNE = np.linalg.norm(centroids_tSNE[0] - centroids_tSNE[2])
        dist_CHNSP_MAN_PCA = np.linalg.norm(centroids_PCA[0]-centroids_PCA[2])
        dist_CHNSP_MAN = np.linalg.norm(centroids[0] - centroids[2])

    # From spectrogram features
    # TODO: later after abstract maybe
    pitch = []
    mean_pitch = []
    max_pitch = []
    min_pitch = []
    wiener = []
    for i in range(0,len(classes)):
        pitch_aux = []
        wiener_aux = []
        mean_pitch_aux = []
        max_pitch_aux = []
        min_pitch_aux = []
        #for j in range(0,len(all_filename[i])):
            #samples, sr = librosa.load(args.audio_dir + '/' + args.baby_id + '/' + all_filename[i][j] + '.wav', sr=16000)

            # Pitch detection
            #pitches, magnitudes = librosa.core.piptrack(samples, sr=sr, n_fft=100, fmin=500, fmax=8000)

            #pitches_all = 0
            #for interval in range(0, magnitudes.shape[1]):
                #index = magnitudes[:, interval].argmax()
                #pitches_all = np.append(pitches_all, pitches[index, interval])

            #pitches_all = pitches_all[np.nonzero(pitches_all)]
            #pitch_aux.append(pitches_all)
            #min_pitch_aux.append(np.min(pitches_all))
            #mean_pitch_aux.append(np.mean(pitches_all))
            #max_pitch_aux.append(np.max(pitches_all))

            # Spectral flatness (Wiener entropy)
            #wiener_aux.append(np.mean(librosa.feature.spectral_flatness(samples.astype(np.float))))

        #pitch.append(pitch_aux)
        #wiener.append(wiener_aux)
        #min_pitch.append(min_pitch_aux)
        #max_pitch.append(max_pitch_aux)
        #mean_pitch.append(mean_pitch_aux)

    # Save a dictionary with the quantities
    dataset_summary = {'how_many': how_many, 'entropy': entropy, 'cov_adultFANpre_babyCHNSP': cov_adultFANpre_babyCHNSP,
                       'cov_adultFANpost_babyCHNSP': cov_adultFANpost_babyCHNSP, 'cov_BBself': cov_babyCHNSP_self,
                       'cov_BBself_pre': cov_babyCHNSP_self_pre,
                       'corr_adultFANpre_babyCHNSP': corr_coeff_adultFANpre_babyCHNSP,
                       'corr_adultFANpost_babyCHNSP': corr_coeff_adultFANpost_babyCHNSP, 'corr_BBself': corr_coeff_baby_CHNSP_self,
                       'Dist_centrMAN': dist_CHNSP_MAN, 'Dist_centrFAN': dist_CHNSP_FAN, 'centroids': centroids, 'centroid_mean_dis': centroids_mean_dist,
                       'Dist_centrMAN_PCA': dist_CHNSP_MAN_PCA, 'Dist_centrFAN_PCA': dist_CHNSP_FAN_PCA, 'centroids_PCA': centroids_PCA, 'centroid_mean_dis_PCA': centroids_mean_dist_PCA,
                       'Dist_centrMAN_tSNE': dist_CHNSP_MAN_tSNE, 'Dist_centrFAN_tSNE': dist_CHNSP_FAN_tSNE, 'centroids_tSNE': centroids_tSNE, 'centroid_mean_dis_tSNE': centroids_mean_dist_tSNE,
                       'Ratio': ratio, 'Wiener_entropy': wiener, 'Mean_pitch': mean_pitch, 'Min_pitch': min_pitch,
                       'Max_pitch': max_pitch, 'All_pitches': pitch, 'L2norm': L2norm}
    np.save(args.data_dir + '/' + baby_id + '_stat_summary.npy', dataset_summary)

    print('Done')

def my_plot(classes, args):
    """
    Manual UMAP plot:
    - classes comparison
    - age comparison
    - example of one family: classes and centroids
    """
    # Define colormap
    classes_colors = ['darkblue', 'red', 'gold']
    classes_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("manual", classes_colors)
    plt.register_cmap("manual", classes_cmap)

    age_classes = ['3mo', '6mo', '9mo', '18mo', 'FAN-MAN']
    age_classes_colors = ['red', 'gold', 'navy', 'darkgreen', 'lightgray']
    age_classes_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("age_manual", age_classes_colors)
    plt.register_cmap("age_manual", age_classes_cmap)

    legend_elements = []
    for l in range(0, len(classes)):
        legend_elements.append(
            Line2D([], [], marker='o', color=classes_colors[l], markersize=10, label=classes[l]))

    legend_elements_age = []
    for l in range(0, len(age_classes)):
        legend_elements_age.append(
            Line2D([], [], marker='o', color=age_classes_colors[l], markersize=10, label=age_classes[l]))

    # Load data
    #summary = pd.read_csv(args.data_dir + '/' + 'summary_' + args.baby_id +'_LENAlabels__opensmile_day_UMAP_mfcc.csv')
    summary = pd.read_csv(args.data_dir + '/' + 'ALL_BABYADULT_summary_LENAlabels_' + '_opensmile_day_UMAP_mfcc.csv')
    # Sort data frame
    summary = summary.sort_values("time ID", ignore_index=True)
    #MEMO ['umap ID', 'filename', 'time ID', 'label', 'umap sum x', 'umap sum y', 'umap avg x', 'umap avg y']
    # Check order
    #print(summary["time ID"])
    #input()

    umapX = summary["umap sum x"]
    umapY = summary["umap sum y"]

    colors = []
    i = 0
    while i < len(umapX):
        j = 0
        while j < len(classes):
            if summary["label"][i] == classes[j]:
                colors.append(classes_colors[j])
            j = j + 1
        i = i + 1

    colors_age = ['gray' for i in range(0, len(umapX))]
    for i in range(0, len(umapX)):
        #if summary["label"][i] == 'CHNSP' or summary["label"][i] == 'CHNNSP':
        if summary["label"][i] == 'CHNSP':
            for k in range(0, len(age_classes) -1):
                if summary["age"][i] == age_classes[k]:
                    colors_age[i] = age_classes_colors[k]

    #plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(umapX, umapY, c=colors, cmap='manual', s=0.1, alpha=0.8)
    ax.set_xlim(np.min(umapX)-0.5, np.max(umapX)+0.5)
    ax.set_ylim(np.min(umapY)-0.5, np.max(umapY)+0.5)
    #plt.savefig(args.data_dir + '/' + args.baby_id + "_UMAPmanual.pdf")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(handles=legend_elements, fontsize=20)
    plt.savefig(args.data_dir + '/' + "_UMAPmanual.pdf")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(umapX, umapY, c=colors_age, cmap='age_manual', s=0.1, alpha=0.8)
    ax.set_xlim(np.min(umapX)-0.5, np.max(umapX)+0.5)
    ax.set_ylim(np.min(umapY)-0.5, np.max(umapY)+0.5)
    #plt.savefig(args.data_dir + '/' + args.baby_id + "_UMAPmanual.pdf")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(handles=legend_elements_age, fontsize=20)
    plt.savefig(args.data_dir + '/' + "_UMAPmanual_age.pdf")

    print('Done')

def aux_stat(classes, args):
    """
    Final summary to plot the results and prepare the table to use it later to fit the data (R routine).
    """
    # List of babies
    summary_table = pd.read_csv(args.data_dir + '/' + 'baby_list_basic.csv')
    summary = pd.DataFrame.to_numpy(summary_table)
    babies = summary[:, 0]
    age = summary[:, 1]

    # List of colors (each color is a particular baby)
    colors = summary[:,2]
    #colors = [ 'b', 'r', 'k', 'g', 'r', 'k', 'b', 'g', 'b', 'g', 'g', 'k', 'y'] ICIS colors
    for i in range(0, len(colors)):
        colors[i] = 'k'

    # From stat summary
    entropy_CHNSP = np.zeros((len(babies),))
    L2norm_CHNSP = np.zeros((len(babies),))
    ratio = []
    how_many_all = []
    how_many_classes = np.zeros((len(babies), len(classes)))
    MEAN_cov_adultFANpre_babyCHNSP = []
    MEAN_cov_adultFANpost_babyCHNSP = []
    MEAN_cov_babyCHNSP_self = []
    MEAN_cov_babyCHNSP_self_pre = []
    MEAN_corr_adultFANpre_babyCHNSP = []
    MEAN_corr_adultFANpost_babyCHNSP = []
    MEAN_corr_babyCHNSP_self = []
    STD_babyCHNSP_self = []
    dist_CHNSP_MAN = []
    dist_CHNSP_FAN_UMAP = []
    dist_CHNSP_FAN_PCA = []
    dist_CHNSP_FAN_tSNE = []
    baby_ID = []
    agegroup = []
    exp_entropy = np.zeros((len(babies),))
    centroid_CHNSP_self_UMAP = []
    centroid_CHNSP_self_PCA = []
    centroid_CHNSP_self_tSNE = []
    for i in range(0, len(babies)):
        print(babies[i])

        baby_ID.append(babies[i][1:4])
        if age[i] < 180:
            agegroup.append('3mo')
        elif (age[i]>=180 and age[i]<250):
            agegroup.append('6mo')
        elif (age[i]>=250 and age[i]<500):
            agegroup.append('9mo')
        else:
            agegroup.append('18mo')

        dataset_summary = np.load(args.data_dir + '/' + babies[i] + '_stat_summary.npy', allow_pickle=True)
        dataset_summary = dataset_summary.item()
        entropy_CHNSP[i] = dataset_summary['entropy'][0]
        exp_entropy[i] = 2**entropy_CHNSP[i]
        L2norm_CHNSP[i] = dataset_summary['L2norm'][0]
        ratio.append(dataset_summary['Ratio'])
        how_many_all.append(np.sum(dataset_summary['how_many']))
        how_many_classes[i,:] = dataset_summary['how_many']
        MEAN_cov_adultFANpre_babyCHNSP.append(np.mean(np.diagonal(dataset_summary['cov_adultFANpre_babyCHNSP'])))
        MEAN_cov_adultFANpost_babyCHNSP.append(np.mean(np.diagonal(dataset_summary['cov_adultFANpost_babyCHNSP'])))
        MEAN_cov_babyCHNSP_self.append(np.mean(np.diagonal(dataset_summary['cov_BBself'])))
        MEAN_cov_babyCHNSP_self_pre.append(np.mean(np.diagonal(dataset_summary['cov_BBself_pre'])))
        MEAN_corr_adultFANpre_babyCHNSP.append(np.mean(np.diagonal(dataset_summary['corr_adultFANpre_babyCHNSP'])))
        MEAN_corr_adultFANpost_babyCHNSP.append(np.mean(np.diagonal(dataset_summary['corr_adultFANpost_babyCHNSP'])))
        MEAN_corr_babyCHNSP_self.append(np.mean(np.diagonal(dataset_summary['corr_BBself'])))
        STD_babyCHNSP_self.append(np.mean(np.sqrt(np.diagonal(dataset_summary['cov_BBself']))))
        dist_CHNSP_MAN.append(dataset_summary['Dist_centrMAN'])
        dist_CHNSP_FAN_UMAP.append(dataset_summary['Dist_centrFAN'])
        dist_CHNSP_FAN_PCA.append(dataset_summary['Dist_centrFAN_PCA'])
        dist_CHNSP_FAN_tSNE.append(dataset_summary['Dist_centrFAN_tSNE'])
        centroid_CHNSP_self_UMAP.append(dataset_summary['centroid_mean_dis'][0,0])
        centroid_CHNSP_self_PCA.append(dataset_summary['centroid_mean_dis_PCA'][0,0])
        centroid_CHNSP_self_tSNE.append(dataset_summary['centroid_mean_dis_tSNE'][0,0])
        if i == 10:
            centroid_coord_CHNSP = dataset_summary['centroids'][0,:]
            centroid_coord_FAN = dataset_summary['centroids'][1,:]

    how_many_all = np.asarray(how_many_all)
    entropy_CHNSP = np.asarray(entropy_CHNSP)
    baby_ID = np.asarray(baby_ID)
    agegroup = np.asarray(agegroup)
    dist_CHNSP_FAN_UMAP = np.asarray(dist_CHNSP_FAN_UMAP)
    centroid_CHNSP_self_UMAP = np.asarray(centroid_CHNSP_self_UMAP)
    MEAN_cov_babyCHNSP_self_pre = np.asarray(MEAN_cov_babyCHNSP_self_pre)
    MEAN_cov_babyCHNSP_self = np.asarray(MEAN_cov_babyCHNSP_self)

    how_many_age = np.zeros((4, len(classes)))
    ages = ['3mo', '6mo', '9mo', '18mo']
    for i in range(0,len(ages)):
        how_many_age[i,:] = np.sum(how_many_classes[np.where(agegroup==ages[i])], axis=0)

    print('How many per class')
    print('Min')
    print(np.min(how_many_classes, axis=0))
    print('Sum')
    print(np.sum(how_many_classes, axis=0))
    print('Max')
    print(np.max(how_many_classes, axis=0))
    print('Mean')
    print(np.mean(how_many_classes, axis=0))
    input()

    print('How many per age')
    print(how_many_age)
    input()

    # Mean, median, std of the same age. First column: dist CHNSP_FAN. Second column: dist_SELF
    ages = ['3mo', '6mo', '9mo', '18mo']
    mean_age = np.zeros((len(ages),2))
    median_age = np.zeros((len(ages),2))
    std_age = np.zeros((len(ages),2))
    j = 0
    while j<len(ages):
        mean_age[j,0] = np.mean(dist_CHNSP_FAN_UMAP[np.where(agegroup==ages[j])[0]])
        mean_age[j,1] = np.mean(centroid_CHNSP_self_UMAP[np.where(agegroup==ages[j])[0]])
        median_age[j,0] = np.median(dist_CHNSP_FAN_UMAP[np.where(agegroup==ages[j])[0]])
        median_age[j,1] = np.median(centroid_CHNSP_self_UMAP[np.where(agegroup==ages[j])[0]])
        std_age[j,0] = np.std(dist_CHNSP_FAN_UMAP[np.where(agegroup==ages[j])[0]])
        std_age[j,1] = np.std(centroid_CHNSP_self_UMAP[np.where(agegroup==ages[j])[0]])
        j = j+1

    summary_table['CHNSPentropy'] = entropy_CHNSP
    summary_table['#voc'] = how_many_all
    summary_table['CHILDID'] = baby_ID
    summary_table['AGEGROUP'] = agegroup
    summary_table['CENTROID_CHSNP_FAN_UMAP'] = dist_CHNSP_FAN_UMAP
    summary_table['CENTROID_CHSNP_FAN_PCA'] = dist_CHNSP_FAN_PCA
    summary_table['CENTROID_CHSNP_FAN_tSNE'] = dist_CHNSP_FAN_tSNE
    summary_table['CENTROIDdist_CHSNPself_UMAP'] = centroid_CHNSP_self_UMAP
    summary_table['CENTROIDdist_CHSNPself_PCA'] = centroid_CHNSP_self_PCA
    summary_table['CENTROIDdist_CHSNPself_tSNE'] = centroid_CHNSP_self_tSNE
    summary_table['cov_BBself_pre'] = MEAN_cov_babyCHNSP_self_pre
    summary_table['cov_BBself'] = MEAN_cov_babyCHNSP_self
    summary_table.to_csv(args.data_dir + '/' + 'baby_list.csv')

    print('Done')

def plot_stat(classes, args):
    import pyreadr as readR

    # List of babies
    summary_table = pd.read_csv(args.data_dir + '/' + 'baby_list_basic.csv')
    summary = pd.DataFrame.to_numpy(summary_table)
    babies = summary[:, 0]
    age = summary[:, 1]

    # List of colors (each color is a particular baby)
    colors = summary[:,2]
    for i in range(0, len(colors)):
        colors[i] = 'k'

    # From stat summary
    entropy_CHNSP = np.zeros((len(babies),))
    L2norm_CHNSP = np.zeros((len(babies),))
    ratio = []
    how_many_all = []
    how_many_classes = np.zeros((len(babies), len(classes)))
    MEAN_cov_adultFANpre_babyCHNSP = []
    MEAN_cov_adultFANpost_babyCHNSP = []
    MEAN_cov_babyCHNSP_self = []
    MEAN_cov_babyCHNSP_self_pre = []
    MEAN_corr_adultFANpre_babyCHNSP = []
    MEAN_corr_adultFANpost_babyCHNSP = []
    MEAN_corr_babyCHNSP_self = []
    STD_babyCHNSP_self = []
    dist_CHNSP_MAN = []
    dist_CHNSP_FAN_UMAP = []
    dist_CHNSP_FAN_PCA = []
    dist_CHNSP_FAN_tSNE = []
    baby_ID = []
    agegroup = []
    exp_entropy = np.zeros((len(babies),))
    centroid_CHNSP_self_UMAP = []
    centroid_CHNSP_self_PCA = []
    centroid_CHNSP_self_tSNE = []
    for i in range(0, len(babies)):
        print(babies[i])

        baby_ID.append(babies[i][1:4])
        if age[i] < 180:
            agegroup.append('3mo')
        elif (age[i]>=180 and age[i]<250):
            agegroup.append('6mo')
        elif (age[i]>=250 and age[i]<500):
            agegroup.append('9mo')
        else:
            agegroup.append('18mo')

        dataset_summary = np.load(args.data_dir + '/' + babies[i] + '_stat_summary.npy', allow_pickle=True)
        dataset_summary = dataset_summary.item()
        entropy_CHNSP[i] = dataset_summary['entropy'][0]
        exp_entropy[i] = 2**entropy_CHNSP[i]
        L2norm_CHNSP[i] = dataset_summary['L2norm'][0]
        ratio.append(dataset_summary['Ratio'])
        how_many_all.append(np.sum(dataset_summary['how_many']))
        how_many_classes[i,:] = dataset_summary['how_many']
        MEAN_cov_adultFANpre_babyCHNSP.append(np.mean(np.diagonal(dataset_summary['cov_adultFANpre_babyCHNSP'])))
        MEAN_cov_adultFANpost_babyCHNSP.append(np.mean(np.diagonal(dataset_summary['cov_adultFANpost_babyCHNSP'])))
        MEAN_cov_babyCHNSP_self.append(np.mean(np.diagonal(dataset_summary['cov_BBself'])))
        MEAN_cov_babyCHNSP_self_pre.append(np.mean(np.diagonal(dataset_summary['cov_BBself_pre'])))
        MEAN_corr_adultFANpre_babyCHNSP.append(np.mean(np.diagonal(dataset_summary['corr_adultFANpre_babyCHNSP'])))
        MEAN_corr_adultFANpost_babyCHNSP.append(np.mean(np.diagonal(dataset_summary['corr_adultFANpost_babyCHNSP'])))
        MEAN_corr_babyCHNSP_self.append(np.mean(np.diagonal(dataset_summary['corr_BBself'])))
        STD_babyCHNSP_self.append(np.mean(np.sqrt(np.diagonal(dataset_summary['cov_BBself']))))
        dist_CHNSP_MAN.append(dataset_summary['Dist_centrMAN'])
        dist_CHNSP_FAN_UMAP.append(dataset_summary['Dist_centrFAN'])
        dist_CHNSP_FAN_PCA.append(dataset_summary['Dist_centrFAN_PCA'])
        dist_CHNSP_FAN_tSNE.append(dataset_summary['Dist_centrFAN_tSNE'])
        centroid_CHNSP_self_UMAP.append(dataset_summary['centroid_mean_dis'][0,0])
        centroid_CHNSP_self_PCA.append(dataset_summary['centroid_mean_dis_PCA'][0,0])
        centroid_CHNSP_self_tSNE.append(dataset_summary['centroid_mean_dis_tSNE'][0,0])
        if i == 10:
            centroid_coord_CHNSP = dataset_summary['centroids'][0,:]
            centroid_coord_FAN = dataset_summary['centroids'][1,:]

    entropy_CHNSP = np.asarray(entropy_CHNSP)
    dist_CHNSP_FAN_UMAP = np.asarray(dist_CHNSP_FAN_UMAP)
    centroid_CHNSP_self_UMAP = np.asarray(centroid_CHNSP_self_UMAP)
    MEAN_cov_babyCHNSP_self_pre = np.asarray(MEAN_cov_babyCHNSP_self_pre)
    MEAN_cov_babyCHNSP_self = np.asarray(MEAN_cov_babyCHNSP_self)

    # Read fit from R
    aux = readR.read_r(args.data_dir + '/' + 'UMAP_CHNSPcentroidSELF.Rdata')
    UMAP_fit_CHNSPselfCENTROID = aux['pred']
    aux = readR.read_r(args.data_dir + '/' + 'UMAP_CHNSP_FAN_centroid.Rdata')
    UMAP_fit_CHNSP_FAN_CENTROID = aux['pred']
    aux = readR.read_r(args.data_dir + '/' + 'PCA_CHNSPcentroidSELF.Rdata')
    PCA_fit_CHNSPselfCENTROID = aux['pred']
    aux = readR.read_r(args.data_dir + '/' + 'PCA_CHNSP_FAN_centroid.Rdata')
    PCA_fit_CHNSP_FAN_CENTROID = aux['pred']
    aux = readR.read_r(args.data_dir + '/' + 'tSNE_CHNSPcentroidSELF.Rdata')
    tSNE_fit_CHNSPselfCENTROID = aux['pred']
    aux = readR.read_r(args.data_dir + '/' + 'tSNE_CHNSP_FAN_centroid.Rdata')
    tSNE_fit_CHNSP_FAN_CENTROID = aux['pred']
    aux = readR.read_r(args.data_dir + '/' + 'UMAP_CHNSPselfPREMeanCOVARIANCE.Rdata')
    UMAP_CHNSPselfPREMeanCOVARIANCE = aux['pred']
    aux = readR.read_r(args.data_dir + '/' + 'UMAP_CHNSPselfMeanCOVARIANCE.Rdata')
    UMAP_CHNSPselfMeanCOVARIANCE = aux['pred']

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        aux_line = []
        age_line = []
        for j in range(0, len(babies)):
            if (babies[j][0:4]==babies[i][0:4]):
                aux_line.append(entropy_CHNSP[j])
                age_line.append(age[j])
        plt.plot(age_line, aux_line, color=colors[i], lw=0.5)
        plt.plot(age[i], entropy_CHNSP[i], color=colors[i], marker='*')
    ax.set_xlabel('Age (in days)', fontsize=15)
    ax.set_ylabel('Entropy', fontsize=15)
    #plt.legend(handles=legend_elements, ncol=2)
    plt.savefig(args.data_dir + '/' + 'entropy.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        aux_line = []
        age_line = []
        for j in range(0, len(babies)):
            if (babies[j][0:4]==babies[i][0:4]):
                aux_line.append(L2norm_CHNSP[j])
                age_line.append(age[j])
        plt.plot(age_line, aux_line, color=colors[i], lw=0.5)
        plt.plot(age[i], L2norm_CHNSP[i], color=colors[i], marker='*')
    ax.set_xlabel('Age (in days)', fontsize=15)
    ax.set_ylabel('Euclidian distance', fontsize=15)
    #plt.legend(handles=legend_elements, ncol=2)
    plt.savefig(args.data_dir + '/' + 'L2norm.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        aux_line = []
        age_line = []
        for j in range(0, len(babies)):
            if (babies[j][0:4]==babies[i][0:4]):
                aux_line.append(exp_entropy[j])
                age_line.append(age[j])
        plt.plot(age_line, aux_line, color=colors[i], lw=0.5)
        plt.plot(age[i], exp_entropy[i], color=colors[i], marker='*')
    ax.set_xlabel('Age (in days)', fontsize=15)
    ax.set_ylabel('Sound variety', fontsize=15)
    #plt.legend(handles=legend_elements, ncol=2)
    plt.savefig(args.data_dir + '/' + 'expEntropy.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        aux_line = []
        age_line = []
        for j in range(0, len(babies)):
            if (babies[j][0:4]==babies[i][0:4]):
                aux_line.append(STD_babyCHNSP_self[j])
                age_line.append(age[j])
        plt.plot(age_line, aux_line, color=colors[i], lw=0.5)
        plt.plot(age[i], STD_babyCHNSP_self[i], color=colors[i], marker='*')
    ax.set_xlabel('Age (in days)', fontsize=15)
    ax.set_ylabel('Standard deviation', fontsize=15)
    #plt.legend(handles=legend_elements, ncol=2)
    plt.savefig(args.data_dir + '/' + 'std.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], ratio[i][0], color=colors[i], marker='*')
    plt.savefig(args.data_dir + '/' + 'ratioCHNSP.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], ratio[i][1], color=colors[i], marker='*')
    plt.savefig(args.data_dir + '/' + 'ratioFAN.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], ratio[i][2], color=colors[i], marker='*')
    plt.savefig(args.data_dir + '/' + 'ratioMAN.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], dist_CHNSP_MAN[i], color=colors[i], marker='*')
    plt.savefig(args.data_dir + '/' + 'dist_CHNSP_MAN.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], dist_CHNSP_FAN_UMAP[i], color=colors[i], marker='*')
        ax.set_xlabel('Age (in days)', fontsize=15)
        ax.set_ylabel('Distance between centroids', fontsize=15)
        #plt.legend(handles=legend_elements, ncol=2)
    plt.plot(sorted(age), UMAP_fit_CHNSP_FAN_CENTROID, 'k', lw=0.5)
    plt.savefig(args.data_dir + '/' + 'dist_CHNSP_FAN_UMAP.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], dist_CHNSP_FAN_PCA[i], color=colors[i], marker='*')
        ax.set_xlabel('Age (in days)', fontsize=15)
        ax.set_ylabel('Distance between centroids', fontsize=15)
        #plt.legend(handles=legend_elements, ncol=2)
    plt.plot(sorted(age), PCA_fit_CHNSP_FAN_CENTROID, 'k', lw=0.5)
    plt.savefig(args.data_dir + '/' + 'dist_CHNSP_FAN_PCA.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], dist_CHNSP_FAN_tSNE[i], color=colors[i], marker='*')
        ax.set_xlabel('Age (in days)', fontsize=15)
        ax.set_ylabel('Distance between centroids', fontsize=15)
        #plt.legend(handles=legend_elements, ncol=2)
    plt.plot(sorted(age), tSNE_fit_CHNSP_FAN_CENTROID, 'k', lw=0.5)
    plt.savefig(args.data_dir + '/' + 'dist_CHNSP_FAN_tSNE.pdf')

    plt.close('all')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], centroid_CHNSP_self_UMAP[i], color=colors[i], marker='*')
        ax.set_xlabel('Age (in days)', fontsize=15)
        ax.set_ylabel('Mean distance from the centroid', fontsize=15)
        #plt.legend(handles=legend_elements, ncol=2)
    plt.plot(sorted(age), UMAP_fit_CHNSPselfCENTROID, 'k', lw=0.5)
    plt.savefig(args.data_dir + '/' + 'MEAN_points_dist_CHNSP_self_UMAP.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], centroid_CHNSP_self_PCA[i], color=colors[i], marker='*')
        ax.set_xlabel('Age (in days)', fontsize=15)
        ax.set_ylabel('Mean distance from the centroid', fontsize=15)
        #plt.legend(handles=legend_elements, ncol=2)
    plt.plot(sorted(age), PCA_fit_CHNSPselfCENTROID, 'k', lw=0.5)
    plt.savefig(args.data_dir + '/' + 'MEAN_points_dist_CHNSP_self_PCA.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], centroid_CHNSP_self_tSNE[i], color=colors[i], marker='*')
        ax.set_xlabel('Age (in days)', fontsize=15)
        ax.set_ylabel('Mean distance from the centroid', fontsize=15)
        #plt.legend(handles=legend_elements, ncol=2)
    plt.plot(sorted(age), tSNE_fit_CHNSPselfCENTROID, 'k', lw=0.5)
    plt.savefig(args.data_dir + '/' + 'MEAN_points_dist_CHNSP_self_tSNE.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], MEAN_cov_babyCHNSP_self[i], color=colors[i], marker='*')
    plt.plot(sorted(age), UMAP_CHNSPselfMeanCOVARIANCE, 'k', lw=0.5)
    plt.savefig(args.data_dir + '/' + 'MEAN_cov_babyCHNSP_self.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], MEAN_cov_babyCHNSP_self_pre[i], color=colors[i], marker='*')
        ax.set_xlabel('Age (in days)', fontsize=15)
        ax.set_ylabel('Mean covariance between CHNSP', fontsize=15)
    plt.plot(sorted(age), UMAP_CHNSPselfPREMeanCOVARIANCE, 'k', lw=0.5)
    plt.savefig(args.data_dir + '/' + 'MEAN_cov_babyCHNSP_self_pre.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], MEAN_cov_adultFANpre_babyCHNSP[i], color=colors[i], marker='*')
    plt.savefig(args.data_dir + '/' + 'MEAN_cov_adultFANpre_babyCHNSP.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], MEAN_cov_adultFANpost_babyCHNSP[i], color=colors[i], marker='*')
    plt.savefig(args.data_dir + '/' + 'MEAN_cov_adultFANpost_babyCHNSP.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], MEAN_corr_babyCHNSP_self[i], color=colors[i], marker='*')
    plt.savefig(args.data_dir + '/' + 'MEAN_corr_babyCHNSP_self.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], MEAN_corr_adultFANpre_babyCHNSP[i], color=colors[i], marker='*')
    plt.savefig(args.data_dir + '/' + 'MEAN_corr_adultFANpre_babyCHNSP.pdf')

    fig, ax = plt.subplots()
    for i in range(0, len(babies)):
        plt.plot(age[i], MEAN_corr_adultFANpost_babyCHNSP[i], color=colors[i], marker='*')
    plt.savefig(args.data_dir + '/' + 'MEAN_corr_adultFANpost_babyCHNSP.pdf')

    plt.close('all')

    # Plot one example baby
    # Load data
    summary = pd.read_csv(
        args.data_dir + '/' + 'ALL_BABYADULTsummary_' + babies[10] + '_summary_LENAlabels__opensmile_day_UMAP_mfcc.csv')
    # Sort data frame
    summary = summary.sort_values("time ID", ignore_index=True)
    # MEMO ['umap ID', 'filename', 'time ID', 'label', 'umap sum x', 'umap sum y', 'umap avg x', 'umap avg y']

    umapX = summary["umap sum x"]
    umapY = summary["umap sum y"]

    labels = ['CHNSP', 'centroid CHNSP', 'FAN', 'centroid FAN']
    classes_colors = ['darkblue', 'red']
    handles_colors = ['darkblue', 'deepskyblue', 'red', 'pink']
    legend_elements = []
    for l in range(0, len(labels)):
        legend_elements.append(
            Line2D([], [], marker='o', color=handles_colors[l], markersize=10, label=labels[l]))

    new_umapX = []
    new_umapY = []
    colors_new = []
    i = 0
    while i < len(umapX):
        j = 0
        while j < 2:
            if summary["label"][i] == classes[j]:
                colors_new.append(classes_colors[j])
                new_umapX.append(umapX[i])
                new_umapY.append(umapY[i])
            j = j + 1
        i = i + 1

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(new_umapX, new_umapY, c=colors_new, cmap='manual', s=1, alpha=0.8)
    ax.scatter(centroid_coord_CHNSP[0], centroid_coord_CHNSP[1], c='deepskyblue', s=20)
    ax.scatter(centroid_coord_FAN[0], centroid_coord_FAN[1], c='pink', s=20)
    ax.set_xlim(np.min(umapX)-0.5, np.max(umapX)+0,5)
    ax.set_ylim(np.min(umapY)-0.5, np.max(umapY)+0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(handles=legend_elements)
    plt.savefig(args.data_dir + '/' + babies[10] + "_exampleBABY.pdf")

    print('Done')

if __name__ == '__main__':
    import argparse
    import glob2
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, choices=['multidim_all', 'plot', 'stat', 'aux_stat', 'plot_stat'])
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--baby_id', type=str)
    parser.add_argument('--audio_dir', type=str)

    UMAP_parameters_args = parser.add_argument_group('UMAP_params')
    UMAP_parameters_args.add_argument('--seed', type=int, default=42)
    UMAP_parameters_args.add_argument('--n_neigh', type=int,
                                      help='How much local is the topology. The smaller the more local')
    UMAP_parameters_args.add_argument('--min_d', type=float,
                                      help='How tightly UMAP is allowed to pack points together. Higher values provide more details. Range 0.0 to 0.99',
                                      default=0.1)
    UMAP_parameters_args.add_argument('--spread', type=float,
                                      help='Additional parameter to change when min_d is changed, it has to be such that spread >= min_d',
                                      default=0.1)
    UMAP_parameters_args.add_argument('--portion', type=bool,
                                      help='To visualize only a part of the day',
                                      default=False)
    UMAP_parameters_args.add_argument('--portion_size', type=float,
                                      help='How long we want to visualize, in hours or in number of sounds per class, depending on the function',
                                      default=0)
    UMAP_parameters_args.add_argument('--step', type=int, help='How many steps in one hour. For example, if slices of 1s, then 3600. If slices of 0.5s, then 7200',
                                      default=3600)
    UMAP_parameters_args.add_argument('--labels', type=str,
                                      help='Lables on/off, and which type',
                                      default='off', choices=['off', 'lena', 'my'])
    UMAP_parameters_args.add_argument('--n_comp', type=int, help='How many components', default=2)
    UMAP_parameters_args.add_argument('--all_labels', type=list, help='All the possible labels',
                                      default=['CHF', 'CHNNSP', 'CHNSP', 'CXF', 'CXN', 'FAF', 'FAN', 'MAF', 'MAN', 'NOF', 'NON', 'OLF', 'OLN', 'SIL', 'TVF', 'TVN'])

    args = parser.parse_args()

    if args.output_dir != None:
        if not os.path.isdir(args.data_dir + '/' + args.output_dir):
            os.makedirs(args.data_dir + '/' + args.output_dir)

    if args.option == 'multidim_all':
        # Classes
        if args.labels == 'lena':
            # classes = ['CHNNSP', 'CHNSP', 'FAF', 'FAN', 'MAN', 'TVF', 'TVN']
            # classes = ['CHNSP', 'FAN', 'MAN']
            classes = ['CHNSP', 'CHNNSP']

        # List of babies
        summary = pd.read_csv(args.data_dir + '/' + 'baby_list_basic.csv')
        summary = pd.DataFrame.to_numpy(summary)
        babies = summary[:, 0]
        age = summary[:,1]

        multidim_all(classes, babies, age, args)

    if args.option == 'plot':
        #classes = ['CHNSP', 'FAN', 'MAN']
        classes = ['CHNSP', 'CHNNSP']
        my_plot(classes, args)

    if args.option == 'stat':
        # List of babies
        summary = pd.read_csv(args.data_dir + '/' + 'baby_list_basic.csv')
        summary = pd.DataFrame.to_numpy(summary)
        babies = summary[:,0]

        #classes = ['CHNSP', 'FAN', 'MAN']  # ['CHNNSP', 'CHNSP', 'FAF', 'FAN', 'MAN']
        classes = ['CHNSP', 'CHNNSP']
        for i in range(0,len(babies)):
            print(babies[i])
            if babies[i] != '0932_000602a':   #ICIS without '0583_000605':
                stat(classes, babies[i], args)

    if args.option == 'aux_stat':
        #classes = ['CHNSP', 'FAN', 'MAN']
        classes = ['CHNSP', 'CHNNSP']
        aux_stat(classes, args)

    if args.option == 'plot_stat':
        #classes = ['CHNSP', 'FAN', 'MAN']
        classes = ['CHNSP', 'CHNNSP']
        plot_stat(classes, args)

    ### Example: python3 BabyExperience.py --data_dir /Users/silviapagliarini/Documents/opensmile/HumanData_analysis --option comparison