from sklearn.manifold import TSNE, Isomap;
from sklearn.decomposition import PCA;
import numpy as np;
from tp2_aux import images_as_matrix,report_clusters;
from sklearn.manifold import Isomap;
from sklearn.cluster import DBSCAN;
from sklearn.preprocessing import StandardScaler;
from sklearn.cluster import KMeans;
from sklearn.metrics import silhouette_score;
from sklearn.neighbors import KNeighborsClassifier;
import matplotlib.pyplot as plt;
from sklearn.metrics.cluster import adjusted_rand_score;

matrix = images_as_matrix();
scaler = StandardScaler();
scaledMatrix = scaler.fit_transform(matrix);

def readFromFile(fileName):
    text = open(fileName).readlines();
    values=[];
    idd = [];
    for lin in text:
        va = lin.split(",");
        values.append(int(va[1].split('\n')[0]));
        idd.append(int(va[0]));
    return np.array(values),np.array(idd);
labels,ids = readFromFile("labels.txt");

def combineHorizontaly(A,B):
    shA=np.shape(A);
    shB=np.shape(B);
    colTot=shA[1]+shB[1];
    rowMax=np.max((shA[0],shB[0]));
    CHorz=np.zeros((rowMax,colTot));
    CHorz[0:shA[0],0:shA[1]]=A;
    CHorz[0:shB[0],shA[1]:colTot]=B;
    return CHorz;

def computeAlgRand(labels, noLabel):
    trueP = trueN = sameC = sameP = 0.0;
    for i in range(len(labels)-1):
        sc = labels[i] == labels[i+1:];
        sp = noLabel[i] == noLabel[i+1:];
        sameC += np.sum(sc);
        sameP += np.sum(sp);
        trueP += np.sum(np.logical_and(sc, sp));
        trueN += np.sum(np.logical_not(np.logical_or(sc,sp)));
    rand = (trueP+trueN)/((len(labels)*(len(labels)-1))/2);
    precision = trueP/sameC;
    recall = trueP/sameP;
    f1measure = 2*((precision*recall)/(precision+recall));
    return rand, precision, recall, f1measure, adjusted_rand_score(labels, noLabel);

def getFeatsPCA():
    pca = PCA(n_components=6);
    return pca.fit_transform(scaledMatrix);

def getFeatsTSNE():
    tsne = TSNE(n_components = 6, method='exact');
    return tsne.fit_transform(scaledMatrix);

def getFeatsIsomap():
    isoMap = Isomap(n_components = 6);
    return isoMap.fit_transform(scaledMatrix);

def getFeatures():
    feats = getFeatsPCA();
    feats = combineHorizontaly(feats,getFeatsTSNE());
    feats = combineHorizontaly(feats,getFeatsIsomap());
    return feats;
    
def neighbor(feats):
    region = KNeighborsClassifier(n_neighbors=5);
    region.fit(feats, np.zeros(563));
    neighbors = region.kneighbors();
    orderedMaxD = np.sort(neighbors[0][:,-1])[::-1];
    return orderedMaxD;

def bestKMeans(feats):
    results = [];
    bestP = 0;
    bestK = 2;
    bestLabels = []
    for n_clusters in range(2,30):
        region = KMeans(n_clusters=n_clusters).fit_predict(feats);
        randIndex, precision, recall, f1, adjRandIndex = computeAlgRand(region,labels);
        results.append([n_clusters,randIndex,precision,recall,f1,adjRandIndex,silhouette_score(feats,labels)]);
        if bestP<=precision:
            bestP = precision
            bestK = n_clusters
            bestLabels = region
    results = np.array(results);
    #drawAlgStats(results, 'Kmeans stats');
    print(bestK)
    report_clusters(ids,bestLabels,"text.html");
    return results;

def bestDBScan(feats):
    results = [];
    for epsil in range(10,50):
        region = DBSCAN(eps=epsil, min_samples=5).fit_predict(feats);
        randIndex, precision, recall, f1, adjRandIndex = computeAlgRand(region,labels);
        results.append([epsil,randIndex,precision,recall,f1,adjRandIndex,silhouette_score(feats,labels)]);
    results = np.array(results);
    drawAlgStats(results, 'DbScan stats');
    return results;

def drawGraph(orderedMaxD):
    plt.rcParams['axes.facecolor'] = 'lightgrey';
    plt.title('5-dist graph');
    plt.xlabel('Points sorted by distance');
    plt.ylabel('5 min distance');
    plt.plot(range(len(orderedMaxD)), orderedMaxD, '-r');
    plt.show();
    plt.close();
    
def drawAlgStats(results, title):
    plt.rcParams['axes.facecolor'] = 'lightgrey';
    plt.title(title);
    plt.plot(results[:,0], results[:,1], '-r', label='randIndex');
    plt.plot(results[:,0], results[:,2], '-b', label='precision');
    plt.plot(results[:,0], results[:,3], '-w', label='f1');
    plt.plot(results[:,0], results[:,4], '-g', label='adjRand');
    plt.show();
    plt.close();
    
bestKMeans(getFeatures());
#drawGraph(neighbor(getFeatures()));
#standartscaler
    
    
