from sklearn.manifold import TSNE, Isomap;
from sklearn.decomposition import PCA;
import numpy as np;
from tp2_aux import images_as_matrix;
from sklearn.manifold import Isomap;
from sklearn.preprocessing import StandardScaler;
from sklearn.cluster import KMeans;
from sklearn.metrics import silhouette_score;
from sklearn.neighbors import KNeighborsClassifier;
import matplotlib.pyplot as plt;

matrix = images_as_matrix();
scaler = StandardScaler();
scaledMatrix = scaler.fit_transform(matrix);

def combineHorizontaly(A,B):
    shA=np.shape(A);
    shB=np.shape(B);
    colTot=shA[1]+shB[1];
    rowMax=np.max((shA[0],shB[0]));
    CHorz=np.zeros((rowMax,colTot));
    CHorz[0:shA[0],0:shA[1]]=A;
    CHorz[0:shB[0],shA[1]:colTot]=B;
    return CHorz;

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
    resultados = [];
    for n_clusters in range(2,30):
        region = KMeans(n_clusters=n_clusters).fit(feats);
        randIndex, precision, recall, f1, adjRandIndex = computeAlgRand(region.labels_,labels);
        resultados.append([n_clusters,rendIndex,precision,recall,f1,adjRandIndex,silhouette_score(feats,labels)]);
    resultados = np.array(resultados);
    return resultados;
        
getFeatures();

def drawGraph(orderedMaxD):
    plt.rcParams['axes.facecolor'] = 'lightgrey';
    plt.title('5-dist graph');
    plt.xlabel('Points sorted by distance');
    plt.ylabel('5 min distance');
    plt.plot(range(len(orderedMaxD)), orderedMaxD, '-r');
    plt.show();
    
drawGraph(neighbor(getFeatures()));

#standartscaler
    
    
