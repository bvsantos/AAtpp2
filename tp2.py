from sklearn.manifold import TSNE, Isomap;
from sklearn.decomposition import PCA;
import numpy as np;
from tp2_aux import images_as_matrix;
from sklearn.manifold import Isomap;

matrix = images_as_matrix();

def getFeatsPCA():
    pca = PCA(n_components=6);
    return pca.fit_transform(matrix);

def getFeatsTSNE():
    tsne = TSNE(n_components = 6, method='exact');
    return tsne.fit_transform(matrix);

def getFeatsIsomap():
    isoMap = Isomap(n_components = 6);
    return isoMap.fit_transform(matrix);

def getFeatures():
    featsPCA = getFeatsPCA();
    featsTSNE = getFeatsTSNE(); 
    featsIsomap = getFeatsIsomap()
    print(featsPCA);
    print(featsTSNE);
    print(featsIsomap);

getFeatures();

    
    
