from tp2_aux import images_as_matrix;
from sklearn.decomposition import PCA;
import numpy as np;

matrix = images_as_matrix();

def getFeatsPCA():
	pca = PCA(n_components=6);
	pca.fit(matrix);
	return pca.components_;

def getFeatures():
	feats = getFeatsPCA();

getFeatures();