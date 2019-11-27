from sklearn.manifold import TSNE;
import numpy as np;
from tp2_aux import images_as_matrix;
from sklearn.manifold import Isomap;

matrix = images_as_matrix();

def tsne():
    print(matrix);
    TSNE(n_components = 6, method='exact').fit(matrix);
    print(matrix);

tsne();
    
    