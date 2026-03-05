import numpy as np
from sklearn.decomposition import PCA

def get_safe_pca(X: np.ndarray, target_components: int):
    """
    Calcula el número máximo seguro de componentes PCA.
    Asegura que nunca pidamos más componentes que muestras o características.
    """
    n_samples, n_features = X.shape
    
    # El PCA no puede tener más componentes que el menor entre:
    # 1. El objetivo deseado (target)
    # 2. Las características totales (n_features)
    # 3. La cantidad de muestras (n_samples) - 1, para tener grados de libertad
    safe_n = min(target_components, n_features, n_samples - 1)
    
    # Siempre al menos 1 componente
    n_pca = max(1, safe_n)
    
    return PCA(n_components=n_pca, random_state=42)
  
