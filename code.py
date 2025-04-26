import random
import cupy as cp
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import entropy, wasserstein_distance
from tqdm import tqdm
from collections import defaultdict

def compute_H1(features, weights=None):
    """Morphological complexity"""
    n_particles, n_features = features.shape
    weights = weights / np.sum(weights) if weights is not None else np.ones(n_particles)/n_particles

    weighted_mean = np.average(features, axis=0, weights=weights)

    centered = features - weighted_mean

    weighted_cov = np.cov(centered.T, aweights=weights)

    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    eigvals = np.clip(eigvals, 0, None)
    sorted_idx = np.argsort(eigvals)[::-1]
    explained_variance = eigvals[sorted_idx]
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    

    cum_var = np.cumsum(explained_variance_ratio)
    # n_components = np.argmax(cum_var >= 0.95) + 1
    # kept_var_ratio = explained_variance_ratio[:n_components]
    n_components = 40
    kept_var_ratio = explained_variance_ratio[:n_components]
    

    H1 = entropy(kept_var_ratio)
    H1_MAX_ENTROPY = np.log(n_components)
    H1_scaled = H1 / H1_MAX_ENTROPY if H1_MAX_ENTROPY > 0 else 0.0
    
    return H1, n_components, H1_scaled

def compute_H2(diameters, weights=None, min_d=0.2, max_d=10.0, num_bins=20):
    """Size distribution complexity"""
    n_particles = len(diameters)
    weights = weights / np.sum(weights) if weights is not None else np.ones(n_particles)/n_particles
    

    # bins = np.linspace(min_d, max_d, num=num_bins+1)
    bins = np.logspace(np.log10(min_d), np.log10(max_d), num=num_bins+1)
    # bins = np.array([0.2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    counts, _ = np.histogram(diameters, bins=bins, weights=weights)
    counts += 1e-6  
    
    probabilities = counts / np.sum(counts)
    H2 = entropy(probabilities)
    H2_MAX_ENTROPY = np.log(num_bins)
    H2_scaled = H2 / H2_MAX_ENTROPY
    
    return H2, H2_MAX_ENTROPY, H2_scaled

def compute_H3(elements_matrix, weights=None, detection_threshold=0.00):
    """Elemental complexity"""
    elements_matrix = np.array(elements_matrix)
    n_particles, n_elements = elements_matrix.shape
    weights = weights / np.sum(weights) if weights is not None else np.ones(n_particles)/n_particles
    
    weighted_mean = np.average(elements_matrix, axis=0, weights=weights)

    centered = elements_matrix - weighted_mean

    weighted_cov = np.cov(centered.T, aweights=weights)

    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    eigvals = np.clip(eigvals, 0, None)  
    sorted_idx = np.argsort(eigvals)[::-1]
    explained_variance = eigvals[sorted_idx]
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    
    H3 = entropy(explained_variance_ratio)  
    H3_MAX_ENTROPY = np.log(n_elements)
    H3_scaled = H3 / H3_MAX_ENTROPY if H3_MAX_ENTROPY > 0 else 0.0
    
    return H3, n_elements, H3_scaled

def compute_N1(features, n_pt_list, Sampling_method = "CCSEM", n_clusters = 20, times = 100, weights=None):
    """BPSA(Mor)"""
    n_particles, n_features = features.shape
    weights = weights / np.mean(weights) if weights is not None else np.ones(n_particles)

    # pca = PCA(n_components=0.95)
    pca = PCA(n_components = 40)
    reduced_data = pca.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters,  n_init=10, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)
    
    exp_values = np.bincount(clusters, weights = weights) / len(clusters)
    
    df_clusters = pd.DataFrame({'clusters': clusters, 'weights': weights})
    error_data = {}
    for n_pt in tqdm(n_pt_list, desc="compute_N1", leave=False):
        errors = []
        for i in list(range(0, times)):

            if Sampling_method == "CCSEM":
                selec_df = SEM_random(df_clusters, n_pt = int(n_pt))
            else:
                selec_df = df_clusters.sample(frac=1).reset_index(drop=True)[:int(n_pt)] 
            
            obs_values = np.bincount(selec_df['clusters'], minlength=len(exp_values), weights = selec_df['weights']) / int(n_pt)
            relative_deviations = np.abs((obs_values - exp_values) / exp_values)
            # relative_deviations = np.nan_to_num(relative_deviations, nan=0)
            error = np.nansum(exp_values * relative_deviations)
            errors.append(error)
        error_data[n_pt] = errors
    error_data_df = pd.DataFrame(error_data)
    error_means = error_data_df.mean().tolist()
    
    min_pt_5index = next((idx for idx, value in enumerate(error_means) if value < 0.05), None)
    
    min_pt_5pct = n_pt_list[min_pt_5index] if min_pt_5index is not None else np.nan
    
    return min_pt_5pct


def compute_N2(diameters, n_pt_list, Sampling_method = "CCSEM", times = 100, weights=None, num_bins=20):
    """BPSA (Size)"""
    n_particles = len(diameters)
    weights = weights / np.mean(weights) if weights is not None else np.ones(n_particles)
    df_diameters = pd.DataFrame({'diameters': diameters, 'weights': weights})
    
    def compute_counts(diameters,  weights, min_d=0.2, max_d=10.0, num_bins = num_bins):
        bins = np.logspace(np.log10(min_d), np.log10(max_d), num=num_bins+1)
        counts, _ = np.histogram(diameters, bins=bins, weights = weights)
        counts += 1e-6  
        return counts
    
    exp_counts = compute_counts(diameters,  weights)
    exp_probs = exp_counts / np.sum(exp_counts)
    
    error_data = {}
    for n_pt in tqdm(n_pt_list, desc="compute_N2", leave=False):
        errors = []
        for i in list(range(0, times)):
            
            if Sampling_method == "CCSEM":
                selec_df = SEM_random(df_diameters, n_pt = int(n_pt))
            else:
                selec_df = df_diameters.sample(frac=1).reset_index(drop=True)[:int(n_pt)] 
            
            obs_counts = compute_counts(selec_df['diameters'].tolist(), selec_df['weights'].tolist())
            obs_probs = obs_counts / np.sum(obs_counts)

            relative_deviations = np.abs((obs_probs - exp_probs) / exp_probs)
            # relative_deviations = np.nan_to_num(relative_deviations, nan=0)
            error = np.nansum(exp_probs * relative_deviations)
            errors.append(error)
        error_data[n_pt] = errors
    error_data_df = pd.DataFrame(error_data)
    error_means = error_data_df.mean().tolist()
    

    min_pt_5index = next((idx for idx, value in enumerate(error_means) if value < 0.05), None)
    
    min_pt_5pct = n_pt_list[min_pt_5index] if min_pt_5index is not None else np.nan
    
    return min_pt_5pct  


def compute_N3(elements_dataframe, n_pt_list, Sampling_method = "CCSEM", times = 100, weights=None):
    """BPSA (Ele)"""
    weights = weights / np.mean(weights) if weights is not None else np.ones(len(elements_dataframe))
    
    weights_array = np.array(weights).reshape(-1, 1)
    ele_data_weight = elements_dataframe.multiply(weights_array, axis=0)
    
    exp_values = np.array(ele_data_weight.mean()/sum(ele_data_weight.mean()))
    mask = exp_values != 0
    
    error_data = {}
    for n_pt in tqdm(n_pt_list, desc="compute_N3", leave=False):
        errors = []
        for i in list(range(0, times)):
            
            if Sampling_method == "CCSEM":
                selec_df = SEM_random(ele_data_weight, n_pt = int(n_pt))
            else:
                selec_df = ele_data_weight.sample(frac=1).reset_index(drop=True)[:int(n_pt)]
            
            obs_values = np.array(selec_df.mean()/sum(selec_df.mean()))
            relative_deviations = np.abs((obs_values[mask] - exp_values[mask]) / exp_values[mask])
            error = np.nansum(exp_values[mask] * relative_deviations)
            errors.append(error)
        error_data[n_pt] = errors
    
    error_data_df = pd.DataFrame(error_data)
    error_means = error_data_df.mean().tolist()

    min_pt_5index = next((idx for idx, value in enumerate(error_means) if value < 0.05), None)
    
    min_pt_5pct = n_pt_list[min_pt_5index] if min_pt_5index is not None else np.nan
    
    return min_pt_5pct


def sliced_wasserstein_gpu(X, Y, n_proj=1000):
    X_gpu = cp.array(X, dtype=cp.float64)
    Y_gpu = cp.array(Y, dtype=cp.float64)
    d = X_gpu.shape[1]
    
    theta_gpu = cp.empty((n_proj, d), dtype=cp.float64)
    for i in range(0, n_proj, 1000):
        batch = min(1000, n_proj-i)
        theta_gpu[i:i+batch] = cp.random.randn(batch, d)
        theta_gpu[i:i+batch] /= cp.linalg.norm(theta_gpu[i:i+batch], axis=1)[:, None]
    
    proj_X = X_gpu @ theta_gpu.T
    proj_Y = Y_gpu @ theta_gpu.T
    
    quantiles = cp.linspace(0, 1, 1000)
    X_quantiles = cp.quantile(proj_X, quantiles, axis=0)
    Y_quantiles = cp.quantile(proj_Y, quantiles, axis=0)
    
    return cp.mean(cp.abs(X_quantiles - Y_quantiles)).get()
    

def wasserstein_convergence(df, n_pt_list, Sampling_method = "CCSEM", n_trials=100):
    
    reference = df if isinstance(df, list) else list(df) if len(df.shape) == 1 else np.array(df)
    n_pt_list = [int(num) for num in n_pt_list]
    
    distances = []
    
    for n in tqdm(n_pt_list, desc = "wasserstein_convergence", leave=False):
        trial_dists = []
        
        for _ in range(n_trials):
            
            if isinstance(reference,np.ndarray) and len(reference.shape) == 2:
                if Sampling_method == "CCSEM":
                    subsample = SEM_random(reference, n_pt = n).to_numpy()
                else:
                    reference_copy = reference.copy()
                    np.random.shuffle(reference_copy)
                    subsample = reference_copy[:n]
                dist = sliced_wasserstein_gpu(reference, subsample)
            else: 
                if Sampling_method == "CCSEM":
                    sample_ = reference[random.randint(1, len(reference)):] + reference[:random.randint(1, len(reference))]
                    subsample = sample_[:n]
                else:
                    reference_copy = reference.copy()
                    random.shuffle(reference_copy)
                    subsample = reference_copy[:n]                
                dist = wasserstein_distance(reference, subsample)
            trial_dists.append(dist)
        avg_dist = np.average(trial_dists)
        distances.append(avg_dist)
    
    return distances


def cobb(args, method='percentile', sampling_method='circular', bootstrap_method='block'):

    df, n, L, B, confidence, p_true, labels, m = args
    np.random.seed(42 + m)
    
    if sampling_method == 'circular':
        start_idx = np.random.randint(0, len(df))
        indices = [(start_idx + i) % len(df) for i in range(n)]
        sample_df = df.iloc[indices]
    elif sampling_method == 'simple':
        sample_df = df.sample(n=n, replace=False)
    else:
        raise ValueError("sampling_method必须是'circular'或'simple'")
    

    presence = {lbl: (sample_df['class'] == lbl).any() for lbl in labels}
    

    original_counts = sample_df['class'].value_counts(normalize=True)
    p_observed = {lbl: original_counts.get(lbl, 0.0) for lbl in labels}
    

    blocks = []
    if bootstrap_method == 'block':
        step_size = L // 2
        for i in range(0, n, step_size):
            block = sample_df.iloc[i:i+L]
            if sampling_method == 'circular' and len(block) < L:
                remaining = L - len(block)
                block = pd.concat([block, sample_df.iloc[0:remaining]])
            blocks.append(block)
    
    jackknife_stats = defaultdict(list)
    if method == 'bca':
        if bootstrap_method == 'block':
            n_units = len(blocks)
            for i in range(n_units):
                jack_units = [blocks[j] for j in range(n_units) if j != i]
                jack_sample = pd.concat(jack_units).iloc[:n]
                counts = jack_sample['class'].value_counts(normalize=True)
                for lbl in labels:
                    jackknife_stats[lbl].append(counts.get(lbl, 0.0))
        else:
            n_units = len(sample_df)
            for i in range(n_units):
                jack_sample = sample_df.drop(sample_df.index[i])
                counts = jack_sample['class'].value_counts(normalize=True)
                for lbl in labels:
                    jackknife_stats[lbl].append(counts.get(lbl, 0.0))
    
    bootstrap_props = defaultdict(list)
    for b in range(B):
        if bootstrap_method == 'block':
            n_units = len(blocks)
            selected_indices = np.random.choice(n_units, size=n_units, replace=True)
            bootstrap_df = pd.concat([blocks[i] for i in selected_indices]).iloc[:n]
        else:
            n_units = len(sample_df)
            selected_indices = np.random.choice(n_units, size=n_units, replace=True)
            bootstrap_df = sample_df.iloc[selected_indices]
        
        class_counts = bootstrap_df['class'].value_counts(normalize=True)
        for lbl in labels:
            bootstrap_props[lbl].append(class_counts.get(lbl, 0.0))
    
    records = []
    for lbl in labels:
        if not presence[lbl]:
            continue  
        
        stats = np.array(bootstrap_props[lbl])
        

        if method == 'percentile':
            alpha = (1 - confidence) / 2
            ci_low, ci_high = np.quantile(stats, [alpha, 1 - alpha])
        elif method == 'bca':
            ci_low, ci_high = bca_interval(
                stats, 
                original_stat=p_observed[lbl],
                jackknife_stats=jackknife_stats[lbl],
                alpha=1-confidence
            )
        else:
            raise ValueError("method必须是'percentile'或'bca'")
        
        records.append({
            'simulation_id': m,
            'sample_size': n,
            'label': lbl,
            'p_observed': p_observed[lbl],
            'p_true': p_true[lbl],
            'ci_low': ci_low,
            'ci_high': ci_high,
            'ci_width': ci_high - ci_low,
            'covered': int(ci_low <= p_true[lbl] <= ci_high)
        })
    
    return records
