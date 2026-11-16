"""
===============================================================================
RIDGE PATTERN CLASSIFIER & DERMATOGLYPHIC DOMAIN PRIOR ENGINE
===============================================================================

Scientific Context & Dermatoglyphic Literature:
-----------------------------------------------
Epidermal ridge patterns (dermatoglyphics) develop between the 10th and 24th
weeks of gestation under complex polygenic control alongside embryonic antigen
development (Cummins & Midlo, 1961; Bloterogel, 1927).

Numerous anthropometric and clinical studies have identified statistically
significant correlations between ABO/Rh blood antigen expressions and the
relative frequency of dermatoglyphic pattern configurations across the digits:
  1. Blood Group O: Consistently displays a statistically elevated frequency
     of Loop patterns (ulnar/radial) across multiple digits (Sangam et al., 2011;
     Bharadwaja et al., 2004).
  2. Blood Group B: Associated with a significantly higher prevalence of complex
     Whorl patterns (plain whorls, double loops) relative to baseline populations
     (Mahajan et al., 2017; Kshirsagar et al., 2003).
  3. Blood Group A: Characterized by a relatively balanced mixture of Loops and
     Whorls, alongside an elevated occurrence of Arch patterns (arches remain
     the least frequent pattern in the general human population, ~5-7%).
  4. Blood Group AB: Displays elevated structural complexity (accidental/composite
     whorls, high minutiae density) accompanied by a notable paucity/near-absence
     of Arch patterns (Eboh, 2013).

CRITICAL SCIENTIFIC NOTE FOR DEFENSE / VIVA:
--------------------------------------------
These associations represent STATISTICAL CORRELATIONS observed across population
cohorts, NOT deterministic 1-to-1 biological laws. Accordingly, this module encodes
these heuristics as SOFT DOMAIN PRIORS that bias decision boundaries in uncertain
regimes, while never overriding strong, direct deep-learning representations.
"""

import numpy as np
from PIL import Image

PATTERN_CLASSES = ['loop', 'whorl', 'arch', 'composite']

# Configurable Domain Multipliers for Soft Prior Estimation
# These multipliers bias the class likelihood based on pattern prevalence.
DEFAULT_PRIOR_MULTIPLIERS = {
    'O': {
        'loop_bias': 1.45,       # Higher likelihood when loops dominate
        'whorl_penalty': 0.85,
        'arch_penalty': 0.90,
        'composite_penalty': 0.85
    },
    'B': {
        'whorl_bias': 1.50,      # Higher likelihood when complex whorls are elevated
        'loop_penalty': 0.90,
        'arch_penalty': 0.85,
        'composite_bias': 1.15
    },
    'A': {
        'arch_bias': 1.65,       # Elevated relative to other groups when arches appear
        'balanced_mix_bias': 1.30, # Balanced loop/whorl ratio
        'composite_penalty': 0.85
    },
    'AB': {
        'composite_bias': 1.55,  # High pattern complexity / composite configurations
        'whorl_bias': 1.25,
        'arch_suppression': 0.45 # Near-absence of simple arches
    }
}

def compute_orientation_field(gray_arr, block_size=16):
    """
    Compute dense ridge orientation field (least-mean-square 2D gradients).
    Returns:
        theta: Local orientation angle in radians [0, pi]
        coherence: Normalized orientation strength [0, 1]
    """
    h, w = gray_arr.shape
    gx = np.zeros_like(gray_arr, dtype=np.float32)
    gy = np.zeros_like(gray_arr, dtype=np.float32)
    
    gx[:, 1:-1] = (gray_arr[:, 2:] - gray_arr[:, :-2]) * 0.5
    gy[1:-1, :] = (gray_arr[2:, :] - gray_arr[:-2, :]) * 0.5
    
    # Gradient covariances
    gxx = gx * gx
    gyy = gy * gy
    gxy = gx * gy
    
    # Block aggregation
    h_blocks = h // block_size
    w_blocks = w // block_size
    
    theta_field = np.zeros((h_blocks, w_blocks), dtype=np.float32)
    coherence_field = np.zeros((h_blocks, w_blocks), dtype=np.float32)
    
    for r in range(h_blocks):
        for c in range(w_blocks):
            r_start, r_end = r * block_size, (r + 1) * block_size
            c_start, c_end = c * block_size, (c + 1) * block_size
            
            b_gxx = np.sum(gxx[r_start:r_end, c_start:c_end])
            b_gyy = np.sum(gyy[r_start:r_end, c_start:c_end])
            b_gxy = np.sum(gxy[r_start:r_end, c_start:c_end])
            
            two_gxy = 2.0 * b_gxy
            gxx_m_gyy = b_gxx - b_gyy
            
            # Dominant orientation angle in [0, pi)
            theta = 0.5 * np.arctan2(two_gxy, gxx_m_gyy) + (np.pi / 2.0)
            theta_field[r, c] = theta % np.pi
            
            # Coherence
            denom = b_gxx + b_gyy + 1e-5
            num = np.hypot(gxx_m_gyy, two_gxy)
            coherence_field[r, c] = num / denom
            
    return theta_field, coherence_field

def detect_singularities(theta_field, coherence_field, coh_threshold=0.3):
    """
    Detect Singular Points (Core and Delta) via Poincare Index over orientation field.
    Core: Poincare index = +1/2 (+pi)
    Delta: Poincare index = -1/2 (-pi)
    """
    h_b, w_b = theta_field.shape
    cores = []
    deltas = []
    
    for r in range(1, h_b - 1):
        for c in range(1, w_b - 1):
            if coherence_field[r, c] < coh_threshold:
                continue
                
            # 8-connected neighbors in closed loop
            neighbors = [
                theta_field[r-1, c-1], theta_field[r-1, c], theta_field[r-1, c+1],
                theta_field[r, c+1], theta_field[r+1, c+1], theta_field[r+1, c],
                theta_field[r+1, c-1], theta_field[r, c-1]
            ]
            
            diffs = []
            for k in range(8):
                d = neighbors[(k + 1) % 8] - neighbors[k]
                while d > np.pi / 2.0:
                    d -= np.pi
                while d < -np.pi / 2.0:
                    d += np.pi
                diffs.append(d)
                
            poincare = np.sum(diffs)
            
            if np.isclose(poincare, np.pi, atol=0.85):
                cores.append((r, c))
            elif np.isclose(poincare, -np.pi, atol=0.85):
                deltas.append((r, c))
                
    return len(cores), len(deltas)

def classify_ridge_pattern(gray_img_arr):
    """
    Classify fingerprint into Henry classification types (Loop, Whorl, Arch, Composite)
    using structural orientation curvature, core/delta singularities, and circular variance.
    Returns:
        pattern_type: str ('loop', 'whorl', 'arch', 'composite')
        pattern_probabilities: dict of soft scores per pattern
        features_dict: extracted topological features
    """
    theta_field, coherence_field = compute_orientation_field(gray_arr=gray_img_arr, block_size=16)
    n_cores, n_deltas = detect_singularities(theta_field, coherence_field)
    
    # Global circular variance of orientation
    sin_theta = np.sin(2.0 * theta_field)
    cos_theta = np.cos(2.0 * theta_field)
    mean_sin = np.mean(sin_theta)
    mean_cos = np.mean(cos_theta)
    circular_variance = 1.0 - np.hypot(mean_sin, mean_cos) # High for whorls/composites, low for arches
    
    # Top-half vs bottom-half curvature gradient
    h_b = theta_field.shape[0]
    top_theta = theta_field[:h_b//2, :]
    bot_theta = theta_field[h_b//2:, :]
    curvature_divergence = float(np.mean(np.abs(np.diff(top_theta, axis=0))) + np.mean(np.abs(np.diff(bot_theta, axis=0))))
    
    # Compute soft pattern probability vector
    # 1. Whorl: 2+ cores, 2+ deltas, or high circular variance (> 0.48)
    whorl_score = 0.15 + (0.35 if n_cores >= 2 else 0.0) + (0.30 if n_deltas >= 2 else 0.0) + (0.25 if circular_variance > 0.45 else 0.0)
    
    # 2. Loop: 1 core, 1 delta, moderate circular variance
    loop_score = 0.20 + (0.45 if n_cores == 1 else 0.0) + (0.30 if n_deltas == 1 else 0.0) + (0.15 if 0.20 <= circular_variance <= 0.50 else 0.0)
    
    # 3. Arch: 0 cores, 0 deltas, low circular variance (< 0.25), smooth unidirectional flow
    arch_score = 0.10 + (0.50 if (n_cores == 0 and n_deltas == 0) else 0.0) + (0.35 if circular_variance < 0.28 else 0.0)
    
    # 4. Composite / Accidental: high singularity count or complex curvature divergence
    comp_score = 0.05 + (0.40 if (n_cores + n_deltas >= 3) else 0.0) + (0.35 if curvature_divergence > 1.2 else 0.0)
    
    # Softmax normalization
    raw_scores = np.array([loop_score, whorl_score, arch_score, comp_score], dtype=np.float32)
    exp_scores = np.exp(raw_scores * 2.5) # Temperature scale
    pattern_probs = exp_scores / np.sum(exp_scores)
    
    pattern_dict = {
        'loop': float(pattern_probs[0]),
        'whorl': float(pattern_probs[1]),
        'arch': float(pattern_probs[2]),
        'composite': float(pattern_probs[3])
    }
    
    best_pattern = PATTERN_CLASSES[np.argmax(pattern_probs)]
    
    features = {
        'n_cores': n_cores,
        'n_deltas': n_deltas,
        'circular_variance': round(float(circular_variance), 4),
        'curvature_divergence': round(curvature_divergence, 4),
        'mean_coherence': round(float(np.mean(coherence_field)), 4)
    }
    
    return best_pattern, pattern_dict, features

def extract_ridge_domain_feature_vector(img_or_arr):
    """
    Extract modular 16-dimensional ridge pattern & biometric domain feature vector
    consisting of pattern distribution probabilities, singularity metrics,
    ridge orientation coherence, and spatial minutiae density descriptors.
    """
    if isinstance(img_or_arr, (str, Image.Image)):
        if isinstance(img_or_arr, str):
            pil_img = Image.open(img_or_arr).convert('L')
        else:
            pil_img = img_or_arr.convert('L')
        arr = np.array(pil_img, dtype=np.float32)
    else:
        arr = np.array(img_or_arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = np.mean(arr, axis=2)
            
    best_pattern, pattern_probs, feats = classify_ridge_pattern(arr)
    
    # Gradients & minutiae proxy
    gx, gy = np.gradient(arr)
    mag = np.hypot(gx, gy)
    mean_density = float(np.mean(mag))
    std_contrast = float(np.std(arr))
    
    # Build compact engineered vector:
    # [4 pattern probs, n_cores, n_deltas, circular_var, curvature, coherence, mean_density, std_contrast, 4 spatial quadrants]
    h, w = arr.shape
    q1 = float(np.mean(mag[:h//2, :w//2]))
    q2 = float(np.mean(mag[:h//2, w//2:]))
    q3 = float(np.mean(mag[h//2:, :w//2]))
    q4 = float(np.mean(mag[h//2:, w//2:]))
    
    vec = np.array([
        pattern_probs['loop'],
        pattern_probs['whorl'],
        pattern_probs['arch'],
        pattern_probs['composite'],
        feats['n_cores'],
        feats['n_deltas'],
        feats['circular_variance'],
        feats['curvature_divergence'],
        feats['mean_coherence'],
        mean_density,
        std_contrast,
        q1, q2, q3, q4
    ], dtype=np.float32)
    
    return vec, best_pattern, pattern_probs, feats

def compute_dermatoglyphic_prior_scores(pattern_probs, multipliers=None):
    """
    Compute rule-based, literature-informed prior score (0-1) per blood group class:
    ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
    
    Formulation:
      Score(O)  = Base * (loop_prob * loop_bias + whorl_penalty * whorl_prob)
      Score(B)  = Base * (whorl_prob * whorl_bias + composite_bias * comp_prob)
      Score(A)  = Base * (arch_prob * arch_bias + balanced_mix_bias * sqrt(loop_prob * whorl_prob))
      Score(AB) = Base * (comp_prob * comp_bias + whorl_prob * whorl_bias) * (1 - arch_prob * arch_suppression)
    """
    cfg = multipliers or DEFAULT_PRIOR_MULTIPLIERS
    
    p_loop = pattern_probs['loop']
    p_whorl = pattern_probs['whorl']
    p_arch = pattern_probs['arch']
    p_comp = pattern_probs['composite']
    
    # 1. Group O Prior Calculation
    o_cfg = cfg['O']
    score_o = (
        p_loop * o_cfg['loop_bias'] +
        p_whorl * o_cfg['whorl_penalty'] +
        p_arch * o_cfg['arch_penalty'] +
        p_comp * o_cfg['composite_penalty']
    )
    
    # 2. Group B Prior Calculation
    b_cfg = cfg['B']
    score_b = (
        p_whorl * b_cfg['whorl_bias'] +
        p_comp * b_cfg['composite_bias'] +
        p_loop * b_cfg['loop_penalty'] +
        p_arch * b_cfg['arch_penalty']
    )
    
    # 3. Group A Prior Calculation
    a_cfg = cfg['A']
    balanced_term = 2.0 * np.sqrt(max(1e-5, p_loop * p_whorl))
    score_a = (
        p_arch * a_cfg['arch_bias'] +
        balanced_term * a_cfg['balanced_mix_bias'] +
        p_comp * a_cfg['composite_penalty']
    )
    
    # 4. Group AB Prior Calculation
    ab_cfg = cfg['AB']
    arch_factor = max(0.1, 1.0 - (p_arch * ab_cfg['arch_suppression']))
    score_ab = (
        (p_comp * ab_cfg['composite_bias'] + p_whorl * ab_cfg['whorl_bias']) * arch_factor
    )
    
    # Ensure positive scores
    raw_abo = np.array([score_a, score_b, score_ab, score_o], dtype=np.float32)
    raw_abo = np.maximum(raw_abo, 0.05)
    
    # Normalize ABO prior distribution to sum to 1.0
    abo_norm = raw_abo / np.sum(raw_abo)
    
    # Expand into 8 classes: A+, A-, AB+, AB-, B+, B-, O+, O-
    # Distribute ABO across Rh+ and Rh- with default Rh+ dominance (~85% Rh+, 15% Rh-)
    rh_pos_factor = 0.85
    rh_neg_factor = 0.15
    
    prior_dict = {
        'A+': float(abo_norm[0] * rh_pos_factor),
        'A-': float(abo_norm[0] * rh_neg_factor),
        'B+': float(abo_norm[1] * rh_pos_factor),
        'B-': float(abo_norm[1] * rh_neg_factor),
        'AB+': float(abo_norm[2] * rh_pos_factor),
        'AB-': float(abo_norm[2] * rh_neg_factor),
        'O+': float(abo_norm[3] * rh_pos_factor),
        'O-': float(abo_norm[3] * rh_neg_factor)
    }
    
    # Renormalize to ensure exact sum to 1.0
    total = sum(prior_dict.values())
    for k in prior_dict:
        prior_dict[k] /= total
        
    return prior_dict
