"""
===============================================================================
POPULATION PREVALENCE CALIBRATION & BAYESIAN POSTERIOR ADJUSTMENT
===============================================================================

Scientific Rationale:
---------------------
In natural human populations, ABO and Rh blood antigens exhibit substantial
inherent class imbalance. For example, globally and in most regional cohorts:
  - O+ and B+ / A+ represent ~85-90% of all individuals.
  - AB- is universally the rarest blood group (~0.5 - 1.0%).

When training ML/DL classifiers on balanced or artificially sampled datasets,
the empirical training prior P_train(y) departs significantly from real-world
prevalence P_true(y). Uncorrected models overpredict rare classes (generating
false positives for AB-) or suffer from poor probabilistic calibration.

Bayesian Posterior Correction:
------------------------------
By Bayes' theorem, the true posterior is given by:
  P(y = c | x) = P(x | y = c) * P_true(y = c) / P(x)

Since the classifier learns P_model(y = c | x) proportional to P(x | y = c) * P_train(y = c),
the population-calibrated posterior is computed via:
  P_calibrated(y = c | x) proportional to P_model(y = c | x) * ( P_true(y = c) / P_train(y = c) )^alpha

Where:
  alpha in [0, 1]: Temperature / calibration factor (alpha=0 gives raw model, alpha=1 full Bayes).
  beta in [0, 1]:  Domain prior influence from ridge-pattern features.
"""

import numpy as np

CLASSES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Real-World ABO/Rh Population Prevalence (Configurable Baseline)
# Normalized frequencies across global & regional blood bank distributions
REAL_WORLD_PREVALENCE = {
    'O+': 0.370,
    'A+': 0.240,
    'B+': 0.285,
    'AB+': 0.065,
    'O-': 0.025,
    'A-': 0.008,
    'B-': 0.005,
    'AB-': 0.002
}

# Ensure normalization
_sum_prev = sum(REAL_WORLD_PREVALENCE.values())
for _k in REAL_WORLD_PREVALENCE:
    REAL_WORLD_PREVALENCE[_k] /= _sum_prev


class BayesianPrevalenceCalibrator:
    """
    Applies Bayesian prevalence adjustment and fuses soft dermatoglyphic priors
    with SVM class probabilities.
    """
    def __init__(self, true_prevalence=None, alpha_prevalence=0.35, beta_pattern=0.25):
        """
        Args:
            true_prevalence: Dict of class name -> true population frequency.
            alpha_prevalence: Exponent controlling strength of prevalence correction.
                              0.0 = no correction; 1.0 = full Bayesian shift.
                              (Recommended default 0.30 - 0.40 to prevent extreme suppression).
            beta_pattern: Exponent controlling strength of dermatoglyphic prior bias.
        """
        self.true_prevalence = true_prevalence or REAL_WORLD_PREVALENCE.copy()
        self.alpha_prevalence = alpha_prevalence
        self.beta_pattern = beta_pattern
        self.training_prevalence = None
        
    def fit_training_distribution(self, y_train_labels, classes=CLASSES):
        """
        Compute empirical training prior from dataset labels.
        """
        unique, counts = np.unique(y_train_labels, return_counts=True)
        total = len(y_train_labels)
        
        self.training_prevalence = {}
        for idx, cls in enumerate(classes):
            if idx in unique:
                c_count = counts[np.where(unique == idx)[0][0]]
                self.training_prevalence[cls] = float(c_count) / total
            else:
                self.training_prevalence[cls] = 1.0 / len(classes)
                
        return self.training_prevalence

    def calibrate_probabilities(self, svm_probs_matrix, pattern_priors_list=None, classes=CLASSES):
        """
        Perform joint Bayesian prevalence correction and domain-prior fusion.
        
        Args:
            svm_probs_matrix: 2D numpy array (N, num_classes) of raw SVM predict_proba outputs.
            pattern_priors_list: Optional list/array (N, num_classes) or list of dicts with pattern priors.
            classes: Ordered list of class names.
            
        Returns:
            calibrated_probs: 2D numpy array (N, num_classes) of normalized posteriors.
        """
        if self.training_prevalence is None:
            # Default to uniform if fit_training_distribution was not explicitly called
            self.training_prevalence = {cls: 1.0 / len(classes) for cls in classes}
            
        # Compute prevalence ratio: (P_true / P_train)^alpha
        prev_ratios = np.zeros(len(classes), dtype=np.float32)
        for i, cls in enumerate(classes):
            p_true = self.true_prevalence.get(cls, 1.0 / len(classes))
            p_train = self.training_prevalence.get(cls, 1.0 / len(classes))
            prev_ratios[i] = (p_true / (p_train + 1e-6)) ** self.alpha_prevalence
            
        # Ensure svm_probs_matrix is 2D float32
        probs = np.array(svm_probs_matrix, dtype=np.float32)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
            
        # 1. Apply Prevalence Multiplier
        calibrated = probs * prev_ratios[np.newaxis, :]
        
        # 2. Fuse Pattern Priors if provided
        if pattern_priors_list is not None:
            if isinstance(pattern_priors_list, list) and isinstance(pattern_priors_list[0], dict):
                p_matrix = np.zeros_like(probs)
                for row_idx, p_dict in enumerate(pattern_priors_list):
                    for c_idx, cls in enumerate(classes):
                        p_matrix[row_idx, c_idx] = p_dict.get(cls, 1.0 / len(classes))
            else:
                p_matrix = np.array(pattern_priors_list, dtype=np.float32)
                if p_matrix.ndim == 1:
                    p_matrix = p_matrix.reshape(1, -1)
                    
            calibrated = calibrated * (p_matrix ** self.beta_pattern)
            
        # 3. Renormalize to valid probability distribution summing to 1.0
        row_sums = np.sum(calibrated, axis=1, keepdims=True) + 1e-8
        calibrated = calibrated / row_sums
        
        return calibrated

def compare_distribution_alignment(y_pred_labels, true_prevalence=REAL_WORLD_PREVALENCE, classes=CLASSES):
    """
    Compare predicted test set class distribution vs real-world population prevalence.
    Returns:
        dict with observed%, expected%, and absolute deviation per class.
    """
    total = len(y_pred_labels)
    unique, counts = np.unique(y_pred_labels, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    alignment_table = []
    total_tv_dist = 0.0 # Total Variation Distance
    
    for idx, cls in enumerate(classes):
        obs_count = count_dict.get(idx, 0)
        obs_pct = (obs_count / total) * 100.0
        exp_pct = true_prevalence.get(cls, 0.0) * 100.0
        diff = obs_pct - exp_pct
        total_tv_dist += abs(obs_pct - exp_pct) * 0.5
        
        alignment_table.append({
            'class': cls,
            'observed_pct': round(obs_pct, 2),
            'expected_prevalence_pct': round(exp_pct, 2),
            'deviation_pct': round(diff, 2)
        })
        
    return alignment_table, round(total_tv_dist, 2)
