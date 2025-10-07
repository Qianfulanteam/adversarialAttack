import numpy as np

attack_configs = {
    'autoattack': {'norm': np.inf, 'eps': 0.3, 'seed': None, 'verbose': False, 'attacks_to_run': [],
                   'version': 'standard', 'is_tf_model': False, 'logger': None},
    'bim': {'norm': np.inf, 'eps': 4 / 255, 'stepsize': 1 / 255, 'steps': 20, 'loss': 'ce'},
    'dim': {'norm': np.inf, 'eps': 4 / 255, 'stepsize': 1 / 255, 'steps': 20, 'decay_factor': 1.0, 'resize_rate': 0.85,
            'diversity_prob': 0.7, 'loss': 'ce'},
    'fgsm': {'norm': np.inf, 'eps': 4 / 255, 'loss': 'ce'},
    'mim': {'norm': np.inf, 'eps': 4 / 255, 'stepsize': 1 / 255, 'steps': 20, 'decay_factor': 0.9, 'loss': 'ce'},
    'tim': {'norm': np.inf, 'kernel_name': 'gaussian', 'len_kernel': 15, 'nsig': 3, 'eps': 4 / 255, 'stepsize': 1 / 255,
            'steps': 20, 'decay_factor': 1.0, 'resize_rate': 0.85, 'diversity_prob': 0.7, 'loss': 'ce'},
    'cw': {'norm': 2, 'kappa': 0.0, 'lr': 0.2, 'init_const': 0.01, 'max_iter': 4,
           'binary_search_steps': 20, 'num_classes': 1000},
    'deepfool': {'norm': np.inf, 'overshoot': 0.02, 'max_iter': 20},
    'pgd': {'norm': np.inf, 'eps': 4 / 255, 'stepsize': 1 / 255, 'steps': 20, 'loss': 'ce'},
    
    'badnet': {'poisoning_rate': 0.1, 'epochs': 100, 'trigger_type':'square', 'trigger_size': 5,},
    
}
