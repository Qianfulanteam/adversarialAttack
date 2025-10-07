import numpy as np

attack_configs = {
    'autoattack': {'norm': np.inf, 'eps': 0.3, 'seed': None, 'verbose': False, 'attacks_to_run': [],
                   'version': 'standard', 'is_tf_model': False, 'logger': None},
    'bim': {'norm': np.inf, 'eps': 4 / 255, 'stepsize': 1 / 255, 'steps': 20, 'loss': 'face'},
    'dim': {'norm': np.inf, 'eps': 4 / 255, 'stepsize': 1 / 255, 'steps': 20, 'decay_factor': 1.0, 'resize_rate': 0.85,
            'diversity_prob': 0.7, 'loss': 'face'},
    'fgsm': {'norm': np.inf, 'eps': 4 / 255, 'loss': 'face'},
    'mim': {'norm': np.inf, 'eps': 4 / 255, 'stepsize': 1 / 255, 'steps': 20, 'decay_factor': 0.9, 'loss': 'face'},
    'tim': {'norm': np.inf, 'kernel_name': 'gaussian', 'len_kernel': 15, 'nsig': 3, 'eps': 4 / 255, 'stepsize': 1 / 255,
            'steps': 20, 'decay_factor': 1.0, 'resize_rate': 0.85, 'diversity_prob': 0.7, 'loss': 'face'},
    'cim': {'norm': np.inf, 'eps': 4 / 255, 'loss': 'face', 'crop_prob': 1.0, 'crop_size_range':(20, 254)},
}
