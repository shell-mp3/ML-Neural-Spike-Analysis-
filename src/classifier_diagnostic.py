# Enhanced configurations for better classification accuracy

# PROBLEM: Current patterns are too subtle - AI can't distinguish them
# SOLUTION: Use more extreme parameters that create clear differences

def get_enhanced_pattern_configs():
    """Enhanced pattern configurations with more extreme differences"""
    
    return {
        'Healthy_Rate': {
            'coding_type': 'rate',
            'oscillatory_power': 0.0,        # No pathological oscillations
            'population_synchrony': 0.05,    # Very low synchrony (was 0.15)
            'spike_regularity': 0.95,        # Very regular (was 0.8)
            'pathological_bursting': 0.0     # No bursting
        },
        'Healthy_Temporal': {
            'coding_type': 'temporal',
            'oscillatory_power': 0.0,
            'population_synchrony': 0.05,    # Very low synchrony
            'spike_regularity': 0.95,        # Very regular
            'pathological_bursting': 0.0
        },
        'Parkinsonian': {
            'coding_type': 'rate',
            'oscillatory_power': 0.9,        # Very strong beta (was 0.7)
            'population_synchrony': 0.9,     # High synchrony (was 0.8)
            'spike_regularity': 0.1,         # Very irregular (was 0.3)
            'pathological_bursting': 0.4     # Moderate bursting (was 0.2)
        },
        'Epileptiform': {
            'coding_type': 'temporal',
            'oscillatory_power': 0.2,        # Low beta (different from Parkinson's)
            'population_synchrony': 1.0,     # Maximum synchrony (was 0.9)
            'spike_regularity': 0.2,         # Very irregular (was 0.5)
            'pathological_bursting': 0.9     # Very high bursting (was 0.7)
        },
        'Mixed_Pathology': {
            'coding_type': 'mixed',
            'oscillatory_power': 0.7,        # Moderate beta
            'population_synchrony': 0.8,     # High synchrony
            'spike_regularity': 0.3,         # Irregular
            'pathological_bursting': 0.6     # High bursting
        }
    }

def get_training_configs():
    """Configurations for training with parameter variation"""
    
    import numpy as np
    
    configs = {}
    
    # Healthy patterns - tight ranges around healthy values
    configs['Healthy_Rate'] = {
        'coding_type': 'rate',
        'oscillatory_power': lambda: np.random.uniform(0.0, 0.05),
        'population_synchrony': lambda: np.random.uniform(0.05, 0.15),
        'spike_regularity': lambda: np.random.uniform(0.85, 0.95),
        'pathological_bursting': lambda: np.random.uniform(0.0, 0.05)
    }
    
    configs['Healthy_Temporal'] = {
        'coding_type': 'temporal',
        'oscillatory_power': lambda: np.random.uniform(0.0, 0.05),
        'population_synchrony': lambda: np.random.uniform(0.05, 0.15),
        'spike_regularity': lambda: np.random.uniform(0.85, 0.95),
        'pathological_bursting': lambda: np.random.uniform(0.0, 0.05)
    }
    
    # Parkinsonian - high beta, high sync, low regularity, moderate bursting
    configs['Parkinsonian'] = {
        'coding_type': np.random.choice(['rate', 'temporal']),
        'oscillatory_power': lambda: np.random.uniform(0.8, 1.0),      # Very high beta
        'population_synchrony': lambda: np.random.uniform(0.8, 0.95),  # High sync
        'spike_regularity': lambda: np.random.uniform(0.05, 0.2),      # Very irregular
        'pathological_bursting': lambda: np.random.uniform(0.3, 0.5)   # Moderate bursting
    }
    
    # Epileptiform - low beta, maximum sync, irregular, maximum bursting
    configs['Epileptiform'] = {
        'coding_type': np.random.choice(['rate', 'temporal']),
        'oscillatory_power': lambda: np.random.uniform(0.1, 0.3),      # Low beta (different from Parkinson's)
        'population_synchrony': lambda: np.random.uniform(0.95, 1.0),  # Maximum sync
        'spike_regularity': lambda: np.random.uniform(0.1, 0.3),       # Irregular
        'pathological_bursting': lambda: np.random.uniform(0.8, 1.0)   # Maximum bursting
    }
    
    # Mixed - intermediate values across all parameters
    configs['Mixed_Pathology'] = {
        'coding_type': np.random.choice(['rate', 'temporal', 'mixed']),
        'oscillatory_power': lambda: np.random.uniform(0.5, 0.8),
        'population_synchrony': lambda: np.random.uniform(0.6, 0.9),
        'spike_regularity': lambda: np.random.uniform(0.2, 0.5),
        'pathological_bursting': lambda: np.random.uniform(0.5, 0.8)
    }
    
    return configs

# Key insight: Make the differences MORE EXTREME
# 
# Current problem: 
# - Healthy synchrony: 0.15, Parkinsonian: 0.8  → Not different enough
# - Healthy regularity: 0.8, Parkinsonian: 0.3 → Not different enough
#
# Solution:
# - Healthy synchrony: 0.05, Parkinsonian: 0.9  → Much clearer difference
# - Healthy regularity: 0.95, Parkinsonian: 0.1 → Much clearer difference
# - Epileptiform bursting: 0.9 vs Parkinsonian: 0.4 → Clear distinction

def test_extreme_patterns():
    """Test function to verify extreme patterns work better"""
    
    from spike_data_loader import NeuralPatternGenerator
    from decoding_analysis import OptimizedMultiClassDecoder
    
    generator = NeuralPatternGenerator(n_neurons=20, trial_duration=2.0)
    decoder = OptimizedMultiClassDecoder()
    
    configs = get_enhanced_pattern_configs()
    
    print("Testing Enhanced Pattern Configurations")
    print("=" * 50)
    
    for pattern_name, config in configs.items():
        print(f"\nTesting {pattern_name}:")
        
        # Generate pattern
        data = generator.generate_synthetic_spikes(
            n_stimuli=3, n_trials_per_stimulus=1, **config
        )
        
        # Extract features
        features = decoder.extract_optimized_features(data)
        
        # Show key discriminative features
        print(f"  Firing Rate: {features[0]:.2f}")
        print(f"  Burst Index: {features[3]:.3f}")
        print(f"  Regularity:  {features[6]:.3f}")
        print(f"  Sync Index:  {features[9]:.3f}")

if __name__ == "__main__":
    test_extreme_patterns()