import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from spike_data_loader import NeuralPatternGenerator
from spike_analyzer import EnhancedSpikeAnalyzer

def test_feature_extraction():
    """Test enhanced feature extraction on pathological patterns"""
    
    print("Testing Enhanced Feature Extraction...")
    print("=" * 50)
    
    # Initialize
    generator = NeuralPatternGenerator(n_neurons=30, trial_duration=2.0)
    analyzer = EnhancedSpikeAnalyzer(trial_duration=2.0)
    
    # Generate test patterns
    patterns = {}
    
    # Healthy Rate
    patterns['Healthy_Rate'] = generator.generate_synthetic_spikes(
        coding_type='rate',
        n_stimuli=5, n_trials_per_stimulus=5,
        oscillatory_power=0.0,
        population_synchrony=0.15,
        spike_regularity=0.8,
        pathological_bursting=0.0
    )
    
    # Healthy Temporal
    patterns['Healthy_Temporal'] = generator.generate_synthetic_spikes(
        coding_type='temporal',
        n_stimuli=5, n_trials_per_stimulus=5,
        oscillatory_power=0.0,
        population_synchrony=0.15,
        spike_regularity=0.8,
        pathological_bursting=0.0
    )
    
    # Parkinsonian
    patterns['Parkinsonian'] = generator.generate_synthetic_spikes(
        coding_type='rate',
        n_stimuli=5, n_trials_per_stimulus=5,
        oscillatory_power=0.7,
        population_synchrony=0.8,
        spike_regularity=0.3,
        pathological_bursting=0.2
    )
    
    # Epileptiform
    patterns['Epileptiform'] = generator.generate_synthetic_spikes(
        coding_type='temporal',
        n_stimuli=5, n_trials_per_stimulus=5,
        oscillatory_power=0.4,
        population_synchrony=0.9,
        spike_regularity=0.5,
        pathological_bursting=0.7
    )
    
    # Mixed Pathology
    patterns['Mixed_Pathology'] = generator.generate_synthetic_spikes(
        coding_type='mixed',
        n_stimuli=5, n_trials_per_stimulus=5,
        oscillatory_power=0.6,
        population_synchrony=0.7,
        spike_regularity=0.4,
        pathological_bursting=0.4
    )
    
    # Extract features for each pattern
    all_features = {}
    feature_categories = ['rate_features', 'temporal_features', 'spectral_features', 
                         'synchrony_features', 'burst_features', 'regularity_features', 
                         'pathology_features']
    
    print("Extracting features...")
    for pattern_name, data in patterns.items():
        print(f"  Analyzing {pattern_name}...")
        analysis_results = analyzer.analyze_spike_data(data)
        all_features[pattern_name] = analysis_results
    
    # Create feature comparison
    compare_features(all_features, feature_categories)
    
    # Test feature extraction timing
    test_extraction_speed(analyzer, patterns)
    
    print("\nFeature extraction testing complete!")

def compare_features(all_features, feature_categories):
    """Compare features across different pattern types"""
    
    print("\nFeature Comparison Across Pattern Types:")
    print("=" * 50)
    
    # Create comprehensive feature matrix
    feature_matrix = []
    pattern_labels = []
    feature_names = []
    
    # Get all unique feature names
    all_feature_names = set()
    for pattern_name, features in all_features.items():
        for category in feature_categories:
            if category in features:
                for feature_name in features[category].keys():
                    all_feature_names.add(f"{category}_{feature_name}")
    
    feature_names = sorted(list(all_feature_names))
    
    # Build feature matrix
    for pattern_name, features in all_features.items():
        feature_vector = []
        for feature_name in feature_names:
            category, feat_name = feature_name.split('_', 1)
            if category in features and feat_name in features[category]:
                value = features[category][feat_name]
                # Handle NaN/inf values
                if np.isnan(value) or np.isinf(value):
                    value = 0
                feature_vector.append(value)
            else:
                feature_vector.append(0)
        
        feature_matrix.append(feature_vector)
        pattern_labels.append(pattern_name)
    
    feature_matrix = np.array(feature_matrix)
    
    # Display key features
    key_features = [
        'rate_features_mean_population_rate',
        'temporal_features_mean_timing_precision',
        'spectral_features_beta_power',
        'spectral_features_beta_ratio',
        'synchrony_features_mean_cross_correlation',
        'synchrony_features_population_sync_index',
        'burst_features_burst_index',
        'regularity_features_regularity_index',
        'pathology_features_parkinsonian_index',
        'pathology_features_epileptiform_index'
    ]
    
    print("\nKey Feature Values:")
    print("-" * 80)
    print(f"{'Pattern':<15} {'Rate':<8} {'Timing':<8} {'Beta':<8} {'Sync':<8} {'Burst':<8} {'Park':<8} {'Epil':<8}")
    print("-" * 80)
    
    for i, pattern in enumerate(pattern_labels):
        values = []
        for feature in key_features[:7]:  # First 7 features
            if feature in feature_names:
                idx = feature_names.index(feature)
                values.append(f"{feature_matrix[i, idx]:.3f}")
            else:
                values.append("N/A")
        
        print(f"{pattern:<15} {values[0]:<8} {values[1]:<8} {values[2]:<8} {values[4]:<8} {values[6]:<8} {values[-2]:<8} {values[-1]:<8}")
    
    # Create feature heatmap
    plot_feature_heatmap(feature_matrix, pattern_labels, feature_names, key_features)
    
    # Statistical summary
    print(f"\nFeature Extraction Summary:")
    print(f"  Total patterns analyzed: {len(pattern_labels)}")
    print(f"  Total features extracted: {len(feature_names)}")
    print(f"  Feature categories: {len(feature_categories)}")
    
    # Check for feature separation
    analyze_feature_separation(feature_matrix, pattern_labels, feature_names)

def plot_feature_heatmap(feature_matrix, pattern_labels, feature_names, key_features):
    """Plot heatmap of key features"""
    
    # Select key features for visualization
    key_indices = []
    key_names_short = []
    
    for feature in key_features:
        if feature in feature_names:
            key_indices.append(feature_names.index(feature))
            # Shorten feature names for display
            short_name = feature.split('_')[-1]  # Take last part
            key_names_short.append(short_name)
    
    if key_indices:
        key_matrix = feature_matrix[:, key_indices]
        
        # Normalize features for better visualization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_matrix = scaler.fit_transform(key_matrix.T).T
        
        # Create heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(normalized_matrix, 
                   xticklabels=key_names_short,
                   yticklabels=pattern_labels,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0)
        plt.title('Normalized Feature Values Across Pattern Types')
        plt.xlabel('Features')
        plt.ylabel('Pattern Types')
        plt.tight_layout()
        plt.show()

def analyze_feature_separation(feature_matrix, pattern_labels, feature_names):
    """Analyze how well features separate different pattern types"""
    
    print(f"\nFeature Separation Analysis:")
    print("-" * 40)
    
    # Calculate pairwise distances between pattern types
    from scipy.spatial.distance import pdist, squareform
    from sklearn.preprocessing import StandardScaler
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    
    # Calculate distances
    distances = pdist(normalized_features, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Show separation between pattern types
    print("Euclidean distances between pattern types:")
    for i, pattern1 in enumerate(pattern_labels):
        for j, pattern2 in enumerate(pattern_labels):
            if i < j:
                print(f"  {pattern1} vs {pattern2}: {distance_matrix[i, j]:.2f}")
    
    # Find most discriminative features
    find_discriminative_features(feature_matrix, pattern_labels, feature_names)

def find_discriminative_features(feature_matrix, pattern_labels, feature_names):
    """Find features that best discriminate between pattern types"""
    
    print(f"\nMost Discriminative Features:")
    print("-" * 30)
    
    # Calculate ANOVA F-statistic for each feature
    from scipy.stats import f_oneway
    
    f_scores = []
    p_values = []
    
    # Group feature values by pattern type
    pattern_groups = {}
    for i, pattern in enumerate(pattern_labels):
        if pattern not in pattern_groups:
            pattern_groups[pattern] = []
        pattern_groups[pattern].append(i)
    
    for feat_idx in range(feature_matrix.shape[1]):
        feature_values = feature_matrix[:, feat_idx]
        
        # Split by pattern type
        groups = []
        for pattern, indices in pattern_groups.items():
            group_values = feature_values[indices]
            groups.append(group_values)
        
        # Calculate F-statistic
        if len(groups) > 1:
            f_stat, p_val = f_oneway(*groups)
            f_scores.append(f_stat)
            p_values.append(p_val)
        else:
            f_scores.append(0)
            p_values.append(1)
    
    # Sort by F-score
    feature_importance = list(zip(feature_names, f_scores, p_values))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Display top discriminative features
    print("Top 10 discriminative features (F-score):")
    for i, (name, f_score, p_val) in enumerate(feature_importance[:10]):
        short_name = name.split('_')[-1] if '_' in name else name
        print(f"  {i+1:2d}. {short_name:<20} F={f_score:.2f}, p={p_val:.3f}")

def test_extraction_speed(analyzer, patterns):
    """Test feature extraction speed"""
    
    print(f"\nFeature Extraction Speed Test:")
    print("-" * 30)
    
    import time
    
    pattern_name = list(patterns.keys())[0]
    test_data = patterns[pattern_name]
    
    # Time the analysis
    start_time = time.time()
    results = analyzer.analyze_spike_data(test_data)
    end_time = time.time()
    
    extraction_time = end_time - start_time
    
    print(f"  Pattern: {pattern_name}")
    print(f"  Trials: {len(test_data['spike_trains'])}")
    print(f"  Neurons: {len(test_data['spike_trains'][0])}")
    print(f"  Extraction time: {extraction_time:.3f} seconds")
    print(f"  Features per second: {len(results)*100/extraction_time:.0f}")

if __name__ == "__main__":
    test_feature_extraction()