import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from spike_data_loader import NeuralPatternGenerator
from spike_analyzer import EnhancedSpikeAnalyzer

def simple_feature_comparison():
    """Simple, direct feature comparison without complex aggregation"""
    
    print("Simple Feature Comparison Test")
    print("=" * 40)
    
    # Initialize
    generator = NeuralPatternGenerator(n_neurons=20, trial_duration=2.0)
    analyzer = EnhancedSpikeAnalyzer(trial_duration=2.0)
    
    # Generate test patterns
    patterns = {}
    
    print("Generating patterns...")
    
    # Healthy Rate
    patterns['Healthy_Rate'] = generator.generate_synthetic_spikes(
        coding_type='rate',
        n_stimuli=3, n_trials_per_stimulus=3,
        oscillatory_power=0.0,
        population_synchrony=0.15,
        spike_regularity=0.8,
        pathological_bursting=0.0
    )
    
    # Parkinsonian (should have high beta power and synchrony)
    patterns['Parkinsonian'] = generator.generate_synthetic_spikes(
        coding_type='rate',
        n_stimuli=3, n_trials_per_stimulus=3,
        oscillatory_power=0.7,      # Strong beta
        population_synchrony=0.8,   # High sync
        spike_regularity=0.3,       # Irregular
        pathological_bursting=0.2
    )
    
    # Epileptiform (should have very high synchrony and bursting)
    patterns['Epileptiform'] = generator.generate_synthetic_spikes(
        coding_type='temporal',
        n_stimuli=3, n_trials_per_stimulus=3,
        oscillatory_power=0.4,
        population_synchrony=0.9,   # Very high sync
        spike_regularity=0.5,
        pathological_bursting=0.7   # High bursting
    )
    
    # Extract and display features
    results = {}
    
    print("\nExtracting features...")
    for pattern_name, data in patterns.items():
        print(f"  Analyzing {pattern_name}...")
        analysis = analyzer.analyze_spike_data(data)
        results[pattern_name] = analysis
    
    # Create simple comparison table
    create_comparison_table(results)
    
    # Create focused visualizations
    create_focused_plots(results)
    
    print("\nFeature extraction test complete!")

def create_comparison_table(results):
    """Create a simple comparison table of key features"""
    
    print("\n" + "=" * 80)
    print("KEY FEATURE COMPARISON")
    print("=" * 80)
    
    # Select key discriminative features
    key_features = [
        ('rate_features', 'mean_population_rate', 'Firing Rate'),
        ('spectral_features', 'beta_power', 'Beta Power'),
        ('spectral_features', 'beta_ratio', 'Beta Ratio'),
        ('synchrony_features', 'population_sync_index', 'Sync Index'),
        ('synchrony_features', 'mean_cross_correlation', 'Cross Corr'),
        ('burst_features', 'burst_index', 'Burst Index'),
        ('regularity_features', 'regularity_index', 'Regularity'),
        ('pathology_features', 'parkinsonian_index', 'Parkinson'),
        ('pathology_features', 'epileptiform_index', 'Epilepsy')
    ]
    
    # Print header
    print(f"{'Pattern':<15}", end="")
    for _, _, display_name in key_features:
        print(f"{display_name:<12}", end="")
    print()
    print("-" * 80)
    
    # Print values for each pattern
    for pattern_name, analysis in results.items():
        print(f"{pattern_name:<15}", end="")
        
        for category, feature, _ in key_features:
            if category in analysis and feature in analysis[category]:
                value = analysis[category][feature]
                print(f"{value:<12.3f}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()
    
    print("-" * 80)
    
    # Analyze differences
    analyze_pattern_differences(results, key_features)

def analyze_pattern_differences(results, key_features):
    """Analyze the differences between patterns"""
    
    print("\nPATTERN ANALYSIS:")
    print("-" * 40)
    
    # Extract values for comparison
    pattern_names = list(results.keys())
    
    for category, feature, display_name in key_features:
        values = []
        labels = []
        
        for pattern_name in pattern_names:
            if (category in results[pattern_name] and 
                feature in results[pattern_name][category]):
                value = results[pattern_name][category][feature]
                values.append(value)
                labels.append(pattern_name)
        
        if len(values) > 1:
            max_val = max(values)
            min_val = min(values)
            max_pattern = labels[values.index(max_val)]
            min_pattern = labels[values.index(min_val)]
            
            if max_val != min_val:  # Only show if there's variation
                print(f"{display_name}:")
                print(f"  Highest: {max_pattern} ({max_val:.3f})")
                print(f"  Lowest:  {min_pattern} ({min_val:.3f})")
                print(f"  Range:   {max_val - min_val:.3f}")
                print()

def create_focused_plots(results):
    """Create focused visualizations of key features"""
    
    # Extract key pathological markers
    patterns = list(results.keys())
    
    # Key features to plot
    features_to_plot = [
        ('spectral_features', 'beta_ratio', 'Beta Power Ratio'),
        ('synchrony_features', 'population_sync_index', 'Population Synchrony'),
        ('burst_features', 'burst_index', 'Burst Index'),
        ('pathology_features', 'parkinsonian_index', 'Parkinsonian Index')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (category, feature, title) in enumerate(features_to_plot):
        ax = axes[i]
        
        values = []
        labels = []
        
        for pattern in patterns:
            if (category in results[pattern] and 
                feature in results[pattern][category]):
                value = results[pattern][category][feature]
                values.append(value)
                labels.append(pattern)
        
        if values:
            bars = ax.bar(labels, values)
            ax.set_title(title)
            ax.set_ylabel('Value')
            
            # Color code bars
            colors = ['green', 'blue', 'red']
            for bar, color in zip(bars, colors[:len(bars)]):
                bar.set_color(color)
            
            # Rotate labels if needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 30)
    
    for category, feature, title in features_to_plot:
        values = []
        for pattern in patterns:
            if (category in results[pattern] and 
                feature in results[pattern][category]):
                values.append(results[pattern][category][feature])
        
        if values:
            print(f"{title}:")
            print(f"  Mean: {np.mean(values):.3f}")
            print(f"  Std:  {np.std(values):.3f}")
            print(f"  Range: {np.min(values):.3f} - {np.max(values):.3f}")
            print()

if __name__ == "__main__":
    simple_feature_comparison()