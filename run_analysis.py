"""
Fixed Neural Analysis Script
============================

Run this script to perform the complete neural analysis without Jupyter.
Just run: python run_analysis_fixed.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('src')

from spike_data_loader import SpikeDataLoader
from spike_analyzer import SpikeAnalyzer
from spike_visualizer import SpikeVisualizer
from decoding_analysis import DecodingAnalysis

def main():
    print("NEURAL SPIKE ANALYSIS")
    print("=" * 50)
    
    # Initialize modules
    loader = SpikeDataLoader()
    analyzer = SpikeAnalyzer()
    visualizer = SpikeVisualizer(style='dark')
    decoder = DecodingAnalysis()
    
    # Generate synthetic data with better parameters
    print("Generating synthetic data with good quality units...")
    # In your existing run_analysis_fixed.py, change this line:
    synthetic_data = loader.create_synthetic_data(
    n_trials=80,
    n_units=12,
    duration=4.0,
    stimulus_effect=1.5  # Reduce from 3.0 to 1.5
)
    
    spike_times = synthetic_data['spike_times']
    unit_ids = synthetic_data['unit_ids']
    event_times = synthetic_data['event_times']
    event_labels = synthetic_data['event_labels']
    
    print(f"Generated {len(np.unique(unit_ids))} units with {len(spike_times)} spikes")
    
    # Create trials
    trials_data = loader.create_trials(spike_times, unit_ids, event_times, event_labels)
    print(f"Created {len(trials_data)} trials")
    
    # Quality control with more lenient thresholds
    unit_stats = analyzer.calculate_unit_stats(trials_data)
    good_units = analyzer.filter_good_units(
        unit_stats,
        min_firing_rate=0.05,  # More lenient
        max_refractory_violations=0.05,  # More lenient
        min_total_spikes=30  # More lenient
    )
    
    print(f"{len(good_units)}/{len(unit_stats)} units passed QC")
    print(f"Good units: {good_units}")
    
    # If still no good units, use all units
    if not good_units:
        print("Warning: No units passed QC, using all units for demo")
        good_units = list(unit_stats.keys())
    
    # Find responsive units
    responsive_units = []
    response_stats = {}
    
    for unit_id in good_units:
        baseline_rates, stim_rates = analyzer.get_baseline_vs_stimulus_rates(trials_data, unit_id)
        try:
            from scipy import stats
            _, p_value = stats.wilcoxon(baseline_rates, stim_rates)
            
            response_stats[unit_id] = {
                'baseline_rate': np.mean(baseline_rates),
                'stimulus_rate': np.mean(stim_rates),
                'p_value': p_value,
                'responsive': p_value < 0.05
            }
            
            if p_value < 0.05:
                responsive_units.append(unit_id)
                print(f"   Unit {unit_id}: responsive (p = {p_value:.4f})")
            else:
                print(f"   Unit {unit_id}: not responsive (p = {p_value:.4f})")
        except Exception as e:
            print(f"   Unit {unit_id}: test failed ({e})")
            # Add to response stats anyway
            response_stats[unit_id] = {
                'baseline_rate': np.mean(baseline_rates),
                'stimulus_rate': np.mean(stim_rates),
                'p_value': 1.0,
                'responsive': False
            }
    
    print(f"{len(responsive_units)} responsive units found")
    
    # Select units for analysis
    if responsive_units:
        analysis_units = responsive_units
    elif good_units:
        analysis_units = good_units[:6]  # Use up to 6 units
    else:
        analysis_units = list(unit_stats.keys())[:6]
    
    print(f"Using {len(analysis_units)} units for decoding: {analysis_units}")
    
    # Extract features
    print("Extracting features...")
    rate_features = decoder.extract_rate_features(trials_data, analysis_units)
    temporal_features = decoder.extract_temporal_features(trials_data, analysis_units)
    labels = decoder.get_trial_labels(trials_data)
    
    print(f"Rate features shape: {rate_features.shape}")
    print(f"Temporal features shape: {temporal_features.shape}")
    print(f"Labels: {np.unique(labels, return_counts=True)}")
    
    # Check if we have valid features
    if rate_features.shape[1] == 0:
        print("ERROR: No rate features extracted!")
        return
    
    if temporal_features.shape[1] == 0:
        print("ERROR: No temporal features extracted!")
        return
    
    # Decoding analysis
    print("Running decoding analysis...")
    results = decoder.compare_decoding_performance(rate_features, temporal_features, labels)
    
    # Results
    print(f"\nDECODING RESULTS:")
    print(f"   Rate accuracy:     {results['rate_accuracy']:.3f} ± {results['rate_std']:.3f}")
    print(f"   Temporal accuracy: {results['temporal_accuracy']:.3f} ± {results['temporal_std']:.3f}")
    print(f"   Shuffle control:   {results['shuffle_accuracy']:.3f} ± {results['shuffle_std']:.3f}")
    print(f"   Improvement:       {results['temporal_accuracy'] - results['rate_accuracy']:+.3f}")
    
    # Statistical significance
    p_value = results.get('p_value_rate_vs_temporal', 1.0)
    if p_value < 0.05:
        significance = "SIGNIFICANT"
    else:
        significance = "not significant"
    print(f"   Statistical test:  p = {p_value:.4f} ({significance})")
    
    # Conclusion
    improvement = results['temporal_accuracy'] - results['rate_accuracy']
    print(f"\nCONCLUSION:")
    if improvement > 0.05 and p_value < 0.05:
        print("   TEMPORAL CODING detected!")
        print("   Spike timing carries more information than rates")
    elif improvement < -0.05 and p_value < 0.05:
        print("   RATE CODING detected!")
        print("   Spike counts are sufficient for decoding")
    else:
        print("   MIXED/UNCLEAR CODING detected!")
        print("   Both rate and timing may contribute")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    
    try:
        # Unit quality plot
        fig = visualizer.plot_unit_quality(unit_stats, good_units)
        plt.savefig('results/unit_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: results/unit_quality.png")
        
        # Raster plot for first analysis unit
        if analysis_units:
            fig = visualizer.plot_raster_psth(trials_data, analysis_units[0])
            plt.savefig('results/raster_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   Saved: results/raster_plot.png")
        
        # Response statistics
        if response_stats:
            fig = visualizer.plot_response_statistics(response_stats)
            plt.savefig('results/response_stats.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   Saved: results/response_stats.png")
        
        # Decoding results
        fig = visualizer.plot_decoding_results(results)
        plt.savefig('results/decoding_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: results/decoding_results.png")
        
    except Exception as e:
        print(f"   Warning: Some plots failed to generate ({e})")
    
    print(f"\nAnalysis complete!")
    print(f"Check the results/ folder for plots and open them with any image viewer.")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"   Dataset: {len(trials_data)} trials, {len(analysis_units)} units")
    print(f"   Quality: {len(good_units)}/{len(unit_stats)} units passed QC")
    print(f"   Responsive: {len(responsive_units)} units")
    print(f"   Performance: Rate={results['rate_accuracy']:.1%}, Temporal={results['temporal_accuracy']:.1%}")

if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    
    try:
        main()
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        print("This might be due to missing dependencies or file issues.")
        print("Try running: pip install -r requirements.txt")