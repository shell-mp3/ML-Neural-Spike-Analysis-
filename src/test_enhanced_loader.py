import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy
import seaborn as sns

# Import the enhanced spike data loader
from spike_data_loader import NeuralPatternGenerator

def test_pathological_patterns():
    """Test and visualize pathological neural patterns"""
    
    generator = NeuralPatternGenerator(n_neurons=20, trial_duration=2.0)
    
    # Generate test patterns
    patterns = {}
    
    print("Testing pathological pattern generation...")
    
    # 1. Healthy Rate Coding (baseline)
    patterns['healthy_rate'] = generator.generate_synthetic_spikes(
        coding_type='rate',
        n_stimuli=5,
        n_trials_per_stimulus=3,
        oscillatory_power=0.0,
        population_synchrony=0.15,
        spike_regularity=0.8,
        pathological_bursting=0.0
    )
    
    # 2. Parkinsonian Pattern
    patterns['parkinsonian'] = generator.generate_synthetic_spikes(
        coding_type='rate',
        n_stimuli=5,
        n_trials_per_stimulus=3,
        oscillatory_power=0.7,      # Strong beta oscillations
        population_synchrony=0.8,   # High synchrony
        spike_regularity=0.3,       # Irregular firing
        pathological_bursting=0.2   # Some bursting
    )
    
    # 3. Epileptiform Pattern
    patterns['epileptiform'] = generator.generate_synthetic_spikes(
        coding_type='rate',
        n_stimuli=5,
        n_trials_per_stimulus=3,
        oscillatory_power=0.4,      # Mixed oscillations
        population_synchrony=0.9,   # Very high synchrony
        spike_regularity=0.5,       # Variable regularity
        pathological_bursting=0.7   # High bursting
    )
    
    # 4. Mixed Pathology
    patterns['mixed'] = generator.generate_synthetic_spikes(
        coding_type='mixed',
        n_stimuli=5,
        n_trials_per_stimulus=3,
        oscillatory_power=0.6,
        population_synchrony=0.7,
        spike_regularity=0.4,
        pathological_bursting=0.4
    )
    
    # Analyze and visualize patterns
    analyze_patterns(patterns)
    
    # Test multi-class dataset generation
    test_multiclass_generation(generator)

def analyze_patterns(patterns):
    """Analyze and visualize the generated patterns"""
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle('Pathological Neural Pattern Analysis', fontsize=16)
    
    pattern_names = list(patterns.keys())
    
    for i, (pattern_name, data) in enumerate(patterns.items()):
        
        # 1. Raster Plot (first trial)
        ax1 = axes[i, 0]
        plot_raster(data['spike_trains'][0], ax1)
        ax1.set_title(f'{pattern_name.title()}\nRaster Plot')
        ax1.set_ylabel('Neuron')
        if i == 3:
            ax1.set_xlabel('Time (s)')
        
        # 2. Population Activity
        ax2 = axes[i, 1]
        pop_activity = calculate_population_activity(data['spike_trains'][0])
        time_axis = np.linspace(0, 2.0, len(pop_activity))
        ax2.plot(time_axis, pop_activity)
        ax2.set_title('Population Activity')
        ax2.set_ylabel('Spikes/bin')
        if i == 3:
            ax2.set_xlabel('Time (s)')
        
        # 3. Power Spectrum
        ax3 = axes[i, 2]
        plot_power_spectrum(pop_activity, ax3)
        ax3.set_title('Power Spectrum')
        ax3.set_ylabel('Power')
        if i == 3:
            ax3.set_xlabel('Frequency (Hz)')
        
        # 4. ISI Distribution (first neuron)
        ax4 = axes[i, 3]
        plot_isi_distribution(data['spike_trains'][0][0], ax4)
        ax4.set_title('ISI Distribution')
        ax4.set_ylabel('Count')
        if i == 3:
            ax4.set_xlabel('ISI (s)')
    
    plt.tight_layout()
    plt.show()
    
    # Quantitative analysis
    print("\nQuantitative Pattern Analysis:")
    print("=" * 50)
    
    for pattern_name, data in patterns.items():
        metrics = calculate_pattern_metrics(data)
        print(f"\n{pattern_name.upper()}:")
        print(f"  Mean firing rate: {metrics['mean_firing_rate']:.2f} Hz")
        print(f"  Population synchrony: {metrics['population_sync']:.3f}")
        print(f"  Beta power (15-25 Hz): {metrics['beta_power']:.3f}")
        print(f"  Burst index: {metrics['burst_index']:.3f}")
        print(f"  Spike regularity: {metrics['spike_regularity']:.3f}")

def plot_raster(spike_trains, ax):
    """Plot raster plot of spike trains"""
    for neuron_idx, spikes in enumerate(spike_trains):
        if len(spikes) > 0:
            ax.plot(spikes, [neuron_idx] * len(spikes), '|', markersize=2, color='black')
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-1, len(spike_trains))

def calculate_population_activity(spike_trains, bin_size=0.01):
    """Calculate population activity over time"""
    time_bins = np.arange(0, 2.0 + bin_size, bin_size)  # Include endpoint
    pop_activity = np.zeros(len(time_bins) - 1)  # Histogram returns n_bins-1 counts
    
    for spikes in spike_trains:
        if len(spikes) > 0:
            spike_counts, _ = np.histogram(spikes, bins=time_bins)
            # Ensure shapes match
            if len(spike_counts) == len(pop_activity):
                pop_activity += spike_counts
            else:
                # Handle edge case where spike times are exactly at boundaries
                min_len = min(len(spike_counts), len(pop_activity))
                pop_activity[:min_len] += spike_counts[:min_len]
    
    return pop_activity

def plot_power_spectrum(signal_data, ax):
    """Plot power spectrum of population activity"""
    if len(signal_data) > 10:
        # Ensure we have enough data points for FFT
        nperseg = min(64, len(signal_data)//4)
        if nperseg < 4:
            nperseg = min(len(signal_data), 16)
            
        try:
            freqs, psd = signal.welch(signal_data, fs=100, nperseg=nperseg)
            ax.semilogy(freqs, psd)
            ax.set_xlim(0, min(50, freqs[-1]))
            
            # Highlight beta band
            beta_range = (freqs >= 15) & (freqs <= 25)
            if np.any(beta_range):
                ax.axvspan(15, 25, alpha=0.3, color='red', label='Beta')
        except ValueError as e:
            # Fallback for very short signals
            ax.text(0.5, 0.5, 'Signal too short\nfor FFT', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_xlim(0, 50)

def plot_isi_distribution(spike_times, ax):
    """Plot inter-spike interval distribution"""
    if len(spike_times) > 1:
        isis = np.diff(spike_times)
        if len(isis) > 0:
            ax.hist(isis, bins=20, alpha=0.7, density=True)
            ax.set_xlim(0, min(0.5, np.max(isis)))

def calculate_pattern_metrics(data):
    """Calculate quantitative metrics for pattern characterization"""
    
    metrics = {}
    
    # Mean firing rate across all neurons and trials
    all_rates = []
    for trial_spikes in data['spike_trains']:
        for spikes in trial_spikes:
            rate = len(spikes) / 2.0  # spikes per second
            all_rates.append(rate)
    metrics['mean_firing_rate'] = np.mean(all_rates)
    
    # Population synchrony (average across trials)
    sync_values = []
    for trial_spikes in data['spike_trains'][:5]:  # First 5 trials
        pop_activity = calculate_population_activity(trial_spikes)
        if len(pop_activity) > 10:
            # Synchrony as coefficient of variation of population activity
            sync_values.append(np.std(pop_activity) / (np.mean(pop_activity) + 1e-6))
    metrics['population_sync'] = np.mean(sync_values) if sync_values else 0
    
    # Beta power
    beta_powers = []
    for trial_spikes in data['spike_trains'][:5]:
        pop_activity = calculate_population_activity(trial_spikes)
        if len(pop_activity) > 10:
            freqs, psd = signal.welch(pop_activity, fs=100, nperseg=min(64, len(pop_activity)//4))
            beta_range = (freqs >= 15) & (freqs <= 25)
            if np.any(beta_range):
                beta_power = np.mean(psd[beta_range])
                total_power = np.mean(psd[(freqs >= 1) & (freqs <= 50)])
                beta_powers.append(beta_power / (total_power + 1e-6))
    metrics['beta_power'] = np.mean(beta_powers) if beta_powers else 0
    
    # Burst index (measure of burstiness)
    burst_indices = []
    for trial_spikes in data['spike_trains'][:5]:
        for spikes in trial_spikes[:5]:  # First 5 neurons
            if len(spikes) > 3:
                isis = np.diff(spikes)
                if len(isis) > 1:
                    # Burst index as ratio of short to long intervals
                    short_isis = np.sum(isis < 0.01)  # < 10ms
                    long_isis = np.sum(isis > 0.1)   # > 100ms
                    if short_isis + long_isis > 0:
                        burst_indices.append(short_isis / (short_isis + long_isis))
    metrics['burst_index'] = np.mean(burst_indices) if burst_indices else 0
    
    # Spike regularity (inverse of CV of ISIs)
    regularities = []
    for trial_spikes in data['spike_trains'][:5]:
        for spikes in trial_spikes[:5]:
            if len(spikes) > 3:
                isis = np.diff(spikes)
                if len(isis) > 1 and np.mean(isis) > 0:
                    cv = np.std(isis) / np.mean(isis)
                    regularities.append(1 / (1 + cv))  # High regularity = low CV
    metrics['spike_regularity'] = np.mean(regularities) if regularities else 0
    
    return metrics

def test_multiclass_generation(generator):
    """Test multi-class dataset generation"""
    
    print("\n" + "="*50)
    print("Testing Multi-Class Dataset Generation")
    print("="*50)
    
    # Generate small test dataset
    dataset, labels = generator.generate_multiclass_dataset(n_trials_per_class=10)
    
    # Verify class balance
    from collections import Counter
    class_counts = Counter(labels)
    print(f"\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} trials")
    
    # Quick metrics comparison
    print(f"\nQuick pattern verification:")
    
    # Group by class
    class_data = {}
    for i, label in enumerate(labels):
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(dataset[i])
    
    # Calculate average metrics per class
    for class_name, class_trials in class_data.items():
        metrics = []
        for trial_data in class_trials[:3]:  # Sample 3 trials
            trial_metrics = calculate_pattern_metrics(trial_data)
            metrics.append(trial_metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics])
        
        print(f"\n{class_name}:")
        print(f"  Avg firing rate: {avg_metrics['mean_firing_rate']:.1f} Hz")
        print(f"  Avg synchrony: {avg_metrics['population_sync']:.3f}")
        print(f"  Avg beta power: {avg_metrics['beta_power']:.3f}")

if __name__ == "__main__":
    test_pathological_patterns()