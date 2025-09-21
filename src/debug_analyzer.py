import numpy as np
import matplotlib.pyplot as plt
from spike_data_loader import NeuralPatternGenerator
from spike_analyzer import EnhancedSpikeAnalyzer

def debug_feature_extraction():
    """Debug the feature extraction step by step"""
    
    print("Debugging Feature Extraction...")
    print("=" * 50)
    
    # Generate simple test data
    generator = NeuralPatternGenerator(n_neurons=10, trial_duration=2.0)
    analyzer = EnhancedSpikeAnalyzer(trial_duration=2.0)
    
    # Generate Parkinsonian pattern (should have clear features)
    data = generator.generate_synthetic_spikes(
        coding_type='rate',
        n_stimuli=3,
        n_trials_per_stimulus=2,
        oscillatory_power=0.7,      # Strong beta
        population_synchrony=0.8,   # High sync
        spike_regularity=0.3,       # Irregular
        pathological_bursting=0.2
    )
    
    print("Test Data Generated:")
    print(f"  Trials: {len(data['spike_trains'])}")
    print(f"  Neurons: {len(data['spike_trains'][0])}")
    print(f"  Sample spike counts: {[len(spikes) for spikes in data['spike_trains'][0][:3]]}")
    
    # Test each feature category individually
    test_rate_features(analyzer, data)
    test_temporal_features(analyzer, data)
    test_spectral_features(analyzer, data)
    test_synchrony_features(analyzer, data)
    test_burst_features(analyzer, data)
    test_regularity_features(analyzer, data)

def test_rate_features(analyzer, data):
    """Test rate feature extraction"""
    print("\n1. Testing Rate Features:")
    print("-" * 30)
    
    try:
        rate_features = analyzer.extract_rate_features(data)
        print("  Rate features extracted successfully!")
        for key, value in rate_features.items():
            print(f"    {key}: {value:.4f}")
    except Exception as e:
        print(f"  ERROR in rate features: {e}")
        import traceback
        traceback.print_exc()

def test_temporal_features(analyzer, data):
    """Test temporal feature extraction"""
    print("\n2. Testing Temporal Features:")
    print("-" * 30)
    
    try:
        temporal_features = analyzer.extract_temporal_features(data)
        print("  Temporal features extracted successfully!")
        for key, value in temporal_features.items():
            print(f"    {key}: {value:.4f}")
    except Exception as e:
        print(f"  ERROR in temporal features: {e}")
        import traceback
        traceback.print_exc()

def test_spectral_features(analyzer, data):
    """Test spectral feature extraction"""
    print("\n3. Testing Spectral Features:")
    print("-" * 30)
    
    try:
        # First test population activity calculation
        pop_activity = analyzer._calculate_population_activity(data['spike_trains'][0])
        print(f"  Population activity calculated: length={len(pop_activity)}, max={np.max(pop_activity):.2f}")
        
        # Plot population activity for visual inspection
        plt.figure(figsize=(10, 4))
        time_axis = np.linspace(0, 2.0, len(pop_activity))
        plt.plot(time_axis, pop_activity)
        plt.title('Population Activity (should show oscillations if pathological)')
        plt.xlabel('Time (s)')
        plt.ylabel('Spike count')
        plt.show()
        
        # Test spectral analysis
        spectral_features = analyzer.extract_spectral_features(data)
        print("  Spectral features extracted successfully!")
        for key, value in spectral_features.items():
            print(f"    {key}: {value:.4f}")
            
    except Exception as e:
        print(f"  ERROR in spectral features: {e}")
        import traceback
        traceback.print_exc()

def test_synchrony_features(analyzer, data):
    """Test synchrony feature extraction"""
    print("\n4. Testing Synchrony Features:")
    print("-" * 30)
    
    try:
        synchrony_features = analyzer.extract_synchrony_features(data)
        print("  Synchrony features extracted successfully!")
        for key, value in synchrony_features.items():
            print(f"    {key}: {value:.4f}")
    except Exception as e:
        print(f"  ERROR in synchrony features: {e}")
        import traceback
        traceback.print_exc()

def test_burst_features(analyzer, data):
    """Test burst feature extraction"""
    print("\n5. Testing Burst Features:")
    print("-" * 30)
    
    try:
        burst_features = analyzer.extract_burst_features(data)
        print("  Burst features extracted successfully!")
        for key, value in burst_features.items():
            print(f"    {key}: {value:.4f}")
    except Exception as e:
        print(f"  ERROR in burst features: {e}")
        import traceback
        traceback.print_exc()

def test_regularity_features(analyzer, data):
    """Test regularity feature extraction"""
    print("\n6. Testing Regularity Features:")
    print("-" * 30)
    
    try:
        regularity_features = analyzer.extract_regularity_features(data)
        print("  Regularity features extracted successfully!")
        for key, value in regularity_features.items():
            print(f"    {key}: {value:.4f}")
    except Exception as e:
        print(f"  ERROR in regularity features: {e}")
        import traceback
        traceback.print_exc()

def test_simple_spectral_analysis():
    """Test basic spectral analysis with known signal"""
    print("\n" + "=" * 50)
    print("Testing Basic Spectral Analysis with Known Signal")
    print("=" * 50)
    
    from scipy import signal
    
    # Create test signal with known frequency content
    fs = 1000  # 1000 Hz sampling rate
    t = np.linspace(0, 2, 2000)  # 2 seconds
    
    # Signal with 20 Hz component (beta band)
    test_signal = np.sin(2 * np.pi * 20 * t) + 0.5 * np.random.randn(len(t))
    
    # Test spectral analysis
    try:
        freqs, psd = signal.welch(test_signal, fs=fs, nperseg=256)
        
        # Find peak frequency
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]
        
        print(f"Test signal peak frequency: {peak_freq:.1f} Hz (should be ~20 Hz)")
        
        # Calculate beta power
        beta_mask = (freqs >= 13) & (freqs <= 30)
        beta_power = np.trapz(psd[beta_mask], freqs[beta_mask])
        total_power = np.trapz(psd, freqs)
        beta_ratio = beta_power / total_power
        
        print(f"Beta power ratio: {beta_ratio:.3f} (should be > 0.1)")
        
        # Plot spectrum
        plt.figure(figsize=(10, 4))
        plt.semilogy(freqs, psd)
        plt.axvspan(13, 30, alpha=0.3, color='red', label='Beta band')
        plt.xlim(0, 50)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title('Test Signal Spectrum (20 Hz peak)')
        plt.legend()
        plt.show()
        
        if peak_freq > 18 and peak_freq < 22:
            print("✓ Basic spectral analysis working correctly!")
        else:
            print("✗ Spectral analysis may have issues")
            
    except Exception as e:
        print(f"ERROR in basic spectral analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_feature_extraction()
    test_simple_spectral_analysis()