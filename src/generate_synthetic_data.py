"""
Generate Realistic Synthetic Neural Data
========================================

This script creates realistic synthetic neural spike data files that mimic
real experimental recordings. The data includes multiple scenarios to test
different aspects of your analysis pipeline.

Run this script to generate data files in your data/ folder.
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path

def create_data_folder():
    """Create data folder if it doesn't exist."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def generate_realistic_spike_data(scenario="mixed_coding", n_trials=120, n_units=15, 
                                 trial_duration=4.0, save_format="npz"):
    """
    Generate realistic synthetic spike data with different coding scenarios.
    
    Parameters:
    -----------
    scenario : str
        Type of coding to simulate:
        - "rate_coding": Information mainly in firing rates
        - "temporal_coding": Information mainly in spike timing
        - "mixed_coding": Both rate and temporal information
        - "no_coding": No stimulus-related information (control)
    n_trials : int
        Number of trials to generate
    n_units : int
        Number of units to simulate
    trial_duration : float
        Duration of each trial in seconds
    save_format : str
        Format to save data ('npz', 'pkl', 'csv', 'mat')
    """
    
    print(f"🧠 Generating {scenario} synthetic data...")
    print(f"   📊 {n_trials} trials, {n_units} units, {trial_duration}s duration")
    
    np.random.seed(42)  # For reproducibility
    
    # Generate stimulus events
    inter_trial_interval = 6.0  # 6 seconds between trials
    event_times = np.arange(0, n_trials * inter_trial_interval, inter_trial_interval)
    
    # Mix of ON and OFF stimuli (60% ON, 40% OFF for slight imbalance - more realistic)
    event_labels = np.random.choice(['ON', 'OFF'], n_trials, p=[0.6, 0.4])
    
    # Initialize spike data
    all_spike_times = []
    all_unit_ids = []
    
    # Unit properties (make some units more responsive than others)
    unit_properties = {}
    for unit_id in range(1, n_units + 1):
        unit_properties[unit_id] = {
            'base_rate': np.random.uniform(0.5, 8.0),  # Baseline firing rate (Hz)
            'responsiveness': np.random.uniform(0.1, 1.0),  # How much the unit responds
            'preferred_stimulus': np.random.choice(['ON', 'OFF']),  # Which stimulus it prefers
            'temporal_precision': np.random.uniform(0.5, 2.0),  # Temporal precision factor
            'noise_level': np.random.uniform(0.1, 0.5)  # Background noise
        }
    
    print(f"   🎯 Unit responsiveness: {len([u for u in unit_properties.values() if u['responsiveness'] > 0.5])}/{n_units} highly responsive")
    
    # Generate spikes for each trial
    for trial_idx, (event_time, stimulus) in enumerate(zip(event_times, event_labels)):
        
        for unit_id in range(1, n_units + 1):
            props = unit_properties[unit_id]
            
            # Time windows
            pre_stim_start = event_time - 1.0
            stim_start = event_time
            stim_end = event_time + 2.0
            post_stim_end = event_time + trial_duration - 1.0
            
            # Generate spikes for different periods
            unit_spikes = []
            
            # 1. Pre-stimulus baseline
            baseline_rate = props['base_rate'] * (1 + props['noise_level'] * np.random.randn() * 0.1)
            baseline_spikes = generate_poisson_spikes(baseline_rate, 1.0)
            baseline_spikes = baseline_spikes + pre_stim_start
            unit_spikes.extend(baseline_spikes)
            
            # 2. Stimulus period - this is where the coding happens!
            stim_duration = 2.0
            
            if scenario == "rate_coding":
                # Rate coding: Change firing rate but keep timing random
                if stimulus == props['preferred_stimulus']:
                    stim_rate = props['base_rate'] * (1 + 2.0 * props['responsiveness'])
                else:
                    stim_rate = props['base_rate'] * (1 - 0.5 * props['responsiveness'])
                
                stim_spikes = generate_poisson_spikes(stim_rate, stim_duration)
                stim_spikes = stim_spikes + stim_start
                
            elif scenario == "temporal_coding":
                # Temporal coding: Precise timing patterns
                base_rate = props['base_rate']
                
                if stimulus == props['preferred_stimulus']:
                    # Create stereotyped temporal pattern
                    if props['responsiveness'] > 0.7:  # High responders get precise patterns
                        # Early precise burst
                        early_burst = np.random.normal(0.05, 0.01, size=int(2 + 3*props['responsiveness']))
                        early_burst = early_burst[early_burst > 0]
                        
                        # Late response  
                        late_response = np.random.normal(1.2, 0.05, size=int(1 + 2*props['responsiveness']))
                        late_response = late_response[(late_response > 0.8) & (late_response < 1.8)]
                        
                        precise_spikes = np.concatenate([early_burst, late_response])
                        precise_spikes = precise_spikes + stim_start
                        
                        # Add some background spikes
                        bg_spikes = generate_poisson_spikes(base_rate * 0.5, stim_duration)
                        bg_spikes = bg_spikes + stim_start
                        
                        stim_spikes = np.concatenate([precise_spikes, bg_spikes])
                    else:
                        # Less responsive units have less precise timing
                        stim_spikes = generate_poisson_spikes(base_rate * 1.2, stim_duration)
                        stim_spikes = stim_spikes + stim_start
                else:
                    # Non-preferred stimulus
                    stim_spikes = generate_poisson_spikes(base_rate * 0.8, stim_duration)
                    stim_spikes = stim_spikes + stim_start
                    
            elif scenario == "mixed_coding":
                # Mixed coding: Both rate and timing matter
                if stimulus == props['preferred_stimulus']:
                    # Increase rate
                    stim_rate = props['base_rate'] * (1 + 1.5 * props['responsiveness'])
                    
                    # Add temporal structure for highly responsive units
                    if props['responsiveness'] > 0.6:
                        # Early response with some jitter
                        early_spikes = np.random.normal(0.1, 0.03, size=int(1 + 2*props['responsiveness']))
                        early_spikes = early_spikes[(early_spikes > 0) & (early_spikes < 0.5)]
                        early_spikes = early_spikes + stim_start
                        
                        # Background poisson
                        bg_spikes = generate_poisson_spikes(stim_rate * 0.7, stim_duration)
                        bg_spikes = bg_spikes + stim_start
                        
                        stim_spikes = np.concatenate([early_spikes, bg_spikes])
                    else:
                        stim_spikes = generate_poisson_spikes(stim_rate, stim_duration)
                        stim_spikes = stim_spikes + stim_start
                else:
                    # Non-preferred: decrease rate, no temporal structure
                    stim_rate = props['base_rate'] * (1 - 0.3 * props['responsiveness'])
                    stim_spikes = generate_poisson_spikes(stim_rate, stim_duration)
                    stim_spikes = stim_spikes + stim_start
                    
            else:  # no_coding
                # No stimulus effect - just baseline firing
                stim_rate = props['base_rate'] * (1 + props['noise_level'] * np.random.randn() * 0.1)
                stim_spikes = generate_poisson_spikes(stim_rate, stim_duration)
                stim_spikes = stim_spikes + stim_start
            
            unit_spikes.extend(stim_spikes)
            
            # 3. Post-stimulus period
            post_rate = props['base_rate'] * (1 + props['noise_level'] * np.random.randn() * 0.1)
            post_spikes = generate_poisson_spikes(post_rate, 1.0)
            post_spikes = post_spikes + stim_end
            unit_spikes.extend(post_spikes)
            
            # Add all spikes for this unit
            unit_spikes = np.array(unit_spikes)
            unit_spikes = unit_spikes[unit_spikes >= pre_stim_start]  # Remove any negative spikes
            unit_spikes = unit_spikes[unit_spikes <= post_stim_end]   # Remove spikes beyond trial
            
            # Add refractory period violations (small percentage)
            unit_spikes = add_refractory_violations(unit_spikes, violation_rate=0.01)
            
            all_spike_times.extend(unit_spikes)
            all_unit_ids.extend([unit_id] * len(unit_spikes))
    
    # Convert to arrays
    all_spike_times = np.array(all_spike_times)
    all_unit_ids = np.array(all_unit_ids)
    
    # Sort by spike time
    sort_idx = np.argsort(all_spike_times)
    all_spike_times = all_spike_times[sort_idx]
    all_unit_ids = all_unit_ids[sort_idx]
    
    # Create data dictionary
    data = {
        'spike_times': all_spike_times,
        'unit_ids': all_unit_ids,
        'event_times': event_times,
        'event_labels': event_labels,
        'metadata': {
            'scenario': scenario,
            'n_trials': n_trials,
            'n_units': n_units,
            'trial_duration': trial_duration,
            'sampling_rate': 30000,  # Typical sampling rate
            'unit_properties': unit_properties
        }
    }
    
    print(f"   ✅ Generated {len(all_spike_times)} spikes across {len(np.unique(all_unit_ids))} units")
    
    return data

def generate_poisson_spikes(rate, duration):
    """Generate spikes from Poisson process."""
    if rate <= 0:
        return np.array([])
    
    n_spikes = np.random.poisson(rate * duration)
    if n_spikes == 0:
        return np.array([])
    
    spikes = np.sort(np.random.uniform(0, duration, n_spikes))
    return spikes

def add_refractory_violations(spike_times, violation_rate=0.01, refractory_period=0.001):
    """Add a small percentage of refractory period violations to make data more realistic."""
    if len(spike_times) < 2:
        return spike_times
    
    spike_times = np.sort(spike_times)
    n_violations = int(len(spike_times) * violation_rate)
    
    for _ in range(n_violations):
        # Pick a random spike and add a violation nearby
        idx = np.random.randint(0, len(spike_times)-1)
        violation_time = spike_times[idx] + np.random.uniform(0.0001, refractory_period*0.8)
        spike_times = np.append(spike_times, violation_time)
    
    return np.sort(spike_times)

def save_data(data, filename, format_type):
    """Save data in specified format."""
    data_dir = create_data_folder()
    filepath = data_dir / filename
    
    if format_type == 'npz':
        np.savez(filepath, **data)
    elif format_type == 'pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format_type == 'csv':
        # Create CSV with spike data
        df = pd.DataFrame({
            'spike_times': data['spike_times'],
            'unit_ids': data['unit_ids']
        })
        df.to_csv(filepath, index=False)
        
        # Save events separately
        events_df = pd.DataFrame({
            'event_times': data['event_times'],
            'event_labels': data['event_labels']
        })
        events_path = str(filepath).replace('.csv', '_events.csv')
        events_df.to_csv(events_path, index=False)
    elif format_type == 'mat':
        try:
            from scipy.io import savemat
            # Convert to MATLAB-compatible format
            mat_data = {k: v for k, v in data.items() if k != 'metadata'}
            savemat(filepath, mat_data)
        except ImportError:
            print("⚠️ scipy not available, skipping .mat file")
            return None
    
    print(f"   💾 Saved: {filepath}")
    return filepath

def generate_all_scenarios():
    """Generate data for all coding scenarios."""
    print("🚀 Generating comprehensive synthetic dataset...")
    print("=" * 60)
    
    scenarios = {
        'rate_coding': "Information encoded in firing rates",
        'temporal_coding': "Information encoded in spike timing",
        'mixed_coding': "Information in both rate and timing",
        'no_coding': "No stimulus information (control)"
    }
    
    generated_files = []
    
    for scenario, description in scenarios.items():
        print(f"\n📊 {scenario.upper().replace('_', ' ')}")
        print(f"   {description}")
        
        # Generate data
        data = generate_realistic_spike_data(
            scenario=scenario,
            n_trials=100,  # Good number for analysis
            n_units=12,    # Realistic for many experiments
            trial_duration=4.0
        )
        
        # Save in multiple formats
        filename_base = f"neural_data_{scenario}"
        
        # NPZ format (recommended)
        npz_file = save_data(data, f"{filename_base}.npz", 'npz')
        generated_files.append(npz_file)
        
        # Pickle format
        pkl_file = save_data(data, f"{filename_base}.pkl", 'pkl')
        generated_files.append(pkl_file)
        
        # CSV format  
        csv_file = save_data(data, f"{filename_base}.csv", 'csv')
        generated_files.append(csv_file)
    
    print(f"\n🎉 DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"📁 Generated {len(generated_files)} files in the 'data/' folder")
    print(f"\n📋 RECOMMENDED FILES TO USE:")
    print(f"   🎯 Start with: data/neural_data_mixed_coding.npz")
    print(f"   🔬 Compare with: data/neural_data_rate_coding.npz")
    print(f"   ⏱️  Temporal test: data/neural_data_temporal_coding.npz")
    print(f"   🎲 Control: data/neural_data_no_coding.npz")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Open: notebooks/main_analysis.ipynb")
    print(f"   2. In the data loading section, use:")
    print(f"      spike_data = loader.load_spike_data('data/neural_data_mixed_coding.npz')")
    print(f"   3. Run the analysis!")
    
    return generated_files

def generate_quick_test_data():
    """Generate a smaller dataset for quick testing."""
    print("⚡ Generating quick test dataset...")
    
    data = generate_realistic_spike_data(
        scenario="mixed_coding",
        n_trials=30,
        n_units=6,
        trial_duration=4.0
    )
    
    filepath = save_data(data, "test_data.npz", 'npz')
    
    print(f"✅ Quick test data saved: {filepath}")
    print(f"💡 Use this for fast testing: loader.load_spike_data('data/test_data.npz')")
    
    return filepath

if __name__ == "__main__":
    print("🧠 NEURAL SPIKE DATA GENERATOR")
    print("=" * 50)
    print("This script generates realistic synthetic neural data for testing your analysis pipeline.")
    print()
    
    # Ask user what they want to generate
    print("Choose an option:")
    print("1. Generate all scenarios (rate, temporal, mixed, control)")
    print("2. Generate quick test data (smaller, faster)")
    print("3. Generate specific scenario")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            generate_all_scenarios()
        elif choice == "2":
            generate_quick_test_data()
        elif choice == "3":
            print("\nScenarios:")
            print("- rate_coding: Information in firing rates")
            print("- temporal_coding: Information in spike timing")  
            print("- mixed_coding: Both rate and timing")
            print("- no_coding: No stimulus information")
            
            scenario = input("Enter scenario: ").strip()
            if scenario in ['rate_coding', 'temporal_coding', 'mixed_coding', 'no_coding']:
                data = generate_realistic_spike_data(scenario=scenario)
                save_data(data, f"neural_data_{scenario}.npz", 'npz')
                print(f"✅ Generated {scenario} data!")
            else:
                print("❌ Invalid scenario")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Generation cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")
        # Generate default data as fallback
        print("🔄 Generating default test data as fallback...")
        generate_quick_test_data()