import numpy as np
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class NeuralPatternGenerator:
    """Enhanced neural spike data generator with pathological pattern simulation"""
    
    def __init__(self, n_neurons=50, trial_duration=2.0, dt=0.001):
        self.n_neurons = n_neurons
        self.trial_duration = trial_duration
        self.dt = dt
        self.time_bins = np.arange(0, trial_duration, dt)
        
    def generate_synthetic_spikes(self, 
                                coding_type='rate',
                                n_stimuli=20,
                                n_trials_per_stimulus=10,
                                base_firing_rate=10.0,
                                noise_level=0.1,
                                refractory_period=0.002,
                                # Pathological parameters
                                oscillatory_power=0.0,
                                population_synchrony=0.2,
                                spike_regularity=0.8,
                                pathological_bursting=0.0,
                                # Disease-specific parameters
                                beta_frequency=20.0,
                                burst_duration=0.05,
                                interburst_interval=0.2):
        """
        Generate synthetic neural spike data with pathological pattern options
        
        Parameters:
        -----------
        coding_type : str
            'rate', 'temporal', or 'mixed'
        oscillatory_power : float (0-1)
            Strength of pathological beta oscillations (Parkinson's-like)
        population_synchrony : float (0-1) 
            Level of population synchronization (epilepsy-like)
        spike_regularity : float (0-1)
            Regularity of spike timing (1=regular, 0=irregular)
        pathological_bursting : float (0-1)
            Amount of burst firing activity
        """
        
        # Generate stimulus parameters
        if coding_type == 'rate':
            stimulus_intensities = np.linspace(0.5, 2.0, n_stimuli)
            stimulus_phases = np.zeros(n_stimuli)  # No phase coding
        elif coding_type == 'temporal':
            stimulus_intensities = np.ones(n_stimuli) * 1.2  # Constant intensity
            stimulus_phases = np.linspace(0, 2*np.pi, n_stimuli)  # Phase coding
        else:  # mixed
            stimulus_intensities = np.linspace(0.8, 1.5, n_stimuli)
            stimulus_phases = np.linspace(0, np.pi, n_stimuli)
        
        all_spike_trains = []
        all_stimulus_labels = []
        all_stimulus_times = []
        
        for stim_idx in range(n_stimuli):
            intensity = stimulus_intensities[stim_idx]
            phase = stimulus_phases[stim_idx]
            
            for trial in range(n_trials_per_stimulus):
                # Generate spike trains for this trial
                spike_trains = self._generate_trial_spikes(
                    intensity, phase, base_firing_rate, noise_level,
                    refractory_period, oscillatory_power, population_synchrony,
                    spike_regularity, pathological_bursting, beta_frequency,
                    burst_duration, interburst_interval
                )
                
                # Generate stimulus onset time (randomized)
                stimulus_time = np.random.uniform(0.2, 0.4)
                
                all_spike_trains.append(spike_trains)
                all_stimulus_labels.append(stim_idx)
                all_stimulus_times.append(stimulus_time)
        
        return {
            'spike_trains': all_spike_trains,
            'stimulus_labels': all_stimulus_labels,
            'stimulus_times': all_stimulus_times,
            'stimulus_intensities': stimulus_intensities,
            'stimulus_phases': stimulus_phases,
            'parameters': {
                'coding_type': coding_type,
                'n_neurons': self.n_neurons,
                'trial_duration': self.trial_duration,
                'base_firing_rate': base_firing_rate,
                'oscillatory_power': oscillatory_power,
                'population_synchrony': population_synchrony,
                'spike_regularity': spike_regularity,
                'pathological_bursting': pathological_bursting
            }
        }
    
    def _generate_trial_spikes(self, intensity, phase, base_firing_rate, noise_level,
                              refractory_period, oscillatory_power, population_synchrony,
                              spike_regularity, pathological_bursting, beta_frequency,
                              burst_duration, interburst_interval):
        """Generate spikes for a single trial with pathological patterns"""
        
        spike_trains = []
        
        # Generate shared oscillatory signal for pathological synchrony
        if oscillatory_power > 0:
            oscillation = self._generate_beta_oscillation(beta_frequency, oscillatory_power)
        else:
            oscillation = np.zeros(len(self.time_bins))
        
        # Generate population synchrony signal
        if population_synchrony > 0:
            sync_signal = self._generate_synchrony_signal(population_synchrony)
        else:
            sync_signal = np.zeros(len(self.time_bins))
            
        for neuron_idx in range(self.n_neurons):
            # Base firing rate with stimulus modulation
            if np.random.rand() < 0.7:  # 70% of neurons are stimulus-responsive
                modulated_rate = base_firing_rate * intensity
                
                # Add temporal coding (phase-locked responses)
                phase_modulation = 1 + 0.5 * np.sin(2 * np.pi * 8 * self.time_bins + phase)
                firing_rate = modulated_rate * phase_modulation
            else:
                firing_rate = np.full(len(self.time_bins), base_firing_rate)
            
            # Add pathological oscillations
            if oscillatory_power > 0:
                firing_rate *= (1 + oscillatory_power * oscillation)
            
            # Add population synchrony
            if population_synchrony > 0:
                firing_rate *= (1 + population_synchrony * sync_signal)
            
            # Generate spike times
            spike_times = self._generate_neuron_spikes(
                firing_rate, spike_regularity, pathological_bursting,
                burst_duration, interburst_interval, refractory_period
            )
            
            # Add noise
            if noise_level > 0:
                n_noise_spikes = np.random.poisson(noise_level * self.trial_duration)
                noise_spikes = np.random.uniform(0, self.trial_duration, n_noise_spikes)
                spike_times = np.concatenate([spike_times, noise_spikes])
            
            spike_times = np.sort(spike_times)
            spike_trains.append(spike_times)
        
        return spike_trains
    
    def _generate_beta_oscillation(self, frequency, power):
        """Generate beta oscillation for Parkinsonian patterns"""
        t = self.time_bins
        # Mix of beta frequencies (characteristic of Parkinson's)
        beta_signal = (np.sin(2 * np.pi * frequency * t) + 
                      0.3 * np.sin(2 * np.pi * (frequency * 1.2) * t) +
                      0.2 * np.sin(2 * np.pi * (frequency * 0.8) * t))
        
        # Add some phase noise for realism
        phase_noise = 0.1 * np.random.randn(len(t))
        beta_signal += phase_noise
        
        # Normalize and apply power scaling
        beta_signal = beta_signal / np.max(np.abs(beta_signal)) * power
        return beta_signal
    
    def _generate_synchrony_signal(self, synchrony_level):
        """Generate population synchrony signal for epileptic patterns"""
        t = self.time_bins
        
        # Multiple frequency components for complex synchrony
        sync_signal = 0
        frequencies = [4, 8, 15, 30, 60]  # Multiple bands
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        for freq, weight in zip(frequencies, weights):
            sync_signal += weight * np.sin(2 * np.pi * freq * t + 
                                         np.random.uniform(0, 2*np.pi))
        
        # Add sharp transients (interictal-like spikes)
        if synchrony_level > 0.6:
            n_spikes = np.random.poisson(3 * synchrony_level)
            spike_times = np.random.uniform(0.5, self.trial_duration - 0.5, n_spikes)
            for spike_time in spike_times:
                spike_idx = int(spike_time / self.dt)
                if spike_idx < len(t) - 50:
                    # Sharp spike with decay
                    sync_signal[spike_idx:spike_idx+50] += (
                        synchrony_level * np.exp(-np.arange(50) * 0.1)
                    )
        
        return sync_signal * synchrony_level
    
    def _generate_neuron_spikes(self, firing_rate, regularity, bursting,
                               burst_duration, interburst_interval, refractory_period):
        """Generate spikes for individual neuron with pathological patterns"""
        
        spike_times = []
        t = 0
        
        while t < self.trial_duration:
            # Get current firing rate
            time_idx = min(int(t / self.dt), len(firing_rate) - 1)
            current_rate = max(0, firing_rate[time_idx])
            
            # Determine if we're in a burst
            in_burst = np.random.rand() < bursting
            
            if in_burst:
                # Generate burst of spikes
                burst_end = min(t + burst_duration, self.trial_duration)
                burst_rate = current_rate * (2 + 3 * bursting)  # Higher rate in burst
                
                while t < burst_end:
                    # More regular timing in bursts
                    if regularity > 0.5:
                        isi = 1.0 / burst_rate + np.random.normal(0, (1-regularity) * 0.01)
                    else:
                        isi = np.random.exponential(1.0 / burst_rate)
                    
                    isi = max(isi, refractory_period)
                    t += isi
                    
                    if t < burst_end:
                        spike_times.append(t)
                
                # Inter-burst interval
                t += interburst_interval * (0.5 + np.random.rand())
                
            else:
                # Normal spike generation
                if current_rate > 0:
                    if regularity > 0.5:
                        # More regular firing
                        mean_isi = 1.0 / current_rate
                        isi = mean_isi + np.random.normal(0, (1-regularity) * mean_isi * 0.5)
                    else:
                        # Poisson-like irregular firing  
                        isi = np.random.exponential(1.0 / current_rate)
                    
                    isi = max(isi, refractory_period)
                    t += isi
                    
                    if t < self.trial_duration:
                        spike_times.append(t)
                else:
                    t += self.dt
        
        return np.array(spike_times)
    
    def generate_multiclass_dataset(self, n_trials_per_class=50):
        """Generate balanced dataset for multi-class classification"""
        
        dataset = []
        labels = []
        
        print("Generating multi-class neural pattern dataset...")
        
        # Class 0: Healthy Rate Coding
        print("  Generating Healthy Rate patterns...")
        for _ in range(n_trials_per_class):
            data = self.generate_synthetic_spikes(
                coding_type='rate',
                oscillatory_power=0.0,
                population_synchrony=0.15,  # Minimal natural synchrony
                spike_regularity=0.8,
                pathological_bursting=0.0
            )
            dataset.append(data)
            labels.append('Healthy_Rate')
        
        # Class 1: Healthy Temporal Coding
        print("  Generating Healthy Temporal patterns...")
        for _ in range(n_trials_per_class):
            data = self.generate_synthetic_spikes(
                coding_type='temporal',
                oscillatory_power=0.0,
                population_synchrony=0.15,
                spike_regularity=0.8,
                pathological_bursting=0.0
            )
            dataset.append(data)
            labels.append('Healthy_Temporal')
        
        # Class 2: Parkinsonian Patterns
        print("  Generating Parkinsonian patterns...")
        for _ in range(n_trials_per_class):
            data = self.generate_synthetic_spikes(
                coding_type=np.random.choice(['rate', 'temporal']),
                oscillatory_power=np.random.uniform(0.5, 0.9),  # Strong beta
                population_synchrony=np.random.uniform(0.6, 0.9),  # High sync
                spike_regularity=np.random.uniform(0.2, 0.5),  # Irregular
                pathological_bursting=np.random.uniform(0.1, 0.4)  # Some bursting
            )
            dataset.append(data)
            labels.append('Parkinsonian')
        
        # Class 3: Epileptiform Patterns  
        print("  Generating Epileptiform patterns...")
        for _ in range(n_trials_per_class):
            data = self.generate_synthetic_spikes(
                coding_type=np.random.choice(['rate', 'temporal']),
                oscillatory_power=np.random.uniform(0.3, 0.7),  # Mixed oscillations
                population_synchrony=np.random.uniform(0.7, 1.0),  # Very high sync
                spike_regularity=np.random.uniform(0.3, 0.7),  # Variable
                pathological_bursting=np.random.uniform(0.4, 0.8)  # High bursting
            )
            dataset.append(data)
            labels.append('Epileptiform')
        
        # Class 4: Mixed Pathology
        print("  Generating Mixed Pathological patterns...")
        for _ in range(n_trials_per_class):
            data = self.generate_synthetic_spikes(
                coding_type=np.random.choice(['rate', 'temporal', 'mixed']),
                oscillatory_power=np.random.uniform(0.4, 0.8),
                population_synchrony=np.random.uniform(0.5, 0.8),
                spike_regularity=np.random.uniform(0.2, 0.6),
                pathological_bursting=np.random.uniform(0.2, 0.6)
            )
            dataset.append(data)
            labels.append('Mixed_Pathology')
        
        print(f"Dataset generation complete: {len(dataset)} trials across 5 classes")
        
        return dataset, labels

# Convenience function for backward compatibility
def load_spike_data(coding_type='rate', n_stimuli=20, **kwargs):
    """Backward compatible function for existing code"""
    generator = NeuralPatternGenerator()
    return generator.generate_synthetic_spikes(coding_type=coding_type, 
                                             n_stimuli=n_stimuli, **kwargs)

if __name__ == "__main__":
    # Quick test when run directly
    print("Testing NeuralPatternGenerator...")
    generator = NeuralPatternGenerator(n_neurons=10, trial_duration=1.0)
    
    # Test basic generation
    data = generator.generate_synthetic_spikes(
        coding_type='rate',
        n_stimuli=3,
        n_trials_per_stimulus=2
    )
    
    print(f"Generated {len(data['spike_trains'])} trials")
    print(f"Each trial has {len(data['spike_trains'][0])} neurons")
    print("Basic generation test passed!")