import numpy as np
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class EnhancedSpikeAnalyzer:
    """Enhanced spike train analyzer with pathological pattern feature extraction"""
    
    def __init__(self, trial_duration=2.0, dt=0.001):
        self.trial_duration = trial_duration
        self.dt = dt
        self.time_bins = np.arange(0, trial_duration, dt)
        
    def analyze_spike_data(self, spike_data):
        """Comprehensive analysis of spike data with pathological features"""
        
        results = {
            'quality_metrics': self.calculate_quality_metrics(spike_data),
            'rate_features': self.extract_rate_features(spike_data),
            'temporal_features': self.extract_temporal_features(spike_data),
            'spectral_features': self.extract_spectral_features(spike_data),
            'synchrony_features': self.extract_synchrony_features(spike_data),
            'burst_features': self.extract_burst_features(spike_data),
            'regularity_features': self.extract_regularity_features(spike_data),
            'pathology_features': self.extract_pathology_features(spike_data)
        }
        
        return results
    
    def calculate_quality_metrics(self, spike_data):
        """Calculate data quality metrics (existing functionality)"""
        
        metrics = {
            'n_trials': len(spike_data['spike_trains']),
            'n_neurons': len(spike_data['spike_trains'][0]) if spike_data['spike_trains'] else 0,
            'mean_firing_rate': 0,
            'refractory_violations': 0,
            'empty_neurons': 0
        }
        
        if not spike_data['spike_trains']:
            return metrics
        
        all_rates = []
        refractory_violations = 0
        empty_neurons = 0
        
        for trial_spikes in spike_data['spike_trains']:
            for neuron_spikes in trial_spikes:
                if len(neuron_spikes) == 0:
                    empty_neurons += 1
                    continue
                
                # Firing rate
                rate = len(neuron_spikes) / self.trial_duration
                all_rates.append(rate)
                
                # Refractory period violations (< 2ms)
                if len(neuron_spikes) > 1:
                    isis = np.diff(neuron_spikes)
                    violations = np.sum(isis < 0.002)
                    refractory_violations += violations
        
        metrics['mean_firing_rate'] = np.mean(all_rates) if all_rates else 0
        metrics['refractory_violations'] = refractory_violations
        metrics['empty_neurons'] = empty_neurons
        
        return metrics
    
    def extract_rate_features(self, spike_data):
        """Extract rate coding features (enhanced)"""
        
        features = {}
        
        # Calculate firing rates for each trial and stimulus
        firing_rates = []
        for trial_idx, trial_spikes in enumerate(spike_data['spike_trains']):
            trial_rates = []
            for neuron_spikes in trial_spikes:
                rate = len(neuron_spikes) / self.trial_duration
                trial_rates.append(rate)
            firing_rates.append(trial_rates)
        
        firing_rates = np.array(firing_rates)
        
        # Basic rate features
        features['mean_population_rate'] = np.mean(firing_rates)
        features['rate_variance'] = np.var(firing_rates)
        features['rate_range'] = np.max(firing_rates) - np.min(firing_rates)
        
        # Stimulus modulation
        if len(spike_data['stimulus_labels']) > 0:
            stimulus_rates = {}
            for trial_idx, stim_label in enumerate(spike_data['stimulus_labels']):
                if stim_label not in stimulus_rates:
                    stimulus_rates[stim_label] = []
                stimulus_rates[stim_label].append(firing_rates[trial_idx])
            
            # Rate modulation across stimuli
            stim_means = [np.mean(rates) for rates in stimulus_rates.values()]
            features['stimulus_modulation'] = np.std(stim_means) / (np.mean(stim_means) + 1e-6)
            features['max_rate_change'] = np.max(stim_means) - np.min(stim_means)
        else:
            features['stimulus_modulation'] = 0
            features['max_rate_change'] = 0
        
        # Population rate dynamics
        features['rate_entropy'] = self._calculate_rate_entropy(firing_rates)
        features['fano_factor'] = np.var(firing_rates) / (np.mean(firing_rates) + 1e-6)
        
        return features
    
    def extract_temporal_features(self, spike_data):
        """Extract temporal coding features (enhanced)"""
        
        features = {}
        
        # Spike timing precision
        timing_precisions = []
        for trial_spikes in spike_data['spike_trains']:
            for neuron_spikes in trial_spikes:
                if len(neuron_spikes) > 2:
                    # Vector strength for temporal precision
                    stim_times = spike_data.get('stimulus_times', [0.5])
                    if stim_times:
                        stim_time = stim_times[0] if isinstance(stim_times, list) else stim_times
                        relative_times = neuron_spikes - stim_time
                        post_stim_spikes = relative_times[(relative_times > 0) & (relative_times < 0.5)]
                        
                        if len(post_stim_spikes) > 2:
                            # Phase consistency (assuming 8 Hz modulation)
                            phases = 2 * np.pi * 8 * post_stim_spikes
                            vector_strength = np.abs(np.mean(np.exp(1j * phases)))
                            timing_precisions.append(vector_strength)
        
        features['mean_timing_precision'] = np.mean(timing_precisions) if timing_precisions else 0
        features['timing_precision_std'] = np.std(timing_precisions) if timing_precisions else 0
        
        # Inter-spike interval statistics
        all_isis = []
        isi_cvs = []
        for trial_spikes in spike_data['spike_trains']:
            for neuron_spikes in trial_spikes:
                if len(neuron_spikes) > 2:
                    isis = np.diff(neuron_spikes)
                    all_isis.extend(isis)
                    
                    if len(isis) > 1:
                        cv = np.std(isis) / (np.mean(isis) + 1e-6)
                        isi_cvs.append(cv)
        
        features['mean_isi'] = np.mean(all_isis) if all_isis else 0
        features['isi_cv'] = np.mean(isi_cvs) if isi_cvs else 0
        features['isi_entropy'] = stats.entropy(np.histogram(all_isis, bins=20)[0] + 1e-6) if all_isis else 0
        
        # Local variation (Shinomoto et al.)
        lv_values = []
        for trial_spikes in spike_data['spike_trains']:
            for neuron_spikes in trial_spikes:
                if len(neuron_spikes) > 3:
                    isis = np.diff(neuron_spikes)
                    if len(isis) > 2:
                        lv = np.mean([3 * (isi1 - isi2)**2 / (isi1 + isi2)**2 
                                    for isi1, isi2 in zip(isis[:-1], isis[1:]) 
                                    if isi1 + isi2 > 0])
                        lv_values.append(lv)
        
        features['local_variation'] = np.mean(lv_values) if lv_values else 0
        
        return features
    
    def extract_spectral_features(self, spike_data):
        """Extract spectral power features for pathological pattern detection"""
        
        features = {}
        
        # Calculate population activity
        pop_activities = []
        for trial_spikes in spike_data['spike_trains']:
            pop_activity = self._calculate_population_activity(trial_spikes)
            pop_activities.append(pop_activity)
        
        # Average spectral analysis across trials
        all_psds = []
        freqs = None
        
        for pop_activity in pop_activities:
            if len(pop_activity) > 20:
                try:
                    f, psd = signal.welch(pop_activity, fs=1000, nperseg=min(128, len(pop_activity)//4))
                    freqs = f
                    all_psds.append(psd)
                except:
                    continue
        
        if all_psds and freqs is not None:
            mean_psd = np.mean(all_psds, axis=0)
            
            # Frequency band powers
            features['delta_power'] = self._band_power(freqs, mean_psd, 1, 4)
            features['theta_power'] = self._band_power(freqs, mean_psd, 4, 8)
            features['alpha_power'] = self._band_power(freqs, mean_psd, 8, 13)
            features['beta_power'] = self._band_power(freqs, mean_psd, 13, 30)
            features['gamma_power'] = self._band_power(freqs, mean_psd, 30, 100)
            
            # Pathological frequency signatures
            features['beta_peak_freq'] = self._find_peak_frequency(freqs, mean_psd, 13, 30)
            features['gamma_peak_freq'] = self._find_peak_frequency(freqs, mean_psd, 30, 100)
            
            # Spectral entropy (measure of spectral complexity)
            features['spectral_entropy'] = stats.entropy(mean_psd + 1e-10)
            
            # Beta/total power ratio (Parkinsonian marker)
            total_power = np.sum(mean_psd[(freqs >= 1) & (freqs <= 100)])
            features['beta_ratio'] = features['beta_power'] / (total_power + 1e-6)
            
        else:
            # Default values if spectral analysis fails
            for key in ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 
                       'gamma_power', 'beta_peak_freq', 'gamma_peak_freq', 
                       'spectral_entropy', 'beta_ratio']:
                features[key] = 0
        
        return features
    
    def extract_synchrony_features(self, spike_data):
        """Extract population synchrony features"""
        
        features = {}
        
        # Cross-correlation analysis
        cross_correlations = []
        for trial_spikes in spike_data['spike_trains']:
            if len(trial_spikes) > 1:
                # Convert to binned spike trains
                binned_trains = []
                for spikes in trial_spikes:
                    binned, _ = np.histogram(spikes, bins=np.arange(0, self.trial_duration + 0.01, 0.01))
                    binned_trains.append(binned)
                
                binned_trains = np.array(binned_trains)
                
                # Pairwise correlations
                if binned_trains.shape[0] > 1:
                    corr_matrix = np.corrcoef(binned_trains)
                    # Remove diagonal and take upper triangle
                    triu_indices = np.triu_indices(corr_matrix.shape[0], k=1)
                    correlations = corr_matrix[triu_indices]
                    correlations = correlations[~np.isnan(correlations)]
                    cross_correlations.extend(correlations)
        
        features['mean_cross_correlation'] = np.mean(cross_correlations) if cross_correlations else 0
        features['max_cross_correlation'] = np.max(cross_correlations) if cross_correlations else 0
        features['correlation_variance'] = np.var(cross_correlations) if cross_correlations else 0
        
        # Population synchrony index
        sync_indices = []
        for trial_spikes in spike_data['spike_trains']:
            pop_activity = self._calculate_population_activity(trial_spikes, bin_size=0.001)
            if len(pop_activity) > 10:
                # Synchrony as coefficient of variation of population activity
                sync_index = np.std(pop_activity) / (np.mean(pop_activity) + 1e-6)
                sync_indices.append(sync_index)
        
        features['population_sync_index'] = np.mean(sync_indices) if sync_indices else 0
        features['sync_index_variance'] = np.var(sync_indices) if sync_indices else 0
        
        # Phase locking analysis (simplified)
        phase_locking_values = []
        for trial_spikes in spike_data['spike_trains']:
            pop_activity = self._calculate_population_activity(trial_spikes)
            if len(pop_activity) > 50:
                # Hilbert transform for instantaneous phase
                analytic_signal = signal.hilbert(pop_activity)
                instantaneous_phase = np.angle(analytic_signal)
                
                # Phase consistency across time
                phase_consistency = np.abs(np.mean(np.exp(1j * instantaneous_phase)))
                phase_locking_values.append(phase_consistency)
        
        features['phase_locking_value'] = np.mean(phase_locking_values) if phase_locking_values else 0
        
        return features
    
    def extract_burst_features(self, spike_data):
        """Extract burst firing features"""
        
        features = {}
        
        burst_indices = []
        burst_durations = []
        interburst_intervals = []
        
        for trial_spikes in spike_data['spike_trains']:
            for neuron_spikes in trial_spikes:
                if len(neuron_spikes) > 3:
                    # Detect bursts (ISI < 10ms followed by ISI > 100ms)
                    isis = np.diff(neuron_spikes)
                    
                    if len(isis) > 2:
                        # Simple burst detection
                        short_isis = isis < 0.01  # < 10ms
                        long_isis = isis > 0.1    # > 100ms
                        
                        # Burst index
                        burst_ratio = np.sum(short_isis) / len(isis)
                        burst_indices.append(burst_ratio)
                        
                        # More sophisticated burst detection
                        bursts = self._detect_bursts(neuron_spikes)
                        if bursts:
                            for burst_start, burst_end in bursts:
                                burst_durations.append(burst_end - burst_start)
                            
                            # Inter-burst intervals
                            if len(bursts) > 1:
                                for i in range(len(bursts) - 1):
                                    ibi = bursts[i+1][0] - bursts[i][1]
                                    interburst_intervals.append(ibi)
        
        features['burst_index'] = np.mean(burst_indices) if burst_indices else 0
        features['mean_burst_duration'] = np.mean(burst_durations) if burst_durations else 0
        features['mean_interburst_interval'] = np.mean(interburst_intervals) if interburst_intervals else 0
        features['burst_frequency'] = len(burst_durations) / (len(spike_data['spike_trains']) * self.trial_duration) if burst_durations else 0
        
        return features
    
    def extract_regularity_features(self, spike_data):
        """Extract spike train regularity features"""
        
        features = {}
        
        fano_factors = []
        cv_values = []
        entropies = []
        
        for trial_spikes in spike_data['spike_trains']:
            for neuron_spikes in trial_spikes:
                if len(neuron_spikes) > 2:
                    # Fano factor (variance/mean of spike counts in windows)
                    window_size = 0.1  # 100ms windows
                    n_windows = int(self.trial_duration / window_size)
                    spike_counts = []
                    
                    for w in range(n_windows):
                        start_time = w * window_size
                        end_time = (w + 1) * window_size
                        count = np.sum((neuron_spikes >= start_time) & (neuron_spikes < end_time))
                        spike_counts.append(count)
                    
                    if spike_counts and np.mean(spike_counts) > 0:
                        fano = np.var(spike_counts) / np.mean(spike_counts)
                        fano_factors.append(fano)
                    
                    # ISI coefficient of variation
                    if len(neuron_spikes) > 3:
                        isis = np.diff(neuron_spikes)
                        if len(isis) > 1 and np.mean(isis) > 0:
                            cv = np.std(isis) / np.mean(isis)
                            cv_values.append(cv)
                    
                    # Spike train entropy
                    if len(neuron_spikes) > 5:
                        # Simple entropy based on ISI distribution
                        isis = np.diff(neuron_spikes)
                        hist, _ = np.histogram(isis, bins=10)
                        prob = hist / np.sum(hist)
                        entropy_val = stats.entropy(prob + 1e-10)
                        entropies.append(entropy_val)
        
        features['mean_fano_factor'] = np.mean(fano_factors) if fano_factors else 1.0
        features['mean_cv'] = np.mean(cv_values) if cv_values else 1.0
        features['mean_entropy'] = np.mean(entropies) if entropies else 0
        features['regularity_index'] = 1 / (1 + np.mean(cv_values)) if cv_values else 0.5
        
        return features
    
    def extract_pathology_features(self, spike_data):
        """Extract composite pathological pattern features"""
        
        features = {}
        
        # Get previously calculated features
        spectral = self.extract_spectral_features(spike_data)
        synchrony = self.extract_synchrony_features(spike_data)
        burst = self.extract_burst_features(spike_data)
        
        # Parkinsonian signature (beta power + synchrony + regularity disruption)
        features['parkinsonian_index'] = (
            spectral['beta_ratio'] * 
            synchrony['mean_cross_correlation'] * 
            (1 - self.extract_regularity_features(spike_data)['regularity_index'])
        )
        
        # Epileptiform signature (high synchrony + bursting + sharp transients)
        features['epileptiform_index'] = (
            synchrony['population_sync_index'] * 
            burst['burst_index'] * 
            synchrony['max_cross_correlation']
        )
        
        # General pathology indicator
        features['pathology_score'] = np.mean([
            features['parkinsonian_index'],
            features['epileptiform_index']
        ])
        
        return features
    
    # Helper methods
    def _calculate_population_activity(self, spike_trains, bin_size=0.01):
        """Calculate population activity over time"""
        time_bins = np.arange(0, self.trial_duration + bin_size, bin_size)
        pop_activity = np.zeros(len(time_bins) - 1)
        
        for spikes in spike_trains:
            if len(spikes) > 0:
                spike_counts, _ = np.histogram(spikes, bins=time_bins)
                min_len = min(len(spike_counts), len(pop_activity))
                pop_activity[:min_len] += spike_counts[:min_len]
        
        return pop_activity
    
    def _band_power(self, freqs, psd, low_freq, high_freq):
        """Calculate power in a frequency band"""
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        if np.any(band_mask):
            return np.trapz(psd[band_mask], freqs[band_mask])
        return 0
    
    def _find_peak_frequency(self, freqs, psd, low_freq, high_freq):
        """Find peak frequency in a band"""
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        if np.any(band_mask):
            band_freqs = freqs[band_mask]
            band_psd = psd[band_mask]
            peak_idx = np.argmax(band_psd)
            return band_freqs[peak_idx]
        return 0
    
    def _detect_bursts(self, spike_times, max_isi=0.01, min_spikes=3):
        """Simple burst detection algorithm"""
        if len(spike_times) < min_spikes:
            return []
        
        isis = np.diff(spike_times)
        burst_spikes = isis <= max_isi
        
        bursts = []
        burst_start = None
        burst_spikes_count = 0
        
        for i, is_burst in enumerate(burst_spikes):
            if is_burst:
                if burst_start is None:
                    burst_start = i
                    burst_spikes_count = 1
                burst_spikes_count += 1
            else:
                if burst_start is not None and burst_spikes_count >= min_spikes:
                    bursts.append((spike_times[burst_start], spike_times[i]))
                burst_start = None
                burst_spikes_count = 0
        
        # Check final burst
        if burst_start is not None and burst_spikes_count >= min_spikes:
            bursts.append((spike_times[burst_start], spike_times[-1]))
        
        return bursts
    
    def _calculate_rate_entropy(self, firing_rates):
        """Calculate entropy of firing rate distribution"""
        if len(firing_rates.flatten()) == 0:
            return 0
        
        hist, _ = np.histogram(firing_rates.flatten(), bins=20)
        prob = hist / np.sum(hist)
        return stats.entropy(prob + 1e-10)

# Backward compatibility function
def analyze_spike_data(spike_data):
    """Backward compatible analysis function"""
    analyzer = EnhancedSpikeAnalyzer()
    return analyzer.analyze_spike_data(spike_data)