"""
Spike Visualizer Module
======================

Advanced visualization tools for neural spike data with cool aesthetics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SpikeVisualizer:
    """
    Class for creating advanced visualizations of neural spike data.
    """
    
    def __init__(self, style='dark'):
        """Initialize with cool visual styling."""
        self.setup_style(style)
        self.colors = {
            'on': '#FF6B6B',      # Coral red for ON stimulus
            'off': '#4ECDC4',     # Teal for OFF stimulus
            'baseline': '#95E1D3', # Light teal
            'spike': '#F38BA8',   # Pink for spikes
            'psth': '#A8E6CF',    # Light green
            'quality_good': '#4CAF50',
            'quality_bad': '#F44336',
            'neural': '#BB86FC'   # Purple for neural activity
        }
        
    def setup_style(self, style='dark'):
        """Setup matplotlib styling for cool visuals."""
        if style == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#1a1a1a'
            self.text_color = 'white'
        else:
            plt.style.use('default')
            self.bg_color = 'white'
            self.text_color = 'black'
            
        # Set custom parameters for cool aesthetics
        plt.rcParams.update({
            'figure.facecolor': self.bg_color,
            'axes.facecolor': self.bg_color,
            'axes.edgecolor': '#333333',
            'axes.linewidth': 0.8,
            'xtick.color': self.text_color,
            'ytick.color': self.text_color,
            'text.color': self.text_color,
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': '#444444',
            'lines.linewidth': 2,
        })

    def plot_raster_psth(self, trials_data: List[Dict], unit_id: int, 
                        bin_size: float = 0.05, smooth_sigma: float = 2.0,
                        figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Create a gorgeous raster plot with PSTH.
        
        Parameters:
        -----------
        trials_data : list
            Trial data
        unit_id : int
            Unit to visualize
        bin_size : float
            PSTH bin size in seconds
        smooth_sigma : float
            Gaussian smoothing parameter
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig = plt.figure(figsize=figsize, facecolor=self.bg_color)
        gs = GridSpec(3, 1, height_ratios=[2, 1, 0.3], hspace=0.3)
        
        # Extract data for this unit
        on_trials, off_trials = [], []
        all_spikes_on, all_spikes_off = [], []
        
        for i, trial in enumerate(trials_data):
            unit_data = next((u for u in trial['units'] if u['unit_id'] == unit_id), None)
            if unit_data is None:
                continue
                
            spikes = unit_data['spike_times']
            if trial['event_label'] == 'ON':
                on_trials.append((i, spikes))
                all_spikes_on.extend(spikes)
            else:
                off_trials.append((i, spikes))
                all_spikes_off.extend(spikes)
        
        # Raster plot
        ax_raster = fig.add_subplot(gs[0])
        
        # Plot ON trials
        on_y_positions = []
        for idx, (trial_idx, spikes) in enumerate(on_trials):
            y_pos = idx
            on_y_positions.append(y_pos)
            ax_raster.vlines(spikes, y_pos - 0.4, y_pos + 0.4, 
                           colors=self.colors['on'], alpha=0.8, linewidth=1.5)
        
        # Plot OFF trials  
        off_y_positions = []
        for idx, (trial_idx, spikes) in enumerate(off_trials):
            y_pos = len(on_trials) + idx + 1
            off_y_positions.append(y_pos)
            ax_raster.vlines(spikes, y_pos - 0.4, y_pos + 0.4,
                           colors=self.colors['off'], alpha=0.8, linewidth=1.5)
        
        # Add stimulus onset line
        ax_raster.axvline(0, color='yellow', linewidth=3, alpha=0.8, 
                         linestyle='--', label='Stimulus Onset')
        
        # Style raster plot
        ax_raster.set_xlim(-1, 3)
        ax_raster.set_ylabel('Trial #', fontweight='bold')
        ax_raster.set_title(f'🧠 Neural Unit {unit_id}: Spike Raster & PSTH', 
                          fontsize=16, fontweight='bold', pad=20)
        
        # Add trial type labels
        if on_y_positions:
            ax_raster.text(-0.9, np.mean(on_y_positions), 'ON', 
                          color=self.colors['on'], fontweight='bold', 
                          fontsize=12, va='center')
        if off_y_positions:
            ax_raster.text(-0.9, np.mean(off_y_positions), 'OFF',
                          color=self.colors['off'], fontweight='bold',
                          fontsize=12, va='center')
        
        # PSTH
        ax_psth = fig.add_subplot(gs[1])
        
        # Calculate PSTH for each condition
        pre_time = trials_data[0]['pre_time']
        post_time = trials_data[0]['post_time']
        time_bins = np.arange(-pre_time, post_time + bin_size, bin_size)
        time_centers = time_bins[:-1] + bin_size/2
        
        # ON condition PSTH
        if all_spikes_on:
            hist_on, _ = np.histogram(all_spikes_on, bins=time_bins)
            psth_on = hist_on / (len(on_trials) * bin_size)  # Convert to Hz
            psth_on_smooth = gaussian_filter1d(psth_on, smooth_sigma)
            
            ax_psth.fill_between(time_centers, 0, psth_on_smooth, 
                               alpha=0.7, color=self.colors['on'], 
                               label=f'ON (n={len(on_trials)})')
        
        # OFF condition PSTH  
        if all_spikes_off:
            hist_off, _ = np.histogram(all_spikes_off, bins=time_bins)
            psth_off = hist_off / (len(off_trials) * bin_size)
            psth_off_smooth = gaussian_filter1d(psth_off, smooth_sigma)
            
            ax_psth.fill_between(time_centers, 0, psth_off_smooth,
                               alpha=0.7, color=self.colors['off'],
                               label=f'OFF (n={len(off_trials)})')
        
        # Style PSTH
        ax_psth.axvline(0, color='yellow', linewidth=3, alpha=0.8, linestyle='--')
        ax_psth.set_xlim(-1, 3)
        ax_psth.set_xlabel('Time from Stimulus Onset (s)', fontweight='bold')
        ax_psth.set_ylabel('Firing Rate (Hz)', fontweight='bold')
        ax_psth.legend(loc='upper right')
        ax_psth.grid(True, alpha=0.3)
        
        # Add stimulus period highlight
        stimulus_patch = Rectangle((0, ax_psth.get_ylim()[0]), 2, 
                                 ax_psth.get_ylim()[1] - ax_psth.get_ylim()[0],
                                 alpha=0.1, color='yellow', label='Stimulus Period')
        ax_psth.add_patch(stimulus_patch)
        
        # Timeline
        ax_timeline = fig.add_subplot(gs[2])
        ax_timeline.barh(0, 1, left=-1, height=0.3, color=self.colors['baseline'], 
                        alpha=0.7, label='Baseline')
        ax_timeline.barh(0, 2, left=0, height=0.3, color='gold', 
                        alpha=0.7, label='Stimulus')  
        ax_timeline.barh(0, 1, left=2, height=0.3, color=self.colors['baseline'],
                        alpha=0.7, label='Post-stim')
        ax_timeline.set_xlim(-1, 3)
        ax_timeline.set_ylim(-0.5, 0.5)
        ax_timeline.set_xlabel('Time (s)', fontweight='bold')
        ax_timeline.set_yticks([])
        ax_timeline.legend(loc='center', bbox_to_anchor=(0.5, -0.5), ncol=3)
        
        plt.tight_layout()
        return fig

    def plot_unit_quality(self, unit_stats: Dict, good_units: List[int],
                         figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Create comprehensive unit quality visualization.
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize, facecolor=self.bg_color)
        fig.suptitle('🔍 Neural Unit Quality Assessment', fontsize=18, 
                    fontweight='bold', y=0.98)
        
        # Prepare data
        units = list(unit_stats.keys())
        colors = [self.colors['quality_good'] if u in good_units 
                 else self.colors['quality_bad'] for u in units]
        
        # 1. Firing Rate Distribution
        firing_rates = [unit_stats[u]['firing_rate'] for u in units]
        axes[0,0].scatter(units, firing_rates, c=colors, s=80, alpha=0.8, edgecolors='white')
        axes[0,0].axhline(0.1, color='red', linestyle='--', alpha=0.7, label='Min threshold')
        axes[0,0].set_ylabel('Firing Rate (Hz)', fontweight='bold')
        axes[0,0].set_title('🎯 Firing Rate by Unit')
        axes[0,0].legend()
        
        # 2. Refractory Violations
        ref_violations = [unit_stats[u]['refractory_violations'] * 100 for u in units]
        axes[0,1].scatter(units, ref_violations, c=colors, s=80, alpha=0.8, edgecolors='white')
        axes[0,1].axhline(2.0, color='red', linestyle='--', alpha=0.7, label='Max threshold')
        axes[0,1].set_ylabel('Refractory Violations (%)', fontweight='bold')
        axes[0,1].set_title('⚡ Refractory Period Violations')
        axes[0,1].legend()
        
        # 3. CV ISI (Regularity)
        cv_isis = [unit_stats[u]['cv_isi'] for u in units]
        axes[0,2].scatter(units, cv_isis, c=colors, s=80, alpha=0.8, edgecolors='white')
        axes[0,2].set_ylabel('CV of ISI', fontweight='bold')
        axes[0,2].set_title('🔄 Spike Timing Regularity')
        
        # 4. Fano Factor (Variability)
        fano_factors = [unit_stats[u]['fano_factor'] for u in units]
        axes[1,0].scatter(units, fano_factors, c=colors, s=80, alpha=0.8, edgecolors='white')
        axes[1,0].axhline(1.0, color='yellow', linestyle='--', alpha=0.7, label='Poisson')
        axes[1,0].set_ylabel('Fano Factor', fontweight='bold')
        axes[1,0].set_xlabel('Unit ID')
        axes[1,0].set_title('📊 Trial-to-Trial Variability')
        axes[1,0].legend()
        
        # 5. Total Spikes
        total_spikes = [unit_stats[u]['total_spikes'] for u in units]
        axes[1,1].scatter(units, total_spikes, c=colors, s=80, alpha=0.8, edgecolors='white')
        axes[1,1].axhline(100, color='red', linestyle='--', alpha=0.7, label='Min threshold')
        axes[1,1].set_ylabel('Total Spikes', fontweight='bold')
        axes[1,1].set_xlabel('Unit ID')
        axes[1,1].set_title('🔢 Total Spike Count')
        axes[1,1].legend()
        
        # 6. Quality Summary
        good_count = len(good_units)
        total_count = len(units)
        quality_data = [good_count, total_count - good_count]
        quality_labels = ['Good Units', 'Poor Quality']
        quality_colors = [self.colors['quality_good'], self.colors['quality_bad']]
        
        wedges, texts, autotexts = axes[1,2].pie(quality_data, labels=quality_labels, 
                                                colors=quality_colors, autopct='%1.0f%%',
                                                startangle=90, textprops={'fontweight': 'bold'})
        axes[1,2].set_title('✅ Quality Control Summary')
        
        # Style all subplots
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            if hasattr(ax, 'spines'):
                for spine in ax.spines.values():
                    spine.set_edgecolor('#333333')
        
        plt.tight_layout()
        return fig

    def plot_isi_distribution(self, trials_data: List[Dict], unit_id: int,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot inter-spike interval distribution with cool styling.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor=self.bg_color)
        fig.suptitle(f'⚡ Unit {unit_id}: Inter-Spike Interval Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Extract all ISIs for this unit
        all_isis = []
        for trial in trials_data:
            unit_data = next((u for u in trial['units'] if u['unit_id'] == unit_id), None)
            if unit_data and len(unit_data['spike_times']) > 1:
                spikes = np.sort(unit_data['spike_times'])
                isis = np.diff(spikes)
                all_isis.extend(isis)
        
        all_isis = np.array(all_isis)
        
        if len(all_isis) == 0:
            ax1.text(0.5, 0.5, 'No ISI data available', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=14)
            ax2.text(0.5, 0.5, 'No ISI data available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            return fig
        
        # 1. ISI histogram
        bins = np.logspace(np.log10(0.001), np.log10(np.max(all_isis)), 50)
        ax1.hist(all_isis, bins=bins, alpha=0.7, color=self.colors['neural'],
                edgecolor='white', linewidth=0.5)
        ax1.axvline(0.001, color='red', linestyle='--', linewidth=2, 
                   label='Refractory Period (1ms)')
        ax1.set_xscale('log')
        ax1.set_xlabel('Inter-Spike Interval (s)', fontweight='bold')
        ax1.set_ylabel('Count', fontweight='bold')
        ax1.set_title('📊 ISI Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ISI vs time (checking for non-stationarity)
        # Sample ISIs if too many
        if len(all_isis) > 5000:
            sample_idx = np.random.choice(len(all_isis), 5000, replace=False)
            sample_isis = all_isis[sample_idx]
        else:
            sample_isis = all_isis
            
        ax2.scatter(range(len(sample_isis)), sample_isis, 
                   alpha=0.6, s=20, color=self.colors['spike'])
        ax2.set_yscale('log')
        ax2.set_xlabel('ISI Number', fontweight='bold')
        ax2.set_ylabel('Inter-Spike Interval (s)', fontweight='bold')
        ax2.set_title('⏱️ ISI Stationarity Check')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text box
        cv_isi = np.std(all_isis) / np.mean(all_isis)
        median_isi = np.median(all_isis)
        refractory_violations = np.sum(all_isis < 0.001) / len(all_isis) * 100
        
        stats_text = f'CV: {cv_isi:.3f}\nMedian: {median_isi:.3f}s\nRef. Viol.: {refractory_violations:.1f}%'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor=self.colors['neural'], alpha=0.8),
                verticalalignment='top', fontweight='bold')
        
        plt.tight_layout()
        return fig

    def plot_cross_correlation(self, trials_data: List[Dict], unit1_id: int, 
                             unit2_id: int, max_lag: float = 0.1,
                             figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot cross-correlation between two units.
        """
        from spike_analyzer import SpikeAnalyzer
        analyzer = SpikeAnalyzer()
        
        # Calculate cross-correlation
        lags, xcorr = analyzer.calculate_cross_correlation(trials_data, unit1_id, unit2_id, max_lag)
        
        if len(lags) == 0:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
            ax.text(0.5, 0.5, 'No cross-correlation data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
        
        # Plot cross-correlogram
        ax.plot(lags, xcorr, color=self.colors['neural'], linewidth=2)
        ax.fill_between(lags, 0, xcorr, alpha=0.3, color=self.colors['neural'])
        
        # Add zero lag line
        ax.axvline(0, color='yellow', linestyle='--', linewidth=2, alpha=0.8,
                  label='Zero Lag')
        
        # Style
        ax.set_xlabel('Lag (s)', fontweight='bold')
        ax.set_ylabel('Cross-Correlation', fontweight='bold')
        ax.set_title(f'🔗 Cross-Correlation: Unit {unit1_id} vs Unit {unit2_id}', 
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Find peak correlation
        peak_idx = np.argmax(np.abs(xcorr))
        peak_lag = lags[peak_idx]
        peak_value = xcorr[peak_idx]
        
        ax.scatter([peak_lag], [peak_value], color='red', s=100, zorder=5,
                  label=f'Peak: {peak_value:.3f} @ {peak_lag:.3f}s')
        ax.legend()
        
        plt.tight_layout()
        return fig

    def plot_response_statistics(self, response_stats: Dict,
                               figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Visualize stimulus response statistics across units.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor=self.bg_color)
        fig.suptitle('📈 Stimulus Response Analysis', fontsize=16, fontweight='bold')
        
        units = list(response_stats.keys())
        responsive_units = [u for u in units if response_stats[u]['responsive']]
        
        # Extract data
        baseline_rates = [response_stats[u]['baseline_rate'] for u in units]
        stimulus_rates = [response_stats[u]['stimulus_rate'] for u in units]  
        p_values = [response_stats[u]['p_value'] for u in units]
        
        colors = [self.colors['quality_good'] if u in responsive_units 
                 else self.colors['quality_bad'] for u in units]
        
        # 1. Baseline vs Stimulus rates
        axes[0,0].scatter(baseline_rates, stimulus_rates, c=colors, s=80, 
                         alpha=0.8, edgecolors='white')
        
        # Add unity line
        max_rate = max(max(baseline_rates), max(stimulus_rates))
        axes[0,0].plot([0, max_rate], [0, max_rate], 'k--', alpha=0.5, label='Unity')
        
        axes[0,0].set_xlabel('Baseline Rate (Hz)', fontweight='bold')
        axes[0,0].set_ylabel('Stimulus Rate (Hz)', fontweight='bold')
        axes[0,0].set_title('🎯 Response Magnitude')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Modulation index
        modulation_idx = [(s - b) / (s + b + 1e-10) for b, s in zip(baseline_rates, stimulus_rates)]
        axes[0,1].scatter(units, modulation_idx, c=colors, s=80, alpha=0.8, edgecolors='white')
        axes[0,1].axhline(0, color='white', linestyle='-', alpha=0.5)
        axes[0,1].set_ylabel('Modulation Index', fontweight='bold')
        axes[0,1].set_title('📊 Response Modulation')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. P-values
        log_p_values = [-np.log10(max(p, 1e-10)) for p in p_values]
        axes[1,0].scatter(units, log_p_values, c=colors, s=80, alpha=0.8, edgecolors='white')
        axes[1,0].axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7,
                         label='p = 0.05')
        axes[1,0].set_xlabel('Unit ID')
        axes[1,0].set_ylabel('-log10(p-value)', fontweight='bold')
        axes[1,0].set_title('📊 Statistical Significance')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Response summary
        resp_count = len(responsive_units)
        total_count = len(units)
        summary_data = [resp_count, total_count - resp_count]
        summary_labels = ['Responsive', 'Non-responsive']
        summary_colors = [self.colors['quality_good'], self.colors['quality_bad']]
        
        wedges, texts, autotexts = axes[1,1].pie(summary_data, labels=summary_labels,
                                                colors=summary_colors, autopct='%1.0f%%',
                                                startangle=90, 
                                                textprops={'fontweight': 'bold'})
        axes[1,1].set_title('✅ Response Summary')
        
        plt.tight_layout()
        return fig

    def plot_decoding_results(self, results: Dict,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Visualize decoding analysis results with cool styling.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor=self.bg_color)
        fig.suptitle('🎯 Neural Decoding: Rate vs Temporal Coding', 
                    fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        methods = ['Rate', 'Temporal', 'Shuffle']
        accuracies = [results['rate_accuracy'], results['temporal_accuracy'], 
                     results['shuffle_accuracy']]
        errors = [results['rate_std'], results['temporal_std'], results['shuffle_std']]
        colors = [self.colors['on'], self.colors['neural'], '#666666']
        
        bars = ax1.bar(methods, accuracies, yerr=errors, capsize=5, 
                      color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add significance indicators
        if results['temporal_accuracy'] > results['rate_accuracy'] + 2*results['rate_std']:
            ax1.text(1, results['temporal_accuracy'] + results['temporal_std'] + 0.02,
                    '***', ha='center', fontsize=20, fontweight='bold', color='yellow')
        
        ax1.set_ylabel('Classification Accuracy', fontweight='bold')
        ax1.set_title('🎯 Decoding Performance')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add accuracy values on bars
        for bar, acc, err in zip(bars, accuracies, errors):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Feature importance (if available)
        if 'feature_importance' in results:
            importance = results['feature_importance']
            feature_names = [f'Feature {i+1}' for i in range(len(importance))]
            
            ax2.barh(range(len(importance)), importance, 
                    color=self.colors['neural'], alpha=0.8)
            ax2.set_yticks(range(len(importance)))
            ax2.set_yticklabels(feature_names)
            ax2.set_xlabel('Feature Importance', fontweight='bold')
            ax2.set_title('🔍 Feature Importance')
        else:
            # Show confusion matrix instead
            if 'confusion_matrix' in results:
                cm = results['confusion_matrix']
                im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
                ax2.set_title('🎯 Confusion Matrix (Temporal)')
                
                # Add text annotations
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax2.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center',
                               fontweight='bold', color='white' if cm[i, j] > 0.5 else 'black')
                
                ax2.set_xlabel('Predicted', fontweight='bold')
                ax2.set_ylabel('Actual', fontweight='bold')
                ax2.set_xticks([0, 1])
                ax2.set_yticks([0, 1])
                ax2.set_xticklabels(['OFF', 'ON'])
                ax2.set_yticklabels(['OFF', 'ON'])
            else:
                # Default message
                ax2.text(0.5, 0.5, 'Additional analysis\ncoming soon!', 
                        ha='center', va='center', transform=ax2.transAxes,
                        fontsize=14, fontweight='bold')
                ax2.set_title('🔬 Extended Analysis')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_glm_kernels(self, glm_results: Dict,
                        figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        Visualize GLM model kernels and components.
        """
        if not glm_results:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
            ax.text(0.5, 0.5, 'No GLM results available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return fig
        
        fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor=self.bg_color)
        fig.suptitle(f'🔬 GLM Model Analysis - Unit {glm_results["unit_id"]}', 
                    fontsize=16, fontweight='bold')
        
        coeffs = glm_results['coefficients']
        
        # 1. Stimulus kernel
        if len(coeffs) > 0:
            axes[0].bar(range(len(coeffs[:10])), coeffs[:10], 
                       color=self.colors['neural'], alpha=0.8, edgecolor='white')
            axes[0].set_xlabel('Feature Index', fontweight='bold')
            axes[0].set_ylabel('Coefficient', fontweight='bold')
            axes[0].set_title('📊 Model Coefficients')
            axes[0].grid(True, alpha=0.3)
        
        # 2. Model performance
        performance_data = [
            glm_results.get('deviance_explained', 0),
            1 - glm_results.get('deviance_explained', 0)
        ]
        performance_labels = ['Explained', 'Unexplained']
        performance_colors = [self.colors['quality_good'], self.colors['quality_bad']]
        
        axes[1].pie(performance_data, labels=performance_labels, colors=performance_colors,
                   autopct='%1.1f%%', startangle=90, textprops={'fontweight': 'bold'})
        axes[1].set_title('📈 Deviance Explained')
        
        # 3. Overdispersion check
        overdispersion = glm_results.get('overdispersion', 1.0)
        axes[2].bar(['Overdispersion'], [overdispersion], 
                   color=self.colors['neural'], alpha=0.8, edgecolor='white')
        axes[2].axhline(1.0, color='yellow', linestyle='--', alpha=0.7, label='Poisson')
        axes[2].set_ylabel('Overdispersion Factor', fontweight='bold')
        axes[2].set_title('⚡ Model Fit Quality')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Add model stats
        stats_text = f'Features: {glm_results.get("n_features", "N/A")}\n'
        stats_text += f'Observations: {glm_results.get("n_observations", "N/A")}\n'
        stats_text += f'Deviance Explained: {glm_results.get("deviance_explained", 0):.1%}'
        
        axes[2].text(0.02, 0.98, stats_text, transform=axes[2].transAxes,
                    bbox=dict(boxstyle='round', facecolor=self.colors['neural'], alpha=0.8),
                    verticalalignment='top', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        return fig

    def create_summary_dashboard(self, trials_data: List[Dict], unit_stats: Dict,
                               good_units: List[int], response_stats: Dict,
                               decoding_results: Dict,
                               figsize: Tuple[int, int] = (20, 16)) -> plt.Figure:
        """
        Create a comprehensive dashboard summary.
        """
        fig = plt.figure(figsize=figsize, facecolor=self.bg_color)
        gs = GridSpec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('🧠 Neural Spike Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Data overview (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        overview_data = {
            'Total Trials': len(trials_data),
            'Total Units': len(unit_stats),
            'Good Units': len(good_units),
            'Responsive Units': len([u for u in response_stats.keys() 
                                   if response_stats[u]['responsive']])
        }
        
        bars = ax1.bar(overview_data.keys(), overview_data.values(),
                      color=[self.colors['neural'], self.colors['on'], 
                            self.colors['quality_good'], self.colors['spike']],
                      alpha=0.8, edgecolor='white', linewidth=2)
        ax1.set_title('📊 Dataset Overview', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, value in zip(bars, overview_data.values()):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Quality distribution (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        firing_rates = [unit_stats[u]['firing_rate'] for u in unit_stats.keys()]
        colors = [self.colors['quality_good'] if u in good_units 
                 else self.colors['quality_bad'] for u in unit_stats.keys()]
        
        ax2.scatter(list(unit_stats.keys()), firing_rates, c=colors, s=60, alpha=0.8)
        ax2.set_ylabel('Firing Rate (Hz)', fontweight='bold')
        ax2.set_title('🎯 Unit Quality', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Example raster (middle-left, spans 2x2)
        ax3 = fig.add_subplot(gs[1:3, :2])
        if good_units:
            example_unit = good_units[0]
            # Mini raster plot
            on_trials, off_trials = [], []
            
            for i, trial in enumerate(trials_data):
                unit_data = next((u for u in trial['units'] if u['unit_id'] == example_unit), None)
                if unit_data is None:
                    continue
                    
                spikes = unit_data['spike_times']
                if trial['event_label'] == 'ON':
                    on_trials.append((i, spikes))
                else:
                    off_trials.append((i, spikes))
            
            # Plot spikes
            for idx, (trial_idx, spikes) in enumerate(on_trials[:20]):  # Limit display
                ax3.vlines(spikes, idx - 0.4, idx + 0.4, 
                          colors=self.colors['on'], alpha=0.8, linewidth=1)
            
            for idx, (trial_idx, spikes) in enumerate(off_trials[:20]):
                y_pos = len(on_trials[:20]) + idx
                ax3.vlines(spikes, y_pos - 0.4, y_pos + 0.4,
                          colors=self.colors['off'], alpha=0.8, linewidth=1)
            
            ax3.axvline(0, color='yellow', linewidth=2, alpha=0.8, linestyle='--')
            ax3.set_xlim(-1, 3)
            ax3.set_xlabel('Time from Stimulus (s)', fontweight='bold')
            ax3.set_ylabel('Trial #', fontweight='bold')
            ax3.set_title(f'🔥 Example Unit {example_unit}', fontweight='bold')
        
        # 4. Response comparison (middle-right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if response_stats:
            responsive_units = [u for u in response_stats.keys() 
                              if response_stats[u]['responsive']]
            resp_count = len(responsive_units)
            total_count = len(response_stats)
            
            pie_data = [resp_count, total_count - resp_count]
            pie_labels = ['Responsive', 'Non-responsive']
            pie_colors = [self.colors['quality_good'], self.colors['quality_bad']]
            
            ax4.pie(pie_data, labels=pie_labels, colors=pie_colors,
                   autopct='%1.0f%%', startangle=90, textprops={'fontweight': 'bold'})
            ax4.set_title('📈 Response Analysis', fontweight='bold')
        
        # 5. Decoding results (bottom-right)
        ax5 = fig.add_subplot(gs[2, 2:])
        if decoding_results:
            methods = ['Rate', 'Temporal', 'Shuffle']
            accuracies = [decoding_results.get('rate_accuracy', 0), 
                         decoding_results.get('temporal_accuracy', 0),
                         decoding_results.get('shuffle_accuracy', 0)]
            colors = [self.colors['on'], self.colors['neural'], '#666666']
            
            bars = ax5.bar(methods, accuracies, color=colors, alpha=0.8, 
                          edgecolor='white', linewidth=1)
            ax5.set_ylabel('Accuracy', fontweight='bold')
            ax5.set_title('🎯 Decoding Performance', fontweight='bold')
            ax5.set_ylim(0, 1)
            ax5.grid(True, alpha=0.3)
            
            # Add values
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Analysis summary (bottom)
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # Create summary text
        summary_text = "🔬 ANALYSIS SUMMARY\n"
        summary_text += "=" * 50 + "\n"
        
        if decoding_results:
            rate_acc = decoding_results.get('rate_accuracy', 0)
            temp_acc = decoding_results.get('temporal_accuracy', 0)
            
            if temp_acc > rate_acc + 0.05:
                conclusion = "🎯 TEMPORAL CODING detected! Spike timing carries more information than rates."
            elif rate_acc > temp_acc + 0.05:
                conclusion = "📊 RATE CODING detected! Spike counts are sufficient for stimulus decoding."
            else:
                conclusion = "🤔 MIXED CODING detected! Both rate and timing contribute to information."
            
            summary_text += f"• {conclusion}\n"
            summary_text += f"• Rate-based accuracy: {rate_acc:.1%}\n"
            summary_text += f"• Temporal accuracy: {temp_acc:.1%}\n"
        
        summary_text += f"• {len(good_units)}/{len(unit_stats)} units passed quality control\n"
        if response_stats:
            resp_count = len([u for u in response_stats.keys() if response_stats[u]['responsive']])
            summary_text += f"• {resp_count}/{len(response_stats)} units are stimulus-responsive\n"
        
        ax6.text(0.5, 0.5, summary_text, ha='center', va='center',
                transform=ax6.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=1', facecolor=self.colors['neural'], 
                         alpha=0.8, edgecolor='white', linewidth=2))
        
        return fig

