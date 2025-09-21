import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from spike_data_loader import NeuralPatternGenerator

class OptimizedMultiClassDecoder:
    """Optimized multi-class neural pattern classifier with fast feature extraction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Multiple classifiers for comparison
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        
        self.lr_classifier = LogisticRegression(
            multi_class='ovr',
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000
        )
        
        # Could add more classifiers
        from sklearn.svm import SVC
        self.svm_classifier = SVC(
            kernel='rbf',
            class_weight='balanced',
            random_state=random_state,
            probability=True
        )
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Data generator
        self.data_generator = NeuralPatternGenerator()
        
        # Training state
        self.is_trained = False
        self.feature_names = [
            'mean_firing_rate', 'firing_rate_std', 'max_firing_rate',
            'burst_index', 'burst_frequency', 'mean_burst_duration',
            'regularity_index', 'mean_cv', 'fano_factor',
            'sync_index', 'sync_variance', 'cross_correlation',
            'population_entropy', 'timing_precision', 'response_latency',
            'parkinsonian_composite', 'epileptiform_composite', 'pathology_score'
        ]
        self.class_names = []
        self.best_classifier = None
        self.feature_importance = None
        
    def extract_optimized_features(self, spike_data):
        """Extract comprehensive but fast features"""
        
        spike_trains = spike_data['spike_trains'][0]  # First trial
        trial_duration = spike_data.get('trial_duration', 2.0)
        
        features = {}
        
        # === RATE FEATURES ===
        firing_rates = [len(spikes) / trial_duration for spikes in spike_trains]
        features['mean_firing_rate'] = np.mean(firing_rates)
        features['firing_rate_std'] = np.std(firing_rates)
        features['max_firing_rate'] = np.max(firing_rates) if firing_rates else 0
        
        # === BURST FEATURES ===
        burst_indices = []
        burst_durations = []
        total_bursts = 0
        
        for spikes in spike_trains:
            if len(spikes) > 3:
                isis = np.diff(spikes)
                
                # Burst detection: ISI < 10ms
                short_isis = np.sum(isis < 0.01)
                burst_index = short_isis / len(isis) if len(isis) > 0 else 0
                burst_indices.append(burst_index)
                
                # Detect burst events
                in_burst = False
                burst_start = 0
                for i, isi in enumerate(isis):
                    if isi < 0.01 and not in_burst:
                        in_burst = True
                        burst_start = spikes[i]
                    elif isi > 0.05 and in_burst:
                        in_burst = False
                        burst_duration = spikes[i] - burst_start
                        if burst_duration > 0.01:  # Minimum burst duration
                            burst_durations.append(burst_duration)
                            total_bursts += 1
        
        features['burst_index'] = np.mean(burst_indices) if burst_indices else 0
        features['burst_frequency'] = total_bursts / trial_duration
        features['mean_burst_duration'] = np.mean(burst_durations) if burst_durations else 0
        
        # === REGULARITY FEATURES ===
        cv_values = []
        fano_factors = []
        
        for spikes in spike_trains:
            if len(spikes) > 2:
                # ISI coefficient of variation
                isis = np.diff(spikes)
                if len(isis) > 1 and np.mean(isis) > 0:
                    cv = np.std(isis) / np.mean(isis)
                    cv_values.append(cv)
                
                # Fano factor (spike count variance/mean in windows)
                window_size = 0.1
                n_windows = int(trial_duration / window_size)
                counts = []
                for w in range(n_windows):
                    start_time = w * window_size
                    end_time = (w + 1) * window_size
                    count = np.sum((spikes >= start_time) & (spikes < end_time))
                    counts.append(count)
                
                if len(counts) > 1 and np.mean(counts) > 0:
                    fano = np.var(counts) / np.mean(counts)
                    fano_factors.append(fano)
        
        features['regularity_index'] = 1 / (1 + np.mean(cv_values)) if cv_values else 0.5
        features['mean_cv'] = np.mean(cv_values) if cv_values else 1.0
        features['fano_factor'] = np.mean(fano_factors) if fano_factors else 1.0
        
        # === SYNCHRONY FEATURES ===
        # Population activity analysis
        bin_size = 0.005  # 5ms bins
        time_bins = np.arange(0, trial_duration + bin_size, bin_size)
        pop_activity = np.zeros(len(time_bins) - 1)
        
        for spikes in spike_trains:
            if len(spikes) > 0:
                spike_counts, _ = np.histogram(spikes, bins=time_bins)
                pop_activity += spike_counts
        
        # Synchrony indices
        features['sync_index'] = np.std(pop_activity) / (np.mean(pop_activity) + 1e-6)
        features['sync_variance'] = np.var(pop_activity)
        
        # Cross-correlation (sample of neuron pairs)
        correlations = []
        n_sample = min(10, len(spike_trains))  # Sample first 10 neurons
        
        if n_sample > 1:
            # Convert to binary trains
            binary_trains = []
            for i in range(n_sample):
                spikes = spike_trains[i]
                binary_train = np.zeros(len(pop_activity))
                for spike_time in spikes:
                    bin_idx = int(spike_time / bin_size)
                    if bin_idx < len(binary_train):
                        binary_train[bin_idx] = 1
                binary_trains.append(binary_train)
            
            # Pairwise correlations (sample pairs to avoid n^2 complexity)
            n_pairs = min(20, n_sample * (n_sample - 1) // 2)
            pair_indices = np.random.choice(range(n_sample * (n_sample - 1) // 2), 
                                          size=min(n_pairs, n_sample * (n_sample - 1) // 2), 
                                          replace=False)
            
            pair_count = 0
            for i in range(n_sample):
                for j in range(i + 1, n_sample):
                    if pair_count in pair_indices:
                        if np.std(binary_trains[i]) > 0 and np.std(binary_trains[j]) > 0:
                            corr = np.corrcoef(binary_trains[i], binary_trains[j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)
                    pair_count += 1
        
        features['cross_correlation'] = np.mean(correlations) if correlations else 0
        
        # === TEMPORAL FEATURES ===
        # Population entropy
        if len(pop_activity) > 0:
            pop_hist = np.histogram(pop_activity, bins=10)[0]
            pop_prob = pop_hist / np.sum(pop_hist)
            features['population_entropy'] = -np.sum(pop_prob * np.log(pop_prob + 1e-10))
        else:
            features['population_entropy'] = 0
        
        # Timing precision (simplified)
        if 'stimulus_times' in spike_data and spike_data['stimulus_times']:
            stim_time = spike_data['stimulus_times'][0]
            response_times = []
            
            for spikes in spike_trains:
                post_stim = spikes[spikes > stim_time]
                if len(post_stim) > 0:
                    response_times.append(post_stim[0] - stim_time)
            
            features['timing_precision'] = 1 / (1 + np.std(response_times)) if len(response_times) > 1 else 0
            features['response_latency'] = np.mean(response_times) if response_times else 0
        else:
            features['timing_precision'] = 0
            features['response_latency'] = 0
        
        # === COMPOSITE PATHOLOGY INDICES ===
        # Parkinsonian signature: high synchrony + irregular firing + some bursting
        features['parkinsonian_composite'] = (
            features['sync_index'] * 
            (1 - features['regularity_index']) * 
            (1 + features['burst_index'])
        )
        
        # Epileptiform signature: very high synchrony + high bursting + irregular
        features['epileptiform_composite'] = (
            features['sync_index'] * 
            features['burst_index'] * 
            features['cross_correlation']
        )
        
        # General pathology score
        features['pathology_score'] = np.mean([
            features['parkinsonian_composite'],
            features['epileptiform_composite']
        ])
        
        # Convert to array in consistent order
        feature_vector = [features[name] for name in self.feature_names]
        return np.array(feature_vector)
    
    def generate_training_data(self, n_trials_per_class=50, verbose=True):
        """Generate training dataset with optimized generation"""
        
        if verbose:
            print(f"Generating training data ({n_trials_per_class} trials per class)...")
        
        X = []
        y = []
        
        # Optimized class configurations
        configs = {
            'Healthy_Rate': {
                'coding_type': 'rate',
                'oscillatory_power': 0.0,
                'population_synchrony': 0.15,
                'spike_regularity': 0.8,
                'pathological_bursting': 0.0
            },
            'Healthy_Temporal': {
                'coding_type': 'temporal',
                'oscillatory_power': 0.0,
                'population_synchrony': 0.15,
                'spike_regularity': 0.8,
                'pathological_bursting': 0.0
            },
            'Parkinsonian': {
                'coding_type': np.random.choice(['rate', 'temporal']),
                'oscillatory_power': lambda: np.random.uniform(0.5, 0.9),
                'population_synchrony': lambda: np.random.uniform(0.6, 0.9),
                'spike_regularity': lambda: np.random.uniform(0.2, 0.5),
                'pathological_bursting': lambda: np.random.uniform(0.1, 0.4)
            },
            'Epileptiform': {
                'coding_type': np.random.choice(['rate', 'temporal']),
                'oscillatory_power': lambda: np.random.uniform(0.3, 0.7),
                'population_synchrony': lambda: np.random.uniform(0.7, 1.0),
                'spike_regularity': lambda: np.random.uniform(0.3, 0.7),
                'pathological_bursting': lambda: np.random.uniform(0.4, 0.8)
            },
            'Mixed_Pathology': {
                'coding_type': np.random.choice(['rate', 'temporal', 'mixed']),
                'oscillatory_power': lambda: np.random.uniform(0.4, 0.8),
                'population_synchrony': lambda: np.random.uniform(0.5, 0.8),
                'spike_regularity': lambda: np.random.uniform(0.2, 0.6),
                'pathological_bursting': lambda: np.random.uniform(0.2, 0.6)
            }
        }
        
        for class_name, base_config in configs.items():
            if verbose:
                print(f"  Generating {class_name}...")
            
            for trial in range(n_trials_per_class):
                # Randomize parameters for pathological classes
                config = {}
                for key, value in base_config.items():
                    if callable(value):
                        config[key] = value()
                    else:
                        config[key] = value
                
                # Generate spike data
                spike_data = self.data_generator.generate_synthetic_spikes(
                    n_stimuli=3, n_trials_per_stimulus=1, **config
                )
                
                # Extract features
                features = self.extract_optimized_features(spike_data)
                
                X.append(features)
                y.append(class_name)
                
                if verbose and (trial + 1) % 25 == 0:
                    print(f"    {trial + 1}/{n_trials_per_class} completed")
        
        return np.array(X), np.array(y)
    
    def train(self, n_trials_per_class=50, test_size=0.2, verbose=True):
        """Train multiple classifiers and select the best"""
        
        if verbose:
            print("=" * 60)
            print("OPTIMIZED MULTI-CLASS NEURAL PATTERN CLASSIFIER")
            print("=" * 60)
        
        # Generate training data
        X, y = self.generate_training_data(n_trials_per_class, verbose)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        if verbose:
            print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Classes: {len(self.class_names)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and evaluate multiple classifiers
        classifiers = {
            'Random Forest': self.rf_classifier,
            'Logistic Regression': self.lr_classifier,
            'SVM': self.svm_classifier
        }
        
        results = {}
        
        for name, classifier in classifiers.items():
            if verbose:
                print(f"\nTraining {name}...")
            
            # Train
            classifier.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = classifier.score(X_train_scaled, y_train)
            test_score = classifier.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(
                classifier, X_train_scaled, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='accuracy'
            )
            
            results[name] = {
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            if verbose:
                print(f"  Train: {train_score:.3f}, Test: {test_score:.3f}")
                print(f"  CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Select best classifier based on CV score
        best_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.best_classifier = classifiers[best_name]
        
        if verbose:
            print(f"\nBest classifier: {best_name}")
        
        # Store feature importance (if available)
        if hasattr(self.best_classifier, 'feature_importances_'):
            self.feature_importance = self.best_classifier.feature_importances_
        
        # Final evaluation
        y_pred = self.best_classifier.predict(X_test_scaled)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        
        if verbose:
            print(f"\nFinal Classification Report ({best_name}):")
            print(classification_report(y_test_labels, y_pred_labels))
            
            # Plot results
            self._plot_confusion_matrix(y_test, y_pred)
            if self.feature_importance is not None:
                self._plot_feature_importance()
        
        self.is_trained = True
        
        return {
            'best_classifier': best_name,
            'results': results,
            'final_accuracy': results[best_name]['test_score']
        }
    
    def predict(self, spike_data):
        """Predict neural pattern class"""
        
        if not self.is_trained:
            raise ValueError("Classifier must be trained first")
        
        # Extract features
        features = self.extract_optimized_features(spike_data)
        
        # Scale
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.best_classifier.predict(features_scaled)[0]
        probabilities = self.best_classifier.predict_proba(features_scaled)[0]
        
        # Convert to class name
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        # Create probability dict
        prob_dict = {}
        for i, class_name in enumerate(self.class_names):
            prob_dict[class_name] = probabilities[i]
        
        return {
            'predicted_class': predicted_class,
            'confidence': np.max(probabilities),
            'probabilities': prob_dict,
            'features': dict(zip(self.feature_names, features))
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.show()
    
    def _plot_feature_importance(self, top_n=12):
        """Plot feature importance"""
        if self.feature_importance is None:
            return
        
        indices = np.argsort(self.feature_importance)[::-1][:top_n]
        top_features = [self.feature_names[i] for i in indices]
        top_importance = self.feature_importance[indices]
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(top_importance[i] / np.max(top_importance)))
        
        plt.tight_layout()
        plt.show()

# Test function
def test_optimized_decoder():
    """Test the optimized decoder"""
    
    # Train
    decoder = OptimizedMultiClassDecoder()
    results = decoder.train(n_trials_per_class=40, verbose=True)
    
    # Test predictions
    print("\n" + "=" * 50)
    print("TESTING PREDICTIONS")
    print("=" * 50)
    
    generator = NeuralPatternGenerator()
    
    test_data = generator.generate_synthetic_spikes(
        coding_type='rate',
        oscillatory_power=0.8,
        population_synchrony=0.9,
        spike_regularity=0.2,
        pathological_bursting=0.4
    )
    
    result = decoder.predict(test_data)
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    return decoder

if __name__ == "__main__":
    decoder = test_optimized_decoder()