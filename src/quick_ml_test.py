import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from spike_data_loader import NeuralPatternGenerator

class FastNeuralClassifier:
    """Fast classifier using only the most discriminative features"""
    
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def extract_key_features(self, spike_data):
        """Extract only the key features we know discriminate between patterns"""
        
        spike_trains = spike_data['spike_trains'][0]  # First trial
        trial_duration = 2.0
        
        # 1. Mean firing rate
        total_spikes = sum(len(spikes) for spikes in spike_trains)
        mean_firing_rate = total_spikes / (len(spike_trains) * trial_duration)
        
        # 2. Burst index (ratio of short ISIs)
        burst_index = 0
        total_isis = 0
        for spikes in spike_trains:
            if len(spikes) > 1:
                isis = np.diff(spikes)
                short_isis = np.sum(isis < 0.01)  # < 10ms
                burst_index += short_isis
                total_isis += len(isis)
        burst_index = burst_index / max(total_isis, 1)
        
        # 3. Regularity (inverse of ISI CV)
        cv_values = []
        for spikes in spike_trains:
            if len(spikes) > 2:
                isis = np.diff(spikes)
                if len(isis) > 1 and np.mean(isis) > 0:
                    cv = np.std(isis) / np.mean(isis)
                    cv_values.append(cv)
        mean_cv = np.mean(cv_values) if cv_values else 1.0
        regularity = 1 / (1 + mean_cv)
        
        # 4. Population synchrony (simplified)
        # Calculate population activity in 10ms bins
        time_bins = np.arange(0, trial_duration + 0.01, 0.01)
        pop_activity = np.zeros(len(time_bins) - 1)
        
        for spikes in spike_trains:
            if len(spikes) > 0:
                spike_counts, _ = np.histogram(spikes, bins=time_bins)
                pop_activity += spike_counts
        
        # Synchrony as coefficient of variation of population activity
        sync_index = np.std(pop_activity) / (np.mean(pop_activity) + 1e-6)
        
        # 5. Cross-correlation (simplified - just first few neurons)
        cross_corr = 0
        if len(spike_trains) > 1:
            # Convert first 5 neurons to binary vectors
            n_neurons = min(5, len(spike_trains))
            bin_size = 0.01
            n_bins = int(trial_duration / bin_size)
            
            binary_trains = []
            for i in range(n_neurons):
                spikes = spike_trains[i]
                binary_train = np.zeros(n_bins)
                for spike_time in spikes:
                    bin_idx = int(spike_time / bin_size)
                    if bin_idx < n_bins:
                        binary_train[bin_idx] = 1
                binary_trains.append(binary_train)
            
            # Average pairwise correlation
            correlations = []
            for i in range(n_neurons):
                for j in range(i+1, n_neurons):
                    if np.std(binary_trains[i]) > 0 and np.std(binary_trains[j]) > 0:
                        corr = np.corrcoef(binary_trains[i], binary_trains[j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
            
            cross_corr = np.mean(correlations) if correlations else 0
        
        return np.array([
            mean_firing_rate,
            burst_index,
            regularity,
            sync_index,
            cross_corr
        ])
    
    def generate_training_data(self, n_trials_per_class=20):
        """Generate training data with minimal trials"""
        
        print(f"Generating training data ({n_trials_per_class} trials per class)...")
        
        generator = NeuralPatternGenerator(n_neurons=15, trial_duration=2.0)
        
        X = []
        y = []
        
        # Class configurations
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
                'coding_type': 'rate',
                'oscillatory_power': 0.7,
                'population_synchrony': 0.8,
                'spike_regularity': 0.3,
                'pathological_bursting': 0.2
            },
            'Epileptiform': {
                'coding_type': 'temporal',
                'oscillatory_power': 0.4,
                'population_synchrony': 0.9,
                'spike_regularity': 0.5,
                'pathological_bursting': 0.7
            },
            'Mixed_Pathology': {
                'coding_type': 'mixed',
                'oscillatory_power': 0.6,
                'population_synchrony': 0.7,
                'spike_regularity': 0.4,
                'pathological_bursting': 0.4
            }
        }
        
        for class_name, config in configs.items():
            print(f"  Generating {class_name}...")
            
            for trial in range(n_trials_per_class):
                # Generate data
                data = generator.generate_synthetic_spikes(
                    n_stimuli=2, n_trials_per_stimulus=1, **config
                )
                
                # Extract features
                features = self.extract_key_features(data)
                
                X.append(features)
                y.append(class_name)
        
        return np.array(X), np.array(y)
    
    def train(self, n_trials_per_class=20):
        """Train the classifier"""
        
        print("Fast Neural Pattern Classifier Training")
        print("=" * 45)
        
        # Generate data
        X, y = self.generate_training_data(n_trials_per_class)
        
        print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train classifier
        print("Training Random Forest...")
        self.classifier.fit(X_train_scaled, y_train_encoded)
        
        # Evaluate
        train_score = self.classifier.score(X_train_scaled, y_train_encoded)
        test_score = self.classifier.score(X_test_scaled, y_test_encoded)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        # Detailed results
        y_pred = self.classifier.predict(X_test_scaled)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_labels))
        
        # Feature importance
        feature_names = ['Firing_Rate', 'Burst_Index', 'Regularity', 'Sync_Index', 'Cross_Corr']
        importances = self.classifier.feature_importances_
        
        print(f"\nFeature Importance:")
        for name, importance in zip(feature_names, importances):
            print(f"  {name}: {importance:.3f}")
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': dict(zip(feature_names, importances))
        }
    
    def predict(self, spike_data):
        """Predict pattern class"""
        
        if not self.is_trained:
            raise ValueError("Classifier must be trained first")
        
        # Extract features
        features = self.extract_key_features(spike_data)
        
        # Scale
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        # Convert to class name
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        # Create probability dict
        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_dict[class_name] = probabilities[i]
        
        return {
            'predicted_class': predicted_class,
            'confidence': np.max(probabilities),
            'probabilities': prob_dict,
            'features': features
        }

def test_fast_classifier():
    """Test the fast classifier"""
    
    # Train classifier
    classifier = FastNeuralClassifier()
    results = classifier.train(n_trials_per_class=25)
    
    print("\n" + "=" * 45)
    print("TESTING PREDICTIONS")
    print("=" * 45)
    
    # Test predictions
    generator = NeuralPatternGenerator(n_neurons=15, trial_duration=2.0)
    
    test_cases = [
        ('Healthy Pattern', {
            'coding_type': 'rate',
            'oscillatory_power': 0.0,
            'population_synchrony': 0.15,
            'spike_regularity': 0.8,
            'pathological_bursting': 0.0
        }),
        ('Parkinsonian Pattern', {
            'coding_type': 'rate',
            'oscillatory_power': 0.8,
            'population_synchrony': 0.9,
            'spike_regularity': 0.2,
            'pathological_bursting': 0.3
        }),
        ('Epileptiform Pattern', {
            'coding_type': 'temporal',
            'oscillatory_power': 0.3,
            'population_synchrony': 0.95,
            'spike_regularity': 0.4,
            'pathological_bursting': 0.8
        })
    ]
    
    for test_name, params in test_cases:
        print(f"\n{test_name}:")
        print("-" * 25)
        
        # Generate test data
        test_data = generator.generate_synthetic_spikes(
            n_stimuli=2, n_trials_per_stimulus=1, **params
        )
        
        # Predict
        result = classifier.predict(test_data)
        
        print(f"Predicted: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Features: {result['features']}")
        
        # Top 2 probabilities
        sorted_probs = sorted(result['probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        print("Top probabilities:")
        for i, (class_name, prob) in enumerate(sorted_probs[:2]):
            print(f"  {class_name}: {prob:.3f}")
    
    return classifier

if __name__ == "__main__":
    classifier = test_fast_classifier()