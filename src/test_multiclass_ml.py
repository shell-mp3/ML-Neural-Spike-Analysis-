from decoding_analysis import MultiClassNeuralDecoder
from spike_data_loader import NeuralPatternGenerator

def test_multiclass_classifier():
    """Test the complete multi-class ML pipeline"""
    
    print("Testing Multi-Class Neural Pattern Classifier")
    print("=" * 50)
    
    # Initialize decoder
    decoder = MultiClassNeuralDecoder(random_state=42)
    
    # Train the classifier
    print("Training classifier...")
    training_results = decoder.train(n_trials_per_class=40, verbose=True)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE - TESTING PREDICTIONS")
    print("=" * 50)
    
    # Test predictions on new data
    generator = NeuralPatternGenerator(n_neurons=20, trial_duration=2.0)
    
    # Test each pattern type
    test_patterns = {
        'Healthy Rate': {
            'coding_type': 'rate',
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
        }
    }
    
    print("\nTesting predictions on new data:")
    print("-" * 40)
    
    for pattern_name, params in test_patterns.items():
        print(f"\nTesting {pattern_name} pattern:")
        
        # Generate test data
        test_data = generator.generate_synthetic_spikes(
            n_stimuli=3, n_trials_per_stimulus=1, **params
        )
        
        # Make prediction
        result = decoder.predict(test_data, use_rf=True)
        
        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Expected: {pattern_name.replace(' ', '_')}")
        
        # Show top 3 probabilities
        sorted_probs = sorted(result['probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        print("  Top 3 probabilities:")
        for i, (class_name, prob) in enumerate(sorted_probs[:3]):
            print(f"    {i+1}. {class_name}: {prob:.3f}")
    
    # Test classification accuracy on held-out data
    test_classification_accuracy(decoder, generator)
    
    print("\nMulti-class testing complete!")

def test_classification_accuracy(decoder, generator):
    """Test classification accuracy on a held-out test set"""
    
    print(f"\n" + "=" * 50)
    print("HELD-OUT TEST SET EVALUATION")
    print("=" * 50)
    
    # Generate small test dataset
    test_patterns = []
    true_labels = []
    
    # Generate 10 samples of each class
    pattern_configs = {
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
            'oscillatory_power': 0.6,
            'population_synchrony': 0.7,
            'spike_regularity': 0.4,
            'pathological_bursting': 0.3
        },
        'Epileptiform': {
            'coding_type': 'temporal',
            'oscillatory_power': 0.3,
            'population_synchrony': 0.8,
            'spike_regularity': 0.6,
            'pathological_bursting': 0.6
        },
        'Mixed_Pathology': {
            'coding_type': 'mixed',
            'oscillatory_power': 0.5,
            'population_synchrony': 0.6,
            'spike_regularity': 0.5,
            'pathological_bursting': 0.4
        }
    }
    
    print("Generating test data...")
    for class_name, config in pattern_configs.items():
        for i in range(10):  # 10 samples per class
            data = generator.generate_synthetic_spikes(
                n_stimuli=3, n_trials_per_stimulus=1, **config
            )
            test_patterns.append(data)
            true_labels.append(class_name)
    
    # Make predictions
    print("Making predictions...")
    predictions = []
    confidences = []
    
    for i, test_data in enumerate(test_patterns):
        result = decoder.predict(test_data, use_rf=True)
        predictions.append(result['predicted_class'])
        confidences.append(result['confidence'])
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_patterns)} samples")
    
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
    accuracy = correct / len(true_labels)
    
    print(f"\nTest Set Results:")
    print(f"  Total samples: {len(true_labels)}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    
    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for class_name in pattern_configs.keys():
        class_true = [i for i, label in enumerate(true_labels) if label == class_name]
        class_correct = sum(1 for i in class_true if predictions[i] == true_labels[i])
        class_accuracy = class_correct / len(class_true) if class_true else 0
        print(f"  {class_name}: {class_accuracy:.3f} ({class_correct}/{len(class_true)})")

if __name__ == "__main__":
    import numpy as np
    test_multiclass_classifier()