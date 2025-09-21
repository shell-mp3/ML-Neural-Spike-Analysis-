"""
Simple Data Generator - Just Run This!
=====================================

This script creates synthetic neural data files in your data/ folder.
Just run: python create_test_data.py
"""

import sys
import os
sys.path.append('src')

# Import the data generator
from generate_synthetic_data import generate_all_scenarios, generate_quick_test_data

def main():
    print("🧠 CREATING SYNTHETIC NEURAL DATA")
    print("=" * 50)
    
    # Create data folder if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Generating realistic synthetic neural spike data...")
    print("This will create several test datasets for you to analyze!")
    print()
    
    try:
        # Generate comprehensive dataset
        generated_files = generate_all_scenarios()
        
        print(f"\n🎉 SUCCESS! Generated synthetic data files.")
        print(f"\n📁 Files created in 'data/' folder:")
        
        # List the files
        data_dir = 'data'
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith(('.npz', '.pkl', '.csv'))]
            for i, file in enumerate(files, 1):
                print(f"   {i}. {file}")
        
        print(f"\n🚀 READY TO ANALYZE!")
        print(f"   1. Open: jupyter notebook notebooks/main_analysis.ipynb")
        print(f"   2. In the data loading cell, replace the synthetic data generation with:")
        print(f"      spike_data = loader.load_spike_data('data/neural_data_mixed_coding.npz')")
        print(f"   3. Run all cells and enjoy the analysis!")
        
        print(f"\n💡 TIP: Try different datasets to see how coding strategies affect results:")
        print(f"   • neural_data_rate_coding.npz - Rate coding dominant")
        print(f"   • neural_data_temporal_coding.npz - Temporal coding dominant") 
        print(f"   • neural_data_mixed_coding.npz - Both rate and temporal")
        print(f"   • neural_data_no_coding.npz - No coding (control)")
        
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        print(f"🔄 Trying to generate simple test data instead...")
        
        try:
            generate_quick_test_data()
            print("✅ Created basic test data!")
        except Exception as e2:
            print(f"❌ Failed to create any data: {e2}")
            print("Please check your Python environment and try again.")

if __name__ == "__main__":
    main()