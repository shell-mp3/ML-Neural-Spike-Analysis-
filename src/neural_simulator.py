import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import time

# Import our enhanced modules
from spike_data_loader import NeuralPatternGenerator
from decoding_analysis import OptimizedMultiClassDecoder

# Configure page
st.set_page_config(
    page_title="Neural Code Research Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling with better contrast
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .subtitle {
        font-size: 1.2rem;
        color: #444;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Step boxes with better contrast */
    .step-box {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .step-number {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Content boxes with white backgrounds */
    .metric-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #333;
    }
    .warning-box {
        background-color: #fff9e6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        color: #856404;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        color: #155724;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
        color: #004085;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Ensure all text is readable */
    .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #333 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #4a90e2;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: 600;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #357abd;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'decoder' not in st.session_state:
        st.session_state.decoder = None
    if 'decoder_trained' not in st.session_state:
        st.session_state.decoder_trained = False
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'user_mode' not in st.session_state:
        st.session_state.user_mode = 'getting_started'

def show_header():
    """Show main header and description"""
    st.markdown('<div class="main-header">🧠 Neural Code Research Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Neural Pattern Analysis for Research & Education</div>', unsafe_allow_html=True)
    
    # Quick explanation
    st.markdown("""
    <div class="info-box">
        <h3>🎯 What This Tool Does</h3>
        <p><strong>Generate realistic neural activity patterns</strong> and use AI to automatically detect:</p>
        <ul>
            <li>🟢 <strong>Healthy brain patterns</strong> (normal neural activity)</li>
            <li>🟡 <strong>Parkinson's-like patterns</strong> (beta oscillations, irregular firing)</li>
            <li>🔴 <strong>Epilepsy-like patterns</strong> (hypersynchrony, bursting activity)</li>
        </ul>
        <p><em>Perfect for neuroscience students, researchers, and anyone curious about how AI can analyze brain activity!</em></p>
    </div>
    """, unsafe_allow_html=True)

def show_disclaimer():
    """Show research disclaimer"""
    with st.expander("⚠️ Important: Educational Use Only"):
        st.markdown("""
        **This tool is for research and educational purposes only.**
        - All neural data is computer-generated (synthetic)
        - Not for medical diagnosis or clinical decisions
        - Designed to teach pattern recognition concepts
        """)

def show_getting_started():
    """Show getting started guide"""
    
    st.markdown("## 🚀 Getting Started")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="step-box">
            <div class="step-number">1️⃣</div>
            <h3>Train the AI Brain</h3>
            <p>First, we need to teach our AI how to recognize different neural patterns. This takes 2-3 minutes and only needs to be done once.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.decoder_trained:
            if st.button("🤖 Train AI to Recognize Patterns", type="primary", use_container_width=True):
                train_classifier()
        else:
            st.markdown("""
            <div class="success-box">
                ✅ <strong>AI is trained and ready!</strong><br>
                You can now generate and analyze neural patterns.
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Retrain AI", use_container_width=True):
                retrain_classifier()
    
    with col2:
        st.markdown("""
        <div class="step-box">
            <div class="step-number">2️⃣</div>
            <h3>Generate & Analyze</h3>
            <p>Once the AI is trained, choose a brain pattern type and watch the AI instantly identify what kind of neural activity it is!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.decoder_trained:
            if st.button("📊 Go to Pattern Analysis", type="primary", use_container_width=True):
                st.session_state.user_mode = 'analysis'
                st.rerun()
        else:
            st.info("Train the AI first, then this button will become active!")
    
    # Show training results if available
    if st.session_state.training_results:
        show_training_summary()

def show_analysis_mode():
    """Show the main analysis interface"""
    
    # Mode switcher
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("⬅️ Back to Getting Started", use_container_width=True):
            st.session_state.user_mode = 'getting_started'
            st.rerun()
    
    st.markdown("## 🧪 Neural Pattern Analysis")
    
    # Pattern selection (simplified)
    st.markdown("### 1️⃣ Choose a Brain Pattern Type")
    
    pattern_descriptions = {
        "Healthy Rate": {
            "description": "Normal healthy brain activity with rate-based information encoding",
            "emoji": "😊",
            "expected": "Should be classified as 'Healthy'",
            "color": "#28a745",
            "bg_color": "#28a74510"
        },
        "Healthy Temporal": {
            "description": "Normal healthy brain activity with timing-based information encoding", 
            "emoji": "😊",
            "expected": "Should be classified as 'Healthy'",
            "color": "#20c997", 
            "bg_color": "#20c99710"
        },
        "Parkinsonian": {
            "description": "Parkinson's disease-like patterns with beta oscillations and irregular firing",
            "emoji": "🤝",
            "expected": "Should be classified as 'Parkinsonian'",
            "color": "#fd7e14",
            "bg_color": "#fd7e1410"
        },
        "Epileptiform": {
            "description": "Epilepsy-like patterns with hypersynchronous bursting activity",
            "emoji": "⚡",
            "expected": "Should be classified as 'Epileptiform'",
            "color": "#dc3545",
            "bg_color": "#dc354510"
        },
        "Mixed Pathology": {
            "description": "Complex patterns with multiple pathological features",
            "emoji": "🔬",
            "expected": "Should be classified as 'Mixed Pathology'",
            "color": "#6f42c1",
            "bg_color": "#6f42c110"
        }
    }
    
    # Custom selectbox styling
    pattern_type = st.selectbox(
        "Select Pattern Type:",
        options=list(pattern_descriptions.keys()),
        format_func=lambda x: f"{pattern_descriptions[x]['emoji']} {x}"
    )
    
    # Show detailed description with color coding
    selected_info = pattern_descriptions[pattern_type]
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {selected_info['bg_color']} 0%, {selected_info['color']}05 100%); 
                padding: 1.5rem; border-radius: 10px; border-left: 5px solid {selected_info['color']}; 
                margin: 1rem 0; color: #333;">
        <h4 style="margin: 0 0 0.5rem 0; color: {selected_info['color']};">
            {selected_info['emoji']} {pattern_type}
        </h4>
        <p style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">
            {selected_info['description']}
        </p>
        <p style="margin: 0; font-style: italic; color: #666;">
            💡 <strong>Expected Result:</strong> {selected_info['expected']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced options (collapsed by default)
    with st.expander("🔧 Advanced Options (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            n_neurons = st.slider("Number of Neurons", 15, 40, 20)
            trial_duration = st.slider("Recording Duration (seconds)", 1.0, 3.0, 2.0, 0.5)
        with col2:
            n_stimuli = st.slider("Number of Stimuli", 3, 8, 5)
            base_firing_rate = st.slider("Base Firing Rate (Hz)", 8.0, 20.0, 12.0)
    
    # Set default values if expander is not used
    if 'n_neurons' not in locals():
        n_neurons = 20
        trial_duration = 2.0
        n_stimuli = 5
        base_firing_rate = 12.0
    
    # Generate button
    st.markdown("### 2️⃣ Generate Pattern and See AI Analysis")
    
    if st.button("🎯 Generate Neural Pattern & Analyze", type="primary", use_container_width=True):
        generate_and_analyze_pattern_simple(pattern_type, n_neurons, trial_duration, n_stimuli, base_firing_rate)
    
    # Show results if available
    if st.session_state.current_data:
        show_results()

def generate_and_analyze_pattern_simple(pattern_type, n_neurons, trial_duration, n_stimuli, base_firing_rate):
    """Generate and analyze pattern with simplified interface"""
    
    # Preset configurations
    presets = {
       "Healthy Rate": {
        'coding_type': 'rate',
        'oscillatory_power': 0.0,
        'population_synchrony': 0.05,    # Much lower
        'spike_regularity': 0.95,        # Much higher
        'pathological_bursting': 0.0
    },
        
        "Healthy Temporal": {
            'coding_type': 'temporal',
            'oscillatory_power': 0.0,
            'population_synchrony': 0.15,
            'spike_regularity': 0.8,
            'pathological_bursting': 0.0
        },
        "Parkinsonian": {
        'coding_type': 'rate',
        'oscillatory_power': 0.9,        # Much higher
        'population_synchrony': 0.9,     # Much higher
        'spike_regularity': 0.1,         # Much lower
        'pathological_bursting': 0.4
         },
          "Epileptiform": {
        'coding_type': 'temporal',
        'oscillatory_power': 0.2,        # Low (different from Parkinson's)
        'population_synchrony': 1.0,     # Maximum
        'spike_regularity': 0.2,         # Low
        'pathological_bursting': 0.9     # Maximum
    },
        "Mixed Pathology": {
            'coding_type': 'mixed',
            'oscillatory_power': 0.6,
            'population_synchrony': 0.7,
            'spike_regularity': 0.4,
            'pathological_bursting': 0.4
        }
    }
    
    with st.spinner(f"🧠 Generating {pattern_type} neural pattern..."):
        # Generate pattern
        generator = NeuralPatternGenerator(n_neurons=n_neurons, trial_duration=trial_duration)
        
        preset = presets[pattern_type]
        spike_data = generator.generate_synthetic_spikes(
            n_stimuli=n_stimuli,
            n_trials_per_stimulus=1,
            base_firing_rate=base_firing_rate,
            **preset
        )
        
        st.session_state.current_data = spike_data
        
        # Analyze with AI
        if st.session_state.decoder_trained:
            prediction = st.session_state.decoder.predict(spike_data)
            st.session_state.prediction_results = prediction
        
    st.success("✅ Pattern generated and analyzed!")
    st.rerun()

def show_results():
    """Show analysis results in a clear, user-friendly way with explanations"""
    
    st.markdown("### 🎯 AI Analysis Results")
    
    if st.session_state.prediction_results:
        prediction = st.session_state.prediction_results
        
        # Main result - big and clear with color coding
        col1, col2 = st.columns(2)
        
        with col1:
            # Determine emoji and color based on prediction
            prediction_info = {
                'Healthy_Rate': {'emoji': '😊', 'color': '#28a745', 'category': 'Healthy'},
                'Healthy_Temporal': {'emoji': '😊', 'color': '#28a745', 'category': 'Healthy'}, 
                'Parkinsonian': {'emoji': '🤝', 'color': '#fd7e14', 'category': 'Pathological'},
                'Epileptiform': {'emoji': '⚡', 'color': '#dc3545', 'category': 'Pathological'},
                'Mixed_Pathology': {'emoji': '🔬', 'color': '#6f42c1', 'category': 'Complex'}
            }
            
            info = prediction_info.get(prediction['predicted_class'], {'emoji': '🧠', 'color': '#6c757d', 'category': 'Unknown'})
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {info['color']}15 0%, {info['color']}05 100%); 
                        padding: 2rem; border-radius: 15px; border-left: 6px solid {info['color']}; 
                        text-align: center; margin: 1rem 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h1 style="margin: 0; color: {info['color']}; font-size: 3rem;">{info['emoji']}</h1>
                <h3 style="margin: 0.5rem 0; color: #333;">AI Detected:</h3>
                <h2 style="margin: 0; color: {info['color']}; font-size: 2rem; font-weight: bold;">
                    {prediction['predicted_class'].replace('_', ' ')}
                </h2>
                <p style="margin: 0.5rem 0; color: #666; font-size: 1.1rem;">
                    <strong>Category:</strong> {info['category']} Pattern
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence = prediction['confidence']
            if confidence > 0.8:
                conf_color = "#28a745"
                conf_bg = "#28a74515"
                message = "Very Confident"
                icon = "🎯"
            elif confidence > 0.6:
                conf_color = "#fd7e14"
                conf_bg = "#fd7e1415"
                message = "Moderately Confident"
                icon = "🤔"
            else:
                conf_color = "#dc3545"
                conf_bg = "#dc354515"
                message = "Uncertain"
                icon = "❓"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {conf_bg} 0%, {conf_color}05 100%); 
                        padding: 2rem; border-radius: 15px; border-left: 6px solid {conf_color}; 
                        text-align: center; margin: 1rem 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h1 style="margin: 0; color: {conf_color}; font-size: 3rem;">{icon}</h1>
                <h3 style="margin: 0.5rem 0; color: #333;">Confidence Level:</h3>
                <h2 style="margin: 0; color: {conf_color}; font-size: 2rem; font-weight: bold;">
                    {confidence:.1%}
                </h2>
                <p style="margin: 0.5rem 0; color: #666; font-size: 1.1rem;">
                    <strong>{message}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed explanation based on the result
        show_result_explanation(prediction)
        
        # Detailed probabilities with better visualization
        st.markdown("### 📊 AI Confidence Breakdown")
        st.markdown("*How confident is the AI about each possible pattern type?*")
        
        prob_df = pd.DataFrame(list(prediction['probabilities'].items()), 
                              columns=['Pattern Type', 'Probability'])
        prob_df['Pattern Type'] = prob_df['Pattern Type'].str.replace('_', ' ')
        prob_df = prob_df.sort_values('Probability', ascending=False)
        
        # Create a colorful bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color scheme for different pattern types
        color_map = {
            'Healthy Rate': '#28a745',
            'Healthy Temporal': '#20c997', 
            'Parkinsonian': '#fd7e14',
            'Epileptiform': '#dc3545',
            'Mixed Pathology': '#6f42c1'
        }
        
        colors = [color_map.get(pattern, '#6c757d') for pattern in prob_df['Pattern Type']]
        
        bars = ax.bar(prob_df['Pattern Type'], prob_df['Probability'], color=colors, alpha=0.8)
        
        # Add percentage labels on bars
        for bar, prob in zip(bars, prob_df['Probability']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Highlight the winner
        bars[0].set_alpha(1.0)
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2)
        
        ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax.set_title('AI Confidence for Each Pattern Type', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(prob_df['Probability']) * 1.2)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add a subtle background gradient
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Show visualizations
    show_visualizations_simple()

def show_result_explanation(prediction):
    """Provide detailed explanation based on the prediction result"""
    
    predicted_class = prediction['predicted_class']
    confidence = prediction['confidence']
    probabilities = prediction['probabilities']
    
    # Get the second highest probability for comparison
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    second_choice = sorted_probs[1] if len(sorted_probs) > 1 else None
    
    st.markdown("### 🔍 What Does This Result Mean?")
    
    # Generate explanation based on the specific case
    explanation = ""
    explanation_color = "#e7f3ff"
    
    if predicted_class in ['Healthy_Rate', 'Healthy_Temporal']:
        if confidence > 0.8:
            explanation = f"""
            **Excellent! Clear Healthy Pattern Detected** ✅
            
            The AI is very confident this shows **normal, healthy brain activity**. This means:
            - Regular, organized neural firing patterns
            - Appropriate synchronization levels  
            - No signs of pathological activity
            - Typical of healthy brain function
            
            **Why so confident?** The neural features clearly match healthy patterns with minimal ambiguity.
            """
            explanation_color = "#d4edda"
        else:
            explanation = f"""
            **Likely Healthy, But Some Mixed Features** ⚠️
            
            The AI thinks this is probably healthy, but detected some **ambiguous characteristics**:
            - Mostly normal firing patterns
            - Some features that could suggest mild irregularities
            - Still within healthy range, but not textbook-perfect
            
            **This is realistic!** Even healthy brains show natural variation.
            """
            explanation_color = "#fff3cd"
    
    elif predicted_class == 'Parkinsonian':
        if confidence > 0.8:
            explanation = f"""
            **Clear Parkinson's-like Pattern Detected** 🎯
            
            The AI found strong evidence of **Parkinsonian neural signatures**:
            - Beta oscillations (13-30 Hz excess activity)
            - Irregular, disrupted firing patterns
            - Increased neural synchronization
            - Reduced movement-related activity modulation
            
            **High confidence** suggests textbook Parkinson's-like features.
            """
            explanation_color = "#fff3cd"
        else:
            second_prob = second_choice[1] if second_choice else 0
            explanation = f"""
            **Parkinson's-like, But With Mixed Features** 🤔
            
            The AI detected Parkinsonian characteristics, but also sees features of **{second_choice[0].replace('_', ' ') if second_choice else 'other patterns'}** ({second_prob:.1%} probability).
            
            **This suggests:**
            - Some beta oscillations present
            - Mixed pathological features
            - Could be early-stage or complex presentation
            - Realistic for real-world cases where symptoms overlap
            
            **Clinical relevance:** Many patients show mixed features, especially in early stages.
            """
            explanation_color = "#f8d7da"
    
    elif predicted_class == 'Epileptiform':
        if confidence > 0.8:
            explanation = f"""
            **Clear Epilepsy-like Pattern Detected** ⚡
            
            The AI identified strong **epileptiform characteristics**:
            - Hypersynchronous neural activity
            - Burst firing patterns
            - Sharp, spike-like transients
            - Abnormal population dynamics
            
            **High confidence** indicates classic seizure-like neural signatures.
            """
            explanation_color = "#f8d7da"
        else:
            second_prob = second_choice[1] if second_choice else 0
            explanation = f"""
            **Epilepsy-like With Some Uncertainty** ❓
            
            Strong epileptiform features detected, but the AI also sees **{second_choice[0].replace('_', ' ') if second_choice else 'other patterns'}** characteristics ({second_prob:.1%}).
            
            **This could mean:**
            - Interictal activity (between seizures)
            - Mixed seizure types
            - Transition periods
            - Complex epilepsy presentation
            
            **Real-world relevance:** Epilepsy patterns can be highly variable.
            """
            explanation_color = "#f8d7da"
    
    elif predicted_class == 'Mixed_Pathology':
        # This is the most complex case - need detailed explanation
        top_other_patterns = [item for item in sorted_probs[:3] if item[0] != 'Mixed_Pathology']
        
        if confidence > 0.7:
            explanation = f"""
            **Complex Mixed Pathological Pattern** 🔬
            
            The AI detected a **combination of multiple pathological features** rather than one clear disease pattern.
            
            **What this means:**
            - Features of both Parkinson's AND epilepsy-like activity
            - Multiple neural systems affected
            - Complex, real-world pathological presentation
            - Could represent comorbid conditions or disease progression
            
            **Top other possibilities:** {', '.join([f"{p[0].replace('_', ' ')} ({p[1]:.1%})" for p in top_other_patterns[:2]])}
            
            **Clinical significance:** Many neurological patients have mixed presentations!
            """
            explanation_color = "#e2e3e5"
        else:
            explanation = f"""
            **Borderline Mixed Pattern - AI is Uncertain** 🤷‍♂️
            
            The AI is having trouble choosing between **Mixed Pathology** and **{second_choice[0].replace('_', ' ') if second_choice else 'other patterns'}** ({second_choice[1]:.1%} vs {confidence:.1%}).
            
            **Why this happens:**
            - Pattern sits on the boundary between categories
            - Subtle or early pathological changes
            - Natural variation in neural activity
            - Realistic complexity of real brain data
            
            **This is actually good!** It shows the AI recognizes when patterns don't fit neat categories.
            
            **Try this:** Generate the same pattern again - do you get consistent results?
            """
            explanation_color = "#fff3cd"
    
    # Display the explanation in a colored box
    st.markdown(f"""
    <div style="background-color: {explanation_color}; padding: 1.5rem; border-radius: 10px; 
                border-left: 5px solid #007bff; margin: 1rem 0; color: #333;">
        {explanation}
    </div>
    """, unsafe_allow_html=True)
    
def show_actionable_suggestions(predicted_class, confidence):
    """Show actionable suggestions for exploring results further"""
    
    st.markdown("### 💡 What Should You Try Next?")
    
    suggestions = []
    
    if confidence < 0.7:
        suggestions.append("🔄 **Generate the same pattern again** - Is the result consistent?")
        suggestions.append("🎯 **Try a more extreme version** - Increase the pathological parameters")
    
    if predicted_class == 'Mixed_Pathology':
        suggestions.append("🔬 **Compare with pure patterns** - Try 'Parkinsonian' and 'Epileptiform' separately")
        suggestions.append("📊 **Look at detailed features** - Check the probability breakdown above")
        suggestions.append("🧪 **Experiment with parameters** - Try reducing some pathological settings")
    
    elif predicted_class in ['Parkinsonian', 'Epileptiform']:
        if confidence > 0.8:
            suggestions.append("✅ **Try the opposite** - Generate a 'Healthy' pattern to see the contrast")
            suggestions.append("🎛️ **Reduce pathological parameters** - See at what point it becomes 'Mixed'")
        else:
            suggestions.append("📈 **Increase pathological strength** - Try higher values for clearer patterns")
    
    elif predicted_class in ['Healthy_Rate', 'Healthy_Temporal']:
        suggestions.append("⚡ **Try pathological patterns** - Compare with 'Parkinsonian' or 'Epileptiform'")
        suggestions.append("🔍 **Experiment with mixed patterns** - See how AI detects subtle changes")
    
    # Always suggest comparison
    suggestions.append("🆚 **Compare different preset types** - Build intuition about pattern differences")
    suggestions.append("📚 **Educational insight** - This shows how AI assists medical diagnosis!")
    
    # Display suggestions in a nice format
    suggestion_text = "\n".join([f"- {suggestion}" for suggestion in suggestions])
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
                padding: 1.5rem; border-radius: 10px; border-left: 5px solid #9c27b0; 
                margin: 1rem 0; color: #333;">
        {suggestion_text}
    </div>
    """, unsafe_allow_html=True)

def show_visualizations_simple():
    """Show simplified visualizations"""
    
    if not st.session_state.current_data:
        return
    
    st.markdown("### 📈 Neural Activity Visualization")
    
    tab1, tab2 = st.tabs(["🎯 Individual Neuron Activity", "📊 Population Activity"])
    
    with tab1:
        show_raster_plot_simple()
    
    with tab2:
        show_population_analysis_simple()

def show_raster_plot_simple():
    """Show simplified raster plot"""
    
    spike_data = st.session_state.current_data
    trial_spikes = spike_data['spike_trains'][0]
    
    st.markdown("**Each line shows when individual neurons fired spikes:**")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for neuron_idx, spikes in enumerate(trial_spikes):
        if len(spikes) > 0:
            ax.plot(spikes, [neuron_idx] * len(spikes), '|', 
                   markersize=4, color='black', alpha=0.8)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Neuron Number', fontsize=12)
    ax.set_title('Neural Spike Activity - Each Tick is a Spike!', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add helpful annotation
    ax.text(0.02, 0.98, 'More spikes = Higher activity\nClusters = Synchronized firing', 
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    st.pyplot(fig)
    plt.close()

def show_population_analysis_simple():
    """Show simplified population analysis"""
    
    spike_data = st.session_state.current_data
    trial_spikes = spike_data['spike_trains'][0]
    trial_duration = 2.0
    bin_size = 0.02
    
    # Calculate population activity
    time_bins = np.arange(0, trial_duration + bin_size, bin_size)
    pop_activity = np.zeros(len(time_bins) - 1)
    
    for spikes in trial_spikes:
        if len(spikes) > 0:
            spike_counts, _ = np.histogram(spikes, bins=time_bins)
            pop_activity += spike_counts
    
    st.markdown("**Total activity across all neurons over time:**")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    time_axis = np.linspace(0, trial_duration, len(pop_activity))
    ax.plot(time_axis, pop_activity, linewidth=3, color='navy')
    ax.fill_between(time_axis, pop_activity, alpha=0.3, color='skyblue')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Total Spikes Across All Neurons', fontsize=12)
    ax.set_title('Population Neural Activity Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add interpretation
    max_activity = np.max(pop_activity)
    mean_activity = np.mean(pop_activity)
    
    if max_activity > 3 * mean_activity:
        annotation = "High peaks suggest\nsynchronized bursting"
    elif np.std(pop_activity) < mean_activity * 0.5:
        annotation = "Steady activity suggests\nregular firing"
    else:
        annotation = "Variable activity\nsuggests mixed patterns"
    
    ax.text(0.7, 0.9, annotation, transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    st.pyplot(fig)
    plt.close()

def train_classifier():
    """Train the classifier with clear progress"""
    
    with st.spinner("🤖 Training AI to recognize neural patterns..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🧠 Setting up AI brain...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        st.session_state.decoder = OptimizedMultiClassDecoder(random_state=42)
        
        status_text.text("📊 Generating training examples (150 neural patterns)...")
        progress_bar.progress(30)
        time.sleep(1)
        
        status_text.text("🎯 Teaching AI to recognize patterns...")
        progress_bar.progress(60)
        
        # Train the model
        results = st.session_state.decoder.train(n_trials_per_class=30, verbose=False)
        
        progress_bar.progress(90)
        status_text.text("✅ Finalizing training...")
        time.sleep(0.5)
        
        st.session_state.training_results = results
        st.session_state.decoder_trained = True
        
        progress_bar.progress(100)
        status_text.text("🎉 AI training complete!")
        
        time.sleep(1)
        st.rerun()

def retrain_classifier():
    """Reset and retrain"""
    st.session_state.decoder = None
    st.session_state.decoder_trained = False
    st.session_state.training_results = None
    st.session_state.prediction_results = None
    st.rerun()

def show_training_summary():
    """Show a simple training summary"""
    
    if not st.session_state.training_results:
        return
    
    results = st.session_state.training_results
    
    st.markdown("### 🎉 AI Training Complete!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best AI Model", results['best_classifier'])
    
    with col2:
        st.metric("Accuracy", f"{results['final_accuracy']:.1%}")
    
    with col3:
        cv_score = results['results'][results['best_classifier']]['cv_mean']
        st.metric("Reliability Score", f"{cv_score:.1%}")
    
    st.success("🧠 Your AI is now ready to analyze neural patterns!")

def main():
    """Main application with clear user flow"""
    
    initialize_session_state()
    
    show_header()
    show_disclaimer()
    
    st.markdown("---")
    
    # Route based on user mode
    if st.session_state.user_mode == 'getting_started' or not st.session_state.decoder_trained:
        show_getting_started()
    else:
        show_analysis_mode()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        🧠 Neural Code Research Lab | Educational Neural Pattern Analysis<br>
        <em>Helping students and researchers understand how AI analyzes brain activity</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()