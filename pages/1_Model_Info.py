import streamlit as st

st.set_page_config(page_title="Model Information", layout="centered")
st.title("📊 Understanding The Chest X-ray Models")

# Introduction
st.markdown("""
This page explains the two chest X-ray classification models devloped by Sam Boesen and why proper evaluation methodology is crucial in medical AI.
""")

# What is AUC?
st.header("📈 Understanding AUC (Area Under the ROC Curve)")
st.markdown("""
AUC (Area Under the ROC Curve) is a key metric in medical imaging that tells us how well my model can:
- Distinguish between patients with and without a condition
- Rank patients correctly (higher probability for actual cases)
- Perform across different decision thresholds

An AUC of:
- 0.5 = Random guessing
- 0.7-0.8 = Good performance
- 0.8-0.9 = Very good performance
- >0.9 = Excellent performance
""")

# Model Comparison
st.header("🔍 Model Comparison")

# Create tabs for different comparison aspects
tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Evaluation Methodology", "Key Differences"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("densenet121_xray.pt (Initial Model)")
        st.markdown("""
        ### Performance
        - **Overall AUC**: 0.7657
        - **Best Class**: Cardiomegaly (AUC = 0.8760)
        
        ### Architecture
        - Base: DenseNet121
        - Input Size: 224x224
        - Output: 15 disease classes
        """)
    
    with col2:
        st.subheader("best_densenet.pt (Final Model)")
        st.markdown("""
        ### Performance
        - **Overall AUC**: 0.5499
        - **Best Class**: Effusion (AUC = 0.6903)
        
        ### Architecture
        - Base: DenseNet121
        - Input Size: 224x224
        - Output: 15 disease classes
        """)

with tab2:
    st.markdown("""
    ### Why Different AUCs?
    
    The significant difference in AUC (0.7657 vs 0.5499) highlights a crucial issue in medical AI: **data leakage**.
    
    #### densenet121_xray.pt (Initial Model)
    - Higher AUC (0.7657)
    - May include same patients in train and test sets
    - Performance likely overestimated
    - Not representative of real-world performance
    
    #### best_densenet.pt (Final Model)
    - Lower AUC (0.5499)
    - Strict patient-level split
    - No overlap between train and test sets
    - More realistic performance estimate
    - Better represents real-world generalization
    """)

with tab3:
    st.markdown("""
    ### Key Differences
    
    1. **Evaluation Methodology**
       - densenet121_xray.pt: Potential data leakage
       - best_densenet.pt: Strict patient-level split
    
    2. **Performance Interpretation**
       - densenet121_xray.pt: Overly optimistic
       - best_densenet.pt: More realistic
    
    3. **Clinical Relevance**
       - densenet121_xray.pt: May not generalize to new patients
       - best_densenet.pt: Better represents real-world performance
    """)

# Disease Classes
st.header("🎯 Disease Classes")
st.markdown("""
My models can detect 15 conditions, with varying performance across classes:

| Condition | densenet121_xray.pt AUC | best_densenet.pt AUC | Notes |
|-----------|------------|------------|-------|
| Cardiomegaly | 0.8760 | 0.5499 | Best in initial model |
| Effusion | 0.7657 | 0.6903 | Best in final model |
| Other conditions | Varies | Varies | See full results |

*Note: best_densenet.pt's lower AUCs reflect more realistic performance on unseen patients.*
""")

# Limitations and Best Practices
st.header("⚠️ Limitations and Best Practices")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Limitations")
    st.markdown("""
    - Not a replacement for professional diagnosis
    - Performance varies by disease class
    - Lower AUC in best_densenet.pt reflects real-world challenges
    - May miss subtle or rare conditions
    - Requires high-quality input images
    """)

with col2:
    st.subheader("Best Practices")
    st.markdown("""
    - Use high-quality, properly oriented X-rays
    - Ensure good image contrast
    - Consider multiple views for complex cases
    - Always verify results with medical professionals
    - Understand that lower AUC in best_densenet.pt is more realistic
    - Use best_densenet.pt for real-world applications
    """)

# Technical Details
st.header("🔧 Technical Details")
st.markdown("""
### Preprocessing
- Image resizing to 224x224
- ImageNet normalization
- RGB conversion if needed

### Inference
- GPU acceleration when available
- MC Dropout for uncertainty estimation
- 30 forward passes for probability distribution

### Visualization
- Grad-CAM heatmaps for interpretability
- Probability scores with uncertainty bounds
- Interactive disease selection for visualization
""")

# Footer
st.markdown("---")
st.markdown("""
*Note: This application is for research and educational purposes only. Always consult healthcare professionals for medical decisions.*

**Created by Sam Boesen**
""") 