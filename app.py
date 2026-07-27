import streamlit as st
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import os
import pickle
import plotly.graph_objects as go

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Page configuration
st.set_page_config(
    page_title="News Classification & Headline Generation",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium glassmorphism dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Core Font overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Card design */
    .header-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .header-title {
        background: linear-gradient(135deg, #60a5fa 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        font-family: 'Outfit', sans-serif;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 400;
    }
    
    /* Glassmorphic Metric Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(96, 165, 250, 0.4);
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.15);
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    
    .metric-detail {
        color: #38bdf8;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    .badge-high {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-low {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #60a5fa;
        font-family: 'Outfit', sans-serif;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.4rem;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    
    .sidebar-text {
        font-size: 0.9rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }
    
    /* Probabilities progress bar */
    .probability-row {
        margin-bottom: 1rem;
    }
    
    .probability-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
        color: #cbd5e1;
    }
    
    /* Custom divider */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(96, 165, 250, 0.3), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Preprocessing helpers
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    cleaned_token = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    return " ".join(cleaned_token)

# Helper to track model file modification time
def get_model_mtime():
    model_path = "models/classification_model.pkl"
    if os.path.exists(model_path):
        return os.path.getmtime(model_path)
    return 0

# Model loading with cache to run fast, checking modification time to auto-reload
@st.cache_resource
def load_assets(mtime):
    model_path = "models/classification_model.pkl"
    vec_path = "models/tfidf_vectorizer.pkl"
    le_path = "models/label_encoder.pkl"
    t5_model_dir = "models/t5_small"
    
    # Load Logistic Regression classifier
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
        
    # Load TF-IDF Vectorizer
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    # Load Label Encoder
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)
        
    # Load T5-small model and tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_dir)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_dir)
    
    return classifier, vectorizer, label_encoder, t5_model, t5_tokenizer

# Page Header
st.markdown("""
<div class="header-container">
    <div class="header-title">News Article Classification & Headline Generation</div>
    <div class="header-subtitle">Analyze, categorize, and generate titles for news descriptions using Logistic Regression & T5-Small Transformer Generator</div>
</div>
""", unsafe_allow_html=True)

# Load Models
try:
    mtime = get_model_mtime()
    classifier, vectorizer, label_encoder, t5_model, t5_tokenizer = load_assets(mtime)
    models_loaded = True
except Exception as e:
    st.error("⚠️ Model Assets Not Loaded Yet!")
    st.markdown(f"""
    The application could not load the saved pre-trained model files. 
    
    **Error:** `{e}`
    
    Please ensure you have run the Jupyter Notebook `news_classification_himakesh.ipynb` to train and save the required files:
    * `models/classification_model.pkl`
    * `models/tfidf_vectorizer.pkl`
    * `models/label_encoder.pkl`
    * `models/t5_small/`
    """)
    models_loaded = False

# Sidebar Content
with st.sidebar:
    st.markdown("📰 **BBC News AI Agent**")
    
    st.markdown('<div class="sidebar-header">Project Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-text">This web application integrates robust text classification and a T5-small transformer generator to analyze news articles, categorize inputs, and synthesize corresponding headlines.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-header">Dataset Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-text"><b>Source:</b> BBC News Dataset<br><b>Target Classes:</b> Business, Politics, Sports, Technology<br><b>Filtered Records:</b> 13,385 clean news descriptions.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-header">Model Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-text"><b>Classifier:</b> Logistic Regression (Balanced)<br><b>Features:</b> TF-IDF (1-3 N-Grams)<br><b>Classifier Validation Accuracy:</b> ~97%<br><b>Headline Gen:</b> T5-Small Transformer</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-header">Target Classes</div>', unsafe_allow_html=True)
    st.markdown('<span class="badge-high" style="margin-right: 5px;">Business</span>'
                '<span class="badge-high" style="margin-right: 5px; background-color: rgba(59, 130, 246, 0.15); color: #60a5fa; border-color: rgba(59, 130, 246, 0.3);">Politics</span>'
                '<span class="badge-high" style="margin-right: 5px; background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3);">Sports</span>'
                '<span class="badge-high" style="background-color: rgba(192, 132, 252, 0.15); color: #c084fc; border-color: rgba(192, 132, 252, 0.3);">Technology</span>', unsafe_allow_html=True)

# Setup Session State for Text Input Clearing
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Main Layout
if models_loaded:
    # Text Area
    user_input = st.text_area(
        "Paste News Article Content or Description here:",
        value=st.session_state.input_text,
        placeholder="Type or paste news article content here to classify and generate headline...",
        height=220,
        key="news_text_area"
    )
    
    # Update session state on input change
    st.session_state.input_text = user_input
    
    # Row of Buttons
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    classify_clicked = False
    headline_clicked = False
    analysis_clicked = False
    
    with col_btn1:
        if st.button("🔍 Classify Article", type="primary", use_container_width=True):
            classify_clicked = True
    with col_btn2:
        if st.button("✍️ Generate Headline", use_container_width=True):
            headline_clicked = True
    with col_btn3:
        if st.button("⚡ Run Complete Analysis", use_container_width=True):
            analysis_clicked = True
    with col_btn4:
        if st.button("🗑️ Clear Input", use_container_width=True):
            st.session_state.input_text = ""
            st.rerun()
            
    # Input Validation
    has_input = len(user_input.strip()) > 0
    
    # Execution Logic
    if classify_clicked or headline_clicked or analysis_clicked:
        if not has_input:
            st.warning("⚠️ Please provide news text input before running analysis!")
        else:
            with st.spinner("Processing NLP models..."):
                # Clean and Preprocess
                cleaned = clean_text(user_input)
                
                # Category Prediction Logic using TF-IDF features and Logistic Regression
                features = vectorizer.transform([cleaned])
                probs = classifier.predict_proba(features)[0]
                best_idx = np.argmax(probs)
                category = label_encoder.classes_[best_idx]
                confidence_score = float(probs[best_idx])
                
                # Confidence Level Mapping
                if confidence_score >= 0.80:
                    conf_level = "High"
                    badge_class = "badge-high"
                elif confidence_score >= 0.50:
                    conf_level = "Medium"
                    badge_class = "badge-medium"
                else:
                    conf_level = "Low"
                    badge_class = "badge-low"
                
                # Probability mapping
                class_probs = dict(zip(label_encoder.classes_, [float(p) for p in probs]))
                
                # Headline Generation Logic using pre-trained and fine-tuned T5-small model
                generated_title = ""
                if headline_clicked or analysis_clicked:
                    input_text = "summarize: " + user_input.lower().strip()
                    inputs = t5_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                    with torch.no_grad():
                        summary_ids = t5_model.generate(
                            inputs["input_ids"],
                            max_length=30,
                            min_length=5,
                            num_beams=4,
                            early_stopping=True,
                            no_repeat_ngram_size=2
                        )
                    generated_title = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # --- DISPLAY OUTPUTS ---
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            st.subheader("📊 Analysis Results")
            
            # Scenario 1: Classify Article ONLY
            if classify_clicked:
                col_res1, col_res2 = st.columns([1, 1])
                
                with col_res1:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Predicted Category</div>
                        <div class="metric-value" style="color: #60a5fa;">{category}</div>
                        <div class="metric-detail">Confidence Level: <span class="{badge_class}">{conf_level}</span></div>
                    </div>
                    
                    <div class="glass-card">
                        <div class="metric-label">Confidence Score</div>
                        <div class="metric-value" style="background: linear-gradient(135deg, #34d399 0%, #059669 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{confidence_score * 100:.2f}%</div>
                        <div class="metric-detail">Logistic Regression probability estimation</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_res2:
                    st.write("**Prediction Probabilities:**")
                    for cls, prob in class_probs.items():
                        st.markdown(f"""
                        <div class="probability-row">
                            <div class="probability-header">
                                <span>{cls}</span>
                                <span>{prob * 100:.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(prob)
                        
                    # Plotly chart
                    fig = go.Figure(go.Bar(
                        x=list(class_probs.values()),
                        y=list(class_probs.keys()),
                        orientation='h',
                        marker=dict(
                            color=['#3b82f6', '#10b981', '#f59e0b', '#ec4899'],
                            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
                        )
                    ))
                    fig.update_layout(
                        xaxis=dict(title="Probability", range=[0, 1], gridcolor='rgba(255, 255, 255, 0.05)'),
                        yaxis=dict(autorange="reversed"),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#cbd5e1'),
                        margin=dict(l=20, r=20, t=10, b=20),
                        height=200
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Scenario 2: Generate Headline ONLY
            elif headline_clicked:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Generated Headline</div>
                    <div class="metric-value" style="font-size: 1.6rem; color: #a78bfa; line-height: 1.4; font-family: 'Plus Jakarta Sans', sans-serif;">
                        “{generated_title}”
                    </div>
                    <div class="metric-detail">Generated using fine-tuned T5-small model</div>
                </div>
                """, unsafe_allow_html=True)

            # Scenario 3: Complete Analysis
            elif analysis_clicked:
                col_c1, col_c2, col_c3 = st.columns(3)
                
                with col_c1:
                    st.markdown(f"""
                    <div class="glass-card" style="height: 160px;">
                        <div class="metric-label">Predicted Category</div>
                        <div class="metric-value" style="color: #60a5fa; font-size: 1.8rem;">{category}</div>
                        <div class="metric-detail">Confidence: <span class="{badge_class}">{conf_level}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c2:
                    st.markdown(f"""
                    <div class="glass-card" style="height: 160px;">
                        <div class="metric-label">Confidence Score</div>
                        <div class="metric-value" style="background: linear-gradient(135deg, #34d399 0%, #059669 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.8rem;">{confidence_score * 100:.2f}%</div>
                        <div class="metric-detail">Softmax probability</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c3:
                    st.markdown(f"""
                    <div class="glass-card" style="height: 160px;">
                        <div class="metric-label">Confidence Level</div>
                        <div class="metric-value" style="color: #fbbf24; font-size: 1.8rem;">{conf_level}</div>
                        <div class="metric-detail">Rule threshold map</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Generated Headline</div>
                    <div class="metric-value" style="font-size: 1.6rem; color: #c084fc; line-height: 1.4; font-family: 'Plus Jakarta Sans', sans-serif;">
                        “{generated_title}”
                    </div>
                    <div class="metric-detail">Synthesized using fine-tuned T5-small model</div>
                </div>
                """, unsafe_allow_html=True)
                
                col_prob_bars, col_prob_plot = st.columns([1, 1])
                
                with col_prob_bars:
                    st.write("**Category Probabilities:**")
                    for cls, prob in class_probs.items():
                        st.markdown(f"""
                        <div class="probability-row">
                            <div class="probability-header">
                                <span>{cls}</span>
                                <span>{prob * 100:.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(prob)
                        
                with col_prob_plot:
                    fig = go.Figure(go.Bar(
                        x=list(class_probs.values()),
                        y=list(class_probs.keys()),
                        orientation='h',
                        marker=dict(
                            color=['#3b82f6', '#10b981', '#f59e0b', '#ec4899'],
                            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
                        )
                    ))
                    fig.update_layout(
                        xaxis=dict(title="Probability", range=[0, 1], gridcolor='rgba(255, 255, 255, 0.05)'),
                        yaxis=dict(autorange="reversed"),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#cbd5e1'),
                        margin=dict(l=20, r=20, t=10, b=20),
                        height=180
                    )
                    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("💡 Please complete the training step in the Jupyter notebook first to generate model weights.")
