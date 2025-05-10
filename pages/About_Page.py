import streamlit as st
import os

def show_about_page():
    st.set_page_config(
        page_title="About",
        page_icon=":musical_note:",
        layout="centered",
    )
    st.title("Indian Classical Music Genre Classifier")
    
    st.header("Project Overview")
    st.write("""
    This project uses a deep neural network model to classify music into different genres.
    The model analyzes various audio features extracted from music files to predict their genres.
    The dataset used for training the model consists of audio files from various Indian classical music genres, like ***Bhajan, Ghazal, Kawali, Rabindra Sangeet, Nazrul Geeti Mantra, and Thumri***.
    """)
    
    st.header("Model Performance")
    st.write("""The Classification Report below shows the performance of the model across different genres.""")
    st.image(os.path.join("assets", "classification-report.png"), caption="Classification Report")
    st.write("""The confusion matrix below shows the model's predictions compared to the actual genres.""")
    st.image(os.path.join("assets", "heatmap.png"), caption="Model Performance Metrics")

    st.header("Model Training and Evaluation")
    st.write("""1-cycle scheduler is used to schedule the learning rate. The learning rate graph below shows the learning rate over epochs.""")
    st.image(os.path.join("assets", "lr-graph.png"), caption="Learning Rate Graph")
    
    

if __name__ == "__main__":
    show_about_page()