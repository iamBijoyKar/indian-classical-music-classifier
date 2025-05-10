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
    """)
    
    st.header("Model Performance")
    st.write("""The heatmap below shows the performance of the model across different genres.""")
    st.image(os.path.join("assets", "heatmap.png"), caption="Model Performance Metrics")
    
    

if __name__ == "__main__":
    show_about_page()