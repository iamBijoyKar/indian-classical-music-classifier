import streamlit as st
import os

def show_about_page():
    st.set_page_config(
        page_title="About",
        page_icon=":musical_note:",
        layout="centered",
    )
    st.title("CLASSIFYING INDIAN CLASSICAL MUSIC FOR THERAPEUTIC CAUSE USING LIGHTWEIGHT DEEP NEURAL NETWORK")
    st.write("""
*Bijoy Kar, Parijat Das, Arnab Kundu, 
Pratingya Sahoo, Sweta Kumari & Prof. Somsubhra Gupta* 
""")
    st.image(os.path.join("assets", "team.jpg"), caption="Our Research Team")
    st.write("""We are student of ***Swami Vivekananda University, Barrackpore department of Computer Science and Engineering.*** This research is the final year project of our graduating year, 2025 under the supervision of Prof. Somsubhra Gupta, School of Computer Science. """)
    
    

if __name__ == "__main__":
    show_about_page()