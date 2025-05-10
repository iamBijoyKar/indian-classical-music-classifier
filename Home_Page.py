import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# path
cwd_path = os.path.dirname(os.path.abspath(__file__))

# navigation

# load model
model_path = os.path.join(os.path.dirname(__file__), "7-class-indian-classical-music-classifier-6000samples-model@v1.keras")
model = tf.keras.models.load_model(model_path)

song_label_dict = {0: 'Ghazal', 1: 'Kawali', 2: "Mantra", 3: 'Nazrul Geeti', 4: 'Rabindra Sangeet', 5: 'Thumri', 6: 'Bhajan'} 
song_health_benifit_dict = {'Ghazal': 'Emotional release, helps with grief, bipolar therapy. Relaxes nervous system', 'Kawali': 'Reduce stress, depression, induces meditative state. Boosts immune system, lowers cortisol', 'Nazrul Geeti': ' Inspires strength, fights depression. Stimulates nervous system', 'Rabindra Sangeet': ' Used in psychotherapy, helps with depression & clarity. Benefits cardiovascular health', 'Thumri': 'Relaxes nervous system', 'Bhajan': 'Reduce anxiety, brings calm and spiritual grounding. Lowers blood pressure, calms heart rate', 'Mantra': 'Lowers blood pressure, calms heart rate'}

def get_peak_frequency(audio_data, sr):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = np.abs(librosa.stft(audio_data))

    # Compute the frequencies corresponding to each bin in the STFT
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0])

    # Find the peak frequency
    peak_bin = np.argmax(np.mean(stft, axis=1))  # Find the bin with the highest mean amplitude
    peak_frequency = frequencies[peak_bin]

    return peak_frequency

def get_peak_pitch(audio_data, sr):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = np.abs(librosa.stft(audio_data))

    # Compute the frequencies corresponding to each bin in the STFT
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0])

    # Find the peak frequency
    peak_bin = np.argmax(np.mean(stft, axis=1))  # Find the bin with the highest mean amplitude
    peak_frequency = frequencies[peak_bin]

    # Convert the peak frequency to pitch
    peak_pitch = librosa.hz_to_midi(peak_frequency)

    if peak_pitch < 0:
        peak_pitch = 0

    return peak_pitch

def get_power_in_dB(audio_data, sr):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = np.abs(librosa.stft(audio_data))

    # Compute the power in dB
    power = librosa.amplitude_to_db(np.mean(stft))

    return power


def audio_data_dividor(audio_data, sr, segment_length_sec = 5):
    # Convert segment length to samples
    segment_length_samples = segment_length_sec * sr

    # Split audio into segments
    segments = [audio_data[i:i + segment_length_samples] for i in range(0, len(audio_data), segment_length_samples)]
    return segments

def get_spectral_centroids(audio_data, sr):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = np.abs(librosa.stft(audio_data))

    # Compute the spectral centroids
    spectral_centroids = librosa.feature.spectral_centroid(S=stft, sr=sr)

    return spectral_centroids[0]

def get_spectral_rolloff(audio_data, sr):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = np.abs(librosa.stft(audio_data))

    # Compute the spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)

    return spectral_rolloff[0]

def get_zero_crossing_rate(audio_data):
    # Compute the zero-crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)

    return zero_crossing_rate[0]

def get_mfccs(audio_data, sr):
    # Compute the Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr)

    return mfccs

def get_chroma(audio_data, sr):
    # Compute the chroma feature
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)

    return chroma

def get_tempo(audio_data, sr):
    # Compute the tempo
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    return tempo[0]

def get_rms(audio_data):
    # Compute the root-mean-square (RMS) energy for each segment
    rms = librosa.feature.rms(y=audio_data)

    return rms[0]

max_pad_len = 174

def pad_feature(feature):
    if len(feature) > max_pad_len:
        return feature[:max_pad_len]  # Truncate if too long
    else:
        return np.pad(feature, (0, max_pad_len - len(feature)))  # Pad if too short

def extract_data_scaler(audio_data, sr):
    # Get features
    peak_frequency = get_peak_frequency(audio_data, sr)
    peak_pitch = get_peak_pitch(audio_data, sr)
    power = get_power_in_dB(audio_data, sr)
    spectral_centroids = get_spectral_centroids(audio_data, sr)
    spectral_rolloff = get_spectral_rolloff(audio_data, sr)
    zero_crossing_rate = get_zero_crossing_rate(audio_data)
    mfccs = get_mfccs(audio_data, sr)
    chroma = get_chroma(audio_data, sr)
    tempo = get_tempo(audio_data, sr)
    rms = get_rms(audio_data)

    avg_spectral_centroid = np.mean(spectral_centroids)
    max_spectral_centroid = np.max(spectral_centroids)
    avg_spectral_rolloff = np.mean(spectral_rolloff)
    max_spectral_rolloff = np.max(spectral_rolloff)
    avg_zero_crossing_rate = np.mean(zero_crossing_rate)
    max_zero_crossing_rate = np.max(zero_crossing_rate)
    avg_rms = np.mean(rms)
    max_rms = np.max(rms)
    avg_chroma = np.mean(np.mean(chroma, axis=1))
    max_chroma = np.max(np.max(chroma, axis=1))
    avg_mfccs = np.mean(np.mean(mfccs, axis=1))
    max_mfccs = np.max(np.max(mfccs, axis=1))

    return peak_frequency, power, avg_spectral_centroid,  avg_spectral_rolloff, avg_zero_crossing_rate, avg_rms, avg_chroma, max_mfccs, tempo


# Mock prediction function
def predict_audio(audio_data, sr):
    # Replace this logic with your actual prediction model
    features = extract_features(audio_data, sr)
    features = np.array(features)
    # print(features)
    y_hat = model.predict(features)
    args = np.argmax(y_hat, axis=1)
    pred_labels = [song_label_dict[arg] for arg in args]
    final_pred = max(set(pred_labels), key = pred_labels.count)
    count = pred_labels.count(final_pred)
    confidence = count / len(pred_labels)
    return final_pred, confidence

# Extract features from the audio file (Example: MFCCs)
def extract_features(audio_data, sr):
    audio_segments = audio_data_dividor(audio_data, sr)
    data_table = []
    for i, segment in enumerate(audio_segments):
        try:
            data = extract_data_scaler(segment, sr)
            data_table.append(data)
            
        except Exception as e:
            print(f"Error processing segment {i}: {e}")
            continue

    return data_table

def main():
    # Set up the Streamlit app 
    st.set_page_config(page_title="Indian Classical Music Genre Classifier", page_icon=":musical_note:", layout="centered")
    st.title("Indian Classical Music Genre Classifier")
    st.write("This is a simple demo of an Indian Classical Music Genre Classifier using a Deep Neural Network. The model is trained on 6 Indian Classical Music genres: Ghazal, Kawali, Nazrul Geeti, Rabindra Sangeet, Thumri, Mantra and Bhajan.")

    #sidebar
    st.sidebar.subheader("ℹ️ Classifier Model Complications & Information")
    st.sidebar.info("Kawali, Ghazal and Thumri have similar characteristics and tone, as well as bhajan and mantra are also similar, so the model may confuse them. On the other hand, Rabindra Sangeet and Nazrul Geeti are not similar, so the model may perform well on them. The model is trained on 6000 segment samples of 7 classes of Indian Classical Music with over 400 hundred of music samples.")

    # Upload audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    
    if uploaded_file is not None:
        # Load the audio file
        audio_data, sr = librosa.load(uploaded_file, sr=None)
        
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Display waveform
        st.subheader("Waveform of the Audio")
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(audio_data, sr=sr, ax=ax)
        ax.set_title("Waveform")
        st.pyplot(fig)

        # Predict using the mock function
        st.subheader("Prediction")
        prediction, confidence = predict_audio(audio_data, sr)
        st.write(f"Model Prediction: **{prediction}**")
        st.write(f"Model Confidence: **{confidence:.2f}**")

        # Display health benefits
        st.subheader("Health Benefits")
        if prediction in song_health_benifit_dict:
            st.write(f"Health Benefits of **{prediction}**: {song_health_benifit_dict[prediction]}")
        else:
            st.write("No health benefits information available for this genre.")

        # Recommendations
        st.subheader("Recommendations")
        if prediction == 'Ghazal':
            st.write("Recommended Songs with Similar Health Benefit: 'Tumhare Hawaale Watan Saathiyo', 'Dil Dhoondta Hai'")
            st.write("Here is a recommended sample song 🎵 with similar health benefit, to listen 👇 ")
            st.audio(os.path.join(cwd_path, 'audio', 'Aaj Jane Ki Zid Na Karo.mp3'), format="audio/mp3")
        elif prediction == 'Kawali':
            st.write("Recommended Songs with Similar Health Benefit: 'Bhar Do Jholi Meri', 'Tajdar-e-Haram'")
            st.write("Here is a recommended sample song 🎵 with similar health benefit, to listen 👇 ")
            st.audio(os.path.join(cwd_path, 'audio', 'Aaj Rang Hai.mp3'), format="audio/mp3")
        elif prediction == 'Nazrul Geeti':
            st.write("Recommended Songs with Similar Health Benefit: 'Dhana Dhanya Pushpa Bhora', 'Chhaya Manik Ranga'")
            st.write("Here is a recommended sample song 🎵 with similar health benefit, to listen 👇 ")
            st.audio(os.path.join(cwd_path, 'audio', 'Aajo Madhuro Banshori Baaje.mp3'), format="audio/mp3")
        elif prediction == 'Rabindra Sangeet':
            st.write("Recommended Songs with Similar Health Benefit: 'Ekla Cholo Re', 'Amar Sonar Bangla'")
            st.write("Here is a recommended sample song 🎵 with similar health benefit, to listen 👇 ")
            st.audio(os.path.join(cwd_path, 'audio', 'Phule Phule Dhole Dhole.mp3'), format="audio/mp3")
        elif prediction == 'Thumri':
            st.write("Recommended Songs with Similar Health Benefit: 'Piya Milenge Na', 'Rang Barse Bheege Chunar Wali'")
            st.write("Here is a recommended sample song 🎵 with similar health benefit, to listen 👇 ")
            st.audio(os.path.join(cwd_path, 'audio', 'Ab Ke Sawan Ghar Aaja.mp3'), format="audio/mp3")
        elif prediction == 'Bhajan':
            st.write("Recommended Songs with Similar Health Benefit: 'Raghupati Raghav Raja Ram', 'Vaishnav Jan To Tene Kahiye'")
            st.write("Here is a recommended sample song 🎵 with similar health benefit, to listen 👇 ")
            st.audio(os.path.join(cwd_path, 'audio', 'SpotifyMate.com - Raghupati Raghav Raja Ram Original - Kavita Raam.mp3'), format="audio/mp3")
        elif prediction == 'Mantra':
            st.write("Recommended Songs with Similar Health Benefit: 'Om Namah Shivaya', 'Gayatri Mantra'")
            st.write("Here is a recommended sample song 🎵 with similar health benefit, to listen 👇 ")
            st.audio(os.path.join(cwd_path, 'audio', 'Aarambh.mp3'), format="audio/mp3")
        else:
            st.write("No recommendations available for this genre.")
            
        # warning
        st.warning("Classification of music genres is a complex task, difference between music genres can be very thin and the model may not always be accurate. Please use the predictions as a guide and not as a definitive answer. View the sidebar for more information.",icon="⚠️")
        st.warning("The health benefits mentioned are based on general observations and may not apply to everyone. Please consult a healthcare professional for personalized advice.",icon="⚠️")


if __name__ == "__main__":
    main()
