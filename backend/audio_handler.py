import speech_recognition as sr
import base64
import io
import wave

def transcribe_audio(audio_base64: str) -> str:
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Create an in-memory WAV file
        audio_file = io.BytesIO()
        with wave.open(audio_file, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(44100)  # Sample rate
            wav_file.writeframes(audio_bytes)
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Convert audio to AudioFile
        audio_file.seek(0)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            
        # Perform speech recognition
        text = recognizer.recognize_google(audio_data)
        return text
        
    except sr.UnknownValueError:
        raise ValueError("Could not understand audio")
    except sr.RequestError as e:
        raise Exception(f"Error with speech recognition service: {e}")
    except Exception as e:
        raise Exception(f"Error processing audio: {e}")