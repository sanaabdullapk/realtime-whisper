import whisper
import pyaudio
import wave
import threading
import time
import numpy as np
from collections import deque
import os
from scipy.signal import resample


class RealTimeTranscriber:
    def __init__(self, model_name="base"):
        """
        Initialize the real-time transcriber
        
        Args:
            model_name (str): Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
        """
        print("Loading Whisper model... This may take a moment.")
        self.model = whisper.load_model(model_name)
        print(f"Loaded {model_name} model successfully!")
        
        # Audio settings
        self.CHUNK = 1024  
        self.FORMAT = pyaudio.paInt16 
        self.CHANNELS = 1  
        self.RATE = 16000 
        self.RECORD_SECONDS = 3
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Audio buffer to store recent audio
        self.audio_buffer = deque(maxlen=self.RATE * 10)  
        
        # Control flags
        self.is_recording = False
        self.is_transcribing = False
        
    def list_microphones(self):
        """List available microphones"""
        print("\nAvailable microphones:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  {i}: {info['name']}")
        print()
    
    def start_recording(self, device_index=None):
        """Start recording audio from microphone"""
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK
            )
            
            self.is_recording = True
            print("🎤 Recording started! Speak into your microphone...")
            print("Press Ctrl+C to stop\n")
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self._record_audio)
            self.record_thread.daemon = True
            self.record_thread.start()
            
            # Start transcription thread
            self.transcribe_thread = threading.Thread(target=self._transcribe_audio)
            self.transcribe_thread.daemon = True
            self.transcribe_thread.start()
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.list_microphones()
            return False
        
        return True
    
    def _record_audio(self):
        """Record audio in a separate thread"""
        while self.is_recording:
            try:
                # Read audio data
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Add to buffer
                self.audio_buffer.extend(audio_data)
                
            except Exception as e:
                print(f"Error recording audio: {e}")
                break
    
    def _transcribe_audio(self):
        """Transcribe audio in a separate thread"""
        while self.is_recording:
            try:
                # Wait for enough audio data
                if len(self.audio_buffer) < self.RATE * self.RECORD_SECONDS:
                    time.sleep(0.1)
                    continue
                
                # Skip if already transcribing
                if self.is_transcribing:
                    time.sleep(0.1)
                    continue
                
                self.is_transcribing = True
                
                # Get recent audio data
                audio_data = np.array(list(self.audio_buffer)[-self.RATE * self.RECORD_SECONDS:])

                max_vol = np.max(np.abs(audio_data))
                print(f"🎧 Max volume in chunk: {max_vol:.4f}")

                if max_vol < 0.005:
                    print("🔇 No speech detected")
                    self.is_transcribing = False
                    time.sleep(0.1)
                    continue

            # Normalize only if safe
                if max_vol > 0:
                    audio_data = audio_data / max_vol

                
                # Resample to 16kHz if needed
                if self.RATE != 16000:
                    num_samples = int(len(audio_data) * 16000 / self.RATE)
                    audio_data = resample(audio_data, num_samples)

                print("🔄 Transcribing...", end=" ", flush=True)
                result = self.model.transcribe(audio_data, language="en", fp16=False)

                
            
                # Get the transcribed text
                text = result["text"].strip()
                
                if text:
                    print(f"\r📝 Transcription: {text}")
                else:
                    print("\r🔇 No speech detected")
                
                self.is_transcribing = False
                
                # Wait before next transcription
                time.sleep(1)
                
            except Exception as e:
                print(f"Error during transcription: {e}")
                self.is_transcribing = False
                time.sleep(1)
    
    def stop_recording(self):
        """Stop recording and clean up"""
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        print("\n🛑 Recording stopped.")
    
    def test_microphone(self):
        """Test if microphone is working"""
        print("Testing microphone for 3 seconds...")
        
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            max_volume = 0
            for _ in range(int(self.RATE / self.CHUNK * 3)):  # 3 seconds
                data = stream.read(self.CHUNK)
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.max(np.abs(audio_data))
                max_volume = max(max_volume, volume)
                print(f"Volume: {'█' * min(50, int(volume / 1000))}")
                time.sleep(0.1)
            
            stream.stop_stream()
            stream.close()
            
            if max_volume > 1000:
                print("✅ Microphone is working!")
                return True
            else:
                print("❌ Microphone might not be working. Try speaking louder or check your microphone.")
                return False
                
        except Exception as e:
            print(f"❌ Microphone test failed: {e}")
            return False

def main():
    """Main function to run the real-time transcriber"""
    print("🎯 Real-Time Whisper Transcription")
    print("=" * 40)
    
    # Create transcriber instance
    transcriber = RealTimeTranscriber(model_name="base")
    
    # List available microphones
    transcriber.list_microphones()
    
    # Ask user to select microphone (optional)
    try:
        mic_choice = input("Enter microphone number (or press Enter for default): ").strip()
        device_index = int(mic_choice) if mic_choice else None
    except ValueError:
        device_index = None

    if device_index is not None:
        device_info = transcriber.audio.get_device_info_by_index(device_index)
        supported_rate = int(device_info['defaultSampleRate'])
        print(f"👉 Microphone #{device_index} default sample rate: {supported_rate}")
        
        # Update transcriber's rate to match microphone
        transcriber.RATE = supported_rate

    
    # Test microphone
    print("\nTesting microphone...")
    if not transcriber.test_microphone():
        print("Please check your microphone and try again.")
        return
    
    try:
        # Start recording and transcription
        if transcriber.start_recording(device_index):
            # Keep running until user stops
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nStopping transcription...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        transcriber.stop_recording()

if __name__ == "__main__":
    main()