from openai import OpenAI
import sounddevice as sd
import numpy as np
import tempfile
import pygame
import threading
import random
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
import yaml
import cv2
import base64
import os
from queue import Queue
import ollama
from groq import Groq
from io import BytesIO
from scipy.io import wavfile
import time
import queue

class TalkingPet:
    def __init__(self, config, debug=False):
        """Initialize the talking pet with configuration dictionary.
        
        Args:
            config (dict): Configuration dictionary
            debug (bool): Enable debug output (default: False)
        """
        self.debug = debug
        self.load_config(config)
        self.initialize_components()
        self.is_speaking = False
        self.on_speaking_callbacks = []
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.pygame_initialized = True
        self.talking = False

    def _debug_print(self, *args, **kwargs):
        """Print debug information if debug mode is enabled."""
        if self.debug:
            print(*args, **kwargs)

    def load_config(self, config):
        """Load configuration from provided dictionary."""
        # Required settings with their types
        required_settings = {
            'sampling_rate': int,
            'num_channels': int,
            'dtype': str,
            'silence_threshold': float,
            'ambient_noise_level_threshold_multiplier': float,
            'max_file_size_bytes': int,
            'enable_squeak': bool,
            'system_prompt': str,
            'voice_id': str,
            'greetings': list,
            'enable_vision': bool,
            'silence_count_threshold': int,
            'model': str,
        }

        # Optional settings with their types
        optional_settings = {
            'character_closed_mouth': str,
            'character_open_mouth': str,
        }

        # Validate and set required settings
        for key, expected_type in required_settings.items():
            if key not in config:
                raise ValueError(f"Missing required setting: {key}")
            
            try:
                value = expected_type(config[key])
                setattr(self, key, value)
            except (ValueError, TypeError):
                raise TypeError(f"Cannot convert {key} value '{config[key]}' to {expected_type}")

        # Set optional settings
        for key, expected_type in optional_settings.items():
            if key in config:
                setattr(self, key, config[key])
            else:
                setattr(self, key, None)

        # Initialize display mode based on character images
        self.enable_display = (self.character_closed_mouth is not None and 
                             self.character_open_mouth is not None)

        # Convert dtype string to numpy dtype
        self.dtype = np.dtype(self.dtype)

        # Initialize AI client based on model
        if self.model.startswith("gpt"):
            self.client = OpenAI()
        elif self.model.startswith("groq"):
            self.client = Groq()
        else:
            self.client = ollama.Client()

    def initialize_components(self):
        """Initialize necessary components like pygame, audio, etc."""
        # Initialize ElevenLabs client
        self.eleven_client = ElevenLabs(api_key=os.environ.get("ELEVEN_API_KEY"))

        # Initialize pygame and display only if character images are provided
        if self.enable_display:
            self._debug_print("Initializing display with character images")
            pygame.mixer.init()
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
            pygame.display.set_caption("Talking Pet")

            # Load character images
            self.skull_closed = pygame.image.load(self.character_closed_mouth)
            self.skull_open = pygame.image.load(self.character_open_mouth)
            self.pygame_initialized = True
        else:
            self._debug_print("No character images provided, running without display")
            pygame.mixer.init()  # Still need mixer for audio
            self.pygame_initialized = False

        # Calculate and set silence threshold
        self.ambient_noise_level = self._calculate_ambient_noise_level()
        self.silence_threshold = max(
            self.ambient_noise_level * self.ambient_noise_level_threshold_multiplier,
            10.0
        )

        # Initialize sound effects if enabled
        if self.enable_squeak:
            pygame.mixer.music.load("filler.mp3")

    def add_speaking_callback(self, callback):
        """Add a callback function that will be called when the pet starts/stops speaking."""
        if callable(callback):
            self.on_speaking_callbacks.append(callback)
        else:
            raise TypeError("Callback must be callable")

    def _notify_speaking_state(self, is_speaking):
        """Notify all callbacks about speaking state changes."""
        self.is_speaking = is_speaking
        for callback in self.on_speaking_callbacks:
            try:
                callback(is_speaking)
            except Exception as e:
                print(f"Error in speaking callback: {e}")

    def listen(self):
        """Listen for user input and return the transcribed text."""
        print("Listening...")
        audio_data = self._listen_until_silence()
        if audio_data is None or len(audio_data) == 0:
            return None
        
        return self._transcribe_audio(audio_data)

    def _listen_until_silence(self):
        """Record audio until silence is detected."""
        audio_queue = Queue()
        silence_count = 0
        stop_recording = False
        first_sound_detected = False
        total_frames = 0

        def audio_callback(indata, frames, time, status):
            nonlocal silence_count, stop_recording, first_sound_detected, total_frames
            
            # Compute the mean amplitude of the audio chunk
            chunk_mean = np.abs(indata).mean()
            self._debug_print(f"Chunk mean: {chunk_mean}")  # Debug line
            
            if chunk_mean > self.silence_threshold:
                self._debug_print("Sound detected, adding to audio queue.")
                audio_queue.put(indata.copy())
                total_frames += frames
                silence_count = 0
                first_sound_detected = True
            elif first_sound_detected:
                silence_count += 1
            
            # Print silence count
            self._debug_print(f"Silence count: {silence_count}")

            # Stop recording after silence, but ensure minimum duration
            current_duration = total_frames / self.sampling_rate
            if (current_duration >= 0.25 and 
                silence_count >= self.silence_count_threshold and 
                first_sound_detected):
                stop_recording = True
                self._debug_print("Silence detected, stopping recording.")

        try:
            # Print available devices
            self._debug_print("Available devices:")
            self._debug_print(sd.query_devices())
            
            self._debug_print("Listening for audio...")
            with sd.InputStream(
                callback=audio_callback,
                samplerate=self.sampling_rate,
                channels=self.num_channels,
                dtype=self.dtype
            ):
                while not stop_recording:
                    sd.sleep(250)

            self._debug_print("Recording stopped.")

            # Process audio queue
            self._debug_print("Processing audio data...")
            audio_data = np.empty((0, self.num_channels), dtype=self.dtype)
            while not audio_queue.empty():
                indata = audio_queue.get()
                audio_data = np.concatenate((audio_data, indata), axis=0)

            return audio_data

        except sd.PortAudioError as e:
            self._debug_print(f"PortAudio error: {e}")
            self._debug_print(f"Current audio settings:")
            self._debug_print(f"Sampling rate: {self.sampling_rate}")
            self._debug_print(f"Channels: {self.num_channels}")
            self._debug_print(f"Dtype: {self.dtype}")
            return None

    def _transcribe_audio(self, audio_data):
        """Transcribe audio data to text."""
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            audio_segment = AudioSegment(
                data=np.array(audio_data).tobytes(),
                sample_width=self.dtype.itemsize,
                frame_rate=self.sampling_rate,
                channels=self.num_channels
            )
            audio_segment.export(f.name, format="mp3")
            
            audio_file = open(f.name, "rb")
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
            return transcript.text

    def speak(self, text, callback=None):
        """Generate and play audio for the given text."""
        self._notify_speaking_state(True)
        
        try:
            # Generate audio stream
            audio_stream = self.eleven_client.generate(
                voice=self.voice_id,
                text=text,
                model="eleven_turbo_v2",
                stream=True
            )

            # Save audio to buffer
            audio_buffer = BytesIO()
            for chunk in audio_stream:
                if chunk:
                    audio_buffer.write(chunk)

            audio_buffer.seek(0)
            output_filename = "pet_response.mp3"
            with open(output_filename, "wb") as f:
                f.write(audio_buffer.getvalue())

            # Display animation and play audio if display is enabled
            if self.enable_display:
                self._display_talking_animation(output_filename)
            else:
                # Just play the audio without animation
                pygame.mixer.music.load(output_filename)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
            
            # Update conversation history
            self.messages.append({"role": "assistant", "content": text})
            if len(self.messages) > 12:
                self.messages = [self.messages[0]] + self.messages[-10:]
                
        finally:
            self._notify_speaking_state(False)
            if callback:
                callback()

    def think_response(self, user_input):
        """Process user input and generate a response."""
        if user_input is None:
            return None
            
        # Set talking flag
        self.talking = True
        
        try:
            base64_image = self._capture_image() if self.enable_vision else None
            
            # Prepare the user message
            if self.enable_vision and base64_image:
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            else:
                user_message = {"role": "user", "content": user_input}
            
            self.messages.append(user_message)
            
            # Get AI response based on model type
            if self.model.startswith("gpt"):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=1,
                )
                reply = response.choices[0].message.content
            elif self.model.startswith("groq"):
                model_name = self.model[5:]  # Remove "groq-" prefix
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    temperature=1,
                )
                reply = response.choices[0].message.content
            else:
                response = ollama.chat(model=self.model, messages=self.messages)
                reply = response['message']['content']
                
            # Check if we should ignore the response
            if reply.lower() == "ignore":
                print("Ignoring conversation...")
                self.messages = self.messages[:-1]
                return None
                
            return reply
            
        finally:
            self.talking = False

    def get_speaking_state(self):
        """Return whether the pet is currently speaking or processing speech."""
        return self.talking or self.is_speaking

    def cleanup(self):
        """Clean up resources."""
        pygame.quit()
        self.pygame_initialized = False

    def _calculate_ambient_noise_level(self):
        """Calculate the ambient noise level by sampling the environment."""
        self._debug_print("Calculating ambient noise level, please wait...")
        ambient_noise_data = []
        duration = 5  # seconds
        
        for _ in range(int(duration / duration)):
            audio_chunk = sd.rec(
                int(self.sampling_rate * duration), 
                samplerate=self.sampling_rate, 
                channels=self.num_channels, 
                dtype=self.dtype
            )
            sd.wait()
            ambient_noise_data.extend(audio_chunk)
            
        ambient_noise_level = np.abs(np.array(ambient_noise_data)).mean()
        self._debug_print(f"Ambient noise level: {ambient_noise_level}")
        return ambient_noise_level

    def _capture_image(self):
        """Capture an image from the webcam and return it as base64 string."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return None
        
        # Save the frame as an image file (optional, for debugging)
        cv2.imwrite('view.jpeg', frame)

        retval, buffer = cv2.imencode('.jpg', frame)
        if not retval:
            print("Failed to encode image.")
            return None

        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text

    def _display_talking_animation(self, audio_file):
        """Display talking animation while playing audio."""
        if not self.pygame_initialized:
            return

        # Load and play the audio
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Animation loop
        clock = pygame.time.Clock()
        mouth_open = False
        frame_count = 0

        while pygame.mixer.music.get_busy():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)

            # Switch mouth state every 5 frames
            if frame_count % 5 == 0:
                mouth_open = not mouth_open

            # Clear screen
            self.screen.fill((255, 255, 255))  # White background

            # Get current window size
            window_width, window_height = self.screen.get_size()

            # Select current image
            current_image = self.skull_open if mouth_open else self.skull_closed

            # Scale image to fit window while maintaining aspect ratio
            img_width = current_image.get_width()
            img_height = current_image.get_height()
            scale = min(window_width/img_width, window_height/img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            scaled_image = pygame.transform.scale(current_image, (new_width, new_height))

            # Center the image
            x = (window_width - new_width) // 2
            y = (window_height - new_height) // 2
            self.screen.blit(scaled_image, (x, y))

            pygame.display.flip()
            clock.tick(30)  # 30 FPS
            frame_count += 1

        # Ensure mouth is closed at the end
        self.screen.fill((255, 255, 255))
        scaled_closed = pygame.transform.scale(self.skull_closed, (new_width, new_height))
        self.screen.blit(scaled_closed, (x, y))
        pygame.display.flip()
