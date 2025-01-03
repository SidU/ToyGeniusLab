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
from audio_processor import AudioProcessor

class AICharacterState:
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def can_proceed(self):
        now = time.time()
        # Remove old requests
        self.requests = [req for req in self.requests 
                        if now - req < self.time_window]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

class AICharacterMetrics:
    def __init__(self):
        self.total_interactions = 0
        self.successful_interactions = 0
        self.failed_interactions = 0
        self.average_response_time = 0
        self.start_time = time.time()

class AICharacter:
    def __init__(self, config, debug=False):
        """Initialize an AI character with configuration dictionary.
        
        Args:
            config (dict): Configuration dictionary containing character settings
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
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.volume = 1.0  # Default volume
        self.state = AICharacterState.IDLE
        self.state_callbacks = []
        self.rate_limiter = RateLimiter(max_requests=20, time_window=60)  # 20 requests per minute
        self.metrics = AICharacterMetrics()
        self.audio_processor = AudioProcessor(config, debug=debug)

        # Create temp directory for audio files
        self.temp_dir = os.path.join(os.getcwd(), "temp_audio")
        os.makedirs(self.temp_dir, exist_ok=True)

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

        # Optional settings with their types and defaults
        optional_settings = {
            'character_closed_mouth': (str, None),
            'character_open_mouth': (str, None),
            'max_message_history': (int, 20),  # Added with default value of 20
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

        # Set optional settings with defaults
        for key, (expected_type, default_value) in optional_settings.items():
            if key in config:
                try:
                    value = expected_type(config[key])
                    setattr(self, key, value)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid value for {key}, using default: {default_value}")
                    setattr(self, key, default_value)
            else:
                setattr(self, key, default_value)

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
        try:
            # Initialize ElevenLabs client
            self.eleven_client = ElevenLabs(api_key=os.environ.get("ELEVEN_API_KEY"))
            if not os.environ.get("ELEVEN_API_KEY"):
                raise ValueError("ELEVEN_API_KEY environment variable not set")

            # Initialize pygame and display only if character images are provided
            if self.enable_display:
                self._debug_print("Initializing display with character images")
                pygame.init()
                self.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
                pygame.display.set_caption("Talking Pet")

                # Load character images
                self.skull_closed = pygame.image.load(self.character_closed_mouth)
                self.skull_open = pygame.image.load(self.character_open_mouth)
                self.pygame_initialized = True
            else:
                self._debug_print("No character images provided, running without display")
                self.pygame_initialized = False

            # Initialize sound effects if enabled
            if self.enable_squeak:
                pygame.mixer.music.load("filler.mp3")

        except Exception as e:
            print(f"Error initializing components: {e}")
            self.pygame_initialized = False

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
        self.set_state(AICharacterState.LISTENING)
        try:
            audio_data = self._listen_until_silence()
            if audio_data is None or len(audio_data) == 0:
                return None
            
            return self._transcribe_audio(audio_data)
        finally:
            self.set_state(AICharacterState.IDLE)

    def _listen_until_silence(self):
        """Record audio until silence is detected."""
        return self.audio_processor.listen_until_silence()

    def _transcribe_audio(self, audio_data):
        """Transcribe audio data to text."""
        if audio_data is None:
            return None
            
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            audio_segment = self.audio_processor.create_audio_segment(audio_data)
            audio_segment.export(f.name, format="mp3")
            
            audio_file = open(f.name, "rb")
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
            return transcript.text

    def speak(self, text, callback=None):
        """Generate and play audio for the given text asynchronously."""
        def audio_worker():
            self.set_state(AICharacterState.SPEAKING)
            try:
                self._notify_speaking_state(True)
                output_filename = os.path.join(self.temp_dir, f"pet_response_{int(time.time())}.mp3")
                
                try:
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
                    with open(output_filename, "wb") as f:
                        f.write(audio_buffer.getvalue())

                    # Display animation and play audio if display is enabled
                    if self.enable_display:
                        self._display_talking_animation(output_filename)
                    else:
                        pygame.mixer.music.load(output_filename)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            pygame.time.wait(100)
                    
                    # Update conversation history
                    self.messages.append({"role": "assistant", "content": text})
                    if len(self.messages) > self.max_message_history:
                        self.messages = [self.messages[0]] + self.messages[-(self.max_message_history-1):]
                        
                finally:
                    self._notify_speaking_state(False)
                    if callback:
                        callback()
                    # Clean up the temporary file
                    if os.path.exists(output_filename):
                        os.remove(output_filename)

            finally:
                self.set_state(AICharacterState.IDLE)
                self._notify_speaking_state(False)
                if callback:
                    callback()
                # Clean up the temporary file
                if os.path.exists(output_filename):
                    os.remove(output_filename)

        self.audio_thread = threading.Thread(target=audio_worker)
        self.audio_thread.start()

    def think_response(self, user_input):
        """Process user input and generate a response with rate limiting."""
        if not self.rate_limiter.can_proceed():
            return "I need a moment to process. Please try again shortly."
        
        if user_input is None:
            return None
            
        self.set_state(AICharacterState.THINKING)
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
                
            # Update message history with size limit
            if len(self.messages) > self.max_message_history:
                self.messages = [self.messages[0]] + self.messages[-(self.max_message_history-1):]
                
            return reply
            
        finally:
            self.set_state(AICharacterState.IDLE)

    def get_speaking_state(self):
        """Return whether the pet is currently speaking or processing speech."""
        return self.talking or self.is_speaking

    def cleanup(self):
        """Clean up resources and ensure graceful shutdown."""
        try:
            # Stop any ongoing audio
            if self.pygame_initialized:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                pygame.quit()
            
            # Wait for audio thread to complete
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=1.0)
            
            # Clear message history
            self.messages.clear()
            
            # Reset state
            self.state = AICharacterState.IDLE
            self.pygame_initialized = False
            
            # Clean up temp directory
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Error removing file {file_path}: {e}")
                try:
                    os.rmdir(self.temp_dir)
                except Exception as e:
                    print(f"Error removing temp directory: {e}")
                
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def _capture_image(self):
        """Capture an image from the webcam and return it as base64 string."""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Cannot open camera")
                return None

            ret, frame = cap.read()
            cap.release()  # Important: manually release the capture

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return None
            
            # Save the frame as an image file (optional, for debugging)
            cv2.imwrite('view.jpeg', frame)

            retval, buffer = cv2.imencode('.jpg', frame)
            if not retval:
                print("Failed to encode image.")
                return None

            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None

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

    def set_volume(self, volume):
        """Set the volume level (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
        if self.pygame_initialized:
            pygame.mixer.music.set_volume(self.volume)

    def set_state(self, new_state):
        """Update the character's state and notify callbacks."""
        self.state = new_state
        for callback in self.state_callbacks:
            try:
                callback(new_state)
            except Exception as e:
                self._debug_print(f"Error in state callback: {e}")

    def _safe_retry(self, func, max_retries=3, delay=1):
        """Safely retry a function with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self._debug_print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(delay * (2 ** attempt))

    def get_metrics(self):
        """Return current metrics."""
        uptime = time.time() - self.metrics.start_time
        return {
            "uptime": uptime,
            "total_interactions": self.metrics.total_interactions,
            "success_rate": (self.metrics.successful_interactions / 
                            max(1, self.metrics.total_interactions)),
            "average_response_time": self.metrics.average_response_time
        }
