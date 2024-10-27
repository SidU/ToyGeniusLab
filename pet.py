from openai import OpenAI
import sounddevice as sd
import numpy as np
import tempfile
import pygame
import threading
import random
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import yaml
import cv2
import base64
import os
import requests
import concurrent.futures
import sys
from queue import Queue
import ollama
from groq import Groq
from io import BytesIO
from pydub.playback import play
import queue
from scipy.io import wavfile
import time

eleven_client = ElevenLabs(
    api_key=os.environ.get("ELEVEN_API_KEY")
)

# Get your OpenAI API Key from the environment variable
api_key = os.environ.get("OPENAI_API_KEY")

# Check if the filename is provided as a command-line argument
if len(sys.argv) < 2:
    print("Error: No YAML file name provided.")
    sys.exit(1)

filename = sys.argv[1]

try:
    with open(filename, 'r') as file:
        settings = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Error: The file {filename} was not found.")
    sys.exit(1)

sampling_rate = settings['sampling_rate']
num_channels = settings['num_channels']
dtype = np.dtype(settings['dtype'])  # Convert string to numpy dtype
silence_threshold = settings['silence_threshold']
ambient_noise_level_threshold_multiplier = settings['ambient_noise_level_threshold_multiplier']
max_file_size_bytes = settings['max_file_size_bytes']
enable_lonely_sounds = settings['enable_lonely_sounds']
enable_squeak = settings['enable_squeak']
system_prompt = settings['system_prompt']
voice_id = settings['voice_id']
greetings = settings['greetings']
lonely_sounds = settings['lonely_sounds']
enable_vision = settings['enable_vision']
silence_count_threshold = settings['silence_count_threshold']
model = settings['model']

# Initialize the client based on the model
if model.startswith("gpt"):
    client = OpenAI()
elif model.startswith("groq"):
    groqClient = Groq()
else:
    ollama_client = ollama.Client()

# Initialize messages
messages = [
    {"role": "system", "content": system_prompt}
]

# Function to get a random greeting
def get_random_greeting():
    return random.choice(greetings)

def play_lonely_sound():
    global silence_threshold  # Access the global silence_threshold
    if not talking:
        original_silence_threshold = silence_threshold
        silence_threshold = 1000  # Increase threshold temporarily
        lonely_file = random.choice(lonely_sounds)
        print(f"Playing lonely sound: {lonely_file}")
        pygame.mixer.music.load(lonely_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass  # Wait for the sound to finish playing
        silence_threshold = original_silence_threshold  # Reset to original threshold
    else:
        print("Not playing lonely sound because the mouse is talking.")

def calculate_ambient_noise_level(duration=5):
    print("Calculating ambient noise level, please wait...")
    ambient_noise_data = []
    for _ in range(int(duration / duration)):  # duration / duration = number of chunks
        audio_chunk = sd.rec(int(sampling_rate * duration), samplerate=sampling_rate, channels=num_channels, dtype=dtype)
        sd.wait()
        ambient_noise_data.extend(audio_chunk)
    ambient_noise_level = np.abs(np.array(ambient_noise_data)).mean()
    print(f"Ambient noise level: {ambient_noise_level}")

    return ambient_noise_level

def get_pet_reply(user_input, base64_image=None):
    global messages
    
    # Prepare the user message
    if enable_vision and base64_image:
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
    
    messages.append(user_message)
    
    # If model starts with "gpt" then use the chat endpoint from OpenAI
    if model.startswith("gpt"):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
        )
        return response.choices[0].message.content
    
    elif model.startswith("groq"):
        # Remove "grok-" from the model name
        model_name = model[5:]
        response = groqClient.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1,
        )
        return response.choices[0].message.content

    else: # Use ollama local deployment
        print(f"Using Ollama model: {model}")
        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']

def say(text):
    global messages

    if enable_squeak:
        # Stop the sound effect because we are about to speak.
        pygame.mixer.music.stop()

    # Generate audio stream for the assistant's reply
    audio_stream = eleven_client.generate(
        voice=voice_id,
        text=text,
        model="eleven_turbo_v2",
        stream=True
    )

    # Create a BytesIO object to hold audio data
    audio_buffer = BytesIO()

    # Stream the generated audio
    for chunk in audio_stream:
        if chunk:
            audio_buffer.write(chunk)

    # Reset buffer position to the beginning
    audio_buffer.seek(0)

    # Save the audio buffer to an MP3 file
    output_filename = "pet_response.mp3"
    with open(output_filename, "wb") as f:
        f.write(audio_buffer.getvalue())

    print(f"Saved audio response to {output_filename}")

    # Display the talking pet while playing the audio
    display_talking_pet(output_filename)

    messages.append({"role": "assistant", "content": text})

    if len(messages) > 12:
        messages = [messages[0]] + messages[-10:]

def get_user_input_from_audio(audio_data):

    global talking

    # Create a temporary mp3 file to save the audio and then transcribe it.
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        audio_segment = AudioSegment(
            data=np.array(audio_data).tobytes(),
            sample_width=dtype.itemsize,
            frame_rate=sampling_rate,
            channels=num_channels
        )
        audio_segment.export(f.name, format="mp3")
        f.seek(0)

        talking = True  # Set the talking variable to True so that the lonely sound doesn't play

        # Print file name
        print(f"File name: {f.name}")
        
        # Copy file to current directory
        os.system(f"cp {f.name} ./audio.mp3")

        if enable_squeak:
            # Play the sound effect
            pygame.mixer.music.play()

        # Transcribe audio using OpenAI API
        audio_file = open(f.name, "rb")
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )

        # Get the user input from the transcript
        user_input = transcript.text

        return user_input

def initialize():
    global silence_threshold, screen, skull_closed, skull_open, pygame_initialized

     # Calculate the ambient noise level and set it as the silence threshold
    ambient_noise_level = calculate_ambient_noise_level()
    silence_threshold = ambient_noise_level * ambient_noise_level_threshold_multiplier  # Adjust multiplier as needed

    # Set silence_threshold to a minimum value
    silence_threshold = max(silence_threshold, 10.0)

    # Print the silence threshold
    print(f"Silence threshold: {silence_threshold}")

    if enable_lonely_sounds:
        # Initialize the periodic lonely sound timer
        timer = threading.Timer(60, play_lonely_sound)
        timer.start()

     # Initialize pygame mixer for sound effects
    pygame.mixer.init()

    if enable_squeak:
        pygame.mixer.music.load("filler.mp3")

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Talking Pet")
    pygame_initialized = True

    # Load skull images
    skull_closed = pygame.image.load(settings['character_closed_mouth'])
    skull_open = pygame.image.load(settings['character_open_mouth'])
    
    # Print image sizes for debugging
    print(f"Closed mouth image size: {skull_closed.get_size()}")
    print(f"Open mouth image size: {skull_open.get_size()}")

def analyze_audio_chunk(audio_chunk):
    samples = np.array(audio_chunk.get_array_of_samples())
    if len(samples) == 0:
        return False
    rms = np.sqrt(np.mean(samples.astype(float)**2))
    mouth_open = rms > 50  # Lowered threshold, adjust as needed
    print(f"RMS: {rms:.2f}, Mouth open: {mouth_open}")  # Debug print
    return mouth_open

def analyze_audio_file(audio_file):
    # Convert mp3 to wav
    audio = AudioSegment.from_mp3(audio_file)
    audio.export("temp.wav", format="wav")
    
    # Read the wav file
    sample_rate, audio_data = wavfile.read("temp.wav")
    
    # If stereo, convert to mono
    if len(audio_data.shape) == 2:
        audio_data = audio_data.mean(axis=1)
    
    # Normalize the audio data
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Calculate energy
    frame_length = int(sample_rate * 0.03)  # 30ms frames
    energy = []
    for i in range(0, len(audio_data), frame_length):
        frame = audio_data[i:i+frame_length]
        energy.append(np.sum(frame**2))
    
    # Normalize energy
    energy = np.array(energy)
    energy = energy / np.max(energy)
    
    return energy, len(audio_data) / sample_rate

def play_and_analyze_audio(audio_file, mouth_queue):
    energy, duration = analyze_audio_file(audio_file)
    
    # Play the audio file
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    
    # Send mouth states to the queue
    start_time = time.time()
    for e in energy:
        mouth_open = e > 0.1  # Adjust this threshold as needed
        mouth_queue.put(mouth_open)
        time.sleep(0.03)  # 30ms per frame
        
        # Check if we've reached the end of the audio
        if time.time() - start_time > duration:
            break
    
    # Signal end of audio
    mouth_queue.put(None)

def resize_image(image, window_size):
    """Resize image to fit the window while maintaining aspect ratio."""
    img_w, img_h = image.get_size()
    win_w, win_h = window_size
    aspect_ratio = img_w / img_h
    if win_w / win_h > aspect_ratio:
        new_h = win_h
        new_w = int(new_h * aspect_ratio)
    else:
        new_w = win_w
        new_h = int(new_w / aspect_ratio)
    return pygame.transform.smoothscale(image, (new_w, new_h))

def display_talking_pet(audio_file):
    global pygame_initialized
    mouth_queue = queue.Queue()

    # Start playing audio and analyzing in a separate thread
    audio_thread = threading.Thread(target=play_and_analyze_audio, args=(audio_file, mouth_queue))
    audio_thread.start()

    running = True
    mouth_open = False
    frame_count = 0
    fullscreen = False

    # Set the initial window size
    screen = pygame.display.set_mode((1024, 1024), pygame.RESIZABLE)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame_initialized = False
            elif event.type == pygame.VIDEORESIZE:
                if not fullscreen:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:  # Press 'f' to toggle fullscreen
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode((1024, 1024), pygame.RESIZABLE)

        # Set background to black
        screen.fill((0, 0, 0))

        # Update mouth state from the queue
        try:
            mouth_state = mouth_queue.get_nowait()
            if mouth_state is None:
                mouth_open = False  # Close mouth when audio ends
                running = False  # Stop the loop when we receive the end signal
            else:
                mouth_open = mouth_state
        except queue.Empty:
            pass  # No new mouth state, keep the current state

        # Get the appropriate skull image
        skull_image = skull_open if mouth_open else skull_closed

        # Resize the image to fit the current window size
        resized_image = resize_image(skull_image, screen.get_size())
        
        # Calculate position to center the image
        pos_x = (screen.get_width() - resized_image.get_width()) // 2
        pos_y = (screen.get_height() - resized_image.get_height()) // 2

        # Blit the resized image
        screen.blit(resized_image, (pos_x, pos_y))

        pygame.display.update()

        frame_count += 1
        if frame_count % 10 == 0:  # Print debug info every 10 frames
            print(f"Frame {frame_count}, Mouth open: {mouth_open}")

    # Ensure the mouth is closed at the end
    screen.fill((0, 0, 0))
    resized_closed = resize_image(skull_closed, screen.get_size())
    pos_x = (screen.get_width() - resized_closed.get_width()) // 2
    pos_y = (screen.get_height() - resized_closed.get_height()) // 2
    screen.blit(resized_closed, (pos_x, pos_y))
    pygame.display.update()

    # Wait for the audio thread to finish
    audio_thread.join()
    pygame.mixer.music.stop()

    # Keep the closed mouth image displayed for a short time
    pygame.time.wait(500)  # Wait for 500 ms

# Global list to accumulate audio data
audio_queue = Queue()
silence_count = 0
stop_recording = False
first_sound_detected = False
total_frames = 0

def audio_callback(indata, frames, time, status):
    global audio_queue, silence_count, stop_recording, first_sound_detected, total_frames
    
    # Compute the mean amplitude of the audio chunk
    chunk_mean = np.abs(indata).mean()
    print(f"Chunk mean: {chunk_mean}")  # Debugging line
    
    if chunk_mean > silence_threshold:
        
        print("Sound detected, adding to audio queue.")
        audio_queue.put(indata.copy())
        total_frames += frames

        silence_count = 0
        first_sound_detected = True

    elif first_sound_detected:

        silence_count += 1
    
    # Print silence count
    print(f"Silence count: {silence_count}")

    # Stop recording after a certain amount of silence, but make sure it's at least 0.25 second long.
    current_duration = total_frames / sampling_rate

    if current_duration >= 0.25 and silence_count >= silence_count_threshold and first_sound_detected:
        stop_recording = True
        print("Silence detected, stopping recording.")

def process_audio_from_queue():

    print("Processing audio data...")

    audio_data = np.empty((0, num_channels), dtype=dtype)

    while not audio_queue.empty():
        # Get data from the queue
        indata = audio_queue.get()
 
        # Append the chunk to the audio data
        audio_data = np.concatenate((audio_data, indata), axis=0)

    return audio_data

def listen_audio_until_silence():
    global audio_queue, silence_count, stop_recording, first_sound_detected, total_frames

    # Reset audio data and silence count
    print("Listening for audio...")
    audio_queue = Queue()
    silence_count = 0
    stop_recording = False
    first_sound_detected = False
    total_frames = 0

    # Print available devices
    print("Available devices:")
    print(sd.query_devices())

    try:
        # Start recording audio
        with sd.InputStream(callback=audio_callback, samplerate=sampling_rate, channels=num_channels, dtype=dtype):
            while not stop_recording:
                sd.sleep(250)  # Sleep for a short time to prevent busy waiting

        print("Recording stopped.")

        # Process audio data from the queue
        audio_data = process_audio_from_queue()

        return audio_data

    except sd.PortAudioError as e:
        print(f"PortAudio error: {e}")
        print(f"Current audio settings:")
        print(f"Sampling rate: {sampling_rate}")
        print(f"Channels: {num_channels}")
        print(f"Dtype: {dtype}")
        return None

def capture_webcam_as_base64():

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None
    
    # Save the frame as an image file
    cv2.imwrite('view.jpeg', frame)

    retval, buffer = cv2.imencode('.jpg', frame)
    if not retval:
        print("Failed to encode image.")
        return None

    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

def main():
    global talking, messages, stop_recording, pygame_initialized

    initialize()

    say(get_random_greeting())

    print("Listening...")

    # Main loop
    while pygame_initialized:
        talking = False  # We start off quiet

        # Listen for audio until we detect silence
        audio_data = listen_audio_until_silence()

        # Use ThreadPoolExecutor to run tasks in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_user_input = executor.submit(get_user_input_from_audio, audio_data)
            
            if enable_vision:
                future_pet_view = executor.submit(capture_webcam_as_base64)

            # Wait for tasks to complete
            user_input = future_user_input.result()
            base64_image = future_pet_view.result() if enable_vision else None

        print(f"User: {user_input}")

        pet_reply = get_pet_reply(user_input, base64_image)
        print(f"Pet: {pet_reply}")

        # convert to lowercase and check if string is "ignore"
        if pet_reply.lower() == "ignore":
            print("Ignoring conversation...")
            messages = messages[:-1]
        else:
            # Speak the assistant's reply
            say(pet_reply)

    pygame.quit()

if __name__ == "__main__":
    main()
