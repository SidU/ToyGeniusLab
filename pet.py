from openai import OpenAI
import sounddevice as sd
import numpy as np
import tempfile
import pygame
import threading
import random
from pydub import AudioSegment
from elevenlabs import generate, stream
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

client = OpenAI()
groqClient = Groq()

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

def get_pet_reply(user_input):
    global messages
    
    # Now get ready to generate the assistant's reply
    messages.append({"role": "user", "content": user_input})
    
    # If model starts with "gpt" then use the chat endpoint from OpenAI
    if model.startswith("gpt"):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
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
    audio_stream = generate(
        voice=voice_id,
        text=text,
        model="eleven_turbo_v2",
        stream=True
    )

    # Stream the generated audio
    stream(audio_stream)

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
    global silence_threshold

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

    # Start recording audio
    with sd.InputStream(callback=audio_callback, samplerate=sampling_rate, channels=num_channels, dtype=dtype):
        while not stop_recording:
            sd.sleep(250)  # Sleep for a short time to prevent busy waiting (gap between pause and processing?)

    print("Recording stopped.")

    # Process audio data from the queue
    audio_data = process_audio_from_queue()

    return audio_data


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

def get_image_description(base64_image):
    messages = [
    {
        "role": "system",
        "content": [
                {"type": "text", "text": "You are a pet animal."},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Imagine you are a pet observing this scene. Describe what you see from your perspective. Focus on the humans, objects, and the environment around you. How do these elements appear and interact from your viewpoint as a pet?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                },
            },
        ],
    }
    ]

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        "max_tokens": 300,
    }

    # Make the API call to GPT-4 Vision Model
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Extract the content from the GPT-4 Vision Model's response
    vision_response = response.json()
    vision_content = vision_response['choices'][0]['message']['content']

    return vision_content

def get_pet_view():
    # Capture image from webcam
    image = capture_webcam_as_base64()

    # Get image description from OpenAI API
    image_description = get_image_description(image)

    return image_description

def update_vision(view_description):
    # Update system prompt by adding suffix, "; Pet's current view: {view_description}"
  
    print(f"Updating vision: {view_description}")
    global system_prompt
    system_prompt = settings['system_prompt'] + f"; You currently see: {view_description}"

    # Update the first message in the messages list
    global messages
    messages[0]["content"] = system_prompt

def main():

    global talking, messages, stop_recording

    initialize()

    say(get_random_greeting())

    print("Listening...")

    # Main loop
    while True:
        talking = False  # We start off quiet

        # Listen for audio until we detect silence
        audio_data = listen_audio_until_silence()

        # Use ThreadPoolExecutor to run tasks in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the operations and mark each future with its operation

            if enable_vision:
                future_pet_view = executor.submit(get_pet_view)

            future_user_input = executor.submit(get_user_input_from_audio, audio_data)

            # Wait for both tasks to complete
            if enable_vision:
                view_description = future_pet_view.result()

            user_input = future_user_input.result()

        # Update view with pet vision
        if enable_vision:
            update_vision(view_description)

        print(f"User: {user_input}")

        pet_reply = get_pet_reply(user_input)
        print(f"Pet: {pet_reply}")

        # convert to lowercase and check if string is "ignore"
        if pet_reply.lower() == "ignore":
            print("Ignoring conversation...")
            messages = messages[:-1]

        # Speak the assistant's reply
        say(pet_reply)

if __name__ == "__main__":
    main()
