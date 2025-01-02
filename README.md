# 🌈 ToyGeniusLab: Unleash Kids' Creativity with AI Toys 🚀

## Introduction
ToyGeniusLab invites kids into a world where they can create and personalize AI-powered toys. By blending technology with imaginative play, we not only empower young minds to explore their creativity but also help them become comfortable with harnessing AI, fostering tech skills in a fun and interactive way.

![image](https://github.com/SidU/ToyGeniusLab/assets/4107912/ff22c5e9-ba1c-4f59-9c65-897e06419352)

[View Demo](https://www.youtube.com/shorts/VtCAA2GemKI)

## Features
- **🎨 Customizable AI Toys:** Kids design their toy's personality and interactions.
- **📚 Educational:** A hands-on introduction to AI, programming, and technology.
- **💡 Open-Source:** A call to the community for ongoing enhancement of software and 3D-printed parts.
- **🤖 Future Enhancements:** Plans to add servos, displays, and more for a truly lifelike toy experience.
- **🔊 Enhanced Audio Detection:** Improved silence detection and audio processing for better interactions.
- **🎯 Debug Mode:** Detailed feedback about audio levels and device status for easier troubleshooting.

## Quick Start

### Prerequisites

- Python 3.x
- OpenAI API key
- Eleven Labs API key
- FFmpeg (`brew install ffmpeg`)
- MPV (`brew install mpv`)
- Required Python packages (see requirements.txt)

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/sidu/toygeniuslab.git
cd toygeniuslab

# Install requirements
pip install -r requirements.txt

# Install system dependencies (macOS)
brew install ffmpeg mpv
```

### API Keys Setup

Set up your API keys as environment variables:

```bash
# OpenAI API Key
export OPENAI_API_KEY="your-api-key-here"

# Eleven Labs API Key
export ELEVEN_API_KEY="your-eleven-api-key-here"
```

### Running a Pet

The easiest way to get started is using the example application:

```bash
python example_usage.py --config configs/ghost.yaml
```

Try other configurations in the `configs/` directory or create your own!

### Audio Setup and Debugging

The system now includes enhanced audio debugging features:
- Displays available audio devices
- Shows real-time audio levels
- Provides detailed silence detection feedback
- Reports ambient noise levels

To optimize audio detection:
1. Run the program to see audio device information
2. Monitor the debug output for audio levels
3. Adjust silence threshold in config if needed

### Connecting Bluetooth Microphone and Speaker

Before running the project, make sure you have a portable Bluetooth microphone and speaker connected to your computer. Ensure that they are selected as the default input and output devices. For best experience, we recommend purchasing a mini bluetooth speaker/mic combo, like [LEICEX Mini Speaker from Amazon](https://www.amazon.com/LEICEX-Bluetooth-Portable-Wireless-Speakers/dp/B0BPNYY61M/) (costs ~$10).

#### Steps to Connect

1. Connect your Bluetooth microphone and speaker to your computer following the manufacturer's instructions.
  
2. **On Windows:**
    - Right-click on the Speaker icon in the taskbar and select "Open Sound settings."
    - Under the "Input" section, select your Bluetooth microphone from the dropdown.
    - Under the "Output" section, select your Bluetooth speaker from the dropdown.
  
3. **On macOS:**
    - Open "System Preferences" and click on "Sound."
    - Go to the "Input" tab and select your Bluetooth microphone.
    - Go to the "Output" tab and select your Bluetooth speaker.

## Bringing Your Mario AI Toy to Life
1. Download and print the [Mario template](https://www.cubeecraft.com/cubees/new-mario).
2. After pairing a Bluetooth speaker/microphone with your computer, insert it into the paper toy.
3. Execute the AI toy program by running python `pet.py mario.yaml` in your terminal. Get ready for interactive fun!

![image](https://github.com/SidU/ToyGeniusLab/assets/4107912/b37b084e-22e5-4c55-800e-9c57f1b6305a)

## Crafting Your Custom AI Toy
1. Begin with downloading the [blank template](https://cubeecraft-production.s3.us-east-2.amazonaws.com/public/about_downloads_cubeecraft_template.pdf.zip). You can digitally color it or use markers and crayons for a hands-on approach. You can also grab a slightly edited version of it [from our repo here](blank_template.png) (has a blank face for more creative options).
2. Insert a Bluetooth speaker/microphone into your custom-designed toy, ensuring it's paired with your computer first.
3. Make a copy of an existing toy's config by running `cp mario.yaml mytoy.yaml`.
4. Update the `system_prompt` property in `mytoy.yaml` according to the personality you want your toy to have.
5. Optionally, update the `voice_id` property in `mytoy.yaml` with the value of the voice you'd like your toy to have from [ElevenLabs.io](https://elevenlabs.io/app/voice-library).
6. Activate your AI toy by executing python `pet.py mytoy.yaml` in your terminal. Enjoy your creation's company!

## Share Your Adventures 📸 
Caught a fun moment with your AI toy? We'd love to see it! Share your experiences and creative toy designs on social media using the hashtag #ToyGeniusLab. Let's spread the joy and inspiration far and wide!

## Stay Updated with ToyGeniusLab
Love ToyGeniusLab? Give us a ⭐ on GitHub to stay connected and receive updates on new features, enhancements, and community contributions. Your support helps us grow and inspire more creative minds!

## Future Horizons 🌈
We're dreaming big for ToyGeniusLab's next steps and welcome your brilliance to bring these ideas to life. Here's what's on our horizon:

* More pets
* Solid local E2E execution: local LLM, local transcription, local TTS
* Local fast transcription and TTS
* SD based generation of custom pets
* Latency improvements
* Interruption handling
* Vision reasoning, with local VLLM support
* Servos for movement
* 3D printable characters
* "Pet in a box" (Raspberry-Pi)

Help shape ToyGeniusLab's tomorrow: Raise PRs for innovative features or spark conversations in our Discussions. 🌟

## How it works
Overview of how the toy works.

![image](https://github.com/SidU/ToyGeniusLab/assets/4107912/464578fe-9fb2-4f1e-9917-70838e1b8a85)

## Using TalkingPet in Your Projects

Want to create your own AI pet application? The `TalkingPet` class provides a simple way to create your own AI-powered interactive toys. Here's a basic example:

```python
from pet import TalkingPet

class PetApplication:
    def __init__(self, config_path):
        # Initialize pet with configuration
        self.pet = TalkingPet(config=self.load_config(config_path))
        
        # Add callback for speaking state changes
        self.pet.add_speaking_callback(self.speaking_state_changed)
        
    def speaking_state_changed(self, is_speaking):
        if is_speaking:
            print("\nSpeaking", end='', flush=True)
        else:
            print("\nSpeech finished!")
            
    def run(self):
        while True:
            # Listen for user input
            user_input = self.pet.listen()
            
            # Get AI response
            response = self.pet.think_response(user_input)
            
            # Speak the response
            self.pet.speak(response)
```

### TalkingPet Main APIs

#### Initialization
```python
pet = TalkingPet(config={
    'sampling_rate': 44100,
    'num_channels': 1,
    'dtype': 'float32',
    'silence_threshold': 0.01,
    'system_prompt': 'Your character prompt here',
    # ... other config options
})
```

#### Core Methods
- **listen()**: Records audio until silence is detected and returns transcribed text
  ```python
  user_input = pet.listen()  # Returns transcribed text or None
  ```

- **think_response(user_input)**: Generates AI response based on user input
  ```python
  response = pet.think_response("Hello!")  # Returns AI-generated response
  ```

- **speak(text)**: Converts text to speech and animates the pet
  ```python
  pet.speak("Hello, I'm your AI pet!")
  ```

#### Speaking State Callbacks
```python
def on_speaking_changed(is_speaking):
    print("Pet is speaking:" if is_speaking else "Pet finished speaking")

pet.add_speaking_callback(on_speaking_changed)
```

#### Other Useful Methods
- **get_speaking_state()**: Returns whether the pet is currently speaking
- **cleanup()**: Properly closes resources when done

### Example Configuration File
```yaml
sampling_rate: 44100
num_channels: 1
dtype: "float32"
silence_threshold: 0.01
ambient_noise_level_threshold_multiplier: 2.0
silence_count_threshold: 30
max_file_size_bytes: 10485760
enable_lonely_sounds: false
enable_squeak: false
system_prompt: "You are a friendly AI pet..."
voice_id: "your-eleven-labs-voice-id"
model: "gpt-4-vision-preview"
character_closed_mouth: "assets/closed.png"
character_open_mouth: "assets/open.png"
enable_vision: true
greetings:
  - "Hello!"
  - "Hi there!"
lonely_sounds:
  - "sound1.mp3"
  - "sound2.mp3"
```

See [example_usage.py](example_usage.py) for a complete implementation example.

## License
MIT