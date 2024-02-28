🎃 MouseGPT: A Halloween Toy Mouse Powered by AI 🐭

## Introduction

MouseGPT is a spook-tacular Halloween toy mouse that talks to trick-or-treaters. Leveraging GPT-3.5 Turbo and ElevenLabs, the mouse offers a unique conversational experience, packed with humor and wit.

![mousegpt_cover](https://github.com/SidU/mousegpt/assets/4107912/84985806-d443-4801-af23-4b3c6bff49d1)

[Watch ChatGPT vs MouseGPT demo](https://www.youtube.com/watch?v=aFIaXpRkP18)
Gets really fun around 2:48 timestamp into the video 🎃

## Features

- 🗨️ Conversational AI using OpenAI's GPT-3.5 Turbo
- 🎤 Real-time audio recording and playback
- 🐭 Quirky character: a funny, sarcastic mouse trapped in a Halloween box
- 🌐 Easy-to-run Python script

## Installation and Setup

### Prerequisites

- Python 3.x
- OpenAI API key

### Clone the Repository

```bash
git clone https://github.com/yourusername/HauntedMouseGPT.git
```

### Install Requirements

Navigate to the project directory and run:

```bash
pip install -r requirements.txt
```

Install ffmpeg
`brew install ffmpeg`

Install mpv
`brew install mpv`

## Environmental Variables Setup

Before running the project, you'll need to set up two essential environment variables: `OPENAI_API_KEY` and `ELEVEN_API_KEY`.

### Setting up OPENAI_API_KEY

1. Visit [OpenAI API Dashboard](https://platform.openai.com/account/api-keys) to obtain your OpenAI API key.
2. Once you have your key, set it as an environment variable. On Unix-based systems, you can use the following command:
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```
    On Windows, you can set it through the command prompt:
    ```bash
    set OPENAI_API_KEY=your-api-key-here
    ```

### Setting up ELEVEN_API_KEY

1. To get the Eleven API key, follow the guide available at [Eleven Labs Documentation](https://docs.elevenlabs.io/introduction).
2. Similar to the OpenAI API key, set the Eleven API key as an environment variable:
    ```bash
    # On Unix-based systems
    export ELEVEN_API_KEY="your-eleven-api-key-here"
    ```
    ```bash
    # On Windows
    set ELEVEN_API_KEY=your-eleven-api-key-here
    ```

Now you're ready to run the project with both API keys set up.

### Connecting Bluetooth Microphone and Speaker

Before running the project, make sure you have a portable Bluetooth microphone and speaker connected to your computer. Ensure that they are selected as the default input and output devices.

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

By ensuring these settings, you'll get the optimal audio experience while interacting with the project.

## How to Use

1. Run `pet.py`.
    ```bash
    python pet.py
    ```
2. Place the hardware setup near your Halloween decorations.
3. Enjoy the quirky conversations your haunted mouse has with trick-or-treaters!

## Configuration

Edit the `pet.py` file to configure system parameters such as recording duration, silence threshold, and so on.

## Spook away!
<img width="519" alt="image" src="https://github.com/SidU/mousegpt/assets/4107912/820e8273-891a-4bcd-b835-53e946e1e067">


## Credits

This project uses the following libraries:

- OpenAI
- ElevenLabs
- sounddevice
- pydub
- numpy
- Mouse squeak sound-effect from [FreeSoundsLibrary](https://www.freesoundslibrary.com/mouse-squeaking-noise)

## License

MIT

---

For any questions or suggestions, feel free to open an issue or a pull request. Happy Haunting! 🎃👻