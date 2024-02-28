
# 🌈 ToyGeniusLab: Unleash Kids' Creativity with AI Toys 🚀

## Introduction
ToyGeniusLab invites kids into a world where they can create and personalize AI-powered toys. By blending technology with imaginative play, we not only empower young minds to explore their creativity but also help them become comfortable with harnessing AI, fostering tech skills in a fun and interactive way.

## Features
- **🎨 Customizable AI Toys:** Kids design their toy's personality and interactions.
- **📚 Educational:** A hands-on introduction to AI, programming, and technology.
- **💡 Open-Source:** A call to the community for ongoing enhancement of software and 3D-printed parts.
- **🤖 Future Enhancements:** Plans to add servos, displays, and more for a truly lifelike toy experience.


## Installation and Setup

### Prerequisites

- Python 3.x
- OpenAI API key
- Eleven Labs API key

### Clone the Repository

```bash
git clone https://github.com/sidu/toygeniuslab.git
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

By ensuring these settings, you'll get the optimal audio experience while interacting with the project.

## Bringing Your Mario AI Toy to Life
1. Download and print the [Mario template](https://www.cubeecraft.com/cubees/new-mario).
2. After pairing a Bluetooth speaker/microphone with your computer, insert it into the paper toy.
3. Execute the AI toy program by running python `pet.py mario.yaml` in your terminal. Get ready for interactive fun!

![image](https://github.com/SidU/ToyGeniusLab/assets/4107912/b37b084e-22e5-4c55-800e-9c57f1b6305a)


## Crafting Your Custom AI Toy
1. Begin with downloading the [blank template](https://cubeecraft-production.s3.us-east-2.amazonaws.com/public/about_downloads_cubeecraft_template.pdf.zip). You can digitally color it or use markers and crayons for a hands-on approach.
2. Insert a Bluetooth speaker/microphone into your custom-designed toy, ensuring it's paired with your computer first.
3. Make a copy of an existing toy's config by running `cp mario.yaml mytoy.yaml`.
4. Update the `system_prompt` property in `mytoy.yaml` according to the personality you want your toy to have.
5. Optionally, update the `voice_id` property in `mytoy.yaml` with the value of the voice you'd like your toy to have from [ElevenLabs.io](https://elevenlabs.io/app/voice-library).
6. Activate your AI toy by executing python `pet.py mytoy.yaml` in your terminal. Enjoy your creation's company!

## Share Your Adventures 📸 
Caught a fun moment with your AI toy? We'd love to see it! Share your experiences and creative toy designs on social media using the hashtag #ToyGeniusLab. Let's spread the joy and inspiration far and wide!

## License
MIT
