from abc import ABC, abstractmethod
from openai import OpenAI
from groq import Groq
import ollama

class ModelHandler(ABC):
    @abstractmethod
    def get_response(self, messages):
        pass

    @classmethod
    def create(cls, model_name):
        if model_name.startswith("gpt"):
            return GPTHandler(model_name)
        elif model_name.startswith("groq"):
            return GroqHandler(model_name)
        else:
            return OllamaHandler(model_name)

class OpenAIHandler(ModelHandler):
    def transcribe_audio(self, audio_file):
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
        return transcript.text 

class GPTHandler(OpenAIHandler):
    def __init__(self, model_name):
        self.client = OpenAI()
        self.model_name = model_name

    def get_response(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=1,
        )
        return response.choices[0].message.content

class GroqHandler(ModelHandler):
    def __init__(self, model_name):
        self.client = Groq()
        self.model_name = model_name[5:]  # Remove "groq-" prefix

    def get_response(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=1,
        )
        return response.choices[0].message.content

    def transcribe_audio(self, audio_file):
        # Implement Groq's audio transcription if available
        raise NotImplementedError("Audio transcription not supported for Groq")

class OllamaHandler(ModelHandler):
    def __init__(self, model_name):
        self.client = ollama.Client()
        self.model_name = model_name

    def get_response(self, messages):
        response = ollama.chat(model=self.model_name, messages=messages)
        return response['message']['content']

    def transcribe_audio(self, audio_file):
        # Implement Ollama's audio transcription if available
        raise NotImplementedError("Audio transcription not supported for Ollama")