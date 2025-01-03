import sounddevice as sd
import numpy as np
import tempfile
from pydub import AudioSegment
from queue import Queue
import pygame
import os
import time
from io import BytesIO

class AudioProcessor:
    def __init__(self, config, debug=False):
        self.debug = debug
        self.sampling_rate = config['sampling_rate']
        self.num_channels = config['num_channels']
        self.dtype = np.dtype(config['dtype'])
        self.silence_threshold = config['silence_threshold']
        self.silence_count_threshold = config['silence_count_threshold']
        self.ambient_noise_level_threshold_multiplier = config['ambient_noise_level_threshold_multiplier']
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init()
            self.pygame_initialized = True
        except pygame.error as e:
            print(f"Warning: Could not initialize pygame mixer: {e}")
            self.pygame_initialized = False

        # Calculate initial ambient noise level
        self.ambient_noise_level = self._calculate_ambient_noise_level()
        self.silence_threshold = max(
            self.ambient_noise_level * self.ambient_noise_level_threshold_multiplier,
            10.0
        )

    def _debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

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

    def listen_until_silence(self):
        """Record audio until silence is detected."""
        audio_queue = Queue()
        silence_count = 0
        stop_recording = False
        first_sound_detected = False
        total_frames = 0

        def audio_callback(indata, frames, time, status):
            try:
                if status:
                    self._debug_print(f"PortAudio Status: {status}")

                nonlocal silence_count, stop_recording, first_sound_detected, total_frames
                
                chunk_mean = np.abs(indata).mean()
                self._debug_print(f"Chunk mean: {chunk_mean}, threshold: {self.silence_threshold}")
                
                if chunk_mean > self.silence_threshold:
                    self._debug_print("Sound detected, adding to audio queue.")
                    audio_queue.put(indata.copy())
                    total_frames += frames
                    silence_count = 0
                    first_sound_detected = True
                elif first_sound_detected:
                    silence_count += 1
                
                self._debug_print(f"Silence count: {silence_count}")

                current_duration = total_frames / self.sampling_rate
                if (current_duration >= 0.25 and 
                    silence_count >= self.silence_count_threshold and 
                    first_sound_detected):
                    stop_recording = True
                    self._debug_print(f"Silence detected, stopping recording. Duration: {current_duration:.2f} seconds")
            
            except Exception as e:
                self._debug_print(f"Error in audio callback: {e}")
                stop_recording = True

        try:
            self._debug_print("Listening for audio...")
            with sd.InputStream(
                device=0,
                callback=audio_callback,
                samplerate=self.sampling_rate,
                blocksize=1024,
                latency='high',
                channels=self.num_channels,
                dtype=self.dtype
            ):
                while not stop_recording:
                    sd.sleep(250)

            audio_data = np.empty((0, self.num_channels), dtype=self.dtype)
            while not audio_queue.empty():
                indata = audio_queue.get()
                audio_data = np.concatenate((audio_data, indata), axis=0)

            return audio_data

        except Exception as e:
            self._debug_print(f"Error recording audio: {e}")
            return None

    def play_audio(self, audio_file, callback=None):
        """Play audio file with pygame mixer."""
        if not self.pygame_initialized:
            return False

        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            if callback:
                callback()
            return True
        except Exception as e:
            self._debug_print(f"Error playing audio: {e}")
            return False

    def create_audio_segment(self, audio_data):
        """Convert numpy audio data to AudioSegment."""
        return AudioSegment(
            data=np.array(audio_data).tobytes(),
            sample_width=self.dtype.itemsize,
            frame_rate=self.sampling_rate,
            channels=self.num_channels
        ) 