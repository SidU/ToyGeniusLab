import os
import yaml
from ai_character import AICharacter

class CharacterApplication:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.character = AICharacter(config=self.config, debug=True)
        self.character.add_speaking_callback(self.speaking_state_changed)
        self.running = False
        
    def speaking_state_changed(self, is_speaking):
        if is_speaking:
            print("\nSpeaking", end='', flush=True)
        else:
            print("\nSpeech finished!")
            
    def run(self):
        self.running = True
        try:
            while self.running:
                # Listen for input
                user_input = self.character.listen()

                # Print user input
                print(f"\nUser: {user_input}")
                
                # Get AI response (renamed from chat to think_response)
                response = self.character.think_response(user_input)
                
                # Print response
                print(f"\nPet: {response}")
                
                # Speak response (no need for progress callback anymore)
                self.character.speak(response)
                
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        self.running = False
        # Add any cleanup code here if needed
        
    def load_config(self, config_path):
        if config_path is None:
            print("No config file specified. Using default settings.")
        elif os.path.exists(config_path):
            # Load config
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(
                f"Configuration file not found: '{config_path}'\n"
                f"Please specify a valid path using --config"
            )
            
if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run the Talking Pet application')
    parser.add_argument('--config', '-c',
                       default='config.yaml',
                       help='Path to the configuration file (default: config.yaml)')
    
    args = parser.parse_args()
    
    # Create and run application with specified config
    app = CharacterApplication(config_path=args.config)
    app.run() 