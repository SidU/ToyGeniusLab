import argparse
from ai_character import AICharacter
import yaml
import sys
import time

class AICharacterAgent:
    """An agent that manages an AICharacter's interactions and lifecycle.
    
    The agent handles:
    - Character initialization and cleanup
    - Continuous interaction loop
    - User feedback during interactions
    - Conversation state management
    - Progress indicators
    """
    
    def __init__(self, config_path, debug=False):
        """Initialize the agent with a character configuration.
        
        Args:
            config_path (str): Path to character configuration YAML
            debug (bool): Enable debug output
        """
        self.config = self.load_config(config_path)
        self.character = AICharacter(config=self.config, debug=debug)
        self.character.add_speaking_callback(self.on_speaking_state_changed)
        self.running = True

    def load_config(self, config_path):
        """Load character configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: Configuration file {config_path} not found.")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)

    def on_speaking_state_changed(self, is_speaking):
        """Handle character speaking state changes.
        
        Provides visual feedback during character speech.
        """
        if is_speaking:
            print("\nCharacter is speaking...", end='', flush=True)
        else:
            print("\nCharacter finished speaking!")

    def show_listening_indicator(self):
        """Display a visual indicator while listening for user input."""
        print("\nListening...", end='', flush=True)

    def show_thinking_indicator(self):
        """Display a visual indicator while character is thinking."""
        print("\nThinking...", end='', flush=True)

    def wait_for_speaking_to_complete(self):
        """Wait until the character has finished speaking."""
        while self.character.is_speaking:
            time.sleep(0.1)

    def run(self):
        """Run the main interaction loop.
        
        Continuously:
        1. Listen for user input
        2. Process input and generate response
        3. Speak the response
        4. Provide feedback throughout
        """
        try:
            # Say greeting before first listen
            self.character.say_greeting()
            self.wait_for_speaking_to_complete()

            while self.running:
                # Listen for user input
                self.show_listening_indicator()
                user_input = self.character.listen()
                
                if user_input:
                    # Process and respond
                    self.show_thinking_indicator()
                    response = self.character.think_response(user_input)
                    
                    if response:
                        self.character.speak(response)
                        self.wait_for_speaking_to_complete()
                
                time.sleep(0.1)  # Small delay to prevent CPU overuse
                
        except KeyboardInterrupt:
            print("\nStopping character interaction...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources when agent is stopping."""
        self.running = False
        if self.character:
            self.character.cleanup()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run an AI Character agent')
    parser.add_argument('--config', required=True, help='Path to character configuration YAML file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    # Create and run the agent
    agent = AICharacterAgent(args.config, debug=args.debug)
    agent.run()

if __name__ == "__main__":
    main() 