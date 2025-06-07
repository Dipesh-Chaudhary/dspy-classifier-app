import os
import dspy
import logging
import sys
from dotenv import load_dotenv

# Configure a logger for the application
def configure_logging(logger_name="dspy_app", log_level=logging.INFO):
    """Configures a standardized logger."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = configure_logging()

def load_environment_config():
    """Loads environment variables from .env file."""
    load_dotenv()
    return {
        "api_base": os.getenv("OLLAMA_API_BASE"),
        "model": os.getenv("OLLAMA_MODEL"),
        "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
    }

def initialize_lm():
    """Initializes and configures the DSPy Language Model."""
    config = load_environment_config()
    if not config["api_base"] or not config["model"]:
        error_msg = "OLLAMA_API_BASE and OLLAMA_MODEL must be set in the .env file."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Initializing LM with model: {config['model']} at base: {config['api_base']}")
    
    lm = dspy.Ollama(
        model=config["model"].replace("ollama_chat/", ""), # dspy.Ollama does not need the prefix
        base_url=config["api_base"],
        max_tokens=4096
    )
    dspy.configure(lm=lm)
    return lm

def get_program_dir():
    """Returns the directory where programs are saved and creates it if it doesn't exist."""
    program_dir = os.path.join(".", "programs")
    os.makedirs(program_dir, exist_ok=True)
    return program_dir

def get_available_programs():
    """Lists all available .json programs in the programs directory."""
    program_dir = get_program_dir()
    return [f for f in os.listdir(program_dir) if f.endswith('.json')]

def save_program(program: dspy.Module, filename: str):
    """Saves a DSPy program to a file."""
    program_dir = get_program_dir()
    full_path = os.path.join(program_dir, filename)
    program.save(full_path)
    logger.info(f"Program saved to {full_path}")

def load_program(filename: str) -> dspy.Module:
    """Loads a DSPy program from a file."""
    from classifier import Classifier, load_banking_classes

    program_dir = get_program_dir()
    full_path = os.path.join(program_dir, filename)
    
    if not os.path.exists(full_path):
        error_msg = f"Program file not found: {full_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # Create a base classifier instance first
        classes = load_banking_classes()
        program = Classifier(classes=classes)
        # Load the state from the JSON file
        program.load(filepath=full_path)
        logger.info(f"Successfully loaded program from {filename}")
        return program
    except Exception as e:
        logger.error(f"Error loading program {filename}: {e}", exc_info=True)
        raise