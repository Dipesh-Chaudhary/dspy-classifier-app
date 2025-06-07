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
    """Loads environment variables from .env file for Groq."""
    load_dotenv()
    return {
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "model_name": os.getenv("MODEL_NAME"), # Default to groq/llama3
    }

def initialize_lm():
    """Initializes and configures the DSPy Language Model for Groq Cloud."""
    config = load_environment_config()
    if not config["groq_api_key"]:
        error_msg = "GROQ_API_KEY must be set in the .env file."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Initializing LM with Groq model: {config['model_name']}")

    # --- THIS IS THE CORRECT CODE FOR GROQ INTEGRATION ---
    lm = dspy.LM(
        model=config["model_name"], # Pass the model name with the groq prefix
        api_key=config["groq_api_key"]
    )
    # ----------------------------------------------------

    dspy.configure(lm=lm, cache=None)
    return lm

# --- The rest of the file remains the same ---

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
        classes = load_banking_classes()
        program = Classifier(classes=classes)
        program.load(filepath=full_path)
        logger.info(f"Successfully loaded program from {filename}")
        return program
    except Exception as e:
        logger.error(f"Error loading program {filename}: {e}", exc_info=True)
        raise