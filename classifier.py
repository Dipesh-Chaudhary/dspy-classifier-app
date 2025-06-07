import dspy
import random
from dspy.datasets import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import logging

logger = logging.getLogger("dspy_app")

def load_banking_classes():
    """Loads and returns the class labels from the Banking77 dataset."""
    try:
        features = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features
        return features['label'].names
    except Exception as e:
        logger.error(f"Failed to load dataset from HuggingFace: {e}")
        # Fallback to a hardcoded list if offline or API changes
        return ['card_payment_wrong_exchange_rate', 'wrong_exchange_rate_for_cash_withdrawal', ...]


class Classifier(dspy.Module):
    """A DSPy module for classifying banking-related text."""
    def __init__(self, classes):
        super().__init__()
        # The `ChainOfThought` module provides a zero-shot prompt with reasoning.
        self.prog = dspy.ChainOfThought(f"text -> label: Literal{classes}")

    def forward(self, text):
        return self.prog(text=text, config=dict(cache=False))

def create_datasets(num_train=500, num_dev=200, num_test=200):
    """Creates and splits the Banking77 dataset for training, development, and testing."""
    logger.info("Creating datasets...")
    CLASSES = load_banking_classes()
    
    kwargs = dict(fields=("text", "label"), input_keys=("text",), split="train", trust_remote_code=True)
    
    try:
        # Load up to a certain number of examples
        total_examples = num_train + num_dev + num_test
        dataset = DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)
        
        # Convert labels to strings
        all_data = [dspy.Example(x, label=CLASSES[x.label]).with_inputs("text") for x in dataset[:total_examples]]
        
        random.Random(0).shuffle(all_data)
        
        trainset = all_data[:num_train]
        devset = all_data[num_train : num_train + num_dev]
        testset = all_data[num_train + num_dev : total_examples]
        
        logger.info(f"Datasets created: Train ({len(trainset)}), Dev ({len(devset)}), Test ({len(testset)})")
        return {"trainset": trainset, "devset": devset, "testset": testset}
    except Exception as e:
        logger.error(f"Failed to create datasets: {e}", exc_info=True)
        return {"trainset": [], "devset": [], "testset": []}


def custom_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """A custom metric to evaluate classification accuracy."""
    return float(gold.label == pred.label)






# # Add this to the end of classifier.py and explicitly run only this file to test
# if __name__ == '__main__':
#     from utils import initialize_lm
#     lm = initialize_lm()
#     classes = load_banking_classes()
#     classifier = Classifier(classes=classes)
#     prediction = classifier(text="My card got stuck")
#     print(prediction)