import dspy
import os
import json
from datetime import datetime
from dspy.teleprompt import MIPROv2
from classifier import custom_metric
import logging

logger = logging.getLogger("dspy_app")

class FeedbackManager:
    """Manages user feedback and triggers feedback-driven program optimization."""
    
    def __init__(self, feedback_dir="feedback"):
        self.feedback_examples = []
        self.feedback_dir = feedback_dir
        os.makedirs(self.feedback_dir, exist_ok=True)
        self.load_feedback_from_disk()

    def add_feedback(self, text: str, predicted_label: str, correct_label: str, reasoning: str = None):
        """Adds a new piece of feedback."""
        feedback_entry = dspy.Example(
            text=text,
            label=correct_label,
            reasoning=reasoning
        ).with_inputs("text")
        
        self.feedback_examples.append(feedback_entry)
        self.save_feedback_to_disk(feedback_entry)
        logger.info(f"Added feedback for text: '{text[:30]}...'. Total feedback: {len(self.feedback_examples)}")

    def get_feedback_examples(self):
        """Returns all collected feedback examples."""
        return self.feedback_examples

    def get_feedback_count(self):
        """Returns the number of feedback examples."""
        return len(self.feedback_examples)

    def optimize_with_feedback(
        self,
        base_program: dspy.Module,
        num_trials: int = 5,
        max_demos: int = 3
    ) -> dspy.Module:
        """
        Optimizes a program using the collected feedback.
        The feedback is used as both the training and validation set to force the model
        to learn from its past mistakes.
        """
        if self.get_feedback_count() == 0:
            logger.warning("No feedback available for optimization. Returning base program.")
            return base_program

        logger.info(f"Starting feedback-driven optimization with {self.get_feedback_count()} examples.")


        # This is the model that is already configured in dspy.settings.
        student_model = dspy.settings.lm
        logger.info(f"Student model (task_model): {student_model.model}")

        # We will use the powerful Llama3 70B model from Groq for this.
        try:
            teacher_model = dspy.LM(model='groq/llama3-70b-8192', api_key=student_model.api_key)
            logger.info(f"Teacher model (prompt_model): {teacher_model.model}")
        except Exception as e:
            logger.warning(f"Could not initialize teacher model. Falling back to student model. Error: {e}")
            teacher_model = student_model

        optimizer = MIPROv2(
            metric=custom_metric,
            prompt_model=student_model,
            task_model=teacher_model,
            auto=None,
            num_candidates=4,  # Fewer candidates for faster feedback loop
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=max_demos,
        )

        # Use feedback as both trainset and valset. This strongly guides the optimizer
        # to correct the specific mistakes captured in the feedback.
        optimized_program = optimizer.compile(
            student=base_program,
            trainset=self.feedback_examples,
            valset=self.feedback_examples,
            num_trials=num_trials,
        )
        
        logger.info("Feedback-driven optimization complete.")
        return optimized_program

    def save_feedback_to_disk(self, example: dspy.Example):
        """Saves a single feedback example to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.feedback_dir, f"feedback_{timestamp}.json")
        
        feedback_data = {
            "text": example.text,
            "label": example.label,
            "reasoning": getattr(example, 'reasoning', None),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(filename, "w") as f:
            json.dump(feedback_data, f, indent=2)
        logger.info(f"Feedback saved to {filename}")

    def load_feedback_from_disk(self):
        """Loads all feedback examples from the feedback directory."""
        logger.info(f"Loading feedback from {self.feedback_dir}...")
        loaded_count = 0
        for filename in os.listdir(self.feedback_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.feedback_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        example = dspy.Example(
                            text=data["text"],
                            label=data["label"],
                            reasoning=data.get("reasoning")
                        ).with_inputs("text")
                        self.feedback_examples.append(example)
                        loaded_count += 1
                except Exception as e:
                    logger.error(f"Error loading feedback file {filename}: {e}")
        logger.info(f"Loaded {loaded_count} feedback examples from disk.")