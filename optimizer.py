import dspy
from dspy.teleprompt import MIPROv2
from classifier import custom_metric, create_datasets
import logging

logger = logging.getLogger("dspy_app")

def run_mipro_optimization(
    base_program: dspy.Module,
    num_trials: int = 10,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 5
) -> dspy.Module:
    """
    Runs a full MIPROv2 optimization.

    Args:
        base_program: The DSPy program to optimize.
        num_trials: The number of optimization trials to run.
        ...

    Returns:
        The optimized DSPy program.
    """
    logger.info("Starting full MIPROv2 optimization...")
    datasets = create_datasets()
    
    optimizer = MIPROv2(
        metric=custom_metric,
        prompt_model=dspy.settings.lm,
        task_model=dspy.settings.lm,
        num_candidates=10, # Number of instruction candidates to generate
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )
    
    optimized_program = optimizer.compile(
        student=base_program,
        trainset=datasets["trainset"],
        valset=datasets["devset"],
        num_trials=num_trials,
    )
    
    logger.info("Full MIPROv2 optimization complete.")
    return optimized_program