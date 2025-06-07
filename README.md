# 🤖 DSPy-Powered Banking Intent Classifier

This project demonstrates a sophisticated, self-optimizing classification system for banking customer intents, built entirely with the **`dspy-ai`** framework from Stanford. It showcases how to move beyond static prompt engineering to a programmatic approach where prompts and model behaviors are optimized automatically based on data and user feedback.

The application provides a web interface (using Streamlit) to:
1.  **Classify** user queries in real-time.
2.  **Collect** user feedback on model performance.
3.  **Trigger** feedback-driven optimizations to automatically improve the model's prompting strategy.
4.  **Inspect and Compare** the internal prompts of different program versions.

## ✨ Key Features

*   **Programmatic Prompting**: Uses `dspy.ChainOfThought` as a base for a modular and optimizable classification program.
*   **Automatic Prompt Optimization**: Leverages `dspy.teleprompt.MIPROv2`, a powerful optimizer, to automatically find the best instructions and few-shot examples for the classifier, significantly boosting accuracy.
*   **Continuous Improvement with Feedback**: Implements a feedback loop where user corrections are collected and used to run targeted optimizations, allowing the system to learn from its mistakes.
*   **Interactive UI**: A Streamlit application provides a user-friendly interface for testing, evaluation, and optimization without needing to touch the code.
*   **Configuration-Driven**: All model endpoints and keys are managed via a `.env` file, preventing hardcoded values.
*   **Prompt Inspection**: A dedicated UI to view the internal prompts of any saved program version and visually compare changes between them.

## 🏛️ Project Architecture

The project is structured in a modular way, separating concerns to make it maintainable and scalable.

```
.
├── 📂 programs/           # Saved, optimized DSPy program versions (.json)
├── 📂 feedback/            # Collected user feedback data (.json)
├── 📜 app.py               # Main Streamlit application
├── 📜 classifier.py         # Defines the core DSPy Classifier module and data handling
├── 📜 optimizer.py          # Contains the logic for running MIPROv2 optimization
├── 📜 feedback_manager.py   # Manages collecting and optimizing with user feedback
├── 📜 prompt_viewer.py      # Utilities for extracting and comparing prompts
├── 📜 utils.py              # Helper functions, logging, and environment setup
├── 📜 .env                  # Environment configuration (MUST BE CREATED)
└── 📜 requirements.txt      # Project dependencies
```

## 🚀 Getting Started

### 1. Prerequisites

*   Python 3.9+
*   An accessible [Ollama](https://ollama.com/) instance running a model (e.g., `deepseek-coder-v2`, `llama3`).

### 2. Installation

Clone the repository and install the required dependencies.

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

### 3. Configuration

This is the **most important step** to resolve connection errors. Create a file named `.env` in the root of the project directory. Populate it with the details of your running Ollama instance.

**Example `.env` file:**

```
# The base URL of your Ollama server
OLLAMA_API_BASE="http://localhost:11434"

# The model you want to use (must be pulled in Ollama)
# The `ollama_chat/` prefix is a convention for LiteLLM.
# Our code handles stripping it for dspy.Ollama.
OLLAMA_MODEL="ollama_chat/deepseek-coder-v2"

# API key (usually just "ollama" for local instances)
OLLAMA_API_KEY="ollama"
```

**Ensure your Ollama server is running and the model is pulled:**

```bash
ollama run deepseek-coder-v2
```

### 4. Running the Application

Launch the Streamlit app with the following command:

```bash
streamlit run app.py
```

Navigate to the URL provided by Streamlit (usually `http://localhost:8501`) in your browser.

## 📖 How to Use the Application

The application is organized into three main tabs:

### Tab 1: Classify & Feedback

This is the main interaction tab.
1.  **Enter a query**: Type a customer banking query (e.g., "why was my card declined?") into the text area.
2.  **Classify**: The active DSPy program will predict the intent and show its reasoning.
3.  **Provide Feedback**: If the prediction is incorrect, select "No", choose the correct label, and submit. This feedback is saved and becomes crucial for future optimizations.

### Tab 2: Optimize Program

This tab allows you to create new, improved versions of your programs.
1.  **Select a Base Program**: Choose an existing program to use as the starting point.
2.  **Run Optimization**: Click the button to start the `MIPROv2` optimizer. It will use the `Banking77` dataset to generate and test new prompts and few-shot examples.
3.  **Result**: A new, optimized program will be saved to the `programs/` directory and become available for selection.

### Tab 3: Prompt Inspector

This tab lets you look under the hood of your DSPy programs.
*   **View Single Prompt**: Select any program to see its internal instruction prompt.
*   **Compare Prompts**: Select two different programs (e.g., `base_program` and an `optimized_` version) to see a side-by-side and a "diff" view, highlighting exactly what the optimizer changed.

##💡 DSPy Concepts Illustrated

This project is a practical guide to several core `dspy` concepts:

*   **Signatures**: The `text -> label` structure implicitly defines a signature that guides the LM.
*   **Modules (`dspy.Module`)**: The `Classifier` class is a custom module, making the logic reusable and composable.
*   **Optimizers (`Teleprompters`)**: `MIPROv2` is used as a `teleprompter` to automate the complex task of prompt engineering, treating it as a formal optimization problem.
*   **Metrics**: A `custom_metric` is defined to score the performance of different program versions during optimization, guiding the process toward the desired outcome.
*   **Feedback-Driven Development**: The application closes the loop from production errors (user feedback) to automatic program improvement, embodying the data-centric principles of modern AI development.