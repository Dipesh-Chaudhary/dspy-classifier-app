# 🤖 DSPy-Powered Self-Optimizing Classifier

This project demonstrates a sophisticated, self-optimizing classification system for banking customer intents, built entirely with the **`dspy-ai`** framework. It showcases how to move beyond static, brittle prompt engineering to a programmatic and data-centric approach where prompts and model behaviors are **automatically compiled and optimized** based on data and user feedback.

The application provides a web interface to:
1.  **Classify** user queries in real-time using a high-speed Groq Llama 3 model.
2.  **Collect** user feedback on model performance.
3.  **Trigger** advanced, feedback-driven optimizations to automatically improve the model's prompting strategy.
4.  **Inspect and Compare** the internal prompts of different program versions to understand exactly what the optimizer has learned.

## ✨ The Upper Hand: Why This Approach is Superior

This project isn't just another LLM wrapper; it's a demonstration of a new paradigm in developing with language models. Here's why the `dspy` methodology has a significant advantage over traditional approaches:

| Feature | Traditional Method (Manual Prompting) | Traditional Method (Fine-Tuning) | **The DSPy Advantage (This Project)** |
| :--- | :--- | :--- | :--- |
| **Development** | An "art" of trial-and-error. Brittle, time-consuming, and hard to reproduce. | Requires massive labeled datasets (1000s of examples) and huge compute resources. | A **systematic science**. Prompts are "compiled" like code to meet data-driven metrics. Works with very few examples. |
| **Adaptability** | Prompts need manual re-writing for every new model or data pattern. | A new fine-tuned model must be created, which is slow and expensive. | **Extremely agile**. Simply re-compile the same program for a new model or with new feedback data in minutes. |
| **Cost & Speed** | "Free" to write, but engineer time is expensive. High inference latency with complex prompts. | Very expensive to train. Creates a slow, specialized model. | **Highly efficient**. Optimizes for lightweight, fast production models (like Llama 3 8B on Groq) while using powerful models (Llama 3 70B) for offline optimization. |
| **Transparency** | The logic is a black box of text. It's unclear why a prompt works. | The model's weights are a black box. | **Transparent & Modular**. You can inspect the exact prompt `dspy` generates and see how it differs from previous versions. The logic is in Python modules, not just prompt files. |

In essence, `dspy` treats prompting not as a creative writing exercise, but as a **programming and compilation problem**, leading to more robust, efficient, and adaptable AI systems.

## 🏛️ Project Architecture

```
.
├── 📂 notebooks/
│   └── 📜 exploration.ipynb  <-- Exploration & Irrefutable Proof
├── 📂 programs/
│   └── 📜 optimized_program_timestamp.json      <-- Saved, compiled DSPy programs
├── 📂 feedback/
│   └── 📜 feedback_timestamp.json               <-- Collected user feedback data
├── 📜 app.py                                    # Main Streamlit application
├── 📜 classifier.py                             # Defines the core DSPy Classifier
├── 📜 optimizer.py                              # Logic for advanced MIPROv2 optimization
├── 📜 feedback_manager.py                       # Manages feedback and feedback-driven optimization
├── 📜 prompt_viewer.py                          # Utilities for inspecting prompts
├── 📜 utils.py                                  # Helper functions & environment setup
├── 📜 .env                                      # Environment configuration (MUST BE CREATED)
└── 📜 requirements.txt                          # Project dependencies
```

## 🚀 Getting Started

### 1. Prerequisites

*   Python 3.9+
*   A [Groq Cloud](https://console.groq.com/keys) account and API key. The free tier is sufficient. or any Litellm supported provider's api key

### 2. Installation

Clone the repository and install the required dependencies into a virtual environment.

```bash
git clone https://github.com/Dipesh-Chaudhary/dspy-classifier-app
cd dspy-classifier-app
python -m venv .venv
source .venv/bin/activate  # On Windows, use .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration

This is the **most important step**. Create a file named `.env` in the root of the project directory. Populate it with your Groq API key.

**Example `.env` file:**

```.env
# Groq Cloud Configuration
# Get your key from https://console.groq.com/keys
GROQ_API_KEY="gsk_YourGroqApiKeyHere"

# The model you want to use on Groq. The `groq/` prefix is required by litellm.
MODEL_NAME="groq/llama3-8b-8192"
```

## 📖 How to Use the Application

### Running the App

Launch the Streamlit app with the following command:

```bash
streamlit run app.py
```

### Recommended Workflow to See the Magic

1.  **Start with the Base Program:** On the **"1. Classify & Feedback"** tab, use the default `base_program`.
2.  **Use the Tricky Example:** Enter the query designed to challenge the base model:
    > `My card payment was declined, but the transfer still shows as pending. Can I cancel it?`
3.  **Observe the Failure:** The base model will likely misclassify this, focusing on `declined_card_payment` or `pending_transfer` instead of the user's primary intent, which is `cancel_transfer`.
4.  **Provide Feedback:** Mark the prediction as incorrect, select `cancel_transfer` as the correct label, and submit.
5.  **Optimize:** Go to the **"2. Optimize Program"** tab. Click **"Run Full Optimization"**. This will use the powerful Llama 3 70B model to generate a better prompt structure for our fast Llama 3 8B model.
6.  **Witness the Improvement:** After optimization, the app will auto-select your new `mipro_optimized_...` program. Go back to Tab 1 and re-run the same tricky query. It should now be classified correctly.
7.  **Inspect the Difference:** Go to the **"3. Prompt Inspector"** tab. Compare your `base_program` with the new `mipro_optimized_...` program to see the sophisticated prompt `dspy` generated automatically.

## 🔬 The Research Workbench: Irrefutable Proof of the Upper Hand

While the Streamlit app is great for interaction, the **`notebooks/exploration.ipynb`** is where we provide undeniable proof of this system's value. It serves as a transparent research log that any engineer can run to verify our claims.

**To run it, start Jupyter Lab from your activated virtual environment:**
```bash
jupyter lab
```

### The Three Layers of Proof in the Notebook

The notebook walks through a clear, three-part demonstration that proves the effectiveness of the `dspy` compiler:

#### 1. The Quantitative Proof (The Numbers)
The notebook first establishes a baseline accuracy for the simple, zero-shot program. It then runs the `MIPROv2` optimizer and re-evaluates. The output clearly shows a **significant, measurable increase in accuracy** (e.g., from ~65% to over 85%) on a held-out test set. This is the hard data that proves the system is learning.

#### 2. The Qualitative Proof (The "Aha!" Moment)
We test both the base and optimized programs on a specially crafted, ambiguous query:
> `"My card payment was declined, but the transfer still shows as pending. Can I cancel it?"`

*   **The Base Program Fails:** It typically latches onto the first keyword it sees (`declined_card_payment`) and gets the classification wrong.
*   **The Optimized Program Succeeds:** It correctly identifies the user's primary intent (`cancel_transfer`), demonstrating its superior ability to understand nuance.

#### 3. The Mechanistic Proof (The "How")
The most compelling part for a fellow engineer. The notebook uses `dspy`'s inspection tools to print the exact prompts used by both programs. You can visually see the transformation:

**BEFORE (Base Prompt):**
```
Given the fields `text`, produce the fields `label`.
```

**AFTER (Optimized Prompt - Abridged Example):**
```markdown
You are an expert in classifying customer inquiries... Prioritize the user's direct question over background context.

---

Follow the anwser format.

### Instruction:
Given the fields `text`, produce the fields `label`.

### Examples:

**Text:** My card payment was declined, but the transfer still shows as pending. Can I cancel it?
**Label:** cancel_transfer

**Text:** What is the exchange rate for a payment I want to make?
**Label:** exchange_rate

---

**Text:** {{text}}
**Label:**
```
The notebook proves that `dspy` isn't magic; it's a compiler that transforms a simple program into a sophisticated, context-aware, and high-performing one based on data.

---

## 💡 DSPy Concepts Illustrated

This project is a practical, hands-on guide to several core `dspy` concepts. By running the app and the notebook, you can see these powerful ideas in action:

*   **Signatures (`Signature`):** The `text -> label` structure in our `ChainOfThought` module implicitly defines a signature. It's a declarative specification of the task, separating the "what" (the transformation we want) from the "how" (the prompt used to achieve it).

*   **Modules (`dspy.Module`):** The `Classifier` class is a custom module. This object-oriented approach makes our AI logic reusable, composable, and stateful. We can save and load its learned parameters (the optimized prompt) just like a traditional PyTorch model.

*   **Optimizers (Teleprompters):** We use `MIPROv2` as a `teleprompter`. This is the most powerful idea in `dspy`. The optimizer treats prompt engineering not as a manual art but as a formal optimization problem. It programmatically searches the vast space of possible instructions and few-shot examples to find a high-performing prompt that maximizes our `custom_metric` on our data.

*   **The Teacher-Student Architecture:** In our `optimizer.py`, we use a more powerful model (`llama3-70b`) as the `prompt_model` (the "teacher") to generate creative and effective prompts. We then use a faster, more economical model (`llama3-8b`) as the `task_model` (the "student") to execute those prompts. This gives us the best of both worlds: high-quality reasoning during development and high-speed performance in production.

*   **Data-Centric AI Development:** This entire project embodies a data-centric philosophy. Instead of endlessly tweaking a prompt by hand, we improve the system by providing it with better data—either a larger, more diverse training set or, more powerfully, a small set of targeted examples from user feedback. The system then adapts itself to the new data automatically.

By combining these concepts, we build a system that is not just powerful, but also **principled, modular, and self-improving.**
```