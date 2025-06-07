import streamlit as st
import dspy
from datetime import datetime
import logging

# Import project modules
from utils import initialize_lm, save_program, load_program, get_available_programs, configure_logging
from classifier import Classifier, load_banking_classes, create_datasets
from optimizer import run_mipro_optimization
from feedback_manager import FeedbackManager
from prompt_viewer import extract_prompt_from_program, render_prompt_comparison

# --- Page and State Configuration ---
st.set_page_config(page_title="DSPy Banking Classifier", layout="wide", page_icon="🤖")
logger = configure_logging()

# --- Initialization Functions ---
def initialize_session_state():
    """Initializes all necessary session state variables."""
    if 'lm' not in st.session_state:
        try:
            st.session_state.lm = initialize_lm()
        except Exception as e:
            st.error(f"Failed to initialize Language Model: {e}")
            st.stop()
    
    if 'feedback_manager' not in st.session_state:
        st.session_state.feedback_manager = FeedbackManager()

    if 'datasets' not in st.session_state:
        st.session_state.datasets = None # Lazy load

    if 'programs' not in st.session_state:
        st.session_state.programs = {}
        
    if 'selected_program' not in st.session_state:
        st.session_state.selected_program = "base_program"
        
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
        
    if 'banking_classes' not in st.session_state:
        st.session_state.banking_classes = load_banking_classes()

def refresh_programs():
    """Loads all programs from disk and ensures a base program exists."""
    programs = {}
    # Load all saved programs
    for filename in get_available_programs():
        try:
            name = filename.replace(".json", "")
            programs[name] = load_program(filename)
        except Exception as e:
            st.warning(f"Could not load {filename}: {e}")
    
    # Ensure a base program always exists
    if "base_program" not in programs:
        programs["base_program"] = Classifier(classes=st.session_state.banking_classes)
        
    st.session_state.programs = programs
    # If the selected program was deleted, default to base
    if st.session_state.selected_program not in st.session_state.programs:
        st.session_state.selected_program = "base_program"

def get_datasets():
    """Lazy loads datasets and caches them in session state."""
    if st.session_state.datasets is None:
        with st.spinner("Loading datasets for the first time..."):
            st.session_state.datasets = create_datasets()
    return st.session_state.datasets

# --- Main App ---
initialize_session_state()
refresh_programs()

st.title("🤖 DSPy Banking Intent Classifier")
st.markdown("An application to test, evaluate, and optimize a text classifier using the `dspy-ai` framework.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Program selection
    program_names = list(st.session_state.programs.keys())
    try:
        current_index = program_names.index(st.session_state.selected_program)
    except ValueError:
        current_index = 0 # Default to first program
    
    st.session_state.selected_program = st.selectbox(
        "Select Active Program",
        options=program_names,
        index=current_index,
        help="Choose the DSPy program to use for classification."
    )
    
    if st.button("🔄 Refresh Programs List"):
        refresh_programs()
        st.rerun()

    st.header("Feedback")
    st.info(f"Feedback collected: **{st.session_state.feedback_manager.get_feedback_count()}**")

    if st.session_state.feedback_manager.get_feedback_count() > 0:
        if st.button("🚀 Optimize with Feedback"):
            with st.spinner("Running feedback-driven optimization..."):
                base_prog = st.session_state.programs[st.session_state.selected_program]
                optimized = st.session_state.feedback_manager.optimize_with_feedback(base_prog)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                name = f"feedback_optimized_{timestamp}"
                save_program(optimized, f"{name}.json")
                refresh_programs()
                st.session_state.selected_program = name
                st.success(f"Created new program: '{name}'")
                st.rerun()

# --- Main Content Tabs ---
tabs = st.tabs(["**1. Classify & Feedback**", "**2. Optimize Program**", "**3. Prompt Inspector**"])

# --- Tab 1: Classify & Feedback ---
with tabs[0]:
    st.header("Classify Text")
    user_text = st.text_area("Enter a customer query to classify:", height=100, key="user_text_input")

    if st.button("Classify", type="primary"):
        if user_text:
            with st.spinner("Classification in progress..."):
                try:
                    active_program = st.session_state.programs[st.session_state.selected_program]
                    prediction = active_program(text=user_text)
                    st.session_state.last_prediction = {
                        "text": user_text,
                        "predicted_label": prediction.label,
                        "reasoning": prediction.reasoning,
                    }
                except Exception as e:
                    st.error(f"Classification failed: {e}")
                    st.session_state.last_prediction = None
        else:
            st.warning("Please enter text to classify.")

    if st.session_state.last_prediction:
        st.divider()
        st.subheader("Classification Result")
        pred = st.session_state.last_prediction
        st.write(f"**Query:** \"{pred['text']}\"")
        st.success(f"**Predicted Intent:** `{pred['predicted_label']}`")
        with st.expander("Show Model Reasoning"):
            st.info(pred['reasoning'])

        # Feedback Form
        st.subheader("✍️ Provide Feedback")
        is_correct = st.radio("Was this prediction correct?", ("Yes", "No"), index=None, key="feedback_radio")

        if is_correct == "No":
            correct_label = st.selectbox(
                "What was the correct intent?",
                options=st.session_state.banking_classes
            )
            reasoning = st.text_area(
                "Why was the prediction incorrect? (Optional but helpful)",
                key="feedback_reasoning"
            )
            if st.button("Submit Feedback"):
                st.session_state.feedback_manager.add_feedback(
                    text=pred['text'],
                    predicted_label=pred['predicted_label'],
                    correct_label=correct_label,
                    reasoning=reasoning
                )
                st.success("Thank you for your feedback!")
                st.session_state.last_prediction = None # Clear after feedback
                st.rerun()
        elif is_correct == "Yes":
            st.success("Great! Thanks for confirming.")


# --- Tab 2: Optimize Program ---
with tabs[1]:
    st.header("Create a New Optimized Program")
    st.markdown("Use `MIPROv2` to generate a new, optimized version of a program based on a training dataset.")
    
    program_to_optimize = st.selectbox(
        "Select a base program to optimize",
        options=list(st.session_state.programs.keys()),
        key="program_to_optimize_select"
    )

    if st.button("Run Full Optimization", type="primary"):
        with st.spinner("Running MIPROv2 optimization... This may take a long time."):
            try:
                base_program = st.session_state.programs[program_to_optimize]
                optimized_program = run_mipro_optimization(base_program)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                new_name = f"mipro_optimized_{timestamp}"
                save_program(optimized_program, f"{new_name}.json")
                
                refresh_programs()
                st.session_state.selected_program = new_name
                st.success(f"Optimization complete! New program created: '{new_name}'")
                st.rerun()

            except Exception as e:
                st.error(f"Optimization failed: {e}", icon="🚨")


# --- Tab 3: Prompt Inspector ---
with tabs[2]:
    st.header("Prompt Template Inspector")
    st.markdown("View and compare the internal prompt templates of your DSPy programs.")

    if len(st.session_state.programs) < 2:
        st.info("You need at least two programs to use the comparison view.")
        program_to_view = st.selectbox(
            "Select program to view",
            options=list(st.session_state.programs.keys())
        )
        prompt = extract_prompt_from_program(st.session_state.programs[program_to_view])
        st.code(prompt, language="markdown")
    else:
        col1, col2 = st.columns(2)
        with col1:
            prog1_name = st.selectbox("Select Program A", options=list(st.session_state.programs.keys()), index=0)
        with col2:
            prog2_name = st.selectbox("Select Program B", options=list(st.session_state.programs.keys()), index=1)
        
        if prog1_name != prog2_name:
            programs_to_compare = {
                prog1_name: st.session_state.programs[prog1_name],
                prog2_name: st.session_state.programs[prog2_name]
            }
            render_prompt_comparison(programs_to_compare)
        else:
            st.warning("Please select two different programs to compare.")