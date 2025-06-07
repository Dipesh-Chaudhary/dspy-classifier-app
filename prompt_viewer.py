import dspy
import difflib
import streamlit as st

def extract_prompt_from_program(program: dspy.Module) -> str:
    """Extracts the prompt template from a DSPy program with robust handling."""
    try:
        # Most common case for compiled programs
        if hasattr(program, "prog") and hasattr(program.prog, "predict") and hasattr(program.prog.predict, "_signature"):
            return str(program.prog.predict._signature)
        
        # Case for uncompiled ChainOfThought
        if hasattr(program, "prog") and isinstance(program.prog, dspy.ChainOfThought):
            return str(program.prog.signature)
            
        # Last resort: a string representation, which can often be informative
        return str(program)
        
    except Exception as e:
        return f"Error extracting prompt: {str(e)}"

def compare_prompts(prompt1: str, prompt2: str) -> str:
    """Compares two prompts and returns an HTML diff."""
    diff = difflib.HtmlDiff(wrapcolumn=80).make_file(
        prompt1.splitlines(),
        prompt2.splitlines(),
        "Program A",
        "Program B",
    )
    return diff

def render_prompt_comparison(programs_dict: dict):
    """Renders a side-by-side and diff comparison of prompts."""
    names = list(programs_dict.keys())
    prog1_name, prog2_name = names[0], names[1]
    
    prompt1 = extract_prompt_from_program(programs_dict[prog1_name])
    prompt2 = extract_prompt_from_program(programs_dict[prog2_name])

    st.subheader("Side-by-Side View")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{prog1_name}**")
        st.code(prompt1, language="markdown")
    with col2:
        st.markdown(f"**{prog2_name}**")
        st.code(prompt2, language="markdown")
        
    st.subheader("Diff View")
    st.markdown("Lines highlighted in green are new in Program B, red are removed, and yellow are changed.")
    diff_html = compare_prompts(prompt1, prompt2)
    st.components.v1.html(diff_html, height=600, scrolling=True)