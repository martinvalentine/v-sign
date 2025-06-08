"""
LLM handler for rephrasing sign language glosses.
"""
import os
import time
import litellm

def llm_rephrase(prompt, input_api_key):
    """
    Rephrase gloss sentence using Gemini model via LiteLLM.
    
    Args:
        prompt (str): The prompt to send to the LLM.
        input_api_key (str): API key for the LLM service.
        
    Returns:
        tuple: (rephrased_text, llm_time_taken)
            - rephrased_text: The rephrased text or error message
            - llm_time_taken: Time taken for LLM processing
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt}
        ]
    }]

    try:
        llm_start_time = time.time()
        response = litellm.completion(
            model="gemini/gemini-2.0-flash",
            messages=messages,
            api_key=input_api_key
        )
        llm_end_time = time.time()
        llm_time_taken = llm_end_time - llm_start_time
        print(f"LLM rephrasing time: {llm_time_taken:.4f} seconds")
        
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        if not content:
            return "LLM response was empty or improperly formatted.", llm_time_taken
        return content.strip(), llm_time_taken
    except Exception as e:
        return f"LLM rephrase sentence failed: {e}", 0.0

def prepare_prompt(gloss_string, prompt_file_path=None):
    """
    Prepare a prompt for LLM rephrasing using a base prompt file and the gloss string.
    
    Args:
        gloss_string (str): The gloss string to be rephrased.
        prompt_file_path (str, optional): Path to the prompt file.
            If None, uses default path.
            
    Returns:
        tuple: (prompt, error_message)
            - prompt: The prepared prompt (or None if error)
            - error_message: Error message if any, None otherwise
    """
    if not prompt_file_path:
        # Default path relative to this script
        prompt_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.txt')
    
    try:
        if not os.path.exists(prompt_file_path):
            return None, f"Prompt file not found at: {prompt_file_path}"
            
        with open(prompt_file_path, "r") as f:
            prompt_base = f.read()
            
        question = f"VSL Glosses: {gloss_string}\n    Vietnamese:"
        prompt = prompt_base + "\n" + question
        
        return prompt, None
        
    except Exception as e:
        return None, f"Error reading prompt file: {e}"

def rephrase_glosses(gloss_string, input_api_key, prompt_file_path=None):
    """
    Full pipeline for rephrasing glosses with LLM.
    
    Args:
        gloss_string (str): The gloss string to rephrase.
        input_api_key (str): API key for LLM service.
        prompt_file_path (str, optional): Path to prompt template file.
        
    Returns:
        tuple: (rephrased_text, llm_time_taken)
            - rephrased_text: Rephrased text or error message
            - llm_time_taken: Time taken for LLM processing
    """
    # No API key provided
    if not input_api_key or not input_api_key.strip():
        return "Rephrasing skipped: No API key provided.", 0.0
        
    # Empty gloss string
    if not gloss_string or not gloss_string.strip():
        return "Rephrasing skipped: Empty recognition.", 0.0
    
    # Prepare prompt
    prompt, error = prepare_prompt(gloss_string, prompt_file_path)
    if error:
        return error, 0.0
        
    # Call LLM for rephrasing
    rephrased_text, llm_time_taken = llm_rephrase(prompt, input_api_key)
    
    return rephrased_text, llm_time_taken
