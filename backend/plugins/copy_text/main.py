import pyperclip

def run(context):
    """
    Copy the identified description or extracted text to clipboard.
    """
    text = context.get('description', '')
    
    # Try to extract just the title or key info if possible
    # For now, copy the whole description
    if text:
        pyperclip.copy(text)
        return "Copied to clipboard!"
    else:
        return "No text to copy."
