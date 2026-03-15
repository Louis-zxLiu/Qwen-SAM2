import webbrowser
import urllib.parse

def run(context):
    text = context.get('description', '')
    if not text:
        return "No keyword to search."
        
    # Extract first line or title as keyword
    keyword = text.split('\n')[0].strip()
    # Remove markdown bold if present
    keyword = keyword.replace('**', '').replace('__', '')
    
    encoded = urllib.parse.quote(keyword)
    url = f"https://www.google.com/search?q={encoded}"
    
    webbrowser.open(url)
    return f"Searching Google for: {keyword}"
