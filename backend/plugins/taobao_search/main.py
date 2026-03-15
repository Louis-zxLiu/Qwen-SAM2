import webbrowser
import urllib.parse

def run(context):
    text = context.get('description', '')
    if not text:
        return "No product name found."
        
    keyword = text.split('\n')[0].strip()
    keyword = keyword.replace('**', '').replace('__', '')
    
    encoded = urllib.parse.quote(keyword)
    url = f"https://s.taobao.com/search?q={encoded}"
    
    webbrowser.open(url)
    return f"Opening Taobao for: {keyword}"
