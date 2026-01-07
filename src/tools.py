from langchain_core.tools import tool
from datetime import datetime


@tool
def calculator(expression: str) -> str:
    """Perform basic math. Input: expression like '25 * 4'"""
    try:
        allowed = set('0123456789+-*/.()')
        clean = expression.replace(' ', '')
        if not set(clean).issubset(allowed):
            return "Error: Only basic math operations allowed"
        return f"Result: {eval(clean)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_time(format_type: str = "default") -> str:
    """Get current time. Input: 'short', 'long', 'date', or 'default'"""
    now = datetime.now()
    formats = {
        "short": now.strftime("%H:%M"),
        "long": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "default": now.strftime("%B %d, %Y at %I:%M %p")
    }
    return formats.get(format_type, formats["default"])


@tool
def word_counter(text: str) -> str:
    """Count words and characters. Input: text to analyze"""
    words = len(text.split())
    chars = len(text)
    return f"Words: {words}, Characters: {chars}"


def get_all_tools():
    """Return list of all available tools."""
    return [calculator, get_time, word_counter]
