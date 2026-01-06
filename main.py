from src.chains import chat, summarize

def main():
    print("Testing Multilingual Assistant")
    print("=" * 40)
    
    # Test English
    response = chat("English", "What is the capital of France?")
    print(f"English: {response}")
    
    # Test Spanish
    response = chat("Spanish", "What is the capital of France?")
    print(f"Spanish: {response}")
    
    # Test Summarizer
    text = """LangChain is a framework designed to make it easier to build applications that are powered by large language models by providing a structured way to connect those models to prompts, data sources, tools, and application logic. Rather than treating a language model as a one-off API call, LangChain encourages you to think in terms of reusable components—such as prompt templates, output parsers, chains, and agents—that can be composed together to solve more complex, real-world problems. It sits in the middle between raw model APIs and full-fledged applications, handling a lot of the orchestration and plumbing that would otherwise be repetitive or error-prone. In practice, this means developers can focus more on what they want the application to do—like answering questions, reasoning over documents, or interacting with external systems—and less on wiring everything together from scratch each time."""
    summary = summarize(text, "brief")
    print(f"Summary: {summary}")

if __name__ == "__main__":
    main()