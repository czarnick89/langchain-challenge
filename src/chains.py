from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from src.prompts import get_prompt_by_name
from src.client import create_client, create_llm


def build_chain(prompt_name="assistant"):
    """Build and return an LCEL chain."""
    prompt = get_prompt_by_name(prompt_name)
    client = create_client()
    llm = create_llm(client)

    # LCEL: prompt -> model -> output parser
    chain = prompt | llm | StrOutputParser()
    return chain


def chat(language, message):
    """Send a message and get a response."""
    chain = build_chain("assistant")
    response = chain.invoke({
        "language": language,
        "message": message
    })
    return response


def summarize(text, length="brief"):
    """Summarize text."""
    chain = build_chain("summarizer")
    response = chain.invoke({
        "text": text,
        "length": length
    })
    return response


def build_simple_chain(llm):
    """Build a two-step idea generation and evaluation chain using LCEL."""
    # Step 1: Generate ideas
    idea_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Generate 3 creative ideas for: {topic}. List them numbered 1-3."
    )

    # Step 2: Evaluate ideas
    eval_prompt = PromptTemplate(
        input_variables=["ideas"],
        template="Evaluate these ideas and pick the best one. Explain why:\n\n{ideas}"
    )

    # LCEL chain: generate ideas, then evaluate them
    chain = (
        idea_prompt
        | llm
        | StrOutputParser()
        | (lambda ideas: {"ideas": ideas})
        | eval_prompt
        | llm
        | StrOutputParser()
    )

    return chain


def build_research_chain(llm):
    """Build a three-step research pipeline with named outputs using LCEL."""
    research_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Research {topic} and provide 3 key facts."
    )

    outline_prompt = PromptTemplate(
        input_variables=["topic", "research"],
        template="Create a 3-point outline about {topic} using:\n{research}"
    )

    summary_prompt = PromptTemplate(
        input_variables=["topic", "outline"],
        template="Write a 2-paragraph summary about {topic} using:\n{outline}"
    )

    # LCEL chain with RunnablePassthrough to preserve topic across steps
    chain = (
        RunnablePassthrough.assign(
            research=research_prompt | llm | StrOutputParser()
        )
        | RunnablePassthrough.assign(
            outline=outline_prompt | llm | StrOutputParser()
        )
        | RunnablePassthrough.assign(
            summary=summary_prompt | llm | StrOutputParser()
        )
    )

    return chain


def generate_and_evaluate(topic):
    """Run the simple chain on a topic."""
    client = create_client()
    llm = create_llm(client)
    chain = build_simple_chain(llm)
    return chain.invoke({"topic": topic})


def research_pipeline(topic):
    """Run the research pipeline on a topic."""
    client = create_client()
    llm = create_llm(client)
    chain = build_research_chain(llm)
    return chain.invoke({"topic": topic})
