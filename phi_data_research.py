"""
phi_data_research.py - agentic AI for a research assistant using ExaTools

Author: Manish Bhobe
My experiments with Python, AI and Gen AI
Code shared for learning purposes only!
"""

from textwrap import dedent
from datetime import datetime

from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.model.google import Gemini
from phi.model.anthropic import Claude
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.exa import ExaTools
from dotenv import load_dotenv

# load all API keys
load_dotenv()

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[
        ExaTools(
            start_published_date=datetime.now().strftime("%Y-%m-%d"), type="keyword"
        )
    ],
    description="You are an advanced AI researcher writing a report on a topic.",
    instructions=[
        "For the provided topic, run 3 different searches.",
        "Read the results carefully and prepare a LinkedIn worthy blog.",
        "Focus on facts and make sure to provide references.",
    ],
    expected_output=dedent(
        """\
    An engaging, informative, and well-structured report in markdown format:

    ## Engaging Report Title

    ### Overview
    {give a brief introduction of the report and why the user should read this report}
    {make this section engaging and create a hook for the reader}

    ### Section 1
    {break the report into sections}
    {provide details/facts/processes in this section}

    ... more sections as necessary...

    ### Takeaways
    {provide key takeaways from the article}

    ### References
    - [Reference 1](link)
    - [Reference 2](link)
    - [Reference 3](link)

    - published on {date} in dd/mm/yyyy
    """
    ),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    save_response_to_file="tmp/{message}.md",
)

agent.print_response(
    "Top 5 applications of Agentic AI in Commercial Insurance", stream=True
)
