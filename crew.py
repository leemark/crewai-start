from crewai import Agent
from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
st.title('🤖 Ask a Crew.')
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")

from langchain_community.llms import DeepInfra
llm_di = DeepInfra(model_id="mistralai/Mixtral-8x7B-Instruct-v0.1")
llm_di.model_kwargs = {
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "max_new_tokens": 250,
    "top_p": 0.9,
}

def generate_response(topic):

  # Creating a senior researcher agent
  researcher = Agent(
    role='Senior Researcher',
    goal=f'Uncover groundbreaking news and insights around {topic}',
    verbose=True,
    backstory="""Driven by curiosity, you're at the forefront of
    innovation, eager to explore and share knowledge that could change
    the world.""",
    llm = llm
  )

  # Creating a writer agent
  writer = Agent(
    role='Writer',
    goal=f'Narrate compelling stories around the requestion question or topic: {topic}',
    verbose=True,
    backstory="""With a flair for simplifying complex topics, you craft
    engaging narratives that captivate and educate, bringing new
    discoveries to light in an accessible manner.""",
    llm = llm
  )

  from crewai import Task

  # Install duckduckgo-search for this example:
  # !pip install -U duckduckgo-search

  from langchain_community.tools import DuckDuckGoSearchRun
  search_tool = DuckDuckGoSearchRun()

  # Research task for identifying trends
  research_task = Task(
    description=f"""Identify the current developments in {topic}.
    Focus on identifying trends and the overall narrative.

    Your final report should clearly articulate the key points,
    impact, and future potential.
    """,
    expected_output='A comprehensive 3-4 paragraphs long report on the latest in {topic}.',
    max_inter=3,
    tools=[search_tool],
    agent=researcher
  )

  # Writing task based on research findings
  write_task = Task(
    description=f"""Compose an insightful article on {topic}.
    Focus on the latest trends and how it's impacting society.
    This article should be easy to understand, engaging, and positive.
    """,
    expected_output=f'A 4 paragraph article on {topic}.',
    tools=[search_tool],
    agent=writer
  )

  from crewai import Crew, Process

  # Forming the crew
  crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential  # Sequential task execution
  )

  # Starting the task execution process
  result = crew.kickoff()
  st.info(result)

with st.form('my_form'):
  topic = st.text_area('Enter text:', 'What should we research?')
  submitted = st.form_submit_button('Submit')
  if submitted:
    generate_response(topic)