from crewai import Agent
from dotenv import load_dotenv
load_dotenv()

# Topic that will be used in the crew run
topic = 'AI in healthcare'

# Creating a senior researcher agent
researcher = Agent(
  role='Senior Researcher',
  goal=f'Uncover groundbreaking technologies around {topic}',
  verbose=True,
  backstory="""Driven by curiosity, you're at the forefront of
  innovation, eager to explore and share knowledge that could change
  the world."""
)

# Creating a writer agent
writer = Agent(
  role='Writer',
  goal=f'Narrate compelling tech stories around {topic}',
  verbose=True,
  backstory="""With a flair for simplifying complex topics, you craft
  engaging narratives that captivate and educate, bringing new
  discoveries to light in an accessible manner."""
)