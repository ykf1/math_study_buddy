from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMMathChain, LLMChain
from langchain.llms import OpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain.utilities import WikipediaAPIWrapper

import langchain_helper as lch


load_dotenv()

# Initialise custom tools

search = DuckDuckGoSearchRun()
python_repl = PythonREPLTool()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
llm_math = LLMMathChain.from_llm(OpenAI())

tools = [
    Tool(
        name="Calculator",
        func=llm_math.run,
        description="useful for when you need to perform calculations to answer questions about math. Inputs must be numbers and operations.",
    ),
    Tool(
        name="Python",
        func=python_repl.run,
        description="useful for when you need to write python code to answer more complex questions about math.",
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="to look up definitions of math theorems and concepts",
    ),
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to do a search on the internet to answer questions that the other tools are unable to give a correct answer. Be specific with your input.",
    )
]

# Prompt template, use few shot prompting to guide the model to produce the desire output

template = """
You are a math tutor who will use the tools available to answer math questions. You have access to the following tools:

{tools}

Check if the input question requires a tool use. When it comes to answering math related questions, \
you will always use the available tools to obtain the anwer and not use your own knowledge.

If the input question is not about math or does not require a tool use, answer the question normally. Use the following format:

Question: the input question you must answer
Thought: This question is not about math or does not require a tool.
Final Answer:

If the input question is about math and requires a use of a tool to answer, use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. Remember to explain how you got your final answer detailing your thought and observation \ 
for each step like a patient tutor explaining the concepts to a student.

Examples:

Question: what is 123 + 456 * 7?
Thought: This question requires performing math calculations, so I will use the calculator tool. 
Action: Calculator
Action Input: 123 + 456 * 7
Observation: 3315
Thought: I now know the final answer
Final Answer: 123 + 456 * 7 = 3315. To solve this expression, we follow the order of operations.  Multiplication is performed first before addition. We compute 456 * 7 and then add 123 to it to get 3315.

Question: What is the formula for calculating the circumference of a circle? Given a circle of radius 2 metres, what is the circumference of the circle?
Thought: This question requires a look up on the formula of the circumference of a circle and then to calculate the circumference of a circle of radius 2 metres. I will first need to look up on Wikipedia for the formula of the circumference of a circle.
Action: Wikipedia
Action Input: Circumference of a circle
Observation: The formula of the circumference of a circle is 2 * pi * radius.
Thought: I will use the formula of 2 * pi * radius to calculate the the circumference of a circle of radius 2 metres. I will use the calculator tool.
Action: Calculator
Action Input: 2 * pi * 2
Observation: Approximately 12.566.
Thought: I now know the final answer
Final Answer: The circumference of a circle with radius 2 metres is approximately 12.566 metres. I found the formula of the circumference of a circle is 2 * pi * radius. For a circle of radius 2 metres, the circumference is calculated as 2 * pi * 2 = 12.566 metres.

Question: Is 11 a prime number?
Thought: This question requires checking if a number is prime. I will write a Python function to check if a number is prime and then use the Python tool to execute the code with the number 11 as the input.
Action: Python
Action Input:
```
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

is_prime(11)

Observation: the python function is_prime returns True with 11 as input. 11 is a prime number.
Final Answer: To check if a number is prime, we can write a Python function that checks if the number is divisible by any number from 2 to the square root of the number. If it is divisible by any number, then it is not prime. Otherwise, it is prime.

Here is the Python code to check if a number is prime:

```
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

To check if 11 is a prime number, we can call the `is_prime` function with the number 11 as the input.
Using the Python tool, I executed the code `is_prime(11)` and observed that it returned `True`.
Therefore, 11 is a prime number.

Question: If Mary has four apples and Giorgio brings two and a half apple boxes (apple box contains eight apples), how many apples are there in total?
Thought: This question requires performing calculations to determine the total number of apples. I will use the calculator tool to perform the necessary calculations.
Action: Calculator
Action Input: 4 + 2.5 * 8
Observation: 24.0
Thought: I now know the final answer
Final Answer: To determine the total number of apples, we first need to calculate the number of apples in the apple boxes brought by Giorgio. Each apple box contains eight apples, and Giorgio brings two and a half apple boxes. So, the number of apples in the apple boxes is 2.5 * 8 = 20.
Next, we add the number of apples Mary has (four apples) to the number of apples in the apple boxes (20 apples) to get the total number of apples.
Therefore, the total number of apples is 24.

Begin! 

Conversation history:
{history}

New question: {input}
{agent_scratchpad}
"""

prompt = lch.CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps", "history"]
)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    temperature=0
    )

agent = LLMSingleActionAgent(
    llm_chain= LLMChain(llm=llm, prompt=prompt),
    output_parser= lch.CustomOutputParser(),
    stop=["\nObservation:"],
    allowed_tools= [tool.name for tool in tools]
)

memory = ConversationBufferWindowMemory(
    memory_key="history", 
    k=1,
    return_messages=True
    )

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    max_iterations=4
    )


def get_response(prompt):
    return agent_executor.run(prompt)


if __name__ == "__main__":
    pass

