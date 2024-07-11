# import for prompt template
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
# import for interacting with chat model
from langchain.chat_models import ChatOpenAI
# import for operations involving language models
from langchain.chains import LLMChain
# import for loading environment variables
from dotenv import load_dotenv
# import for ignoring deprecation warnings
import warnings
from langchain.memory import ConversationBufferMemory


warnings.filterwarnings("ignore", category=DeprecationWarning)
# loads env variables from .env file
load_dotenv()
# creates instance of OpeanAI GPT model
chat = ChatOpenAI()

# memory_key is key used to store any additional input variables; return_messages = True means return objects that wrap up the messages
memory = ConversationBufferMemory(memory_key="messages", return_messages=True)

# creates ChatPromptTemplate instance; defines how input will be formatted before being passed to the model
prompt = ChatPromptTemplate(
    # template expects input var called content to be passed
    input_variables=["content", "messages"],
    # generates a prompt from the input variable
    messages=[
        # go look at input variables called messages
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

# creates instance of LLMChain; responsible for processing input through language model
chain = LLMChain(
    # responsible for processing input
    llm = chat,
    # responsible for formatting input
    prompt = prompt,
    memory = memory
)

# loop to keep the chat running
while True:
    content = input(">> ")
    result = chain({"content": content})
    print(result["text"])