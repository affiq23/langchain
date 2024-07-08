from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import argparse
import warnings

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

llm = OpenAI(
    
)

code_prompt = PromptTemplate(
    template= "Write a very short {language} function that {task}",
    input_variables=["language", "task"]
)

code_chain = LLMChain(
    llm=llm, 
    prompt= code_prompt
)

result = code_chain({
    "language": args.language,
    "task": args.task
})

print(result['text'])

