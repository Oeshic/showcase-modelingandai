# Imports
import os
import json
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


from models import FinancialHealth
from parser import parser

# Configure environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def main():

    # Initialize llm model
    llm = ChatOpenAI(model="gpt-4o", temperature=0,max_tokens=100)

    template = """
        You are a financial analyst evaluating a company's financial health.
        Based on the financial data provided, classify the company's financial health.

        Financial Data:
        {financial_data}

        {format_instructions}
    """
    prompt = PromptTemplate(
        template=template,
        input_variable=["financial_data"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    with open('./data/local/financial_data.json') as f:
        datas = json.load(f)


    classification_response = []

    for data in tqdm(datas):
        data_dict = {}
        data_dict["ticker"] = data['ticker']
        data_dict["cik"] = data['cik']
        
        classification_response.append(data_dict)

    print(classification_response)



if __name__ == "__main__":
    main()