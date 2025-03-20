# Imports
import os
import json
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import PydanticOutputParser


from models import FinancialHealth
from parser import parser

# Configure environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def main():

    # Initialize llm model
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    template = """
        You are a financial analyst evaluating a company's financial health.
        Based on the financial data provided, classify the company's financial health.

        Financial Data:
        {financial_data}

        Your response must follow this structure:
        - classification: (One of: Excellent, Good, Fair, Poor, Critical)
        - reasons: (A list of at least 2 reasons for the classification)
        - recommendations: (A list of at least 2 recommended actions for improvement)

        {format_instructions}
    """
    prompt = PromptTemplate(
        template=template,
        input_variable=["financial_data"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    with open('./data/local/companyData5.json') as f:
        datas = json.load(f)


    classification_response = []

    financial_health_chain = prompt | llm | parser

    def analyze_company_health(financial_data: str):
        result = financial_health_chain.invoke({"financial_data": financial_data})
                
        return result

    
    for data in tqdm(datas):
        data_dict = {}
        data_dict["ticker"] = data['ticker']
        data_dict["cik"] = data['cik']
        result = analyze_company_health(data)
        data_dict["classification"] = result.classification
        data_dict["reasons"] = result.reasons
        data_dict["recommendations"] = result.recommendations
        
        classification_response.append(data_dict)

    print(f'Done Classifying') 

    with open('./data/local/classification_results2.json', 'w') as f:
        json.dump(classification_response, f, indent=4)

    print(f'Done storing Locally')


if __name__ == "__main__":
    main()