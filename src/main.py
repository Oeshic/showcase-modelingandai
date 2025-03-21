# Imports
import os
import json
import pandas as pd
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

    # # Initialize llm model
    # llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # template = """
    #     You are a financial analyst evaluating a company's financial health.
    #     Based on the financial data provided, classify the company's financial health.

    #     Financial Data:
    #     {financial_data}

    #     Your response must follow this structure:
    #     - classification: (One of: Excellent, Good, Fair, Poor, Critical)
    #     - reasons: (A list of at least 2 reasons for the classification)
    #     - recommendations: (A list of at least 2 recommended actions for improvement)

    #     {format_instructions}
    # """
    # prompt = PromptTemplate(
    #     template=template,
    #     input_variable=["financial_data"],
    #     partial_variables={"format_instructions": parser.get_format_instructions()}
    # )

    # with open('./data/local/companyData5.json') as f:
    #     datas = json.load(f)


    # classification_response = []

    # financial_health_chain = prompt | llm | parser

    # def analyze_company_health(financial_data: str):
    #     result = financial_health_chain.invoke({"financial_data": financial_data})
                
    #     return result

    
    # for data in tqdm(datas):
    #     data_dict = {}
    #     data_dict["ticker"] = data['ticker']
    #     data_dict["cik"] = data['cik']
    #     result = analyze_company_health(data)
    #     data_dict["classification"] = result.classification
    #     data_dict["reasons"] = result.reasons
    #     data_dict["recommendations"] = result.recommendations
        
    #     classification_response.append(data_dict)

    # print(f'Done Classifying') 

    # with open('./data/local/classification_results2.json', 'w') as f:
    #     json.dump(classification_response, f, indent=4)

    # print(f'Done storing Locally')


    with open('./data/local/financial_data.json') as f:
        data = json.load(f)

    df_list = []
    for item in data:
        ticker = item["ticker"]
        for entry in item["revenueData"]:
            df_list.append({"Ticker": ticker, "Year": entry["year"], "Revenue": entry["value"]})

    df = pd.DataFrame(df_list)

    print(df.head())

    metrics = [
        "revenueData", "grossProfitData", "netIncomeLossData", "operatingIncomeLossData",
        "assetsData", "ppeNetData", "liabilitiesData", "depreciationData", "cashFromOperationData"
    ]

    # Function to clean and fill missing years in financial data
    def clean_financial_data(financial_data):
        # Return as is if empty
        if not financial_data:
            return financial_data

        df = pd.DataFrame(financial_data)
        
        if df.empty or "year" not in df or "value" not in df:
            return []

        # Ensure data is sorted by year
        df = df.sort_values(by="year")
        df.set_index("year", inplace=True)
        
        # Fill missing years
        full_index = range(df.index.min(), df.index.max() + 1)
        df = df.reindex(full_index)
        
        # Interpolate missing values linearly
        df["value"] = df["value"].interpolate(method="linear")
        
        # Convert back to list of dictionaries
        cleaned_data = [{"year": int(year), "value": float(value)} for year, value in df.dropna().iterrows()]
        return cleaned_data
    
    # Process each company's financial data
    cleaned_data = []
    for company in data:
        cleaned_company = {
            "ticker": company["ticker"],
            "cik": company["cik"]
        }
        
        for metric in metrics:
            cleaned_company[metric] = clean_financial_data(company.get(metric, []))
        
        cleaned_data.append(cleaned_company)

    # Save the cleaned dataset
    with open("./data/local/financial_data_cleaned.json", "w") as file:
        json.dump(cleaned_data, file, indent=4)

    print("\n Cleaned dataset has been saved to: ./data/local/financial_data_cleaned.json")




if __name__ == "__main__":
    main()