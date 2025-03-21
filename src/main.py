# Imports
import os
import json
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_core.output_parsers import PydanticOutputParser

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

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


    # Sample data loading
    with open('./data/local/companyData5.json') as f:
        data = json.load(f)

    # List of metrics to forecast
    metrics = [
        "revenueData", "grossProfitData", "netIncomeLossData", "operatingIncomeLossData",
        "assetsData", "ppeNetData", "liabilitiesData", "depreciationData", "cashFromOperationData"
    ]

    # Function to clean financial data
    def clean_financial_data(financial_data):
        if not financial_data:
            return []  # Return an empty list if the data is missing or empty
        
        df = pd.DataFrame(financial_data)
        
        # Check if 'year' column exists and is not empty
        if 'year' not in df or df['year'].isnull().all():
            print("Error: 'year' column is missing or empty.")
            return []

        df = df.sort_values(by="year")
        
        # Ensure 'year' is set as index for time series forecasting
        df.set_index("year", inplace=True)
        
        full_index = range(df.index.min(), df.index.max() + 1)
        df = df.reindex(full_index)
        
        # Interpolate missing values linearly
        df["value"] = df["value"].interpolate(method="linear")
        
        # Return the cleaned data with 'year' and 'value'
        return [{"year": int(year), "value": float(value.iloc[0])} for year, value in df.dropna().iterrows()]


    # Function to forecast with ARIMA and return combined historical and forecasted data
    def arima_forecast(financial_data, forecast_years=2):
        # Convert the financial data into a DataFrame
        df = pd.DataFrame(financial_data)
        
        # Ensure the DataFrame has the required columns: 'year' and 'value'
        if 'year' not in df or 'value' not in df:
            print(f"Skipping ARIMA forecast: 'year' or 'value' column missing for {financial_data}")
            return []
        
        # Check if the dataset has less than 2 data points
        if len(df) < 2:
            print(f"Skipping ARIMA forecast for data with less than 2 points: {df}")
            return []  # Skip this dataset
        
        df = df.sort_values(by="year")
        df.set_index("year", inplace=True)

        # Ensure there are no missing values in the 'value' column
        if df['value'].isnull().all():
            print(f"Skipping ARIMA forecast: 'value' column is missing or empty for {df['year'].iloc[0]}")
            return []

        # Fit ARIMA model (p, d, q values are assumed to be (1, 1, 1) for simplicity)
        try:
            model = sm.tsa.ARIMA(df['value'], order=(1, 1, 1))
            model_fit = model.fit()
        except Exception as e:
            print(f"Error fitting ARIMA model for {df.index[0]}: {e}")
            return []

        # Forecast the next `forecast_years` years
        forecast = model_fit.forecast(steps=forecast_years)
        
        # Generate the years for forecasted values
        forecast_years = [df.index[-1] + i for i in range(1, forecast_years + 1)]
        forecast_df = pd.DataFrame({'year': forecast_years, 'value': forecast})
        
        # Combine historical data and forecasted data
        result = pd.concat([df, forecast_df.set_index("year")])
        
        # Return the result as a dictionary of records
        return result.reset_index().to_dict(orient="records")




    # Dictionary to store forecasted data for each stock and metric
    forecasted_results = {}

    for item in data:
        ticker = item["ticker"]
        
        print(f"Processing {ticker}...")  # Debugging step
        
        stock_forecasts = {}
        
        for metric in metrics:
            if metric in item and item[metric]:  # Check if the metric exists and is not empty
                print(f"Processing {metric} for {ticker}...")
                
                cleaned_data = clean_financial_data(item[metric])
                
                # Only proceed if cleaned data is available
                if cleaned_data:
                    forecasted_data = arima_forecast(cleaned_data, forecast_years=2)
                    stock_forecasts[metric] = forecasted_data
                else:
                    print(f"Skipping {metric} for {ticker} because it has no valid data.")
            else:
                print(f"Skipping {metric} for {ticker} because data is missing or empty.")
        
        forecasted_results[ticker] = stock_forecasts

    # Save to file
    with open('./data/local/companyData5_forecast.json', 'w') as outfile:
        json.dump(forecasted_results, outfile, indent=4)


    print(f"Forecasted data has been written")
    



if __name__ == "__main__":
    main()