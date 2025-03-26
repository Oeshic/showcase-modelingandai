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

import numpy as np
import statsmodels.api as sm

from models import FinancialHealth
from parser import parser

# Configure environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


##################################
#
# FORECAST FUNCTION DEFINITIONS
#
##################################

# Divide Large JSON file into different parts
def split_data(num_parts: int, classification_file: str) -> None:

    # Load the full dataset
    with open(f'./data/local/{classification_file}.json', 'r') as f:
        data = json.load(f)

    # Number of parts to split into
    chunk_size = len(data) // num_parts

    # Split and save into smaller files
    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx = None if i == num_parts - 1 else (i + 1) * chunk_size
        chunk = data[start_idx:end_idx]

        with open(f'./data/local/classification_parts/{classification_file}_part{i+1}.json', 'w') as f:
            json.dump(chunk, f, indent=4)

    print(f"JSON file successfully split into {num_parts} parts.")

# Return {classification: "", reasons: "",  JSON
def classify_financial_health(num_parts:int, classification_file: str) -> None:

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    template = """
        You are a financial analyst evaluating a company's financial health.
        Analyze the provided financial data, classify the company's health.

        Financial Data:
        {financial_data}

        Output Format:
        - classification: (One of: Excellent, Good, Fair, Poor, Critical)
        - reasons: (At least 2 reasons supporting the classification)
        - recommendations: (At least 2 improvement actions and 1 [Buy, Sell, Hold])

        {format_instructions}
    """
    prompt = PromptTemplate(
        template=template,
        input_variable=["financial_data"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    financial_health_llm_output = prompt | llm
    total_token_count = 0

    def analyze_company_health(financial_data: str):
        nonlocal total_token_count

        try:
            result = financial_health_llm_output.invoke({"financial_data": financial_data})
            total_token_count += result.response_metadata['token_usage']['total_tokens']
            parsed_result = parser.parse(result.content)
            return parsed_result

        except Exception as e:
            print(f"Error processing company data: {financial_data.get('ticker', 'Unknown')}. Error: {e}")
            return None

    # Process each JSON part separately
    for i in range(1, num_parts+1):
        print(f"\nProcessing Part {i}...")
        with open(f'./data/local/classification_parts/{classification_file}_part{i}.json', 'r') as f:
            datas = json.load(f)

        classification_response = []

        for data in tqdm(datas):
            data_dict = {
                "ticker": data.get("ticker", ""),
                "cik": data.get("cik", ""),
                "classification": "",
                "reasons": [],
                "recommendations": []
            }

            result = analyze_company_health(data)
            if result:
                data_dict["classification"] = result.classification
                data_dict["reasons"] = result.reasons
                data_dict["recommendations"] = result.recommendations

            classification_response.append(data_dict)

        # Save results for each part separately
        with open(f'./data/local/classification_parts/{classification_file}_classification_results_part{i}.json', 'w') as f:
            json.dump(classification_response, f, indent=4)

        print(f"Saved Part {i} results.")

    print(f'\nTotal tokens used: {total_token_count}')


def merge_data(num_parts: int, classification_file: str) -> None:
    final_results = []

    # Merge all parts
    for i in range(1, num_parts+1):
        with open(f'./data/local/classification_parts/{classification_file}_classification_results_part{i}.json', 'r') as f:
            part_results = json.load(f)
            final_results.extend(part_results)

    # Save the final merged results
    with open('./data/local/{classification_file}_results_final.json', 'w') as f:
        json.dump(final_results, f, indent=4)

    print("Merged all classification results into one final JSON file.")


##################################
#
# FORECAST FUNCTION DEFINITIONS
#
##################################

# Function to clean financial data
def clean_financial_data(financial_data):
    
    # Return an empty list if the data is missing or empty
    if not financial_data:
        return []
    
    # Convert the financial data into a DataFrame
    df = pd.DataFrame(financial_data)
    
    # Check if 'year' column exists and is not empty
    if 'year' not in df or df['year'].isnull().all():
        print("Error: 'year' column is missing or empty.")
        return []

    df = df.sort_values(by="year")
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
    # Skip this dataset
    if len(df) < 2:
        print(f"Skipping ARIMA forecast for data with less than 2 points: {df}")
        return []
    
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


def forecast_arima():

    # Sample data loading
    with open('./data/local/financial_data.json') as f:
        data = json.load(f)


    # List of metrics to forecast
    metrics = [
        "revenueData", "grossProfitData", "cogsData", "netIncomeLossData", "operatingIncomeLossData",
        "assetsData", "ppeNetData", "liabilitiesData", "depreciationData", "cashFromOperationData"
    ]


    # Dictionary to store forecasted data for each stock and metric
    forecasted_results = {}

    for item in data:
        ticker = item["ticker"]
        
        # print(f"Processing {ticker}...")  # Debugging step
        
        stock_forecasts = {}
        
        for metric in metrics:
            if metric in item and item[metric]:
                print(f"Processing {metric} for {ticker}...")
                
                cleaned_data = clean_financial_data(item[metric])
                
                # Only proceed if cleaned data is available
                if cleaned_data:
                    forecasted_data = arima_forecast(cleaned_data, forecast_years=2)
                    stock_forecasts[metric] = forecasted_data
                else:
                    print(f"Skipping {metric} for {ticker} because it has no valid data.")
            else:
                stock_forecasts[metric] = []
                # print(f"Skipping {metric} for {ticker} because data is missing or empty.")
        
        forecasted_results[ticker] = stock_forecasts

    # Create forecasts folder if it does not exist
    output_dir = './data/local/forecasts/'
    os.makedirs(output_dir, exist_ok=True)

    # Save to file
    with open('./data/local/forecasts/financial_data_forecast_test.json', 'w') as outfile:
        json.dump(forecasted_results, outfile, indent=4)

    print(f"Forecasted data has been written")

def main():

    ##################################
    #
    # AI CLASSIFICATION
    #
    ##################################

    num_parts = 10
    classification_filename_no_extension = 'financial_data'

    split_data(num_parts, classification_filename_no_extension)
    classify_financial_health(num_parts, classification_filename_no_extension)
    merge_data(num_parts, classification_filename_no_extension)

    ##################################
    #
    # FORECAST
    #
    ##################################

    forecast_arima()

if __name__ == "__main__":
    main()