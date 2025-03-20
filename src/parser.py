from langchain_core.output_parsers import PydanticOutputParser
from models import FinancialHealth

parser = PydanticOutputParser(pydantic_object=FinancialHealth)

def parse_input_data(data: dict) -> FinancialHealth:
    
    return parser.parse(data)

