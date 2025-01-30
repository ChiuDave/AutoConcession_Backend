import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import pandas as pd
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

COLUMNS = ['Type', 'Stock', 'VIN', 'Year', 'Make', 'Model', 'Body', 'ModelNumber',
       'Doors', 'ExteriorColor', 'interiorColor', 'EngineCylinders',
       'EngineDisplacement', 'Transmission', 'Miles', 'SellingPrice', 'MSRP',
       'BookValue', 'Invoice', 'Certified', 'Options', 'Style_Description',
       'Ext_Color_Generic', 'Ext_Color_Code', 'int_Color_Generic',
       'int_Color_Code', 'int_Upholstery', 'Engine_Block_Type',
       'Engine_Aspiration_Type', 'Engine_Description', 'Transmission_Speed',
       'Transmission_Description', 'Drivetrain', 'Fuel_Type', 'CityMPG',
       'HighwayMPG', 'EPAClassification', 'Wheelbase_Code', 'internet_Price',
       'MarketClass', 'PassengerCapacity', 'ExtColorHexCode',
       'intColorHexCode', 'EngineDisplacementCubicInches']

QUANT_COLUMNS = {
        'Type': 'str', 'Year': pd.Int64Dtype(), 'Make': 'str', 'Body': 'str', 'Doors': pd.Int64Dtype(), 'EngineCylinders': pd.Int64Dtype(), 'Engine_Block_Type': 'str', 'Engine_Aspiration_Type': 'str', 'Transmission': 'str', 'Drivetrain': 'str', 'Fuel_Type': 'str', 'CityMPG': 'float', 'HighwayMPG': 'float', 'Miles':'float' , 'SellingPrice': 'float', 'Certified': 'bool', 'PassengerCapacity': pd.Int64Dtype()
}

possible_values = {"Type": ["Used", "New"], "Year": {"min": 2009.0, "max": 2024.0}, "Make": ["Mazda", "Maserati", "Volkswagen", "INFINITI", "Hyundai", "Ford", "Honda", "Ram", "Nissan", "Alfa Romeo", "Kia", "BMW", "Audi", "Chevrolet", "Land Rover", "Dodge", "Bentley", "Mercedes-Benz", "Toyota", "Genesis", "Jeep", "Chrysler", "Mitsubishi", "MINI", "Acura", "Volvo", "GMC", "Subaru", "Cadillac", "Jaguar", "Porsche", "Lexus", "FIAT", "Lincoln", "Buick", "Tesla", "Scion"], "Body": ["Sport Utility", "4dr Car", "2dr Car", "Crew Cab Pickup", "Extended Cab Pickup", "Hatchback", "Convertible", "Mini-van, Passenger", "Regular Cab Pickup", "Full-size Cargo Van", "3dr Car", "Mini-van, Cargo", "Full-size Passenger Van", "3D Cargo Van", "Station Wagon"], "Doors": {"min": 2.0, "max": 4.0}, "EngineCylinders": {"min": 3.0, "max": 8.0}, "Engine_Block_Type": ["I", "V", "nan"], "Engine_Aspiration_Type": ["Gasoline Direct Injection", "Sequential MPI", "nan", "Direct Injection", "Port/Direct Injection", "MPI", "Electronic Fuel Injection", "Diesel Direct Injection", "Electric", "SIDI", "EFI"], "Transmission": ["Automatic", "Variable", "Manual", "6-Speed", "nan"], "Drivetrain": ["AWD", "FWD", "RWD", "4WD"], "Fuel_Type": ["Gasoline Fuel", "Electric Fuel System", "Hybrid Fuel", "Flex Fuel", "Diesel Fuel"], "CityMPG": {"min": 12.0, "max": 134.0}, "HighwayMPG": {"min": 17.0, "max": 110.0}, "Miles": {"min": 0.0, "max": 177574.0}, "SellingPrice": {"min": 0.0, "max": 91998.0}, "Certified": [False, True], "PassengerCapacity": {"min": 2.0, "max": 12.0}}

def determine_possible_values(quant_columns):
        possible_values = {key:None for key in quant_columns.keys()}

        csv_file = "vehicles.csv"
        df = pd.read_csv(csv_file, usecols=quant_columns.keys(), dtype=quant_columns)

        for col, type in quant_columns.items():
                if type == 'str':
                        possible_values[col] = [ str(x) for x in df[col].unique()]
                elif type == 'bool':
                        possible_values[col] = [bool(x) for x in df[col].unique()]
                else:
                        possible_values[col] = {
                                'min': float(df[col].min()),
                                'max': float(df[col].max())
                        }



examples = [
    {
        "criteria": "Year > 2020 AND Price < 50000",
        "sql": "SELECT * FROM cars WHERE Year > 2020 AND SellingPrice < 50000;"
    },
    {
        "criteria": "Transmission = 'Automatic' AND Mileage < 100000",
        "sql": "SELECT * FROM cars WHERE Transmission = 'Automatic' AND Miles < 100000;"
    }
]

example_template = PromptTemplate(
    input_variables=["criteria", "sql"],
    template="Criteria: {criteria}\nSQL: {sql}"
)

prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        prefix=f"\
                The user is conversing with a chatbot in search of a car. \
                Use the input string to generate a SQL query that will select every column of the `cars` table.\
                Only generate the SQL query with no other text around it so that it can be used in SQL directly.\
                Be very conservative and only use columns that are in the possible values dictionary.\
                If the query talks about colour you can ignore it.\
                Only generate a single SQL query.\
                This is the dictionary of possible columns and values in the SQL table: {json.dumps(possible_values)}".replace('{', '{{').replace('}', '}}'),
        suffix="Criteria: {criteria}\nSQL:",
        input_variables=['criteria']
)

def generate_sql_from_input(inputString, llm):
        formatted_prompt = prompt.format(
                criteria=inputString
        )
        response = llm.invoke(formatted_prompt)
        return response