import pandas as pd
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

csv_file = "vehicles.csv"
chunk_size = 1000  # Adjust the chunk size based on your memory constraints

def create_detailed_description(row):
    fuel_efficiency = f"{row['CityMPG']} city / {row['HighwayMPG']} highway MPG"
    if pd.isnull(row['HighwayMPG']):
        efficiency_label = "unknown fuel efficiency"
    elif row["HighwayMPG"] >= 35:
        efficiency_label = "high fuel efficiency"
    elif row["HighwayMPG"] >= 25:
        efficiency_label = "average fuel efficiency"
    else:
        efficiency_label = "low fuel efficiency"

    if pd.isnull(row['Miles']):
        mileage_label = "unknown mileage"
    elif row["Miles"] < 30000:
        mileage_label = "low mileage"
    elif row["Miles"] < 60000:
        mileage_label = "moderate mileage"
    else:
        mileage_label = "high mileage"

    if pd.isnull(row['SellingPrice']):
        price_label = "price not available"
    elif row["SellingPrice"] < 20000:
        price_label = "affordable price"
    elif row["SellingPrice"] < 40000:
        price_label = "moderate price"
    else:
        price_label = "premium price"

    engine_displacement = f"{row['EngineDisplacement']}L" if pd.notnull(row['EngineDisplacement']) else "N/A"
    engine_cylinders = f"{row['EngineCylinders']}-cylinder engine" if pd.notnull(row['EngineCylinders']) else "engine"

    description = (
        f"{row['Year']} {row['Make']} {row['Model']} ({row['Body']}) with {mileage_label}\n"
        f"Exterior Color: {row['ExteriorColor']} | Interior Color: {row['InteriorColor']}\n"
        f"{engine_displacement} {engine_cylinders} | {row['Transmission']} transmission\n"
        f"Drivetrain: {row['Drivetrain']} | Fuel Type: {row['Fuel_Type']}\n"
        f"Fuel Efficiency: {fuel_efficiency} ({efficiency_label})\n"
        f"Passenger Capacity: {row['PassengerCapacity']} | Selling Price: ${row['SellingPrice']} ({price_label})\n"
        f"Certified: {'Yes' if row['Certified'] else 'No'}"
    )
    return description

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vectorstore = None

for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
    chunk["description"] = chunk.apply(create_detailed_description, axis=1)
    texts = chunk["description"].tolist()
    metadata = chunk.drop(columns=["description"]).to_dict(orient="records")
    ids = chunk.index.astype(str).tolist()

    if vectorstore is None:
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata, ids=ids)
    else:
        vectorstore.add_texts(texts, metadatas=metadata, ids=ids)

vectorstore.save_local("faiss_vehicle_index")