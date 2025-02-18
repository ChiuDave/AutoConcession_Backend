from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from dotenv import load_dotenv
import textwrap
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import sqlite3
from inputSql import generate_sql_from_input
import numpy as np

conversation_history = []

sqlChat = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("faiss_vehicle_index", embeddings, allow_dangerous_deserialization=True)

chat = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

system =  """You are a helpful assistant with access to a database of vehicle descriptions. 
            Engage in a conversational manner, keeping track of the user's queries and your responses within the current session. 
            You can ask follow up questions to the user in order to further refine the search. 
            When answering follow-up questions, refer to previous exchanges to provide relevant context. 
            In your responses do not include vehicles you decided to exclude. 
            If a new question is unrelated to previous conversations, disregard previous context. Always be clear and concise in your responses.
            Make your answers more in a bullet point format. 
            [CRITICAL RULE: Only recommend vehicles that exist in the provided csv file.]

            THINKING FRAMEWORK:

            1. Customer Understanding Phase
            - Interpret customer's explicit and implicit needs
            - Analyze customer's communication style and mood
            - Identify key buying signals or objections
            - Consider customer's price sensitivity
            - Map customer requests to available inventory

            2. Vehicle Matching Process
            - Compare customer needs with available inventory
            - Consider multiple vehicle options
            - Evaluate price alignment
            - Assess feature relevance
            - Prepare alternative suggestions

            3. Response Strategy Development
            - Choose appropriate communication style
            - Structure information hierarchy
            - Plan closing technique
            - Prepare for potential objections
            - Design next steps

            Core Response Behaviors:

            1. Response Style
            - Keep all responses under 3 sentences unless specifically asked for details
            - Always shows three possible options
            - Lead with the most relevant information first
            - Use natural, conversational language
            - Maintain professionalism even when faced with casual or rude behavior

            2. Sales Strategy
            - Always include price ranges when mentioning specific models
            - When introducing specific models, always bring proper length of details
            - Respond to budget-related keywords (like "broke", "expensive", "cheap") with appropriate options
            - When lacking inventory information, focus on general recommendations and invite store visits
            - Look for opportunities to suggest viewing available vehicles in person

            3. Customer Interaction
            - Match the customer's communication style while staying professional
            - Handle non-serious queries (like jokes) with brief, friendly responses before steering back to sales
            - For unclear requests, provide one quick clarification question followed by a suggestion
            - When faced with rudeness, respond once professionally then wait for serious queries
            - When customer shows interest in test drives, guide them to use the Appointment button

            4. Information Hierarchy
            - Price -> Features -> Technical details
            - Always mention price ranges with vehicle suggestions
            - Keep technical explanations simple unless specifically asked for details
            - Focus on practical benefits over technical specifications

            5. Closing Techniques
            - End each response with a subtle call to action
            - When suggesting test drives, specifically mention the "Appointment" button for easy scheduling
            - Suggest store visits or test drives when interest is shown
            - Provide clear next steps for interested customers
            - Be direct about availability and options

            INTERNAL DIALOGUE GUIDELINES:

            Before each response, think through:
            1. Customer Profile
            - What is their apparent budget level?
            - What style of communication are they using?
            - What signals are they giving about their interests?
            - What potential objections might they have?

            2. Product Selection
            - Which vehicles in our inventory match their needs?
            - What are the key selling points for these options?
            - What alternatives should we have ready?
            - How do our options align with their budget?

            3. Sales Approach
            - What tone should I use in my response?
            - How can I move this conversation toward a sale?
            - What would be the most effective call to action?
            - How can I overcome potential objections?

            Response Templates:
            - For jokes/non-serious queries: Brief acknowledgment + one vehicle suggestion
            - For rude comments: Make a joke and then steer the conversation to sales
            - For specific vehicle interests: Price range + key features + next step
            - For general queries: 2-3 options with price ranges + simple comparison
            - For test drive inquiries: Mention the "Appointment" button convenience (e.g., "Feel free to click the Appointment button above to schedule your test drive!")

            When suggesting vehicles, use this format:
            Brand Model Name Price Range Key Benefit Available Action

            [CRITICAL RULE: Only recommend vehicles that exist in the provided csv file.]


            Retire toute mention de "thinking framework" et "internal dialogue guidelines".
            Example of a good response:     Here's what I found in our inventory that fits your budget and preferences:

            2019 Hyundai Accent (4dr Car) – $9,998

            2019 Nissan Sentra (4dr Car) – $7,498 

            2019 Nissan Sentra (4dr Car) – $9,998 

            Would you like to schedule a test drive for any of these options or visit our store to explore further?
            """

human = (
    "User query: '{query}'.\n"
    "Relevant matches from the database:\n"
    "{results}\n"
    "Use the matches to provide a conversational and context-aware response to the user."
)

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat

df = pd.read_csv('vehicles.csv')
conn = sqlite3.connect('data.db')
df.to_sql('cars', conn, index=False, if_exists='replace')
conn.close()

def sanitize_metadata(metadata):
    if isinstance(metadata, list):
        return [sanitize_metadata(item) for item in metadata]
    elif isinstance(metadata, dict):
        return {
            key: sanitize_metadata(value) for key, value in metadata.items()
        }
    elif isinstance(metadata, (float, int)) and np.isnan(metadata):
        return None 
    return metadata


def reset_chat():
    conversation_history.clear()
    return jsonify({'status': 'success'})

def chat_endpoint():
    data = request.get_json()
    user_input = data.get('message', '').strip()
    if not user_input:
        return jsonify({'error': 'No message provided.'}), 400

    try:
        is_follow_up = "follow-up" in user_input.lower() or (len(conversation_history) > 0 and not user_input.lower().startswith(('new', 'reset', 'start over')))
        session_context = ""
        enriched_user_input = user_input

        if is_follow_up:
            session_context = "\n".join(
                f"User: {entry['query']}\nAssistant: {entry['response']}"
                for entry in conversation_history[-6:]
            )
            enriched_user_input = f"{session_context}\nUser: {user_input}"

        sql_query = generate_sql_from_input(enriched_user_input, sqlChat)

        print("Query is: ", sql_query)
    
        try:
            conn = sqlite3.connect('data.db')
            filtered_df = pd.read_sql_query(sql_query.content, conn)
            conn.close()

            texts = filtered_df["description"].tolist()
            metadata = filtered_df.drop(columns=["description"]).to_dict(orient="records")
            vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
        except Exception as e:

            texts = df["description"].tolist()
            metadata = df.drop(columns=["description"]).to_dict(orient="records")

            vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)


        enriched_query = enriched_user_input if is_follow_up else user_input
        search_results = vectorstore.similarity_search(enriched_query, k=3)

        metadata_results = [sanitize_metadata(result.metadata) for result in search_results]

        formatted_results = "\n".join(
            f"{i+1}. {result.page_content}"
            for i, result in enumerate(search_results)
        )

        response = chain.invoke({
            "query": enriched_user_input,
            "results": formatted_results
        })

        conversation_history.append({
            "query": user_input,
            "response": response.content
        })

        print(f"\nAssistant: {response.content}")

        car_details_links = []
        for result in metadata_results:
            vin = result.get("VIN")
            make = result.get("Make")
            image_link = result.get("Image_Link")
            model = result.get("Model")

            if vin and make:
                if image_link:
                    car_details_links.append(
                        f'<div>'
                        f'  <p>'
                        f'      View Details:'
                        f'      <a href="/AutoConcession/details/{vin}" target="_blank" style="font-weight: bold; color: blue; text-decoration: none;">'
                        f'          {make} {model}'
                        f'          <img src="{image_link}" alt="{make} {model}" style="width:100%;height:auto;border-radius:8px;margin-right:10px;" />'
                        f'      </a>'
                        f'  </p>'
                        f'</div>'
                    )
                else:
                    car_details_links.append(
                        f'<div>'
                        f'  <p>'
                        f'      View Details:'
                        f'      <a href="/AutoConcession/details/{vin}" target="_blank" style="font-weight: bold; color: blue; text-decoration: none;">'
                        f'          {make} {model}'
                        f'      </a>'
                        f'  </p>'
                        f'</div>'
                    )

        response_with_link = response.content + "\n\n" + "\n".join(car_details_links)

        response_data = {
            'response': response_with_link,
            'metadata': metadata_results
        }

        return jsonify(response_data)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500