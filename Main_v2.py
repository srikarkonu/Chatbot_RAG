import logging
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI 
import os
import streamlit as st
from dotenv import load_dotenv
import requests
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory  # Import memory module
from langchain.embeddings.openai import OpenAIEmbeddings
import base64
import json

# Logging setup
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
 #cEncode the image
with open("konu-1.png", "rb") as img_file:
    encoded_img = base64.b64encode(img_file.read()).decode()

# Set up API key
load_dotenv()
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
if not OPEN_AI_KEY:
    raise ValueError("Missing OpenAI API key. Ensure it is set in your environment.")

# Flask API URL
BASE_URL = "https://ppapi.vercel.app"
faiss_db_path = "faiss_index_all"  # Directory for the FAISS vector database

# Initialize Streamlit
st.set_page_config(page_title="Real Estate Assistant", layout="centered")
   
# Display header with logo
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{encoded_img}" alt="KONU Logo" width="200">
        <h1 style="color: #F4004D; margin-top: 10px;"> Real Estate Assistant</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tool Functions for Budget Properties
def budget_properties_tool(locality: str, budget: int):
    """Fetch available properties from the API based on location and budget."""
    try:
        params = {"locality": locality, "budget": budget}
        response = requests.get(f"{BASE_URL}/budget_properties", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as e:
        return f"Error: {str(e)}"

def budget_properties_tool_wrapper(input_str):
    """Wrapper to parse single string input for locality and budget."""
    try:
        parts = input_str.split(",")
        locality = parts[0].strip()
        budget = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else None
        return budget_properties_tool(locality, budget)
    except Exception as e:
        return f"Error parsing input: {str(e)}"
    
def available_properties_tool(location: str):
    """Fetch available properties in a specific location."""
    try:
        params = {"location": location}
        response = requests.get(f"{BASE_URL}/available_properties", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as e:
        return f"Error: {str(e)}"

def available_properties_tool_wrapper(input_str):
    """Wrapper to parse single string input for location."""
    try:
        location = input_str.strip()
        return available_properties_tool(location)
    except Exception as e:
        return f"Error parsing input: {str(e)}"
    
# Tool Function for Market Value
def market_value_tool(location: str, property_category: str = None):
    """Fetch market value for a property based on location and optionally by property category."""
    try:
        params = {"location": location}
        if property_category:
            params["property_category"] = property_category
        
        response = requests.get(f"{BASE_URL}/market_value", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {"Error": f"HTTP error occurred: {http_err}"}
    except Exception as e:
        return {"Error": str(e)}

def market_value_tool_wrapper(input_str: str):
    """Wrapper to parse input for location and optional property category."""
    try:
        parts = input_str.split(",")
        location = parts[0].strip()
        property_category = parts[1].strip() if len(parts) > 1 else None
        return market_value_tool(location, property_category)
    except Exception as e:
        return {"Error": f"Error parsing input: {str(e)}"}

def properties_near_it_hub(hub_name: str, radius: float = 2.0):
    """Fetch available properties near a specified IT hub within a given radius."""
    try:
        params = {"hub_name": hub_name, "radius": radius}
        response = requests.get(f"{BASE_URL}/properties_near_it_hub", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {"Error": f"HTTP error occurred: {http_err}"}
    except Exception as e:
        return {"Error": str(e)}

def properties_near_it_hub_wrapper(input_str: str):
    """Wrapper to parse single string input for IT hub name and radius."""
    try:
        parts = input_str.split(",")
        hub_name = parts[0].strip()
        radius = float(parts[1].strip()) if len(parts) > 1 else 2.0
        return properties_near_it_hub(hub_name, radius)
    except ValueError:
        return "Error: Radius must be a valid number."
    except Exception as e:
        return f"Error parsing input: {str(e)}"

# Tool Functions for Properties Near Metro Station
def properties_near_metro(station_name: str, radius: float = 2.0):
    """Fetch available properties near a metro station within a specified radius."""
    try:
        params = {"station_name": station_name, "radius": radius}
        response = requests.get(f"{BASE_URL}/properties_near_metro_station", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {"Error": f"HTTP error occurred: {http_err}"}
    except Exception as e:
        return {"Error": str(e)}

def properties_near_metro_wrapper(input_str: str):
    """Wrapper to parse single string input for station_name and radius."""
    try:
        parts = input_str.split(",")
        station_name = parts[0].strip()
        radius = float(parts[1].strip()) if len(parts) > 1 else 2.0
        return properties_near_metro(station_name, radius)
    except ValueError:
        return "Error: Radius must be a valid number."
    except Exception as e:
        return f"Error parsing input: {str(e)}"
    
def properties_near(latitude: float, longitude: float, radius: float = 2.0):
    """Fetch available properties near a given latitude and longitude within a specified radius."""
    try:
        params = {"latitude": latitude, "longitude": longitude, "radius": radius}
        response = requests.get(f"{BASE_URL}/properties_near", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {"Error": f"HTTP error occurred: {http_err}"}
    except Exception as e:
        return {"Error": str(e)}

def properties_near_wrapper(input_str: str):
    """Wrapper to parse single string input for latitude, longitude, and optional radius."""
    try:
        parts = input_str.split(",")
        latitude = float(parts[0].strip())
        longitude = float(parts[1].strip())
        radius = float(parts[2].strip()) if len(parts) > 2 else 2.0
        return properties_near(latitude, longitude, radius)
    except ValueError:
        return "Error: Latitude, longitude, and radius must be valid numbers."
    except Exception as e:
        return f"Error parsing input: {str(e)}"
    
# Tool Functions for RERA Approved Properties
def rera_approved_tool(location: str, project_name: str = None):
    """Fetch available properties that are RERA-approved from the API based on location and optional project name."""
    try:
        params = {"location": location}
        if project_name:
            params["project_name"] = project_name  # Add project_name only if provided

        response = requests.get(f"{BASE_URL}/rera_approved", params=params)
        response.raise_for_status()  # Raise an error for non-200 responses
        return response.json()
    
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as e:
        return f"Error: {str(e)}"

def rera_approved_tool_wrapper(input_str: str):
    """Wrapper to parse single string input for project_name (optional) and location (mandatory)."""
    try:
        parts = input_str.split(",")
        location = parts[0].strip()
        project_name = parts[1].strip() if len(parts) > 1 else None
        return rera_approved_tool(location, project_name)
    except Exception as e:
        return f"Error parsing input: {str(e)}"

    
def project_price_tool(project_name: str, area: float = 1.0):
    """Fetch project price details based on the project name and area."""
    try:
        params = {"project_name": project_name, "area": area}
        response = requests.get(f"{BASE_URL}/project_price", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {"Error": f"HTTP error occurred: {http_err}"}
    except Exception as e:
        return {"Error": str(e)}


def project_price_tool_wrapper(input_str: str):
    """Wrapper to parse single string input for project_name and area."""
    try:
        parts = input_str.split(",")
        project_name = parts[0].strip()
        area = float(parts[1].strip()) if len(parts) > 1 else 1.0
        return project_price_tool(project_name, area)
    except ValueError:
        return "Error: Area must be a valid number."
    except Exception as e:
        return f"Error parsing input: {str(e)}"
    

    
def calculate_emi_tool(loan_amount: float, tenure_years: float, annual_interest_rate: float):
    """Fetch EMI calculation from the API based on loan parameters."""
    try:
        payload = {
            "loan_amount": loan_amount,
            "tenure_years": tenure_years,
            "annual_interest_rate": annual_interest_rate
        }
        response = requests.post(f"{BASE_URL}/calculate_emi", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_emi_tool_wrapper(input_str: str):
    """Wrapper to parse single string input for loan_amount, tenure_years, and annual_interest_rate."""
    try:
        parts = input_str.split(",")
        if len(parts) != 3:
            return "Error: Please provide input in 'loan_amount, tenure_years, annual_interest_rate' format."
        
        loan_amount = float(parts[0].strip())
        tenure_years = float(parts[1].strip())
        annual_interest_rate = float(parts[2].strip())
        
        return calculate_emi_tool(loan_amount, tenure_years, annual_interest_rate)
    except ValueError:
        return "Error: All inputs must be numeric."
    except Exception as e:
        return f"Error parsing input: {str(e)}"

# Define Prompt Template for EMI Calculation
emi_prompt_template = PromptTemplate(
    input_variables=["query"],
    template=''' 
    You are an AI assistant specializing in financial calculations. Your task is to compute EMI based on the following parameters:
    
    **Loan Amount**: Amount borrowed (in INR).
    **Tenure (Years)**: Duration of the loan in years.
    **Annual Interest Rate**: Interest rate per annum.
    
    The response must be structured as follows:
    ```
    | Loan Amount | Interest Rate | Tenure (Years) | EMI (Monthly) | Total Interest |
    |------------|--------------|---------------|--------------|--------------|
    | ₹{loan_amount} | {annual_interest_rate}% | {tenure_years} | ₹{emi} | ₹{total_interest} |
    ```
    
    **Input Query:** {query}
    **Output:**
    ''',
)

def filter_properties_tool(city=None, locality=None, pincode=None, property_category=None, bhk=None, area=None, page=1, page_size=10):
    """Fetch filtered properties based on user-specified criteria."""
    try:
        params = {
            "city": city,
            "locality": locality,
            "pincode": pincode,
            "property_category": property_category,
            "bhk": bhk,
            "area": area,
            "page": page,
            "page_size": page_size
        }
        # Remove None values from parameters
        params = {k: v for k, v in params.items() if v is not None}
        
        response = requests.get(f"{BASE_URL}/filter_properties/", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as e:
        return f"Error: {str(e)}"

def filter_properties_tool_wrapper(input_str):
    """Wrapper to parse input for filtering properties."""
    try:
        filters = json.loads(input_str)  # Expecting a JSON formatted string
        
        # If `property_category` contains BHK info, move it to `bhk`
        if "property_category" in filters and "BHK" in filters["property_category"]:
            bhk_value = int(filters["property_category"].split()[0])  # Extract the number
            filters["bhk"] = bhk_value  # Assign to `bhk`
            del filters["property_category"]  # Remove ambiguity
        
        return filter_properties_tool(**filters)
    except json.JSONDecodeError:
        return "Error: Input format should be a valid JSON string."
    except Exception as e:
        return f"Error parsing input: {str(e)}"

    
@st.cache_resource
def load_faiss_db(faiss_db_path):
    if os.path.exists(faiss_db_path):
        vector_store = FAISS.load_local(
            faiss_db_path,
            OpenAIEmbeddings(api_key=OPEN_AI_KEY),
            allow_dangerous_deserialization=True,
        )
        return vector_store
    else:
        st.error(f"FAISS vector store not found at {faiss_db_path}. Please ensure it exists.")
        st.stop()    

# Initialize FAISS vector store
vector_store = load_faiss_db(faiss_db_path)
# Prompt Template for Structured Table Output
structured_table_prompt = PromptTemplate(
    input_variables=["query"],
    template='''
    You are an AI assistant specialized in real estate listings. Your task is to extract and format property details into a **structured tabular format** that can be displayed correctly in markdown and Streamlit.

    **Strict Formatting Rules:**
    - Return data **only as a markdown table**.
    - No extra text, explanations, or headings before or after the table.
    - Format currency values with **₹** and commas (e.g., ₹10,50,000).
    - Ensure the size is always in **sq. ft.** format.
    - Column alignment should be consistent.

    **Output Format Example:**
    ```
    | Project Name          | Type                        | Price       | Size         | BHK | Pincode | Address            | City       | RERA Approved |
    |----------------------|----------------------------|-------------|-------------|-----|---------|------------------|------------|--------------|
    | Sai Vanamali Phase 1 | Gated Community / Apartment | ₹10,000     | 5253 sq. ft. | 3   | 500049  | Miyapur, Hyderabad | Hyderabad  | Yes          |
    | Lakshmis Emperia     | Stand Alone / Apartment     | ₹9,280      | 1537.25 sq. ft. | 2 | 500049  | Miyapur, Hyderabad | Hyderabad  | Yes          |
    | Vertex Viraat        | Gated Community / Apartment | ₹10,350     | 1472 sq. ft. | 3   | 500049  | Miyapur, Hyderabad | Hyderabad  | No           |
    ```

    **Action Format (STRICT)**:
    ```
    Thought: [Your reasoning]
    Action: [Name of the tool to call]
    Action Input: [Properly formatted tool input]
    ```

    **Query:** {query}

    **Final Answer:**
    '''
)

retrieval_tool=    Tool(
        name="FAISS Retrieval",
        func=lambda query: vector_store.similarity_search(query),
        description="Retrieve information from the FAISS vector store.",
        prompt_template=PromptTemplate(
            input_variables=["query"],
            template='''
            You are a Real Estate assistant. Summarize key information related to: {query}. 
            Provide bullet points with concise details (max 50 words). Example:
            1. [Point]
            2. [Point]
            3. [Point]
            Use previous context for follow-up questions.
            ''',
        ))

# Define Tools with the Prompt Template
tools = [
    Tool(
        name="BudgetProperties",
        func=budget_properties_tool_wrapper,
        description="Get properties in a location within a budget. Provide input as 'locality, budget",
        prompt_template=structured_table_prompt
    ),

    Tool(
        name="MarketValue",
        func=market_value_tool_wrapper,
        description="Get the market value of properties in a location. Provide input as 'location' and optionally 'property category'.",
        prompt_template=structured_table_prompt
    ),
    Tool(
        name="PropertiesNearMetroStation",
        func=properties_near_metro_wrapper,
        description="Fetch properties near a given metro station within a specified radius (in km).",
        prompt_template=structured_table_prompt
    ),
    Tool(
        name="PropertiesNearITHub",
        func=properties_near_it_hub_wrapper,
        description="Fetch properties near a given IT hub within a specified radius (in km).",
        prompt_template=structured_table_prompt
    ),
    Tool(
        name="PropertiesNear",
        func=properties_near_wrapper,
        description="Fetch properties near a given latitude and longitude within a specified radius (in km). Provide input as 'latitude, longitude, radius'.",
        prompt_template=structured_table_prompt
    ),
      Tool(
        name="RERA Approved Properties",
        func=rera_approved_tool_wrapper,
        description="Fetch RERA-approved properties in a given location or specify a project name to know whether it is RERA registered or not?.",
        prompt_template=structured_table_prompt
    ),
        Tool(
        name="ProjectPrice",
        func=project_price_tool_wrapper,
        description="Fetch project price details based on project name and area. Provide input as 'project_name, area' or just 'project_name'.",
        prompt_template=structured_table_prompt,
    ),
     Tool(
    name="CalculateEMI",
    func=calculate_emi_tool_wrapper,
    description="Calculate EMI based on loan amount, tenure, and interest rate. Provide input as 'loan_amount, tenure_years, annual_interest_rate'.",
    prompt_template=emi_prompt_template
),
     Tool(
    name="FilterProperties",
    func=filter_properties_tool_wrapper,
    description=(
        "Filter properties based on city, locality, pincode, property_category, bhk, area, etc. "
        "Provide input in the format 'city: Hyderabad, pincode: 500049, property_category: Gated community'."
    ),
    prompt_template=structured_table_prompt,
),
 Tool(
        name="AvailableProperties",
        func=available_properties_tool_wrapper,
        description="Get available properties in a location. Provide input as 'location'.",
        prompt_template=structured_table_prompt
    ),
    retrieval_tool,
]

# Initialize LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
    openai_api_key=OPEN_AI_KEY,
    max_tokens=2000
)

# Add Memory to Store Context
memory = ConversationBufferWindowMemory(k=5, return_messages=True)

def custom_error_handler(e):
    return f"Parsing error: {str(e)}"

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=custom_error_handler,
    memory=memory,
)

# Chat interface
def main_chat():
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Type your question here...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Handle EMI request separately
        if prompt.strip().lower() == "emi":
            st.subheader("EMI Calculator")
            loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
            tenure_years = st.number_input("Tenure (Years)", min_value=0.0)
            annual_interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0)
            
            if st.button("Calculate EMI"):
                result = calculate_emi_tool(loan_amount, tenure_years, annual_interest_rate)
                if isinstance(result, dict):
                    emi = result.get("emi", "N/A")
                    total_interest = result.get("total_interest", "N/A")
                    st.markdown(f"### EMI: ₹{emi}")
                    st.markdown(f"### Total Interest: ₹{total_interest}")
                else:
                    st.error(result)
        else:
            # Process other inputs with the agent
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = agent.run(prompt)
                    except Exception as e:
                        response = f"An error occurred: {e}"
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear button
    if st.button("Clear Chat"):
        st.session_state.messages = []

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if __name__ == "__main__":
    main_chat()
    st.markdown(
        """
        <div style="text-align: center; font-size: small; color: grey; margin-top: 20px;">
            Bujji learning: Small errors may occur as the system evolves.
        </div>
        """,
        unsafe_allow_html=True,
    )
