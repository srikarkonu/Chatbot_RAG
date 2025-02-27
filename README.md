# Konu Real Estate Assistant

## Overview
The Konu Real Estate Assistant is an AI-powered chatbot designed to assist users with real estate-related queries. Built using Python, Streamlit, LangChain, and OpenAI's GPT models, it provides tools for property searches, market value analysis, EMI calculations, and more. The assistant integrates with a custom API to fetch real-time property data and uses a FAISS vector store for retrieval-augmented generation (RAG).

## Features
- **Property Search**: Find properties by budget, location, proximity to metro stations or IT hubs, and RERA approval.
- **Market Value**: Retrieve market value trends for specific locations and property types.
- **EMI Calculator**: Calculate monthly EMIs based on loan amount, tenure, and interest rate.
- **Dynamic Suggestions**: Offers context-aware follow-up questions based on user input.
- **Structured Output**: Responses are formatted as markdown tables for clarity and readability.
- **Chat Memory**: Retains conversation context for better follow-up responses.

## Prerequisites
- Python 3.8+
- OpenAI API Key: Required for LLM functionality.
- Streamlit: For the web-based UI.
- FAISS Index: Pre-built vector store for property data retrieval.
- API Access: The app relies on a custom API hosted at `https://ppapi.vercel.app`.
- Docker: To containerize and deploy the application.
- Ngrok: To expose the local server publicly.

## Setup

### Clone the Repository
```bash
git clone <repository-url>
cd konu-real-estate-assistant
```

### Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

#### Sample `requirements.txt`
```
langchain
langchain-openai
streamlit
python-dotenv
requests
faiss-cpu
openai
```

### Set Up Environment Variables
Create a `.env` file in the root directory and add your OpenAI API key:
```bash
OPEN_AI_KEY=<your-openai-api-key>
```

### Prepare FAISS Vector Store
Ensure the FAISS index (`faiss_index_all`) is available in the project directory. This is used for retrieving property data. If not present, generate it using your dataset and the `OpenAIEmbeddings`.

### Verify API Access
The app connects to `https://ppapi.vercel.app`. Ensure this API is operational and accessible.

## Running the Application Locally

### Start the Streamlit App
Run the following command from the project directory:
```bash
streamlit run app.py
```
Replace `app.py` with the filename of your script if different.

### Access the UI
Open your browser and navigate to `http://localhost:8501` to interact with the chatbot.

## Running the Application in Docker

### Build the Docker Image
```bash
docker build -t konu-real-estate-assistant .
```

### Run the Docker Container
```bash
docker run -p 8501:8501 --env-file .env konu-real-estate-assistant
```

### Exposing the Application via Ngrok
If you need to make your local deployment accessible over the internet, use Ngrok:
```bash
ngrok config add-authtoken $YOUR_AUTHTOKEN
ngrok http 8501
```
Ngrok will provide a public URL that you can share for remote access.

## Usage
- **Ask Questions**: Type real estate-related queries in the chat input box, e.g.:
  - "Show me properties in Hyderabad under ₹50,00,000."
  - "What’s the market value in Miyapur?"
  - "Calculate EMI for a ₹40,00,000 loan, 10 years, 8% interest."
- **Explore Suggestions**: After each response, the assistant provides four dynamic follow-up questions. Click any suggestion to continue the conversation.
- **Clear Chat**: Use the "Clear Chat" button to reset the conversation history.

## Tools and Functionality
The assistant uses the following tools, each tied to a specific API endpoint or calculation:
- `BudgetProperties`: Search properties by locality and budget.
- `MarketValue`: Fetch market value data for a location.
- `PropertiesNearMetroStation`: Find properties near metro stations.
- `PropertiesNearITHub`: Locate properties near IT hubs.
- `RERA Approved Properties`: List RERA-approved projects.
- `ProjectPrice`: Get pricing details for specific projects.
- `CalculateEMI`: Compute loan EMIs.
- `FilterProperties`: Filter properties by various criteria (e.g., city, BHK).
- `FAISS Retrieval`: Retrieve summarized property info from the vector store.

Responses are formatted as markdown tables for clarity, leveraging custom prompt templates.

## Customization
- **Prompt Templates**: Modify `structured_table_prompt` or `emi_prompt_template` to adjust output formatting.
- **Tools**: Add or remove tools in the tools list to extend functionality.
- **API Base URL**: Update `BASE_URL` if using a different API endpoint.

## Logging
Logs are written to `chatbot.log` in the project directory. This includes timestamps, log levels, and messages for debugging.

## Troubleshooting
- **Missing API Key**: Ensure `OPEN_AI_KEY` is set in `.env`.
- **FAISS Index Not Found**: Verify the `faiss_index_all` directory exists.
- **API Errors**: Check the API status at `https://ppapi.vercel.app` and network connectivity.
- **Parsing Errors**: Review `chatbot.log` for detailed error messages.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

### Dockerfile
```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.13

# Set the working directory in the container
WORKDIR /app

# Copy the chatbot code into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the necessary port
EXPOSE 8501

# Run the chatbot application
CMD ["streamlit", "run", "main3.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
