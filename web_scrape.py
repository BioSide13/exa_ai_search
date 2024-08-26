import requests
from transformers import pipeline

def search_and_summarize(query: str, api_key: str, num_results: int = 5):
    """
    Searches the web using the Exa API and summarizes the relevant information.

    Parameters:
    - query: The search term input by the user.
    - api_key: Your Exa API key.
    - num_results: Number of top results to process (default is 5).

    Returns:
    - A summary of the relevant information from the search results.
    """

    # Define the correct Exa API endpoint
    exa_api_url = "https://api.exa.ai/web/search"

    headers = {"x-api-key": api_key}  # Use 'x-api-key' header for authentication
    params = {
        "query": query,
        "num_results": num_results
    }

    try:
        # Send the request to the Exa API
        response = requests.get(exa_api_url, headers=headers, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            # Extract the results
            results = data.get("results", [])
            if not results:
                return "No relevant information found for the given query."

            # Collect snippets from the results for summarization
            snippets = [result.get("snippet", "") for result in results]
            full_text = " ".join(snippets)

            # Initialize the Hugging Face Transformers summarization pipeline
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

            # Summarize the content
            summary = summarizer(full_text, max_length=150, min_length=50, do_sample=False)
            return summary[0]['summary_text']

        else:
            return f"Failed to retrieve data: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}"

def main():
    # User input for search query
    query = input("Enter your search query: ")
    
    # API key for Exa API
    api_key = "YOUR_EXA_API_KEY"  # Replace with your actual Exa API key

    # Perform search and summarization
    summary = search_and_summarize(query, api_key)

    # Output the summary
    print("\nSummary of the relevant information:\n")
    print(summary)

if __name__ == "__main__":
    main()
