# agent.py
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import os
import json
from dotenv import load_dotenv

serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# Import your custom tools from tools.py
from dukebot.tools import (
    get_events_from_duke_api,
    get_curriculum_with_subject_from_duke_api,
    get_detailed_course_information_from_duke_api,
    get_people_information_from_duke_api,
    search_subject_by_code,
    search_group_format,
    search_category_format,
    get_pratt_info_from_serpapi,
    get_specific_pratt_info
)

# Load environment variables from .env file
load_dotenv()

def create_duke_agent():
    """
    Create a LangChain agent with the Duke tools.
    API keys are loaded from .env file.
    
    Returns:
        An initialized LangChain agent
    """
    # Get API keys from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Check if API keys are available
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Define the tools
    tools = [
        Tool(
            name="get_duke_events",
            func=get_events_from_duke_api,
            description=(
                "This tool retrieves upcoming events from Duke University's public calendar API based on a free-form, "
                "natural language query. The tool first processes the query to determine the relevant event filters by "
                "automatically mapping your input to the correct organizer groups and thematic categories. It does so "
                "by reading the full lists of valid groups and categories from local text files, reducing those lists "
                "via fuzzy matching (or retrieval-augmented generation) to a small set of top candidates, and then using "
                "an LLM to select the final filter values. If no relevant filters are identified, it defaults to using ['All'] "
                "for that parameter, ensuring that the API call is still valid. \n\n"
                "Parameters:\n"
                "  - prompt (str): A natural language query describing the event filters you wish to apply. For example, "
                "\"Please give me the events of aipi\" will be processed to select the appropriate group(s) and category(ies).\n"
                "  - feed_type (str): The format in which the event data should be returned. Accepted values include "
                "'rss', 'js', 'ics', 'csv', 'json', and 'jsonp'. The default is 'json'.\n"
                "  - future_days (int): The number of days into the future for which to retrieve events. The default is 45.\n"
                "  - filter_method_group (bool): Determines the logic used for filtering events by groups. When set to True, "
                "an event is included if it matches ANY of the specified groups (logical OR); when False, the event must match ALL "
                "specified groups (logical AND). The default is True.\n"
                "  - filter_method_category (bool): Determines the logic used for filtering events by categories. When set to True, "
                "an event is included if it matches ANY of the specified categories (logical OR); when False, the event must match ALL "
                "specified categories (logical AND). The default is True.\n\n"
                "The tool returns the raw calendar event data as provided by Duke University's API (or an error message if the API request fails). "
                "It is ideal for dynamically retrieving event information without having to manually specify the detailed filter values, "
                "since the mapping process leverages an LLM to interpret your query and select the correct filters from the available lists."
                )
            ),
        Tool(
            name="get_curriculum_with_subject_from_duke_api",
            func=get_curriculum_with_subject_from_duke_api,
            description=(
                "Use this tool to retrieve curriculum information from Duke University's API."
                "IMPORTANT: The 'subject' parameter must be from subjects.txt list. "
                "Parameters:"
                "   subject (str): The subject to get curriculum data for. For example, the subject is 'ARABIC-Arabic'."
                "Return:"
                "   str: Raw curriculum data in JSON format or an error message. If valid result, the response will contain each course's course id and course offer number for further queries."
            )
        ),
        Tool(
            name="get_detailed_course_information_from_duke_api",
            func=get_detailed_course_information_from_duke_api,
            description=(
                "Use this tool to retrieve detailed curriculum information from Duke University's API."
                "The course ID and course offer number can be obtained from get_curriculum_with_subject_from_duke_api."
                "Parameters:"
                "   course_id (str): The course ID to get curriculum data for. For example, the course ID is 029248' for General African American Studies."
                "   course_offer_number (str): The course offer number to get curriculum data for. For example, the course offer number is '1' for General African American Studies."
                "Return:"
                "   str: Raw curriculum data in JSON format or an error message."
            )
        ),
        Tool(
            name="get_people_information_from_duke_api",
            func=get_people_information_from_duke_api,
            description=(
                "Use this tool to retrieve people information from Duke University's API."
                "Parameters:"
                "   name (str): The name to get people data for. For example, the name is 'Brinnae Bent'."
                "Return:"
                "   str: Raw people data in JSON format or an error message."
            )
        ),
        Tool(
            name="search_subject_by_code",
            func=search_subject_by_code,
            description=(
                "Use this tool to find the correct format of a subject before using get_curriculum_with_subject_from_duke_api. "
                "This tool handles case-insensitive matching and partial matches. "
                "Example: 'cs' might return 'COMPSCI - Computer Science'. "
                "Always use this tool first if you're uncertain about the exact subject format."
            )
        ),
        Tool(
            name="search_group_format",
            func=search_group_format,
            description=(
                "Use this tool to find the correct format of a group before using get_events_from_duke_api. "
                "This tool handles case-insensitive matching and partial matches. "
                "Example: 'data science' might return '+DataScience (+DS)'. "
                "Always use this tool first if you're uncertain about the exact group format."
            )
        ),
        Tool(
            name="search_category_format",
            func=search_category_format,
            description=(
                "Use this tool to find the correct format of a category before using get_events_from_duke_api. "
                "This tool handles case-insensitive matching and partial matches. "
                "Example: 'ai' might return 'Artificial Intelligence'. "
                "Always use this tool first if you're uncertain about the exact category format."
            )
        ),
        Tool(
             name="PrattSearch",
             func=lambda query: get_pratt_info_from_serpapi(
                 query="Duke Pratt School of Engineering " + query, 
                 api_key=serpapi_api_key,
                 filter_domain=True  
             ),
             description=(
                 "Use this tool to search for information about Duke Pratt School of Engineering. "
                 "Specify your search query."
             )
         ),
    ]
    
    # Create a memory instance
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Initialize the LLM with the OpenAI API key
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model_name="gpt-4",
        temperature=0
    )
    
    # System prompt for agentic search approach
    system_prompt = """
    You are DukeBot, an authoritative and knowledgeable Duke University assistant with access to a suite of specialized Duke API tools. Your mission is to accurately and professionally provide information on three primary areas:

    1. **AI MEng Program Information**: Deliver detailed and reliable information about the AI MEng program. This includes curriculum details, admissions criteria, faculty expertise, career outcomes, and any unique features of the program.

    2. **Prospective Student Information**: Provide factual and comprehensive information for prospective students about Duke University and Duke Pratt School of Engineering. Include key figures, campus life details, academic programs, admissions statistics, financial aid information, and notable achievements.

    3. **Campus Events**: Retrieve and present up-to-date information on events happening on campus. Ensure that events are filtered correctly by organizer groups and thematic categories.

    For every query, follow these steps:

    1. **THINK**:
    - Carefully analyze the user’s query to determine which domain(s) it touches: AI MEng details, prospective student facts, or campus events.
    - Decide which API tools are the best fit to get accurate data.

    2. **FORMAT SEARCH**:
    - NEVER pass user-provided subject, group, or category formats directly to the API tools.
    - Use the dedicated search functions (e.g., search_subject_by_code, search_group_format, search_category_format) to find and confirm the correct, official formats for any subjects, groups, or categories mentioned.
    - If the query includes ambiguous or multiple potential matches, ask the user for clarification or select the most likely candidate.

    3. **ACT**:
    - Once you have validated and formatted all input parameters, execute the correct API call(s) using the specialized Duke API tools.
    - For example, use the "get_duke_events" tool for event queries or the appropriate tool for retrieving AI MEng program details or prospective student information.

    4. **OBSERVE**:
    - Analyze and verify the data returned from the API tools.
    - Check that the returned results align with the user’s query and that all required formatting is correct.

    5. **RESPOND**:
    - Synthesize the fetched data into a clear, concise, and helpful response. Your answer should be accurate, professional, and tailored to the query’s focus (whether program details, key facts and figures, or event listings).
    - Do not mention internal formatting or search corrections unless necessary to help the user understand any issues.

    Remember:
    - Never bypass input validation: always convert user input into the official formats through your search tools before calling an API.
    - If there is uncertainty or multiple matches, ask the user to clarify rather than guessing.
    - Your responses should reflect Duke University's excellence and the specialized capabilities of Duke Pratt School of Engineering.
    - If you call a tool, always check the input format and pass the correct arguments to the tool.

    By following these steps, you ensure every query about the AI MEng program, prospective student information, or campus events is handled precisely and professionally.
    """
    
    # Create a proper chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    # Initialize the agent with the correct prompt
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        max_iterations=5,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        prompt=prompt  # Use the properly formatted prompt
    )
    
    return agent

# Example usage
def process_user_query(query):
    try:
        # Create the agent
        duke_agent = create_duke_agent()
        
        # Process the query using invoke
        response = duke_agent.invoke({"input": query})
        
        # Extract the agent's response
        return response.get("output", "I couldn't process your request at this time.")
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return f"An error occurred: {str(e)}"

# Example usage
def main():
    # Test queries that demonstrate format compatibility
    test_queries = [
        # "What events are happening at Duke this week?",
        "Get me detailed information about the AIPI courses",
        # "Tell me about Computer Science classes",
        # "Are there any AI events at Duke?",
        # "What cs courses are available?",
        # "Tell me about AI Meng program",
        # "please show me the events related to data science",
        # "please tell me about Brinnae Bent",
        # "tell me some professors who are working on AI",
        # "Introduce me Duke University",
        # "Tell me something about Pratt School of Engineering at Duke",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = process_user_query(query)
        print(f"Response: {response}")
        print("-" * 80)

if __name__ == "__main__":
    main()