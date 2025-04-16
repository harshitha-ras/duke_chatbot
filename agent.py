from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# Import your custom tools from tools.py
from tools import (
    get_curriculum_with_subject_from_duke_api,
    get_events_from_duke_api_single_input,
    get_course_details_single_input,
    get_people_information_from_duke_api,
    search_subject_by_code,
    search_group_format,
    search_category_format,
    get_pratt_info_from_serpapi,
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
            func=get_events_from_duke_api_single_input,
            description=(
                "This tool retrieves upcoming events from Duke University's public calendar API based on a free-form natural language query. "
                "It processes your query by automatically mapping your input to the correct organizer groups and thematic categories. "
                "To do this, it reads the full lists of valid groups and categories from local text files, then uses fuzzy matching or retrieval-augmented generation "
                "to narrow these lists to the most relevant candidates. An LLM is subsequently used to select the final filter values; if no suitable filters "
                "are found, it defaults to ['All'] to maintain a valid API call. \n\n"
                "Parameters:\n"
                "  - prompt (str): A natural language description of the event filters you wish to apply (e.g., 'Please give me the events of AIPI').\n"
                "  - feed_type (str): The desired format for the returned data. Accepted values include 'rss', 'js', 'ics', 'csv', 'json', and 'jsonp'.\n"
                "  - future_days (int): The number of days into the future for which to retrieve events (default is 45).\n"
                "  - filter_method_group (bool): Defines filtering for organizer groups. If True, an event is included if it matches ANY specified group; "
                "if False, it must match ALL specified groups.\n"
                "  - filter_method_category (bool): Defines filtering for thematic categories. If True, an event is included if it matches ANY specified category; "
                "if False, it must match ALL specified categories.\n\n"
                "The tool returns the raw event data from Duke University's calendar API, or an error message if the API request fails."
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
            func=get_course_details_single_input,
            description=(
                "Use this tool to retrieve detailed curriculum information from Duke University's API. "
                "You must provide both a valid course ID (course_id) and a course offer number (course_offer_number), "
                "but **pass them as a single string** in the format 'course_id,course_offer_number'. "
                "\n\nFor example:\n"
                "  '027568,1' for course_id='027568' and course_offer_number='1'.\n\n"
                "These parameters can be obtained from get_curriculum_with_subject_from_duke_api, which returns a list "
                "of courses (each with a 'crse_id' and 'crse_offer_nbr').\n\n"
                "Parameters:\n"
                "  - course_id (str): The unique ID of the course, e.g. '029248'.\n"
                "  - course_offer_number (str): The offer number for that course, e.g. '1'.\n\n"
                "Return:\n"
                "  - str: Raw curriculum data in JSON format, or an error message if something goes wrong."
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
    - If it is a general query, use the PrattSearch tool to find relevant information first, then use the specialized tools for specific details.

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

def process_user_query(query):
    """
    Process a user query using the Duke agent.
    Args:
        query (str): The user query to process.
    Returns:
        str: The response from the agent.
    
    """
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

def main():
    # Test queries that demonstrate format compatibility
    test_queries = [
        "What events are happening at Duke this week?",
        "Get me detailed information about the AIPI courses",
        "Tell me about Computer Science classes",
        "Are there any AI events at Duke?",
        "What cs courses are available?",
        "Tell me about AIPI program",
        "please show me the events related to data science",
        "please tell me about Brinnae Bent",
        "tell me some professors who are working on AI",
        "Introduce me Duke University",
        "Tell me something about Pratt School of Engineering at Duke",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = process_user_query(query)
        print(f"Response: {response}")
        print("-" * 80)

if __name__ == "__main__":
    main()