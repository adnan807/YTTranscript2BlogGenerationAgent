from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs
from typing_extensions import TypedDict, List
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
import os

# --- Define the agent state ---
class AgentState(TypedDict):
    transcript: str
    youtube_url: str
    blog: str

def get_transcript(youtube_url: str) -> AgentState:
    state: AgentState = {'youtube_url': youtube_url, 'transcript': '', 'blog': ''}
    try:
        video_id = parse_qs(urlparse(youtube_url).query).get('v', [None])[0]
        if not video_id:
            state['transcript'] = "Invalid YouTube URL."
            return state

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        state['transcript'] = ' '.join([item['text'] for item in transcript_list])
        return state

    except TranscriptsDisabled:
        state['transcript'] = "This video doesn't contain a transcript."
        return state
    except NoTranscriptFound:
        state['transcript'] = "No transcript was found for this video."
        return state
    except Exception as e:
        state['transcript'] = f"An error occurred: {str(e)}"
        return state


def generate_blog(transcript: str, youtube_url: str = "") -> AgentState:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", api_key=google_api_key)

    prompt = f"""
    Create a blog post using the following transcript. The blog should include:

    1. A **Title**
    2. A **Description** (main body content)
    3. A **Conclusion**

    Transcript:
    {transcript}
    
    if there is no transcript simply return message in blog no blog for given link 
    """

    response = model.invoke(prompt)
    return {
        "transcript": transcript,
        "youtube_url": youtube_url,
        "blog": response.content if hasattr(response, "content") else str(response)
    }

get_transcript_tool = StructuredTool.from_function(
    name="GetTranscript",
    description="Get Transcript of youtube video if available",
    func=get_transcript
)

generate_blog_tool = StructuredTool.from_function(
    name="GenerateBlog",
    description="Generate blog from a YouTube transcript",
    func=generate_blog
)

# --- Agent class using LangGraph ---
class Agent:
    def __init__(self, tools: List[StructuredTool], system_prompt=""):
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in tools}

        graph = StateGraph(AgentState)
        # Add nodes using tool.invoke
        graph.add_node("GetTranscript", self.tools["GetTranscript"].invoke)
        graph.add_node("GenerateBlog", self.tools["GenerateBlog"].invoke)

        # Connect nodes
        graph.add_edge(START, "GetTranscript")
        graph.add_edge("GetTranscript", "GenerateBlog")
        graph.add_edge("GenerateBlog", END)

        # Compile the graph
        self.graph = graph.compile()

# --- Main execution ---
def main(youtube_link):
    agent = Agent([get_transcript_tool, generate_blog_tool])
    final_state = agent.graph.invoke({"youtube_url": youtube_link})
    print(final_state['blog'])

if __name__ == "__main__":
    main("https://www.youtube.com/watch?v=MeyePX5x2Pw&t=1s&ab_channel=BBCNews")
