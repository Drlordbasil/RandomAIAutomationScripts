import os
import logging
import time
import random
from typing import List, Dict, Any, Union
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
import seaborn as sns
import autogen
from autogen import Agent, UserProxyAgent, AssistantAgent, ConversableAgent, GroupChat, GroupChatManager
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
research_collection = chroma_client.get_or_create_collection(name="research_papers")
meta_collection = chroma_client.get_or_create_collection(name="meta_learning")

# Use OpenAI's embedding function
embedding_function = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="mxbai-embed-large",
)

def load_config() -> List[Dict[str, Any]]:
    """Load the configuration from the JSON file."""
    try:
        return autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST.json",filter_dict={"model": "llama3-8b-8192"})
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def get_llm_config(config_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create the LLM configuration."""
    return {
        "cache_seed": 42,
        "temperature": 0,
        "config_list": config_list,
        "timeout": 380,
    }

class RateLimiter:
    def __init__(self, max_calls: int, time_frame: int):
        self.max_calls = max_calls
        self.time_frame = time_frame
        self.calls = []

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            now = time.time()
            self.calls = [call for call in self.calls if now - call < self.time_frame]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_frame - (now - self.calls[0])
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            self.calls.append(now)
            return func(*args, **kwargs)
        return wrapper

@RateLimiter(max_calls=5, time_frame=60)
def retrieve_papers(sources: List[str], keywords: str, timeframe: str) -> List[Dict[str, str]]:
    """Retrieve research papers from various sources."""
    papers = []
    for source in sources:
        try:
            if source == 'arxiv':
                url = f'https://arxiv.org/search/?query={keywords}&start=0&max_results=100&sortBy=submittedDate&sortOrderDescending=yes'
                response = requests.get(url, timeout=30)
                soup = BeautifulSoup(response.content, 'html.parser')
                results = soup.find_all('li', class_='arxiv-result')
                print(f"Found {len(results)} papers from ArXiv")
                for result in results:
                    title = result.find('p', class_='title').text.strip()
                    authors = result.find('p', class_='authors').text.strip()
                    abstract = result.find('p', class_='abstract').text.strip()
                    pdf_url = result.find('a', href=True)['href']
                    published = result.find('p', class_='is-size-7').text.strip()
                    papers.append({'title': title, 'authors': authors, 'abstract': abstract, 'pdf_url': pdf_url, 'published': published, 'source': source})
            elif source == 'ieee':
                url = f'https://ieeexplore.ieee.org/search/searchresult.jsp?queryText={keywords}&newsearch=true'
                response = requests.get(url, timeout=30)
                soup = BeautifulSoup(response.content, 'html.parser')
                results = soup.find_all('div', class_='result-item')
                for result in results:
                    title = result.find('h2', class_='document-title').text.strip()
                    authors = result.find('p', class_='authors').text.strip()
                    abstract = result.find('div', class_='abstract').text.strip()
                    pdf_url = f"https://ieeexplore.ieee.org{result.find('a', class_='document-link')['href']}"
                    published = result.find('div', class_='publication-year').text.strip()
                    papers.append({'title': title, 'authors': authors, 'abstract': abstract, 'pdf_url': pdf_url, 'published': published, 'source': source})
        except Exception as e:
            logger.error(f"Error retrieving papers from {source}: {e}")
    return papers
@RateLimiter(max_calls=5, time_frame=60)
def analyze_papers(papers: List[Dict[str, str]], timeframe: str) -> Dict[str, Any]:
    """Analyze and visualize research papers."""
    papers = [paper for paper in papers if paper['published'] >= timeframe]

    titles = [paper['title'] for paper in papers]
    abstracts = [paper['abstract'] for paper in papers]
    authors = [paper['authors'] for paper in papers]

    vectorizer = TfidfVectorizer(max_features=5000)
    abstract_vectors = vectorizer.fit_transform(abstracts)

    kmeans = KMeans(n_clusters=5)
    clusters = kmeans.fit_predict(abstract_vectors)
    lda = LatentDirichletAllocation(n_components=5)
    topics = lda.fit_transform(abstract_vectors)

    plt.figure(figsize=(10, 8))
    tsne = TSNE(n_components=2)
    reduced_vectors = tsne.fit_transform(abstract_vectors.toarray())
    sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=clusters)
    plt.title('Cluster Analysis of LLM Papers')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('cluster_analysis.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(topics, annot=True, cmap='Blues')
    plt.title('Topic Modeling of LLM Papers')
    plt.xlabel('Topics')
    plt.ylabel('Papers')
    plt.savefig('topic_modeling.png')
    plt.close()

    trend_analysis = []
    for cluster in np.unique(clusters):
        papers_in_cluster = [paper for paper, cluster_label in zip(papers, clusters) if cluster_label == cluster]
        trend = {'cluster': int(cluster), 'papers': papers_in_cluster, 'trend': '', 'impact': ''}
        for paper in papers_in_cluster:
            trend['trend'] += paper['abstract'] + ' '
            trend['impact'] += paper['abstract'] + ' '
        trend_analysis.append(trend)

    return {
        'clusters': clusters.tolist(),
        'topics': topics.tolist(),
        'trend_analysis': trend_analysis,
        'papers': papers
    }
@RateLimiter(max_calls=5, time_frame=60)
def dynamic_speaker_selection(
    last_speaker: Agent,
    groupchat: GroupChat
) -> Union[Agent, str, None]:
    """Dynamic speaker selection based on current research needs and agent capabilities."""
    messages = groupchat.messages
    agents = groupchat.agents

    logger.info(f"Selecting next speaker. Last speaker was: {last_speaker.name}")

    last_message = messages[-1]["content"].lower()
    
    if "research" in last_message or "analyze" in last_message:
        return next(agent for agent in agents if agent.name == "Researcher")
    elif "code" in last_message or "implement" in last_message:
        return next(agent for agent in agents if agent.name == "Coder")
    elif "execute" in last_message or "run" in last_message:
        return next(agent for agent in agents if agent.name == "Executor")
    elif "evaluate" in last_message or "critique" in last_message:
        return next(agent for agent in agents if agent.name == "Critic")
    elif "coordinate" in last_message or "next steps" in last_message:
        return next(agent for agent in agents if agent.name == "Initializer")
    
    return random.choice(agents)
@RateLimiter(max_calls=5, time_frame=60)
def adaptive_task_completion_check(message: Dict[str, str]) -> bool:
    """Adaptive task completion check based on research progress and quality."""
    completion_indicators = [
        "research summary complete",
        "final analysis:",
        "conclusion of the literature review"
    ]
    content = message.get("content", "")
    
    basic_completion = isinstance(content, str) and any(indicator in content.lower() for indicator in completion_indicators)
    
    if basic_completion:
        quality_score = analyze_research_quality(content)
        if quality_score > 0.8:
            logger.info(f"High-quality research completed. Quality score: {quality_score}")
            return True
        else:
            logger.info(f"Research complete but quality score below threshold: {quality_score}")
            return False
    
    return False
@RateLimiter(max_calls=5, time_frame=60)
def analyze_research_quality(content: str) -> float:
    """Analyze the quality of the research based on various factors."""
    factors = [
        len(content) > 1000,
        "methodology" in content.lower(),
        "conclusion" in content.lower(),
        "future work" in content.lower(),
        content.count("\n") > 10
    ]
    return sum(factors) / len(factors)
@RateLimiter(max_calls=5, time_frame=60)
def create_agents(llm_config: Dict[str, Any]) -> List[ConversableAgent]:
    """Create and return a list of agents for the research workflow."""
    initializer = UserProxyAgent(
        name="Initializer",
        system_message="You are the initiator and coordinator of the research workflow. Your tasks include providing initial research topics, adjusting research direction based on findings, and coordinating between other agents.",
        human_input_mode="TERMINATE",
        code_execution_config=False,
    )

    researcher = AssistantAgent(
        name="Researcher",
        llm_config=llm_config,
        system_message="You are an expert researcher specializing in LLM applications. Your task is to analyze research papers, identify trends, and suggest future research directions.",
    )

    coder = AssistantAgent(
        name="Coder",
        llm_config=llm_config,
        system_message="You are an expert Python programmer. Your task is to write and refine code for data retrieval, analysis, and visualization of research papers.",
    )

    executor = UserProxyAgent(
        name="Executor",
        system_message="You are responsible for executing code and reporting results. Run the provided code, handle errors, and provide detailed output.",
        human_input_mode="NEVER",
        code_execution_config={
            "last_n_messages": 3,
            "work_dir": "research_papers",
            "use_docker": False,
        },
    )

    critic = AssistantAgent(
        name="Critic",
        llm_config=llm_config,
        system_message="You are a critical thinker and evaluator. Your task is to evaluate the quality of research findings, identify potential biases, and suggest improvements to the research process.",
    )

    return [initializer, researcher, coder, executor, critic]
@RateLimiter(max_calls=5, time_frame=60)
def create_group_chat(agents: List[ConversableAgent]) -> GroupChat:
    """Create and return a GroupChat instance with the provided agents."""
    return GroupChat(
        agents=agents,
        messages=[],
        max_round=50,
        speaker_selection_method=dynamic_speaker_selection,
        allow_repeat_speaker=False,
    )
@RateLimiter(max_calls=5, time_frame=60)
def create_group_chat_manager(groupchat: GroupChat, llm_config: Dict[str, Any]) -> GroupChatManager:
    """Create and return a GroupChatManager instance."""
    return GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
        is_termination_msg=adaptive_task_completion_check
    )
@RateLimiter(max_calls=5, time_frame=60)
def save_research_results(content: str, filename: str = "research_summary.md"):
    """Save research results to a file and ChromaDB."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Research summary saved to {filename}")
    research_collection.add(
        documents=[content],
        metadatas=[{"type": "research_summary", "filename": filename}],
        ids=[str(hash(content))]
    )
@RateLimiter(max_calls=5, time_frame=60)
def analyze_research_impact(content: str) -> Dict[str, Any]:
    """Analyze the impact of the research findings."""
    word_count = len(content.split())
    unique_words = len(set(content.lower().split()))
    readability_score = 1 - (unique_words / word_count)
    
    impact_score = (word_count / 1000) * readability_score
    
    impact_analysis = {
        "impact_score": impact_score,
        "word_count": word_count,
        "unique_words": unique_words,
        "readability_score": readability_score
    }
    
    logger.info(f"Research Impact Analysis: {json.dumps(impact_analysis, indent=2)}")
    meta_collection.add(
        documents=[json.dumps(impact_analysis)],
        metadatas=[{"type": "impact_analysis"}],
        ids=[str(hash(json.dumps(impact_analysis)))]
    )
    
    return impact_analysis
@RateLimiter(max_calls=5, time_frame=60)
def update_research_strategy(chat_history: List[Dict[str, Any]]):
    """Update research strategy based on intermediate findings and agent interactions."""
    topics = [msg["content"] for msg in chat_history if msg["role"] == "assistant"]
    topic_embeddings = embedding_function(topics)
    
    n_clusters = min(5, len(topics))
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(topic_embeddings)
        
        topic_counts = np.bincount(labels)
        main_topics = [topics[i] for i in np.argsort(topic_counts)[-3:]]
        
        logger.info(f"Main research topics identified: {main_topics}")
        
        strategy_update = {
            "main_topics": main_topics,
            "topic_distribution": topic_counts.tolist()
        }
        meta_collection.add(
            documents=[json.dumps(strategy_update)],
            metadatas=[{"type": "research_strategy_update"}],
            ids=[str(hash(json.dumps(strategy_update)))]
        )
        
        return strategy_update
    else:
        logger.info("Not enough data to update research strategy.")
        return None
@RateLimiter(max_calls=5, time_frame=60)
def perform_meta_learning(chat_history: List[Dict[str, Any]], impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Perform meta-learning to improve the research process for future runs."""
    agent_interactions = {}
    for msg in chat_history:
        sender = msg.get("name", "Unknown")
        if sender not in agent_interactions:
            agent_interactions[sender] = 0
        agent_interactions[sender] += 1

    agent_effectiveness = {}
    for agent, count in agent_interactions.items():
        effectiveness = count / len(chat_history) * impact_analysis['impact_score']
        agent_effectiveness[agent] = effectiveness

    if impact_analysis['impact_score'] < 0.5:
        areas_for_improvement = ["Increase diversity of research sources", "Enhance critical analysis"]
    elif impact_analysis['readability_score'] < 0.6:
        areas_for_improvement = ["Improve clarity of explanations", "Enhance structure of research summary"]
    else:
        areas_for_improvement = ["Explore more interdisciplinary connections", "Deepen analysis of potential impacts"]

    meta_learning_result = {
        "agent_interactions": agent_interactions,
        "agent_effectiveness": agent_effectiveness,
        "areas_for_improvement": areas_for_improvement
    }

    meta_collection.add(
        documents=[json.dumps(meta_learning_result)],
        metadatas=[{"type": "meta_learning"}],
        ids=[str(hash(json.dumps(meta_learning_result)))]
    )

    return meta_learning_result
@RateLimiter(max_calls=5, time_frame=60)
def run_research_workflow():
    """Main research workflow with dynamic adaptation."""
    try:
        config_list = load_config()
        llm_config = get_llm_config(config_list)
        
        agents = create_agents(llm_config)
        groupchat = create_group_chat(agents)
        manager = create_group_chat_manager(groupchat, llm_config)
        
        initializer = next(agent for agent in agents if agent.name == "Initializer")
        
        chat_result = initializer.initiate_chat(
            manager,
            message="Conduct a comprehensive literature review on the latest advancements in LLM applications. "
                    "Focus on papers from the last month, covering at least 5 different domains. "
                    "Provide an in-depth analysis of trends, potential impacts, and suggest future research directions."
        )

        for i, message in enumerate(chat_result.chat_history):
            if message["role"] == "user" and message.get("name") == "Executor":
                logger.info(f"Executed code: {message['content'][:100]}...")
            
            if i % 10 == 0 and i > 0:
                strategy_update = update_research_strategy(chat_result.chat_history[:i])
                if strategy_update:
                    initializer.send(
                        f"Based on our current findings, let's adjust our focus to these main topics: {strategy_update['main_topics']}. "
                        f"Please incorporate these insights into your next steps.",
                        manager
                    )

        final_summary = ""
        for message in reversed(chat_result.chat_history):
            if message["role"] == "assistant" and message.get("name") == "Researcher":
                final_summary = message["content"]
                break

        if not final_summary:
            logger.warning("No final summary found from Researcher. Using the last message in chat history.")
            final_summary = chat_result.chat_history[-1]["content"]

        print("\nFinal Research Summary:")
        print(final_summary)

        save_research_results(final_summary)

        impact_analysis = analyze_research_impact(final_summary)

        meta_learning_result = perform_meta_learning(chat_result.chat_history, impact_analysis)

        logger.info("Full conversation history:")
        for msg in chat_result.chat_history:
            logger.info(f"{msg.get('name', 'Unknown')}: {msg['content'][:100]}...")

        logger.info(f"Research workflow completed. Impact Score: {impact_analysis['impact_score']:.2f}")
        logger.info(f"Meta-learning insights: {json.dumps(meta_learning_result, indent=2)}")

        return {
            'final_summary': final_summary,
            'impact_analysis': impact_analysis,
            'meta_learning_result': meta_learning_result
        }

    except Exception as e:
        logger.error(f"An error occurred during the research workflow: {e}")
        raise

if __name__ == "__main__":
    max_retries = 3
    retry_delay = 15  # seconds

    for attempt in range(max_retries):
        try:
            result = run_research_workflow()
            
            # Save the final results
            with open('final_research_results.json', 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info("Research workflow completed successfully. Results saved to 'final_research_results.json'")
            break
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached. Exiting.")
                raise
