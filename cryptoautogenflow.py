import autogen
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Literal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
]

coding_config = {
    "temperature": 0.2,
    "config_list": config_list,
    "timeout": 600,
}

def load_environmental_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"Environmental data loaded successfully. Shape: {df.shape}")
        print("\nSummary statistics:")
        print(df.describe())
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found. Please ensure the file exists and the path is correct.")

def preprocess_data(data: pd.DataFrame) -> Dict[str, np.ndarray]:
    X = data.drop('profit', axis=1).values
    y = data['profit'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': data.drop('profit', axis=1).columns.tolist()
    }

def create_correlation_heatmap(data: pd.DataFrame) -> plt.Figure:
    corr = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Environmental Variables')
    return plt.gcf()

def plot_profit_trends(data: pd.DataFrame) -> plt.Figure:
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['profit'], label='Profit')
    plt.title('Profit Trends Over Time')
    plt.xlabel('Time')
    plt.ylabel('Profit')
    plt.legend()
    return plt.gcf()

def train_model(preprocessed_data: Dict[str, np.ndarray]) -> Dict[str, Union[float, RandomForestRegressor]]:
    X_train, y_train = preprocessed_data['X_train'], preprocessed_data['y_train']
    X_test, y_test = preprocessed_data['X_test'], preprocessed_data['y_test']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    result = {
        "train_score": train_score,
        "test_score": test_score,
        "model": model
    }
    print(f"Model training complete. Train Score: {train_score:.4f}, Test Score: {test_score:.4f}")
    return result

def plot_feature_importance(model: RandomForestRegressor, feature_names: List[str]) -> plt.Figure:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    return plt.gcf()

def optimize_strategy(model: RandomForestRegressor, scaler: StandardScaler, feature_names: List[str]) -> List[str]:
    num_features = len(feature_names)
    num_samples = 10000
    
    random_samples = np.random.rand(num_samples, num_features)
    scaled_samples = scaler.transform(random_samples)
    
    predictions = model.predict(scaled_samples)
    best_idx = np.argmax(predictions)
    
    best_strategy = random_samples[best_idx]
    best_profit = predictions[best_idx]
    
    strategy = [f"Optimal {feature_names[i]}: {best_strategy[i]:.2f}" for i in range(num_features)]
    strategy.append(f"Expected Profit: {best_profit:.2f}")
    
    print("Strategy optimized:")
    for line in strategy:
        print(line)
    
    return strategy

def plot_optimization_results(model: RandomForestRegressor, scaler: StandardScaler, feature_names: List[str]) -> plt.Figure:
    num_features = len(feature_names)
    num_samples = 1000
    
    random_samples = np.random.rand(num_samples, num_features)
    scaled_samples = scaler.transform(random_samples)
    
    predictions = model.predict(scaled_samples)
    
    plt.figure(figsize=(14, 8))
    plt.scatter(random_samples[:, 0], predictions, alpha=0.5)
    plt.xlabel(feature_names[0])
    plt.ylabel('Predicted Profit')
    plt.title('Optimization Results')
    return plt.gcf()

# Common configuration for all agents
common_config = {
    "human_input_mode": "NEVER",
    "max_consecutive_auto_reply": 10,
    "code_execution_config": {
        "work_dir": "mining_optimization",
        "use_docker": False,
    },
}

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    system_message="You are a user proxy for a cryptocurrency mining optimization team. Your role is to facilitate the optimization process by providing necessary information and executing code when required.",
    **common_config
)

data_scientist = autogen.AssistantAgent(
    name="data_scientist",
    llm_config=coding_config,
    system_message="You are a data scientist specializing in cryptocurrency mining data analysis. Your role is to prepare, analyze, and visualize data for the ML model. Use the load_environmental_data, preprocess_data, create_correlation_heatmap, and plot_profit_trends functions. Ensure data quality and provide insights.",
    **common_config
)

ml_engineer = autogen.AssistantAgent(
    name="ml_engineer",
    llm_config=coding_config,
    system_message="You are a machine learning engineer specializing in predictive modeling. Your role is to design, train, and optimize ML models for cryptocurrency mining strategies. Use the train_model and plot_feature_importance functions. Validate model performance and suggest improvements.",
    **common_config
)

optimization_specialist = autogen.AssistantAgent(
    name="optimization_specialist",
    llm_config=coding_config,
    system_message="You are an optimization specialist. Your role is to interpret the results from the ML model and provide actionable mining strategies. Use the optimize_strategy and plot_optimization_results functions to generate and visualize optimal strategies.",
    **common_config
)

reviewer = autogen.AssistantAgent(
    name="reviewer",
    llm_config=coding_config,
    system_message="You are a senior technical lead and mining expert. Your role is to review and validate the work of the team at each step. Provide constructive feedback, ensure cross-functional collaboration, and only approve when the work meets high standards. Challenge assumptions and ensure robustness, scalability, and security of the proposed solutions.",
    **common_config
)

for agent in [user_proxy, data_scientist, ml_engineer, optimization_specialist, reviewer]:
    agent.register_function(
        function_map={
            "load_environmental_data": load_environmental_data,
            "preprocess_data": preprocess_data,
            "create_correlation_heatmap": create_correlation_heatmap,
            "plot_profit_trends": plot_profit_trends,
            "train_model": train_model,
            "plot_feature_importance": plot_feature_importance,
            "optimize_strategy": optimize_strategy,
            "plot_optimization_results": plot_optimization_results
        }
    )

def custom_speaker_selection(
    current_speaker: autogen.Agent,
    groupchat: autogen.GroupChat
) -> Union[autogen.Agent, Literal['auto', 'manual', 'random'], None]:
    last_message = groupchat.messages[-1]['content'].lower() if groupchat.messages else ""
    
    phase_order = {
        "data_analysis": [data_scientist, reviewer],
        "model_training": [ml_engineer, reviewer],
        "strategy_optimization": [optimization_specialist, reviewer],
        "final_review": [data_scientist, ml_engineer, optimization_specialist, reviewer]
    }
    
    if "data analysis phase complete" in last_message:
        return phase_order["model_training"][0]
    elif "model training phase complete" in last_message:
        return phase_order["strategy_optimization"][0]
    elif "strategy optimization phase complete" in last_message:
        return phase_order["final_review"][0]
    
    for phase, order in phase_order.items():
        if current_speaker in order:
            current_index = order.index(current_speaker)
            next_index = (current_index + 1) % len(order)
            return order[next_index]
    
    return 'auto'

groupchat = autogen.GroupChat(
    agents=[user_proxy, data_scientist, ml_engineer, optimization_specialist, reviewer],
    messages=[],
    max_round=100,
    speaker_selection_method=custom_speaker_selection
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=coding_config)

def main():
    user_proxy.initiate_chat(
        manager,
        message="""
        We need to develop an advanced cryptocurrency mining optimization system using Machine Learning. Please follow these steps:

        1. Data Analysis Phase:
           - Data Scientist: Load and preprocess environmental data.
           - Data Scientist: Perform data analysis and visualization.
           - Reviewer: Examine the data analysis and provide feedback.

        2. Model Training Phase:
           - ML Engineer: Train the model and evaluate its performance.
           - ML Engineer: Analyze feature importance.
           - Reviewer: Evaluate the model and provide feedback.

        3. Strategy Optimization Phase:
           - Optimization Specialist: Generate optimal mining strategies.
           - Optimization Specialist: Visualize optimization results.
           - Reviewer: Examine the proposed strategies and provide feedback.

        4. Final Review Phase:
           - All team members: Contribute to a comprehensive final report.
           - Reviewer: Conduct a final review of the entire system and approve or provide final feedback.

        The project will conclude when the reviewer approves the final comprehensive report.
        
        Let's begin with the Data Analysis Phase. Data Scientist, please start by loading and preprocessing our environmental data.
        """
    )

if __name__ == "__main__":
    main()
