


from autogen import ConversableAgent, UserProxyAgent, AssistantAgent, config_list_from_json


# Configure LLM
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST.json")
llm_config = {
    "config_list": config_list,
    "timeout": 120,
    "cache_seed": None
}




class CreativeAISystem:
    def __init__(self):
        self.creative_assistant = ConversableAgent(
            name="CreativeAssistant",
            system_message="""You are a highly creative AI assistant capable of generating innovative ideas and solutions. 
            Your expertise spans various domains, and you excel at coming up with novel concepts and improving existing ones. 
            Always provide complete Python code or shell commands. The UserProxyAgent will handle the execution.""",
            llm_config=llm_config,
        )

        self.knowledge_assistant = ConversableAgent(
            name="KnowledgeAssistant",
            system_message="""You are an AI with vast knowledge across multiple domains. 
            Your role is to provide accurate and relevant information to support the creative process and suggest improvements.""",
            llm_config=llm_config,
        )

        self.user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            llm_config=llm_config,
            max_consecutive_auto_reply=5,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={
                "work_dir": "creative_ai_output",
                "use_docker": True,
            },
            system_message="""You are a proxy for the user, always asking for complete Python code or shell commands. 
            Encourage continuous refinement and improvement of ideas and code.""",
        )


        self.product_manager = AssistantAgent(
            name="ProductManager",
            system_message="""You are a product manager responsible for guiding the development of creative and profitable AI applications. 
            Your expertise lies in market trends, user needs, and product strategy. 
            Continuously seek ways to improve and refine ideas and code. 
            Always provide executable Python code or shell commands.""",
            llm_config=llm_config,
        )

        self.entrepreneurial_mindset_agent = AssistantAgent(
            name="EntrepreneurialMindsetAgent",
            system_message="""You are an entrepreneurial-minded assistant focused on identifying and leveraging opportunities to monetize AI applications. 
            Your role is to provide strategic insights, suggest revenue models, and guide the development of profitable AI solutions. 
            Always provide actionable Python code or shell commands.""",
            llm_config=llm_config,
        )

        # Initialize context variables
        self.application_idea = ""
        self.application_concept = ""
        self.product_roadmap = ""
        self.prototype_code = ""

    def update_context(self, key, value):
        setattr(self, key, value)

    def get_context(self):
        return f"""
        Current Application Idea: {self.application_idea}
        Current Application Concept: {self.application_concept}
        Current Product Roadmap: {self.product_roadmap}
        Current Prototype Code: {self.prototype_code}
        """

    def brainstorm_creative_application(self):
        brainstorm_message = f"""
        Let's brainstorm a creative and profitable AI application. Consider recent trends in AI and potential market needs. Keep refining and improving the ideas.
        {self.get_context()}
        """
        response = self.user_proxy.initiate_chat(
            self.product_manager,
            message=brainstorm_message,
        )
        self.update_context("application_idea", response.summary)

        idea_refinement_message = f"""
        Based on our initial brainstorming, let's refine our AI application idea further:
        {self.get_context()}
        Suggest improvements and extensions to this idea.
        """
        response = self.user_proxy.initiate_chat(
            self.creative_assistant,
            message=idea_refinement_message,
        )
        self.update_context("application_idea", response.summary)

    def develop_application_concept(self):
        concept_message = f"""
        Based on our brainstorming, develop a detailed concept for our AI application. Include potential features, target audience, and how it leverages recent advancements in AI. Continuously refine and improve this concept.
        {self.get_context()}
        """
        response = self.user_proxy.initiate_chat(
            self.creative_assistant,
            message=concept_message,
        )
        self.update_context("application_concept", response.summary)

    def create_product_roadmap(self):
        roadmap_message = f"""
        Create a comprehensive product roadmap for our AI application concept. Include key milestones, potential challenges, and development process suggestions. Keep updating and optimizing this roadmap.
        {self.get_context()}
        """
        response = self.user_proxy.initiate_chat(
            self.product_manager,
            message=roadmap_message,
        )
        self.update_context("product_roadmap", response.summary)

    def generate_prototype_code(self):
        code_generation_message = f"""
        Generate Python code for a prototype of our AI application. The code should demonstrate the core functionality, including:
        1. Basic structure of the application
        2. Key features outlined in our concept
        3. Placeholder for AI model integration
        Use best practices for code organization, error handling, and include comprehensive comments. Ensure to provide commands for necessary library installations using pip.
        Continuously refine, optimize, and improve this code.
        {self.get_context()}
        """
        response = self.user_proxy.initiate_chat(
            self.creative_assistant,
            message=code_generation_message,
        )
        self.update_context("prototype_code", response.summary)

    def monetize_application(self):
        monetize_message = f"""
        Suggest potential revenue models and strategies to monetize our AI application. 
        Provide actionable Python code or shell commands to implement these strategies. 
        Ensure to include steps for tracking and optimizing revenue generation.
        {self.get_context()}
        """
        response = self.user_proxy.initiate_chat(
            self.entrepreneurial_mindset_agent,
            message=monetize_message,
        )
        self.update_context("monetization_strategy", response.summary)

    def run_creative_ai_system(self):
        print("Starting Creative AI System...")
        self.brainstorm_creative_application()
        self.develop_application_concept()
        self.create_product_roadmap()
        self.generate_prototype_code()
        self.monetize_application()
        print("Creative AI System process completed. Final context:")
        print(self.get_context())
        print("Check the 'creative_ai_output' directory for detailed results.")

if __name__ == "__main__":
    creative_system = CreativeAISystem()
    creative_system.run_creative_ai_system()
