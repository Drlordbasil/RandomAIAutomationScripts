import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from datasets import load_dataset

def load_dataset_from_files(files):
    """
    Load dataset from multiple Python script files.
    
    Args:
        files (list): List of file paths.
        
    Returns:
        TextDataset: Dataset object for language model training.
    
    Source:
    - [Loading Dataset](https://huggingface.co/docs/datasets/loading_datasets.html)
    """
    raw_text = ""
    for file in files:
        with open(file, "r") as f:
            raw_text += f.read() + "\n"
    
    with open("temp_dataset.txt", "w") as temp_f:
        temp_f.write(raw_text)
    
    return TextDataset(
        tokenizer=tokenizer,
        file_path="temp_dataset.txt",
        block_size=128,
    )

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred (tuple): Evaluation predictions.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    
    Source:
    - [HuggingFace Metrics](https://huggingface.co/docs/transformers/v4.4.2/en/performance#metrics)
    """
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == labels).float().mean()
    return {"accuracy": accuracy.item()}

def train_model(files):
    """
    Train the model using provided script files.
    
    Args:
        files (list): List of file paths.
        
    Returns:
        Trainer: Trainer object after training.
    
    Source:
    - [Fine-tuning a Language Model](https://huggingface.co/transformers/training.html)
    """
    dataset = load_dataset_from_files(files)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    return trainer

def chat_with_model(input_text):
    """
    Generate text from the fine-tuned model based on user input.
    
    Args:
        input_text (str): User input text.
        
    Returns:
        str: Generated text from the model.
    
    Source:
    - [Text Generation](https://huggingface.co/transformers/task_summary.html#text-generation)
    """
    model_name = "refine-ai/Power-Llama-3-13b-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    """
    Main function to define and launch the Gradio interface.
    
    Source:
    - [Gradio Documentation](https://gradio.app/docs/)
    """
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Tab("Train Model"):
            file_input = gr.File(file_count="multiple", type="filepath")
            train_button = gr.Button("Train")
            train_output = gr.Textbox()
            
        with gr.Tab("Chat with Model"):
            chat_input = gr.Textbox(label="Input Text")
            chat_button = gr.Button("Submit")
            chat_output = gr.Textbox()
            
        train_button.click(train_model, inputs=file_input, outputs=train_output)
        chat_button.click(chat_with_model, inputs=chat_input, outputs=chat_output)
    
    demo.launch()

if __name__ == "__main__":
    # Load the model and tokenizer
    model_name = "refine-ai/Power-Llama-3-13b-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    main()
