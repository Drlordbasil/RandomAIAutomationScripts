import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir='runs/multimodal_nn')

class MultimodalNN(nn.Module):
    """
    Multimodal Neural Network for integrating text, image, and audio inputs.
    """
    def __init__(self):
        super(MultimodalNN, self).__init__()
        
        # Text processing layers
        self.text_fc1 = nn.Linear(768, 512)
        self.text_fc2 = nn.Linear(512, 256)
        self.text_fc3 = nn.Linear(256, 128)
        self.text_fc4 = nn.Linear(128, 64)
        
        # Image processing layers
        self.image_fc1 = nn.Linear(2048, 1024)
        self.image_fc2 = nn.Linear(1024, 512)
        self.image_fc3 = nn.Linear(512, 256)
        self.image_fc4 = nn.Linear(256, 128)
        
        # Audio processing layers
        self.audio_fc1 = nn.Linear(128, 256)
        self.audio_fc2 = nn.Linear(256, 128)
        self.audio_fc3 = nn.Linear(128, 64)
        self.audio_fc4 = nn.Linear(64, 32)
        
        # Combined layers
        self.combined_fc1 = nn.Linear(64 + 128 + 32, 512)
        self.combined_fc2 = nn.Linear(512, 256)
        self.combined_fc3 = nn.Linear(256, 128)
        self.combined_fc4 = nn.Linear(128, 64)
        
        # Output layer
        self.output = nn.Linear(64, 3)  # 3 outputs: text, image, audio

    def forward(self, text_input, image_input, audio_input):
        """
        Forward pass through the network.
        
        Args:
            text_input (torch.Tensor): Input tensor for text data.
            image_input (torch.Tensor): Input tensor for image data.
            audio_input (torch.Tensor): Input tensor for audio data.
        
        Returns:
            torch.Tensor: Output tensor with 3 values corresponding to the combined input.
        """
        # Preprocess inputs
        text_input = self.preprocess_text(text_input)
        image_input = self.preprocess_image(image_input)
        audio_input = self.preprocess_audio(audio_input)

        # Text branch
        text_out = torch.relu(self.text_fc1(text_input))
        text_out = torch.relu(self.text_fc2(text_out))
        text_out = torch.relu(self.text_fc3(text_out))
        text_out = torch.relu(self.text_fc4(text_out))
        
        # Image branch
        image_out = torch.relu(self.image_fc1(image_input))
        image_out = torch.relu(self.image_fc2(image_out))
        image_out = torch.relu(self.image_fc3(image_out))
        image_out = torch.relu(self.image_fc4(image_out))
        
        # Audio branch
        audio_out = torch.relu(self.audio_fc1(audio_input))
        audio_out = torch.relu(self.audio_fc2(audio_out))
        audio_out = torch.relu(self.audio_fc3(audio_out))
        audio_out = torch.relu(self.audio_fc4(audio_out))
        
        # Concatenate all outputs
        combined = torch.cat((text_out, image_out, audio_out), dim=1)
        combined = torch.relu(self.combined_fc1(combined))
        combined = torch.relu(self.combined_fc2(combined))
        combined = torch.relu(self.combined_fc3(combined))
        combined = torch.relu(self.combined_fc4(combined))
        return self.output(combined)
    
    def preprocess_text(self, text_input, desired_size=768):
        """
        Preprocess text input to match the desired size.
        
        Args:
            text_input (torch.Tensor): Input tensor for text data.
            desired_size (int): Desired size of the tensor's second dimension.
        
        Returns:
            torch.Tensor: Preprocessed text tensor.
        """
        return self._pad_or_trim(text_input, desired_size)

    def preprocess_image(self, image_input, desired_size=2048):
        """
        Preprocess image input to match the desired size.
        
        Args:
            image_input (torch.Tensor): Input tensor for image data.
            desired_size (int): Desired size of the tensor's second dimension.
        
        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        return self._pad_or_trim(image_input, desired_size)

    def preprocess_audio(self, audio_input, desired_size=128):
        """
        Preprocess audio input to match the desired size.
        
        Args:
            audio_input (torch.Tensor): Input tensor for audio data.
            desired_size (int): Desired size of the tensor's second dimension.
        
        Returns:
            torch.Tensor: Preprocessed audio tensor.
        """
        return self._pad_or_trim(audio_input, desired_size)

    def _pad_or_trim(self, tensor, size):
        """
        Pad or trim the input tensor to the specified size.
        
        Args:
            tensor (torch.Tensor): Input tensor.
            size (int): Desired size of the tensor's second dimension.
        
        Returns:
            torch.Tensor: Reshaped tensor.
        """
        if tensor.shape[1] > size:
            return tensor[:, :size]
        elif tensor.shape[1] < size:
            padding = size - tensor.shape[1]
            return torch.cat([tensor, torch.zeros(tensor.shape[0], padding)], dim=1)
        return tensor

class RLAgent:
    """
    Reinforcement Learning Agent using a Multimodal Neural Network.
    """
    def __init__(self, model, lr=0.001):
        """
        Initialize the RLAgent.
        
        Args:
            model (nn.Module): The neural network model to be used.
            lr (float): Learning rate for the optimizer.
        """
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def update(self, inputs, targets, iteration):
        """
        Perform an update of the model parameters based on inputs and targets.
        
        Args:
            inputs (tuple): A tuple containing tensors for text, image, and audio data.
            targets (torch.Tensor): The target tensor.
            iteration (int): Current iteration number.
        """
        assert isinstance(inputs, tuple) and len(inputs) == 3, "Inputs should be a tuple containing three elements: text, image, and audio data."
        assert targets.shape[1] == 3, "Targets shape mismatch"

        self.optimizer.zero_grad()
        outputs = self.model(*inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        # Log the loss
        logging.info(f"Iteration {iteration} - Update loss: {loss.item()}")
        writer.add_scalar('Loss/train', loss.item(), iteration)

    def predict(self, inputs):
        """
        Make predictions using the model.
        
        Args:
            inputs (tuple): A tuple containing tensors for text, image, and audio data.
        
        Returns:
            torch.Tensor: Model predictions.
        """
        with torch.no_grad():
            return self.model(*inputs)

    def save_checkpoint(self, iteration):
        """
        Save a checkpoint of the model's state.
        
        Args:
            iteration (int): Current iteration number.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_checkpoint_{iteration}.pth')
        torch.save(self.model.state_dict(), checkpoint_path)
        logging.info(f"Model checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint of the model's state.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        self.model.load_state_dict(torch.load(checkpoint_path))
        logging.info(f"Model checkpoint loaded from {checkpoint_path}")

if __name__ == "__main__":
    # Example data for testing
    text_input = torch.randn((1, 500))  # Example shape for text input, different from expected
    image_input = torch.randn((1, 3000))  # Example shape for image input, different from expected
    audio_input = torch.randn((1, 100))  # Example shape for audio input, different from expected
   
    target_output = torch.tensor([[1.0, 0.0, 0.0]])  # Example target output
   
    # Initialize model and agent
    model = MultimodalNN()
    agent = RLAgent(model)

    # For extensive testing, we will run multiple iterations
    num_iterations = 100
    for iteration in range(num_iterations):
        print(f"Iteration: {iteration + 1}")
        
        try:
            # Update the model
            agent.update((text_input, image_input, audio_input), target_output, iteration)
            print("Update successful.")
        except Exception as e:
            print(f"Error during model update: {e}")
        
        try:
            # Make a prediction
            outputs = agent.predict((text_input, image_input, audio_input))
            print(f"Predicted outputs: {outputs}")
        except Exception as e:
            print(f"Error during prediction: {e}")
        
        # Save model checkpoint
        agent.save_checkpoint(iteration)
        
        # Add any additional tests or checks here
        print("-" * 50)

    # Unit test for _pad_or_trim method
    def test_pad_or_trim():
        model = MultimodalNN()
        # Test case where tensor needs to be trimmed
        tensor = torch.randn((1, 1000))
        reshaped_tensor = model._pad_or_trim(tensor, 768)
        assert reshaped_tensor.shape[1] == 768, "Test failed for trimming."

        # Test case where tensor needs to be padded
        tensor = torch.randn((1, 600))
        reshaped_tensor = model._pad_or_trim(tensor, 768)
        assert reshaped_tensor.shape[1] == 768, "Test failed for padding."

        # Test case where tensor is already the correct size
        tensor = torch.randn((1, 768))
        reshaped_tensor = model._pad_or_trim(tensor, 768)
        assert reshaped_tensor.shape[1] == 768, "Test failed for correct size."

        # Additional edge case tests
        tensor = torch.randn((2, 500))
        reshaped_tensor = model._pad_or_trim(tensor, 768)
        assert reshaped_tensor.shape == (2, 768), "Test failed for padding with batch size > 1."

        tensor = torch.randn((2, 1000))
        reshaped_tensor = model._pad_or_trim(tensor, 768)
        assert reshaped_tensor.shape == (2, 768), "Test failed for trimming with batch size > 1."

        print("All tests for _pad_or_trim passed.")

    # Run unit tests
    test_pad_or_trim()
    
    # Close the TensorBoard writer
    writer.close()
