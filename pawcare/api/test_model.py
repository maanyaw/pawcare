import torch
import torch.nn as nn
import os

class PetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 4)

    def forward(self, x):
        return self.fc(x)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pet_model.pt")

model = PetModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

sample_input = torch.tensor([[10.0, 2.0, 1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
with torch.no_grad():
    output = model(sample_input)

print("✅ Model loaded successfully (weights only)!")
print("Output:", output)
