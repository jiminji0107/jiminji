import torch
from model import SimpleSelfSupervisedModel

model = SimpleSelfSupervisedModel()
model.load_state_dict(torch.load('model.pth'))
model.eval() 