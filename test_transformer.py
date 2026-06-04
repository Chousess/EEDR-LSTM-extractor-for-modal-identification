from models.transformer_model import TransformerModel
import torch

print('=== Test 1: Basic run on CPU ===')
model = TransformerModel()
x = torch.randn(100, 4)
y = model(x)
print('[OK] Basic run successful')

print('\n=== Test 2: Call regularization_loss ===')
try:
    loss = model.regularization_loss()
    print('[OK] regularization_loss successful')
except Exception as e:
    print(f'[ERROR] {type(e).__name__}: {e}')

print('\n=== Test 3: Import TransformerModel from train_flat.py ===')
try:
    from train_flat import TransformerModel as TFModel
    print('[OK] Successfully imported TransformerModel from train_flat.py')
except Exception as e:
    print(f'[ERROR] Import error: {e}')
