#!/usr/bin/env python
# -*- coding: utf-8 -*-
from models.transformer_model import TransformerModel
import torch
import traceback

def test_cpu():
    print('='*60)
    print('TEST 1: Basic run on CPU')
    print('='*60)
    try:
        model = TransformerModel()
        x = torch.randn(10, 4)  # smaller batch
        y = model(x)
        print('[PASS] Forward pass successful')
        print(f'  Input shape: {x.shape}')
        print(f'  Output shapes: {[o.shape for o in y]}')
        return True
    except Exception as e:
        print(f'[FAIL] {e}')
        traceback.print_exc()
        return False

def test_loss():
    print('\n' + '='*60)
    print('TEST 2: Regularization loss on CPU')
    print('='*60)
    try:
        model = TransformerModel()
        x = torch.randn(10, 4)
        y = model(x)
        loss = model.regularization_loss()
        print('[PASS] Regularization loss computation successful')
        print(f'  Loss value: {loss.item():.6f}')
        return True
    except Exception as e:
        print(f'[FAIL] {e}')
        traceback.print_exc()
        return False

def test_gpu():
    print('\n' + '='*60)
    print('TEST 3: Run on GPU (if available)')
    print('='*60)
    if not torch.cuda.is_available():
        print('[SKIP] GPU not available')
        return True
    
    try:
        device = torch.device('cuda:0')
        model = TransformerModel().to(device)
        x = torch.randn(10, 4).to(device)
        y = model(x)
        print('[PASS] GPU forward pass successful')
        
        loss = model.regularization_loss()
        print('[PASS] GPU regularization loss computation successful')
        print(f'  Loss value: {loss.item():.6f}')
        return True
    except Exception as e:
        print(f'[FAIL] {e}')
        traceback.print_exc()
        return False

if __name__ == '__main__':
    results = []
    results.append(('CPU Basic', test_cpu()))
    results.append(('CPU Loss', test_loss()))
    results.append(('GPU', test_gpu()))
    
    print('\n' + '='*60)
    print('TEST SUMMARY')
    print('='*60)
    for test_name, passed in results:
        status = '[PASS]' if passed else '[FAIL]'
        print(f'{status} {test_name}')
