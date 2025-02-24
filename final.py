import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# CUDA 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================
# Custom Loss Class Definition
# ============================
class CustomLoss(CrossEntropyLoss):
    def __init__(self, ignore_index=-100):
        super().__init__(ignore_index=ignore_index, reduction='none')
        
    def forward(self, input, target):
        loss = super().forward(input.view(-1, input.size(-1)), target.view(-1))
        loss = loss.view(target.size(0), -1)
        return loss.mean()
    
    @staticmethod
    def create_bracket_mask(target):
        return torch.ones_like(target, dtype=torch.bool)

# ============================
# Dataset Class Definition
# ============================
class NumberNormalizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, is_training=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
    def __len__(self):
        return len(self.data)
    
    def preprocess_text(self, text, remove_brackets=False):
        if remove_brackets:
            text = re.sub(r'\[(.*?)\]', r'\1', text)
        return text
    
    def __getitem__(self, idx):
        input_text = self.data.iloc[idx]['scriptITN']
        target_text = self.data.iloc[idx]['scriptTN']
        
        # T5 모델용 task prefix 추가
        input_text = "normalize: " + input_text
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
            'input_text': input_text,
            'target_text': target_text
        }

# ============================
# Metric Calculation Function
# ============================
def calculate_metrics(pred_text, target_text):
    def extract_brackets(text):
        return re.findall(r'\[(.*?)\]', text)
    
    pred_brackets = extract_brackets(pred_text)
    target_brackets = extract_brackets(target_text)
    
    bracket_accuracy = sum(p == t for p, t in zip(pred_brackets, target_brackets)) / max(len(target_brackets), 1)
    
    pred_no_brackets = re.sub(r'\[.*?\]', '', pred_text)
    target_no_brackets = re.sub(r'\[.*?\]', '', target_text)
    text_preservation = 1.0 if pred_no_brackets.strip() == target_no_brackets.strip() else 0.0
    
    return {
        'bracket_accuracy': bracket_accuracy,
        'text_preservation': text_preservation
    }

# ============================
# Training Function
# ============================
def train_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    metrics = {'bracket_accuracy': 0, 'text_preservation': 0}
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    num_samples = len(dataloader.dataset)
    metrics = {k: v/num_samples for k, v in metrics.items()}
    
    return total_loss / len(dataloader), metrics

# ============================
# Evaluation Function
# ============================
def evaluate(model, dataloader, device, criterion, tokenizer):
    model.eval()
    total_loss = 0
    metrics = {'bracket_accuracy': 0, 'text_preservation': 0}
    predictions = []
    
    progress_bar = tqdm(dataloader, desc='Evaluating')
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5
            )
            
            for i in range(len(generated)):
                pred_text = tokenizer.decode(generated[i], skip_special_tokens=True)
                target_text = batch['target_text'][i]
                batch_metrics = calculate_metrics(pred_text, target_text)
                
                metrics['bracket_accuracy'] += batch_metrics['bracket_accuracy']
                metrics['text_preservation'] += batch_metrics['text_preservation']
                
                pred_text_no_brackets = re.sub(r'\[(.*?)\]', r'\1', pred_text)
                
                differences = []
                target_brackets = re.findall(r'\[(.*?)\]', target_text)
                pred_brackets = re.findall(r'\[(.*?)\]', pred_text)
                
                for idx, (t, p) in enumerate(zip(target_brackets, pred_brackets), 1):
                    if t != p:
                        differences.append(f"차이점 {idx}: {t} -> {p}")
                
                differences = ' | '.join(differences) if differences else "차이 없음"
                
                predictions.append({
                    'input_text': batch['input_text'][i],
                    'target_text': target_text,
                    'predicted_text': pred_text,
                    'predicted_text_no_brackets': pred_text_no_brackets,
                    'differences': differences
                })
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    num_samples = len(dataloader.dataset)
    metrics = {k: v/num_samples for k, v in metrics.items()}
    
    return total_loss / len(dataloader), metrics, predictions

# ============================
# Inference Function
# ============================
def predict_text(text, model, tokenizer, device):
    model.eval()
    text = "normalize: " + text
    inputs = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ============================
# Main 실행부
# ============================
def main():
    # 1. 이미 분할된 CSV 파일 로드 (train.csv, val.csv, test.csv)
    if not (os.path.exists('./data/train.csv') and os.path.exists('./data/val.csv') and os.path.exists('./data/test.csv')):
        print("train.csv, val.csv, 또는 test.csv 파일이 존재하지 않습니다. 올바른 경로의 파일들을 준비하세요.")
        return
    
    train_data = pd.read_csv('./data/train.csv')
    val_data = pd.read_csv('./data/val.csv')
    test_data = pd.read_csv('./data/test.csv')
    
    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")
    
    # 2. 모델과 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained("paust/pko-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("paust/pko-t5-large")
    model = model.to(device)
    
    # 3. 데이터셋과 DataLoader 생성
    train_dataset = NumberNormalizationDataset(train_data, tokenizer, is_training=True)
    val_dataset = NumberNormalizationDataset(val_data, tokenizer, is_training=True)
    test_dataset = NumberNormalizationDataset(test_data, tokenizer, is_training=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    
    criterion = CustomLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 5
    best_val_loss = float('inf')
    
    # 4. 이미 훈련된 모델이 있다면 로드하여 학습 스킵, 없으면 학습 진행
    if os.path.exists('best_model.pt'):
        print("기존에 훈련된 모델(best_model.pt)이 존재합니다. 모델을 로드합니다.")
        model.load_state_dict(torch.load('best_model.pt', map_location=device))
    else:
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 20)
            
            train_loss, train_metrics = train_epoch(model, train_dataloader, optimizer, device, criterion)
            val_loss, val_metrics, _ = evaluate(model, val_dataloader, device, criterion, tokenizer)
            
            print(f'Train Loss: {train_loss:.4f}')
            print('Train Metrics:', train_metrics)
            print(f'Val Loss: {val_loss:.4f}')
            print('Val Metrics:', val_metrics)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')
                print('New best model saved!')
    
    # 5. 추론: 학습된(또는 로드된) 모델을 이용해 예시 입력에 대해 예측 수행
    sample_text = "참석자는 10분입니다."
    predicted_text = predict_text(sample_text, model, tokenizer, device)
    print(f"\nInput: {sample_text}")
    print(f"Predicted: {predicted_text}")
    
    # 6. 테스트 데이터셋에 대해 평가 진행 및 결과 저장
    test_loss, test_metrics, test_predictions = evaluate(model, test_dataloader, device, criterion, tokenizer)
    print(f"\nTest Loss: {test_loss:.4f}")
    print("Test Metrics:", test_metrics)
    
    # 테스트 예측 결과를 CSV 파일로 저장
    df_results = pd.DataFrame(test_predictions)
    df_results.to_csv("./data/test_result.csv", index=False)
    print("Test predictions saved to test_result.csv")
    
if __name__ == '__main__':
    main()