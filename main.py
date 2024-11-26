import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from Models.PredictModel import MF
from Utils.dataloader import load_data, RatingDataset, RatingTestDataset


def evaluate(model, dataset, data_loader, device, is_valid=True):
    """评估函数：计算RMSE和MAE"""
    model.eval()
    total_mse = 0
    total_mae = 0
    total_samples = 0
    
    with torch.no_grad():
        while True:
            users, end_epoch = data_loader.get_next_batch_users()
            users = users.to(device)
            
            for user in users:
                # 获取用户的实际评分
                user_ratings = data_loader.get_user_ratings(user.item(), is_valid=is_valid)
                if not user_ratings:
                    continue
                    
                items, true_ratings = zip(*user_ratings)
                items = torch.LongTensor(items).to(device)
                true_ratings = torch.FloatTensor(true_ratings).to(device)
                
                # 预测评分
                pred_ratings = model(user.expand(len(items)), items)
                
                # 还原归一化的评分
                pred_ratings = dataset.denormalize_rating(pred_ratings)
                
                total_mse += ((pred_ratings - true_ratings) ** 2).sum().item()
                total_mae += torch.abs(pred_ratings - true_ratings).sum().item()
                total_samples += len(items)
            
            if end_epoch:
                break
    
    rmse = np.sqrt(total_mse / total_samples)
    mae = total_mae / total_samples
    
    return rmse, mae

def train_and_evaluate(model, train_loader, valid_test_dataset, num_epochs, learning_rate, device):
    """训练和评估函数"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    best_valid_rmse = float('inf')
    best_epoch = 0
    best_test_metrics = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            users = batch['user'].to(device)
            items = batch['item'].to(device)
            ratings = batch['rating'].to(device).float()
            
            # 前向传播
            pred_ratings = model(users, items)
            loss = criterion(pred_ratings, ratings)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # 在验证集上评估
        valid_rmse, valid_mae = evaluate(
            model, 
            train_loader.dataset,
            valid_test_dataset, 
            device, 
            is_valid=True
        )
        
        # 在测试集上评估
        test_rmse, test_mae = evaluate(
            model,
            train_loader.dataset,
            valid_test_dataset,
            device,
            is_valid=False
        )
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}')
        print(f'Validation: RMSE = {valid_rmse:.4f}, MAE = {valid_mae:.4f}')
        print(f'Test: RMSE = {test_rmse:.4f}, MAE = {test_mae:.4f}\n')
        
        # 基于验证集指标保存最佳模型
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_epoch = epoch
            best_test_metrics = (test_rmse, test_mae)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_rmse': valid_rmse,
                'valid_mae': valid_mae,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
            }, 'checkpoint/best_model.pth')
        
        # 基于验证集性能进行early stop
        if epoch - best_epoch > 4:
            print(f'Early stopping at epoch {epoch+1}')
            print(f'Best validation RMSE: {best_valid_rmse:.4f} at epoch {best_epoch+1}')
            print(f'Corresponding test metrics: RMSE = {best_test_metrics[0]:.4f}, MAE = {best_test_metrics[1]:.4f}')
            break
    
    return best_valid_rmse, best_test_metrics



def main():
    # 参数设置
    EMBEDDING_DIM = 64
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    u_count, i_count, train_mat, train_ints, valid_mat, test_mat = load_data(
        file_path='reviews_Digital_Music_5.csv',
        test_ratio=0.2,
        random_seed=42
    )
    
    # 创建数据加载器
    train_dataset = RatingDataset(u_count, i_count, train_mat, train_ints)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
        
    print("一个Train batch的数据:")
    for i, batch in enumerate(train_loader):
        if i == 0:  # 只打印第一个batch
            print(f"Batch size: {len(batch['user'])}")
            print("\n前5个样本:")
            for j in range(min(5, len(batch['user']))):
                print(f"用户: {batch['user'][j].item()}, "
                      f"物品: {batch['item'][j].item()}, "
                      f"归一化评分: {batch['rating'][j].item():.4f}, "
                      f"原始评分: {train_dataset.denormalize_rating(batch['rating'][j].item()):.4f}")
            break
    print("\n")
    
    valid_test_dataset = RatingTestDataset(
        u_count,
        i_count,
        valid_mat,
        test_mat,
        batch_size=BATCH_SIZE
    )
    
    # 创建模型
    model = MF(
        user_count=u_count,
        item_count=i_count,
        dim=EMBEDDING_DIM,
        gpu=DEVICE
    )
    
    # 训练和评估
    best_valid_rmse, best_test_metrics = train_and_evaluate(
        model,
        train_loader,
        valid_test_dataset,
        NUM_EPOCHS,
        LEARNING_RATE,
        DEVICE
    )
    
    print("\nTraining completed!")
    print(f'Best Validation RMSE: {best_valid_rmse:.4f}')
    print(f'Corresponding Test RMSE: {best_test_metrics[0]:.4f}')
    print(f'Corresponding Test MAE: {best_test_metrics[1]:.4f}')

if __name__ == '__main__':
    main()