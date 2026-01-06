import os 
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, AdamW
import math
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from utilities import load_data
from models.model import MLP, DeepDRBPMoE
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef  
from sklearn.metrics import f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np 
import optuna
import time
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# 定义类别映射
CLASS_MAPPING = {
    'DBP': 0,
    'DRBP': 1,
    'Negative': 2,
    'RBP': 3
}

REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

def get_model_init(model_type, input_dim, expert_num=4, top_k=2):
    if model_type == 'mlp':
        model = MLP(input_dim, num_class=4)  # 修改为4分类

    elif model_type == 'DeepDRBPMoE':
        numn_experts = expert_num
        top_k = top_k
        capacity_factor = 1.5
        model = DeepDRBPMoE(prot_input_dim=input_dim, num_class=4,  # 修改为4分类
                    num_experts=numn_experts, top_k=top_k, capacity_factor=capacity_factor)  

    return model


def get_args():
    # 添加参数解析
    parser = argparse.ArgumentParser(description='DRBP Prediction')
    parser.add_argument('--mode', type=str, required=False, default = 'train', choices=['train', 'eval', 'predict','cross_validate'], help='train-training, eval-evaluation, predict-prediction')
    parser.add_argument('--data_path', type=str, default='/mnt/sdb/ZYQ/workspace/ENPD/DATA/human_data/prot_seq_new.csv')
    parser.add_argument('--test_path', required=False, default='./data/example/test.txt', help="data for test")
    parser.add_argument('--pred_path', required=False, default='./output_pred/test_pred.txt', help="data for pred")
    parser.add_argument('--label_data', required=False, default='./data/example/test.txt', help="data for label")
    parser.add_argument('--llm', type=str, default='lucaone', choices=['evo2', 'esm2', 'lucaone','ProtT5','xTrimoPGLM','ProTrek','SaProt'])
    parser.add_argument('--llm_path', type=str, default='/mnt/sdb/ZYQ/workspace/DeepSeqPDBI/LLM/lucaone/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20240906230224/checkpoint-step36000000')
    parser.add_argument('--output_dir', type=str, default='output_all')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--embed_save', type=str, default='embed_save')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_pth', type=str, default=None, help="model for pred")
    parser.add_argument('--model', type=str, default='mlp', choices=['DeepDRBPMoE', 'deepseekmoe', 'smoe','moe','mlp','sidsmoe','mfsidsmoe','SIMLP','sicnn','deepseekmoep','trans','sitrans','drnamoe','mixmoe','mhmoe','cnn','minimoe','simplecnn','rcnn'])
    parser.add_argument('--k_folds', type=int, default=5, help="number of folds for cross validation")
    
    args = parser.parse_args()
    return args



def plot_cm(all_labels, all_preds, output_dir, file_name):
    # 修改混淆矩阵标签为4分类
    cm = confusion_matrix(all_labels, all_preds)
    fig = plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                    annot_kws={
                        'size': 15,
                        'weight': 'bold'
                    },
                    linewidths=0.5,
                    linecolor='grey',
                    cbar_kws={
                        'shrink': 0.8,
                        'label': 'Sample Count'
                    },
                    xticklabels=['DBP', 'RBP', 'DRBP', 'None'],
                    yticklabels=['DBP', 'RBP', 'DRBP', 'None']
    )

    ax.set_xlabel('Predicted Label', fontsize=15, weight='bold')
    ax.set_ylabel('True Label', fontsize=15, weight='bold')
    ax.set_title('Confusion Matrix', fontsize=16, pad=20, weight='bold')
    
    ax.xaxis.set_tick_params(labelsize=12, rotation=0)
    ax.yaxis.set_tick_params(labelsize=12, rotation=0)
    
    plt.tight_layout()
    with open(f'{output_dir}/{file_name}_confusion_matrix.png', 'wb') as f:
        fig.savefig(f, format='png')
    plt.close()


# def plot_roc_pr(all_labels, all_probs, output_dir, file_name):
#     auroc = roc_auc_score(all_labels, all_probs)
#     aupr = average_precision_score(all_labels, all_probs)
        
#     fpr, tpr, _ = roc_curve(all_labels, all_probs)
#     fig = plt.figure(figsize=(12, 9))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, 
#             label=f'ROC curve (AUC = {auroc:.5f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc="lower right")
#     with open(f'{output_dir}/{file_name}_roc_curve.png', 'wb') as f:
#         fig.savefig(f, format='png')
#     plt.close()

    
#     # 绘制PR曲线
#     precision, recall, _ = precision_recall_curve(all_labels, all_probs)
#     fig = plt.figure(figsize=(12, 9))
#     plt.plot(recall, precision, color='blue', label=f'AUPR = {aupr:.5f})')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend()
#     with open(f'{output_dir}/{file_name}_pr_curve.png', 'wb') as f:
#         fig.savefig(f, format='png')
#     plt.close()


def train(model, train_loader, val_loader, output_dir, epochs, lr, device, model_type):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    class_weights = torch.tensor([1.5, 6.0, 1.0, 1.5]).to(device)  # 修改为4分类的权重
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    train_losses = []  
    val_losses = []    
    train_acc = []
    val_acc = []
    patience_limit = 50
    patience_counter = 0 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        i = 0
        for batch in train_loader:
            # 修改：只处理单条蛋白序列
            prot_seq, labels = batch['prot'], batch['label']
            prot_seq, labels = prot_seq.to(device), labels.long().to(device)
            
            optimizer.zero_grad()
            
            # 修改：所有模型都使用单序列输入
            if model_type == 'DeepDRBPMoE':
                outputs, loss_aux = model(prot_seq)
            else:
                outputs = model(prot_seq)
                
            _, predicted = torch.max(outputs.squeeze(0), 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            loss_base = criterion(outputs.squeeze(0), labels)
            if model_type == 'DeepDRBPMoE':
                loss = loss_base + loss_aux
            else:
                loss = loss_base
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * prot_seq.size(0)

        scheduler.step()

        train_losses.append(train_loss / len(train_loader.dataset))
        t_acc = accuracy_score(train_labels, train_preds)
        train_acc.append(t_acc)

        # 验证阶段
        model.eval()
        val_loss = 0
        all_labels = []
        all_probs = []
        all_preds = []  
        with torch.no_grad():
            for batch_val in val_loader:
                prot_seq, labels = batch_val['prot'], batch_val['label']
                prot_seq, labels = prot_seq.to(device), labels.long().to(device)
                
                if model_type == 'DeepDRBPMoE':
                    outputs, loss_aux = model(prot_seq)
                else:
                    outputs = model(prot_seq)

                val_loss += criterion(outputs.squeeze(0), labels).item() * prot_seq.size(0)
                _, predicted = torch.max(outputs.squeeze(0), 1)
                
                # 修改：处理多分类的概率
                probs = torch.softmax(outputs.squeeze(0), dim=1)
                all_probs.extend(probs.cpu().numpy())  # 保存所有类别的概率
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_losses.append(val_loss / len(val_loader.dataset))
        v_acc = accuracy_score(all_labels, all_preds)
        val_acc.append(v_acc)

        # 修改：计算多分类的AUROC（需要one-vs-rest）
        try:
            auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
            aupr = average_precision_score(all_labels, all_probs, average='macro')
        except:
            auroc, aupr = 0, 0
            
        mcc = matthews_corrcoef(all_labels, all_preds)

        # 添加：计算Recall、Precision和F1-score指标
        try:
            f1 = f1_score(all_labels, all_preds, average='macro')
            precision = precision_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
        except:
            f1, precision, recall = 0, 0, 0       

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
            file_name = 'val'
            plot_cm(all_labels, all_preds, output_dir, file_name)
        else:
            patience_counter += 1
        vac = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss/len(train_loader.dataset):.5f} | "
              f"Train ACC: {t_acc:.5f} | "
              f"Val Loss: {val_loss/len(val_loader.dataset):.5f} | "
              f"Val ACC: {vac:.5f} | "
              f"Val AUROC: {auroc:.5f} | "
              f"Val AUPR: {aupr:.5f} | "
              f"Val MCC: {mcc:.5f} | "
              f"Val F1: {f1:.5f} | "
              f"Val Precision: {precision:.5f} | "
              f"Val Recall: {recall:.5f}") 
        
        if patience_counter >= patience_limit:
            print(f"\nEarly stopping triggered at epoch {epoch+1}!")
            break

    # 保存训练曲线
    fig = plt.figure(figsize=(12, 9))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    with open(f'{output_dir}/loss_curves.png', 'wb') as f:
        fig.savefig(f, format='png')
    plt.close()

    fig = plt.figure(figsize=(12, 9))
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    with open(f'{output_dir}/accuracy_curves.png', 'wb') as f:
        fig.savefig(f, format='png')
    plt.close()
            
    return all_labels, all_probs


def evaluate(pred_file, test_indices, data_path, output_dir, file_name):
    df_pred = pd.read_csv(pred_file, sep='\t', header=0)
    if test_indices is None:
        test_indices = list(range(0, len(df_pred)))
    df = pd.read_csv(data_path, sep=',',header=0)
    all_labels = df.loc[test_indices, 'Class']
    all_preds = df_pred['Class']  # 预测类别
    y_true, y_pred = all_labels.tolist(), all_preds.tolist()


    try:
        class_names = [REVERSE_CLASS_MAPPING[i] for i in range(len(REVERSE_CLASS_MAPPING))]
    except KeyError:
        print("错误: REVERSE_CLASS_MAPPING 缺少某些索引 (0-3)，请检查定义。")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 1 MCC (马修斯相关系数)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    #  2 Classification Report (详细的 Precision/Recall/F1)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    # 提取核心指标
    acc = report_dict['accuracy']
    macro_f1 = report_dict['macro avg']['f1-score']
    
    # ==========================================
    # [控制台输出]
    # ==========================================
    print("\n" + "="*40)
    print(f" [评估摘要]")
    print(f" Accuracy : {acc:.5f}")
    print(f" Macro F1 : {macro_f1:.5f}")
    print(f" MCC      : {mcc:.5f}")
    print("="*40 + "\n")

    # --- 3. 保存文本报告 (txt) ---
    txt_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=== Model Evaluation Summary ===\n")
        f.write(f"Accuracy: {acc:.5f}\n")
        f.write(f"Macro F1: {macro_f1:.5f}\n")
        f.write(f"MCC: {mcc:.5f}\n\n")
        f.write("=== Detailed Classification Report ===\n")
        f.write(report_text)
    
    # --- 4. 保存 CSV 表格 ---
    df_metrics = pd.DataFrame(report_dict).transpose()
    df_metrics.to_csv(os.path.join(output_dir, 'metrics.csv'), float_format='%.5f')

    # --- 5. 绘制混淆矩阵 (PNG) ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    # fmt='d' 表示整数计数，不用小数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    print(f"评估完成！结果已保存至: {output_dir}")





def get_prot_embed(df, llm, llm_path, save_path, device):
    """获取蛋白序列的嵌入表示"""
    prot_id = df['id']
    prot_seqs = df['seq']
    labels = df['Class']
    
    # 将文本标签转换为数字标签
    numeric_labels = [CLASS_MAPPING[label] for label in labels]
    
    prot_embed = load_data.get_embed(prot_id, prot_seqs, llm, llm_path, save_path, device)
    if isinstance(prot_embed, torch.Tensor):
        prot_embed = prot_embed.float()
    else:
        prot_embed = torch.tensor(prot_embed, dtype=torch.float32)
            
    return prot_embed, numeric_labels

class ProteinDataset(torch.utils.data.Dataset):
    """单条蛋白序列数据集"""
    def __init__(self, prot_encoded, labels):
        self.prot = prot_encoded
        self.labels = torch.tensor(labels, dtype=torch.long)  # 修改为long类型
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'prot': self.prot[idx],
            'label': self.labels[idx]
        }
    
def predict(model_pred, model_pth, test_loader, output_dir, prot_names, pred_file_name, device, model_type):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_pred.load_state_dict(torch.load(model_pth))
    model_pred = model_pred.to(device)
    model_pred.eval()
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for test_batch in test_loader:
            prot_seq = test_batch['prot']
            prot_seq = prot_seq.to(device)
            
            if model_type == 'DeepDRBPMoE':
                outputs, loss_aux = model_pred(prot_seq)
            else:
                outputs = model_pred(prot_seq)
                
            _, predicted = torch.max(outputs.squeeze(0), 1)
            probs = torch.softmax(outputs.squeeze(0), dim=1)  # 所有类别的概率
            
            all_probs.append(probs.cpu())
            all_preds.extend(predicted.cpu().numpy())

    # 合并结果
    all_probs = torch.cat(all_probs).numpy()
    
    # 保存预测结果（包含所有类别的概率）
    result_path = f'{output_dir}/{pred_file_name}.txt'
    with open(result_path, 'w') as f:
        # 写入表头
        f.write("Protein_ID\tDBP_Prob\tRBP_Prob\tDRBP_Prob\tNone_Prob\tClass\n")
        
        for name, prob, pred in zip(prot_names, all_probs, all_preds):
            pred_class_name = REVERSE_CLASS_MAPPING[pred]
            f.write(f"{name}\t{prob[0]:.5f}\t{prob[1]:.5f}\t{prob[2]:.5f}\t{prob[3]:.5f}\t{pred_class_name}\n")


# 添加一个新的数据划分函数
def split_data_by_protein_uniqueness(df, split_ratios=(0.7, 0.2, 0.1), seed=42):
    """根据蛋白序列的唯一性划分数据集"""
    train_ratio, val_ratio, test_ratio = split_ratios
    
    # 获取所有唯一的蛋白序列
    unique_prot_seqs = df['seq'].unique()
    
    np.random.seed(seed)
    np.random.shuffle(unique_prot_seqs)
    
    train_size = int(len(unique_prot_seqs) * train_ratio)
    train_prot_seqs = unique_prot_seqs[:train_size]
    remaining_prot_seqs = unique_prot_seqs[train_size:]
    
    val_size = int(len(remaining_prot_seqs) * (val_ratio / (val_ratio + test_ratio)))
    val_prot_seqs = remaining_prot_seqs[:val_size]
    test_prot_seqs = remaining_prot_seqs[val_size:]
    
    train_df = df[df['seq'].isin(train_prot_seqs)]
    val_df = df[df['seq'].isin(val_prot_seqs)]
    test_df = df[df['seq'].isin(test_prot_seqs)]
    
    # 检查比例
    total_len = len(df)
    print(f"原始数据大小: {total_len}")
    print(f"训练集大小: {len(train_df)} ({len(train_df)/total_len:.2%}), 唯一蛋白序列数: {len(train_prot_seqs)}")
    print(f"验证集大小: {len(val_df)} ({len(val_df)/total_len:.2%}), 唯一蛋白序列数: {len(val_prot_seqs)}")
    print(f"测试集大小: {len(test_df)} ({len(test_df)/total_len:.2%}), 唯一蛋白序列数: {len(test_prot_seqs)}")
    
    return train_df, val_df, test_df

if  __name__ == "__main__":
    
    args = get_args()
    gpu_id = str(args.cuda)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path = args.data_path
    llm_path = args.llm_path
    llm = args.llm
    lr = args.lr
    epochs = args.epochs
    output_dir = args.output_dir
    model_type = args.model
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.mode == 'train':
        # 加载数据（假设数据格式为：id,seq,type）
        df = pd.read_csv(data_path, sep=',')
        save_path = args.embed_save
        batch_size = args.batch_size
        
        # 获取蛋白嵌入
        prot_embed, numeric_labels = get_prot_embed(df, llm, llm_path, save_path, device)
        input_dim = prot_embed.shape[1]
        
        # 创建数据集
        full_dataset = ProteinDataset(prot_embed, numeric_labels)
        
        # 划分训练/验证/测试集
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.2 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        seed = 42
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        print(f"训练集大小: {len(train_dataset)}") 
        print(f"验证集大小: {len(val_dataset)}")   
        print(f"测试集大小: {len(test_dataset)}")   

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        model = get_model_init(model_type, input_dim)
        model_pred = get_model_init(model_type, input_dim)
        
        # 训练模型
        start_time = time.time()
        all_labels, all_probs = train(model, train_loader, val_loader, output_dir, epochs, lr, device, model_type)
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"训练耗时: {training_duration:.2f} 秒")
        
        # 预测测试集
        model_pth = f'{output_dir}/best_model.pth'
        prot_names = [df.iloc[i]['id'] for i in test_dataset.indices]
        predict(model_pred, model_pth, test_loader, output_dir, prot_names, 'test_best_pred', device, model_type)
        
        # 评估测试集
        pred_file = f'{output_dir}/test_best_pred.txt'
        file_name = 'test_best_pred'
        test_indices = test_loader.dataset.indices  
        evaluate(pred_file, test_indices, data_path, output_dir, file_name)

    elif args.mode == 'predict':
        # 预测模式
        test_path = args.test_path
        df = pd.read_csv(test_path)
        prot_id = df['id']
        prot_seqs = df['seq']
        
        # 获取嵌入
        prot_embed = load_data.get_embed(prot_id, prot_seqs, llm, llm_path, args.embed_save, device)
        input_dim = prot_embed.shape[1]
        
        # 创建测试数据集
        test_dataset = ProteinDataset(prot_embed, [0] * len(prot_embed))  # 标签设为0（不重要）
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 加载模型并进行预测
        model_pth = args.model_pth
        model_pred = get_model_init(model_type, input_dim)
        predict(model_pred, model_pth, test_loader, output_dir, prot_id.tolist(), 'prediction', device, model_type)

    elif args.mode == 'eval':
        # 评估模式
        pred_file = args.pred_path
        file_name = 'evaluation'
        data_path = args.data_path
        auroc, aupr, acc, mcc, f1, precision, recall = evaluate(pred_file, None, data_path, output_dir, file_name)
        print(f"Evaluation Results - AUROC: {auroc:.5f}, AUPR: {aupr:.5F}, ACC: {acc:.5F}, MCC: {mcc:.5F}, F1: {f1:.5F}, Precision: {precision:.5F}, Recall: {recall:.5F}")