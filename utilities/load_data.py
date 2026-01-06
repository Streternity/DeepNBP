import os
import sys
import re 
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

sys.path.append(".")
sys.path.append("..")
sys.path.append('utilities')

def save_embedding(embedding, save_path, id_name, method):
    save_path = os.path.join(save_path, method)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(embedding.cpu(), f'{save_path}/{method}_{id_name}_emb.pt')

def get_molecule_embedding(molecules, smiles, method, llm_path = None, save_path = 'save_embed', device = 'cuda'):
    """
    输入 SMILES 字符串列表，返回嵌入向量, method表示使用的大模型, llm_path表示大模型的路径
    :param smiles: 分子结构的 SMILES 表示法
    :return: 模型输出的嵌入向量
    """

    save_list = []
    if method=="molgpt":
        if llm_path == None:
            tokenizer = AutoTokenizer.from_pretrained("Q-bert/Mol-GPT")
            model = AutoModelForCausalLM.from_pretrained("Q-bert/Mol-GPT")
        else:
            model_path = os.path.join(llm_path, "MolGPT")
            tokenizer = AutoTokenizer.from_pretrained("Q-bert/Mol-GPT", cache_dir = model_path)
            model = AutoModelForCausalLM.from_pretrained("Q-bert/Mol-GPT", cache_dir = model_path)

        for smile, molecule in zip(smiles, molecules):
            # 如果出现重复则跳过
            if molecule in save_list:
                continue
            save_list.append(molecule)
            inputs = tokenizer(smile, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                logits = outputs.logits
            mol_embedding = torch.mean(logits, dim=1)
            mol_embedding = mol_embedding.squeeze(0) 
            save_embedding(mol_embedding, save_path, molecule, method)

    if method=="molformer":
        if llm_path == None:
            tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
            model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
        else:
            model_path = os.path.join(llm_path, "MolFormer")
            tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", cache_dir = model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", cache_dir = model_path, deterministic_eval=True, trust_remote_code=True)
        for smile, molecule in zip(smiles, molecules):
            if molecule in save_list:
                continue
            save_list.append(molecule)
            inputs = tokenizer(smile, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits=outputs.last_hidden_state   
            mol_embedding = torch.mean(logits, dim=1)
            mol_embedding = mol_embedding.squeeze(0) 
            save_embedding(mol_embedding, save_path, molecule, method)

    if method == 'GP-MoLFormer':
        if llm_path == None:
            tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("ibm-research/GP-MoLFormer-Uniq", deterministic_eval=True, trust_remote_code=True)
        else:
            model_path = os.path.join(llm_path, "GP-MoLFormer-Uniq")
            token_path = os.path.join(llm_path, "MolFormer")
            tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", cache_dir = token_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("ibm-research/GP-MoLFormer-Uniq" ,cache_dir = model_path, deterministic_eval=True, trust_remote_code=True)
        for smile, molecule in zip(smiles, molecules):
            if molecule in save_list:
                continue
            save_list.append(molecule)
            inputs = tokenizer(smile, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits 
                logits = logits[:, 1:-1, :]                                 # remove CLS and EOS token
            mol_embedding = torch.mean(logits, dim=1)
            mol_embedding = mol_embedding.squeeze(0)                        # shape [2362]
            save_embedding(mol_embedding, save_path, molecule, method)

    if method == 'MolT5':
        # 加载模型和分词器
    
        tokenizer = T5Tokenizer.from_pretrained(llm_path, model_max_length=512)
        model = T5ForConditionalGeneration.from_pretrained(llm_path)

        for smile, molecule in zip(smiles, molecules):
            if molecule in save_list:
                continue
            save_list.append(molecule)
            inputs = tokenizer(smile, padding=True, return_tensors="pt")
            input_ids = inputs.input_ids
            with torch.no_grad():
                outputs = model.encoder(input_ids=input_ids, attention_mask=inputs.attention_mask)
                hidden_states = outputs.last_hidden_state
            mol_embedding = hidden_states.mean(dim=1).squeeze()     # shape [1024]
            save_embedding(mol_embedding, save_path, molecule, method)

    if method == 'ChemBERTa':
        llm_path = '/mnt/sdb/ZYQ/workspace/LLM/ChemBERTa-100M-MLM'
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
        model = AutoModel.from_pretrained(llm_path)
        for smile, molecule in zip(smiles, molecules):
            if molecule in save_list:
                continue
            save_list.append(molecule)
            inputs = tokenizer(smile, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits                            
            mol_embedding = torch.mean(logits, dim=1)
            mol_embedding = mol_embedding.squeeze(0)                        # shape [7924]
            save_embedding(mol_embedding, save_path, molecule, method)



def get_dna_embedding(seq_name, seqs, method, llm1_path, save_path, device):

    if method == 'dnabert2':
        from transformers.models.bert.configuration_bert import BertConfig
        from transformers import AutoTokenizer, AutoModel      
        tokenizer = AutoTokenizer.from_pretrained(llm1_path, trust_remote_code=True)
        config = BertConfig.from_pretrained(llm1_path)
        model = AutoModel.from_pretrained(llm1_path, trust_remote_code=True, config=config)
        model = model.to(device)
        for seq_id, seq in list(dict.fromkeys(zip(seq_name, seqs))):
            inputs = tokenizer(seq, return_tensors = 'pt')["input_ids"]
            inputs = inputs.to(device)
            hidden_states = model(inputs)[0] # [1, sequence_length, 768]
            embedding_mean = torch.mean(hidden_states[0], dim=0)   # [768]
            save_embedding(embedding_mean, save_path, seq_id, method)

    elif method == 'evo2':
        from evo2 import Evo2
        # llm1_path = '/mnt/sdb/ZYQ/workspace/LLM/evo2_7b'
        # evo2_model = Evo2('evo2_7b',local_path = llm1_path)
        evo2_model = Evo2('evo2_7b')
        layer_name = 'blocks.28.mlp.l3'
        for seq_id, seq in list(dict.fromkeys(zip(seq_name, seqs))):
            input_ids = torch.tensor(evo2_model.tokenizer.tokenize(seq), dtype=torch.int,).unsqueeze(0).to(device)
            outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
            embeddings = embeddings[layer_name]
            embedding_mean = torch.mean(embeddings[0], dim=0)       #  [4096]
            embedding_mean = embedding_mean.float()
            save_embedding(embedding_mean, save_path, seq_id, method)

    elif method == 'lucaone':
        import get_lucaone_embed
        seq_type = 'gene'
        trunc_type="right"
        max_seq_len=4094
        for seq_id, seq in list(dict.fromkeys(zip(seq_name, seqs))):
            emb = get_lucaone_embed.get_embedding_vector(seq_id, seq, seq_type, llm1_path, trunc_type, max_seq_len)
            emb = torch.tensor(emb)
            emb = torch.mean(emb, dim = 0)
            # emb = emb.unsqueeze(0)             # 将维度转为[1, 2560]
            save_embedding(emb, save_path, seq_id, method)

    elif method == 'GENERator':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Load the tokenizer and model.
        tokenizer = AutoTokenizer.from_pretrained(llm1_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(llm1_path)
        config = model.config
        model = model.to(device)
        max_length = config.max_position_embeddings

        for seq_id, seq in list(dict.fromkeys(zip(seq_name, seqs))):
            tokenizer.padding_side = "right"
            inputs = tokenizer(
                seq,
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = inputs.to(device)
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  
                attention_mask = inputs["attention_mask"]
                last_token_indices = attention_mask.sum(dim=1) - 1 
                seq_embedding = hidden_states[0, last_token_indices[0], :]              # [2048]
                save_embedding(seq_embedding, save_path, seq_id, method)

    elif method == 'dnagpt':
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(llm1_path)
        model = AutoModel.from_pretrained(llm1_path)
        model = model.to(device)
        for seq_id, seq in list(dict.fromkeys(zip(seq_name, seqs))):
            inputs = tokenizer(seq, return_tensors = 'pt')["input_ids"]
            inputs = inputs.to(device)
            hidden_states = model(inputs)[0]                             # [1, sequence_length, 768]
            embedding_mean = torch.mean(hidden_states[0], dim=0)         # [768]
            save_embedding(embedding_mean, save_path, seq_id, method)



def get_rna_embedding(seq_name, seqs, method, llm1_path, save_path, device):

    if method == 'dnabert2':
        from transformers.models.bert.configuration_bert import BertConfig
        from transformers import AutoTokenizer, AutoModel      
        tokenizer = AutoTokenizer.from_pretrained(llm1_path, trust_remote_code=True)
        config = BertConfig.from_pretrained(llm1_path)
        model = AutoModel.from_pretrained(llm1_path, trust_remote_code=True, config=config)
        model = model.to(device)
        for seq_id, seq in list(dict.fromkeys(zip(seq_name, seqs))):
            inputs = tokenizer(seq, return_tensors = 'pt')["input_ids"]
            inputs = inputs.to(device)
            hidden_states = model(inputs)[0] # [1, sequence_length, 768]
            embedding_mean = torch.mean(hidden_states[0], dim=0)   # [768]
            save_embedding(embedding_mean, save_path, seq_id, method)

    elif method == 'evo2':
        from evo2 import Evo2
        llm1_path = '/mnt/sdb/ZYQ/workspace/LLM/evo2_7b'
        evo2_model = Evo2('evo2_7b',local_path = llm1_path)
        evo2_model = Evo2('evo2_7b')
        layer_name = 'blocks.28.mlp.l3'
        for seq_id, seq in list(dict.fromkeys(zip(seq_name, seqs))):
            input_ids = torch.tensor(evo2_model.tokenizer.tokenize(seq), dtype=torch.int,).unsqueeze(0).to(device)
            outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
            embeddings = embeddings[layer_name]
            embedding_mean = torch.mean(embeddings[0], dim=0)       #  [4096]
            embedding_mean = embedding_mean.float()
            save_embedding(embedding_mean, save_path, seq_id, method)


    elif method == 'lucaone':
        import get_lucaone_embed
        os.makedirs(save_path, exist_ok=True)
        seq_type = 'gene'
        trunc_type="right"
        max_seq_len=4094
        for seq_id, seq in list(dict.fromkeys(zip(seq_name, seqs))):
            emb = get_lucaone_embed.get_embedding_vector(seq_id, seq, seq_type, llm1_path, trunc_type, max_seq_len)
            emb = torch.tensor(emb)
            emb = torch.mean(emb, dim = 0)
            save_embedding(emb, save_path, seq_id, method)            #  [2560]

    elif method == 'GENERator':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Load the tokenizer and model.
        tokenizer = AutoTokenizer.from_pretrained(llm1_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(llm1_path)
        config = model.config
        model = model.to(device)
        max_length = config.max_position_embeddings

        for seq_id, seq in list(dict.fromkeys(zip(seq_name, seqs))):
            tokenizer.padding_side = "right"
            inputs = tokenizer(
                seq,
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = inputs.to(device)
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  
                attention_mask = inputs["attention_mask"]
                last_token_indices = attention_mask.sum(dim=1) - 1 
                seq_embedding = hidden_states[0, last_token_indices[0], :]              # [2048]
                save_embedding(seq_embedding, save_path, seq_id, method)

    elif method == 'dnagpt':
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(llm1_path)
        model = AutoModel.from_pretrained(llm1_path)
        model = model.to(device)
        for seq_id, seq in list(dict.fromkeys(zip(seq_name, seqs))):
            inputs = tokenizer(seq, return_tensors = 'pt')["input_ids"]
            inputs = inputs.to(device)
            hidden_states = model(inputs)[0]                             # [1, sequence_length, 768]
            embedding_mean = torch.mean(hidden_states[0], dim=0)         # [768]
            save_embedding(embedding_mean, save_path, seq_id, method)



def get_prot_embedding(seq_name, seq_seq, method, llm1_path, save_path, device):
    os.makedirs(save_path, exist_ok=True)
    if method == 'lucaone':
        import get_lucaone_embed
        import torch
        os.makedirs(save_path, exist_ok=True)
        seq_type = 'prot' 
        trunc_type="right"
        max_seq_len=4094
        for seq_id, seq in list(dict.fromkeys(zip(seq_name, seq_seq))):   
            emb = get_lucaone_embed.get_embedding_vector(seq_id, seq, seq_type, llm1_path, trunc_type, max_seq_len)
            emb = torch.tensor(emb)
            emb = torch.mean(emb, dim = 0)                          #  [2560]
            save_embedding(emb, save_path, seq_id, method)


    elif method == 'esm2':
        import torch
        import pandas as pd
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        # 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(llm1_path)
        model = AutoModelForMaskedLM.from_pretrained(llm1_path)
        model = model.to(device) 
        
        model.eval()
        with torch.no_grad():
            for seq_id, seq in list(dict.fromkeys(zip(seq_name, seq_seq))):
                inputs = tokenizer(seq, return_tensors = 'pt')["input_ids"]
                inputs = tokenizer(
                    seq, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    add_special_tokens=False
                ).to(device)  # 将输入移动到设备
                outputs = model(**inputs, output_hidden_states=True)
                last_layer = outputs.hidden_states[-1]                   # [1, seq_len, 2560]
                embedding_mean = last_layer.mean(dim=1).squeeze()        # [2560]
                save_embedding(embedding_mean, save_path, seq_id, method)

    elif method == 'ProtT5':
        from transformers import T5Tokenizer, T5EncoderModel
        import torch
        import re
        llm1_path = '/mnt/sdb/ZYQ/workspace/LLM/prot_t5_xl_uniref50'
        tokenizer = T5Tokenizer.from_pretrained(llm1_path, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(llm1_path)
        model = model.to(device)

        model.eval()
        with torch.no_grad():
            for seq_id, seq in list(dict.fromkeys(zip(seq_name, seq_seq))):
                seq_len = len(seq)
                seq = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in [seq]]
                ids = tokenizer(seq, add_special_tokens=True, padding="longest")
                input_ids = torch.tensor(ids['input_ids']).to(device)
                attention_mask = torch.tensor(ids['attention_mask']).to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                last_layer = outputs.last_hidden_state[0,:seq_len]               # shape [seq_len x 1024]
                embedding_mean = last_layer.mean(dim=0)                          # shape [1024]
                save_embedding(embedding_mean, save_path, seq_id, method)


    elif method == 'ProTrek':
        import torch
        sys.path.append("./source/ProTrek")
        from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
        from utils.foldseek_util import get_struc_seq
        # Load model
        config = {
            "protein_config": f"{llm1_path}/esm2_t33_650M_UR50D",
            "text_config": f"{llm1_path}/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "structure_config": f"{llm1_path}/foldseek_t30_150M",
            "load_protein_pretrained": False,
            "load_text_pretrained": False,
            "from_checkpoint": f"{llm1_path}/ProTrek_650M.pt"
        }
        model = ProTrekTrimodalModel(**config).eval().to(device)

        with torch.no_grad():
            for seq_id, seq in list(dict.fromkeys(zip(seq_name, seq_seq))):
                embedding_mean = model.get_protein_repr([seq]).squeeze()           #[1024]                 
                save_embedding(embedding_mean, save_path, seq_id, method)
            
    elif method == 'SaProt':
        import torch
        from transformers import EsmTokenizer, EsmForMaskedLM
        model_path = llm1_path
        tokenizer = EsmTokenizer.from_pretrained(model_path)
        model = EsmForMaskedLM.from_pretrained(model_path)        
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for seq_id, seq in list(dict.fromkeys(zip(seq_name, seq_seq))):
                seq_len = len(seq)
                seq = [sequence for sequence in seq]
                seq = '#'.join(seq)
                seq += '#'
                inputs = tokenizer(seq, return_tensors="pt")
                inputs = inputs.to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                last_layer = logits[0,1:seq_len+1]               # shape [seq_len x 446]
                embedding_mean = last_layer.mean(dim=0)          # shape [446]            
                save_embedding(embedding_mean, save_path, seq_id, method)

    elif method == 'xTrimoPGLM':
        from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, AutoConfig
        import torch

        tokenizer  = AutoTokenizer.from_pretrained(llm1_path, trust_remote_code=True, use_fast=True)
        config = AutoConfig.from_pretrained(llm1_path,  trust_remote_code=True, torch_dtype=torch.half)
        config.is_causal=False
        config.post_layer_norm=True # use the final layernorm or not, some tasks set to false would be better.
        model = AutoModelForMaskedLM.from_pretrained(llm1_path, config = config, torch_dtype=torch.half,trust_remote_code=True)
        model = model.to(device)
        model.eval()
        # output = tokenizer(seq, add_special_tokens=True, return_tensors='pt')
        with torch.inference_mode():
            for seq_id, seq in list(dict.fromkeys(zip(seq_name, seq_seq))):
                output = tokenizer(seq, add_special_tokens=True, return_tensors='pt')
                inputs = {"input_ids": output["input_ids"].to(device), "attention_mask": output["attention_mask"].to(device)}
                output_embeddings = model(**inputs, output_hidden_states=True, return_last_hidden_state=True)
                output_embeddings = output_embeddings.hidden_states[:-1, 0]             # shape [seq_len x 10240]
                embedding_mean = output_embeddings.mean(dim=0)                          # shape [10240]   
                save_embedding(embedding_mean, save_path, seq_id, method)



def get_embed(seq_name, seq_seq, method, llm1_path, save_path, device):
    llm_embedding = []
    save_list = []
    for idx, seq in zip(seq_name, seq_seq):
        emb_path = f'{save_path}/{method}/{method}_{idx}_emb.pt'
        if os.path.exists(emb_path):
            emb = torch.load(emb_path)
            llm_embedding.append(emb.to('cpu'))
        else:
            get_prot_embedding(seq_name, seq_seq, method, llm1_path, save_path, device)
            emb = torch.load(emb_path)
            llm_embedding.append(emb.to('cpu'))
    embed_llm = torch.stack(llm_embedding) 
    return embed_llm

def get_cell_embedding(cell, method, llm_path = None):
    """留个接口，后续用到再加"""
    pass


def get_gene_embedding(gene, method, llm_path = None):
    """留个接口，后续用到再加"""
    pass


def get_data(data_path, seq1_name_col, seq2_name_col, seq1_seq_col, seq2_seq_col):
    dat = pd.read_csv(data_path, sep='\t')
    seq1_name = dat[seq1_name_col].tolist()
    seq2_name = dat[seq2_name_col].tolist()
    seq1_seq = dat[seq1_seq_col].tolist()
    seq2_seq = dat[seq2_seq_col].tolist()
    try:
        label = dat['Class'].tolist()
    except:
        label = ['unknow'] * len(seq1_seq)
    return seq1_name, seq2_name, seq1_seq, seq2_seq, label

class IteractionDataset(Dataset):
    def __init__(self, seq1_embeddings, seq2_embeddings, labels, batch_size=32, seed=42, split_ratios=(0.8, 0.1, 0.1)):
        """
        参数：
            split_ratios: (训练集比例, 验证集比例, 测试集比例)
        """
        assert len(seq1_embeddings) == len(seq2_embeddings) == len(labels)

        # 初始化数据集
        self.seq1 = seq1_embeddings
        self.seq2 = seq2_embeddings
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.batch_size = batch_size
        self.seed = seed
        
        # 划分数据集
        self._split_dataset(split_ratios)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'seq1': self.seq1[idx].squeeze(),
            'seq2': self.seq2[idx].squeeze(),
            'label': self.labels[idx]
        }
    
    def _split_dataset(self, ratios):
        """实现分层数据集划分"""
        total = sum(ratios)
        train_ratio, val_ratio, test_ratio = [r/total for r in ratios]
        
        # 第一次划分：训练集 vs 临时集
        first_split_size = 1 - (val_ratio + test_ratio)
        indices = list(range(len(self)))
        labels_np = self.labels.numpy()
        
        # 分层划分
        self.train_indices, temp_indices = train_test_split(
            indices,
            test_size=1-first_split_size,
            stratify=labels_np,
            random_state=self.seed
        )
        
        # 第二次划分：验证集 vs 测试集
        self.val_indices, self.test_indices = train_test_split(
            temp_indices,
            test_size=test_ratio/(test_ratio + val_ratio),
            stratify=labels_np[temp_indices],
            random_state=self.seed
        )
        
        # 创建子数据集
        self.train_set = torch.utils.data.Subset(self, self.train_indices)
        self.val_set = torch.utils.data.Subset(self, self.val_indices)
        self.test_set = torch.utils.data.Subset(self, self.test_indices)
    
    def get_dataloaders(self):
        """返回三个数据加载器"""
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader


class TestDataset(Dataset):
    def __init__(self, seq1_embeddings, seq2_embeddings, labels=None):
        self.seq1 = seq1_embeddings
        self.seq2 = seq2_embeddings
        self.labels = labels if labels is not None else None
        # 添加indices属性
        self.indices = list(range(len(seq1_embeddings)))
        
    def __len__(self):
        return len(self.seq1)
    
    def __getitem__(self, idx):
        item = {
            'seq1': self.seq1[idx],
            'seq2': self.seq2[idx]
        }
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item


if __name__ == "__main__":

    # GP-MoLFormer :     llm1_path = '/mnt/sdb/ZYQ/workspace/LLM'
    # MolT5:             llm1_path = '/mnt/sdb/ZYQ/workspace/LLM/molt5-large-smiles2caption'

    # dnabert :         llm1_path = './LLM/DNABERT-2-117M'  
    # esm2:             llm1_path = '/mnt/sdb/ZYQ/workspace/LLM/esm2_t36_3B_UR50D'
    # GeneRator:        llm1_path = '/mnt/sdb/ZYQ/workspace/LLM/GENERator-eukaryote-1.2b-base'
    # dnagpt:           llm1_path = '/mnt/sdb/ZYQ/workspace/LLM/human_gpt2-v1'


    # ProtT5:           llm2_path = '/mnt/sdb/ZYQ/workspace/LLM/prot_t5_xl_uniref50'
    # ProTrek:          llm2_path = '/mnt/sdb/ZYQ/workspace/LLM/ProTrek_650M'
    # SaProt:           llm2_path = '/mnt/sdb/ZYQ/workspace/LLM/SaProt_1.3B_AF2'
    # xTrimoPGLM:       llm2_path = '/mnt/sdb/ZYQ/workspace/LLM/proteinglm-100b-int4'


    # gpu_id = 2
    # gpu_id = str(gpu_id)
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # save_path = "./emb_save_new"

    # ##测试读取文件
    # data_path = '/mnt/sdb/ZYQ/workspace/DeepSeqPDBI/data/example/data.txt'
    # data_path = '/mnt/sdb/ZYQ/workspace/DeepSeqPDBI/data/example/test.txt'
    # seq1_name_col = 'Molecule_ID'
    # seq2_name_col = 'RNA_target'
    # seq1_seq_col = 'SMILES'
    # seq2_seq_col = 'RNA_sequence'
    # seq1_name, seq2_name, seq1_seq, seq2_seq, label = get_data(data_path, seq1_name_col, seq2_name_col, seq1_seq_col, seq2_seq_col)

    # seq1_method = 'molformer'
    # # seq1_method = 'molgpt'
    # seq2_method = 'lucaone' 
    # seq1_type = 'drug'
    # seq2_type = 'rna'
    # llm1_path = '/mnt/sdb/ZYQ/workspace/DeepSeqPDBI/LLM'
    # # llm2_path = '/mnt/sdb/chenzhb/LucaOne/LucaOneTasks/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step5600000'
    # # llm2_path = '/mnt/sdb/chenzhb/LucaOne/LucaOneTasks/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000'
    # # llm2_path = '/mnt/sdb/chenzhb/LucaOne/LucaOneTasks/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000'
    # # llm2_path = '/mnt/sdb/ZYQ/workspace/DeepSeqPDBI/LLM/lucaone/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000'
    # llm2_path = '/mnt/sdb/ZYQ/workspace/DeepSeqPDBI/LLM/lucaone/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20240906230224/checkpoint-step36000000'
    # seq1_embedding = get_embed(seq1_name, seq1_seq, seq1_type, seq1_method, llm1_path, save_path)
    # seq2_embedding = get_embed(seq2_name, seq2_seq, seq2_type, seq2_method, llm2_path, save_path)

#     df = pd.read_csv(data_path, sep = '\t')
#     labels = df['Class']
#     full_dataset = IteractionDataset(seq1_embedding, seq2_embedding, labels, batch_size=32, seed=42, split_ratios=(0.8, 0.1, 0.1))
#     # 获取数据加载器
#     train_loader, val_loader, test_loader = full_dataset.get_dataloaders()

    gpu_id = 0
    gpu_id = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    save_path = "./emb_save_MTI/" 
    data_path = "/home/wym321/RNA_Mol/data/mirtarbase/mirtarbase.txt"
    seq1_name_col = 'RNA_ID'
    seq2_name_col = 'prot_ID'
    seq1_seq_col = 'RNA_sequence'
    seq2_seq_col = 'prot_sequence'
    seq1_name, seq2_name, seq1_seq, seq2_seq, label = get_data(data_path, seq1_name_col, seq2_name_col, seq1_seq_col, seq2_seq_col)

    seq1_method = 'lucaone'
    # seq1_method = 'molgpt'
    seq2_method = 'lucaone' 
    seq1_type = 'rna'
    seq2_type = 'prot'
    # # llm1_path = '/mnt/sdb/ZYQ/workspace/DeepSeqPDBI/LLM'
    # # llm2_path = '/mnt/sdb/chenzhb/LucaOne/LucaOneTasks/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step5600000'
    # # llm2_path = '/mnt/sdb/chenzhb/LucaOne/LucaOneTasks/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000'
    # # llm2_path = '/mnt/sdb/chenzhb/LucaOne/LucaOneTasks/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000'
    # # llm2_path = '/mnt/sdb/ZYQ/workspace/DeepSeqPDBI/LLM/lucaone/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000'
    llm1_path = '/mnt/sdb/ZYQ/workspace/DeepSeqPDBI/LLM/lucaone/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20240906230224/checkpoint-step36000000'
    llm2_path = '/mnt/sdb/ZYQ/workspace/DeepSeqPDBI/LLM/lucaone/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20240906230224/checkpoint-step36000000'

    # seq1_embedding = get_embed(seq1_name, seq1_seq, seq1_type, seq1_method, llm1_path, save_path)
    seq2_embedding = get_embed(seq2_name, seq2_seq, seq2_type, seq2_method, llm2_path, save_path)




    # gpu_id = 2
    # gpu_id = str(gpu_id)
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # save_path = "./emb_save_DNA_RNA"

    # ##测试读取文件
    # data_path = '/mnt/sdb/ZYQ/workspace/DNA_RNA/data/all_data_ids_out_4096_new.txt'
    # seq1_name_col = 'Molecule_ID'
    # seq2_name_col = 'RNA_target'
    # seq1_seq_col = 'SMILES'
    # seq2_seq_col = 'RNA_sequence'
    # seq1_name, seq2_name, seq1_seq, seq2_seq, label = get_data(data_path, seq1_name_col, seq2_name_col, seq1_seq_col, seq2_seq_col)



    # gpu_id = 1
    # gpu_id = str(gpu_id)
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # data_path = '/mnt/sdb/ZYQ/workspace/DNA_RNA/data/all_data_ids_out_4096_new.txt'
    # save_path = "./emb_save_DNA_RNA"
    # seq1_name_col = 'RNA_id'
    # seq1_seq_col = 'RNAseqs'
    # seq1_method = 'lucaone'
    # seq1_type = 'rna'
    # llm1_path = '/mnt/sdb/ZYQ/workspace/DeepSeqPDBI/LLM/lucaone/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20240906230224/checkpoint-step36000000'
    # llm1_path = '/mnt/sdb/chenzhb/LucaOne/LucaOneTasks/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000'

    # seq1_name, seq2_name, seq1_seq, seq2_seq, label = get_data(data_path, seq1_name_col, seq1_name_col, seq1_seq_col, seq1_seq_col)
    # seq1_embedding = get_embed(seq1_name, seq1_seq, seq1_type, seq1_method, llm1_path, save_path)


    
    gpu_id = 1
    gpu_id = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    data_path = '/mnt/sdb/ZYQ/workspace/DNA_RNA/data/sample_20000.txt'
    save_path = "./emb_save_DNA_RNA"
    seq1_name_col = 'RNA_id'
    seq1_seq_col = 'RNAseqs'
    seq1_method = 'lucaone'
    seq1_type = 'rna'
    seq2_name_col = 'DNA_id'
    seq2_seq_col = 'DNAseqs'
    seq2_type = 'dna'
    seq2_method = 'lucaone'
    llm1_path = '/mnt/sdb/ZYQ/workspace/DeepSeqPDBI/LLM/lucaone/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20240906230224/checkpoint-step36000000'
    llm2_path = '/mnt/sdb/chenzhb/LucaOne/LucaOneTasks/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000'

    seq1_name, seq2_name, seq1_seq, seq2_seq, label = get_data(data_path, seq1_name_col, seq2_name_col, seq1_seq_col, seq2_seq_col)
    seq1_embedding = get_embed(seq1_name, seq1_seq, seq1_type, seq1_method, llm1_path, save_path)
    seq2_embedding = get_embed(seq2_name, seq2_seq, seq2_type, seq2_method, llm2_path, save_path)