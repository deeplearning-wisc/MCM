
import os
from tqdm import tqdm
import torch
import argparse
import clip
from transformers import logging
from transformers import BertTokenizer, AutoTokenizer
from utils.common import AverageMeter
from utils.dataset import load_train_data

logging.set_verbosity_warning()
os.environ['TOKENIZERS_PARALLELISM'] = "false"
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# main
parser = argparse.ArgumentParser(description='Training clip')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--patience", type=int, default=2)
parser.add_argument("--factor", type=float, default=0.5)

parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=20)

parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--num_projection_layers", type=int, default=1)
parser.add_argument("--projection_dim", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.07)

parser.add_argument("--image_embedding", type=int, default=768)
parser.add_argument("--text_embedding", type=int, default=768)
parser.add_argument("--max_length", type=int, default=200)

parser.add_argument("--text_encoder_model", type=str, default="distilbert-base-uncased")
parser.add_argument("--text_tokenizer", type=str, default="distilbert-base-uncased")

parser.add_argument("--lang", type=str, default='es', help="Source language")
parser.add_argument("--data", type=str, default='coco', help="Source language")
parser.add_argument("--data_dir", type=str, default='data', help="Source language")


parser.add_argument("--resume", action='store_true')
parser.add_argument("--is_train", action='store_false')

# parse parameters
params = parser.parse_args()
#v1: hard coded for now
params.model_name = "dccuchile/bert-base-spanish-wwm-uncased"
params.image_path = '/nobackup/dataset_myf/COCO/train2014'
params.captions_path = f"data/captions/{params.lang}/processed_captions_train2014.csv"
params.image_prefix = 'COCO_train2014_'
params.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def train_epoch(model, tokenizier, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AverageMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        text_tokens = tokenizer(batch['caption'], padding=True, truncation=True,max_length=params.max_length)
        batch['image'] = batch['image'].to(params.device)
        for k, v in text_tokens.items():
            batch[k] = torch.tensor(v).to(params.device)
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

        torch.cuda.empty_cache()
    lr_scheduler.step()
    return loss_meter


def valid_epoch(model, tokenizer, valid_loader):
    loss_meter = AverageMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        text_tokens = tokenizer(batch['caption'], padding=True, truncation=True,max_length=params.max_length)
        batch['image'] = batch['image'].to(params.device)
        for k, v in text_tokens.items():
            batch[k] = torch.tensor(v).to(params.device)
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def log_write(logf, msg, console_print=True):
    logf.write(msg + '\n')
    if console_print:
        print(msg)
        
def train(model, tokenizer, params):
    logf = open(f'../results/logs/{params.data}_{params.lang}.out', 'w')
    print("Training model")
    train_loader= load_train_data(params)
    # optimizer = torch.optim.AdamW(
        # model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    # )
    optimizer = torch.optim.SGD(model.parameters(), params.lr)
    
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=params.patience, factor=params.factor
    # )
    # lambda1 = lambda epoch: epoch // 30
    # lambda2 = lambda epoch: 0.95 ** epoch
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0002, max_lr=0.002,step_size_up=5,mode="triangular")
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(params.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, tokenizer, train_loader, optimizer, lr_scheduler, step)

        # model.eval()
        # with torch.no_grad():
        #     valid_loss = valid_epoch(model, tokenizer, valid_loader)
        
        # if valid_loss.avg < best_loss:
        #     best_loss = valid_loss.avg
        #     torch.save(model.state_dict(), params.model_path)
        #     log_write(logf, "Saved Best Model!")
        
        log_write(logf, "epoch {} train_loss: {:.4f} val_loss: {:.4f}".format(epoch, train_loss.avg, valid_loss.avg))

if __name__ == "__main__":
    print("Model on " + params.lang)
    params.model_path = f"../results/clips/{params.data}/best_{params.lang}.pt" #path where the model to be saved
    tokenizer = AutoTokenizer.from_pretrained(params.model_name, do_lower_case=True)

    model, preprocess = clip.load("ViT-B/32",device=params.device, jit=False)
    clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    if params.resume:
        model.load_state_dict(torch.load(params.model_path))

    if params.is_train:
        train(model, tokenizer, params)