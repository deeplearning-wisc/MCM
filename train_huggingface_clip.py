
from genericpath import exists
import os
from numpy import count_nonzero
from tqdm import tqdm
import torch
import argparse
from transformers import CLIPProcessor, CLIPModel
from transformers import logging
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

from utils.common import AverageMeter
from utils.coco_dataset import build_coco_loader
from utils.dogs_dataset import build_dogs_loader, get_dogs_cls, labels_from_wordnet_ids

logging.set_verbosity_warning()
os.environ['TOKENIZERS_PARALLELISM'] = "false"
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_params(description = 'Training clip'):
    parser = argparse.ArgumentParser(description=description)
    # training  
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-8)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    # arch
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max_length", type=int, default=77)
    # #model loading
    # parser.add_argument("--text_encoder_model", type=str, default="distilbert-base-uncased")
    # parser.add_argument("--text_tokenizer", type=str, default="distilbert-base-uncased")
    #Misc
    parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--is_train", action='store_false')
    #data loading
    parser.add_argument("--server", type=str, default='inst-01', help="run on which server")
    parser.add_argument("--root_dir", type=str, default='data', help="data root dir")
    parser.add_argument("--lang", type=str, default='es', help="caption language")
    parser.add_argument("--dataset", type=str, default='COCO', help="image dataset")
    #logging
    parser.add_argument("--unique_id", type=str, default='es_32_adamw', help="data root dir")


    # parse parameters
    params = parser.parse_args()
    params.unique_id += f'_{params.lr}'
    params.ckpt = 'openai/clip-vit-large-patch14'
    if params.server == 'A100':
        params.image_dir = '/home/mingyifei/datasets/COCO' 
        params.save_dir = f'checkpoints/clip/{params.dataset}'
        params.batch_size = 256
    elif params.server in ['inst-01', 'inst-04']:
        params.image_dir = '/nobackup/COCO/COCO-14'
        params.save_dir = f'/nobackup/checkpoints/clip/{params.dataset}'
        params.batch_size = 64
    elif params.server in ['galaxy-01', 'galaxy-02']:
        params.image_dir = '/nobackup-slow/dataset/ILSVRC-2012'
        params.save_dir = f'/nobackup/zcai/checkpoints/clip/{params.dataset}' # /nobackup/checkpoints throwing permission denied error
        params.batch_size = 128
    params.captions_dir = f"{params.root_dir}/{params.dataset}/captions/{params.lang}"
    os.makedirs(params.save_dir, exist_ok=True)
    params.device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device('cpu')
    return params

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def log_write(logf, msg, console_print=True):
    logf.write(msg + '\n')
    if console_print:
        print(msg)

# def convert_models_to_fp32(model): 
#     for p in model.parameters(): 
#         p.data = p.data.float() 
#         p.grad.data = p.grad.data.float() 

def train_epoch(model, tokenizer, train_loader, optimizer, global_step):
    loss_meter = AverageMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        text_tokens = tokenizer(batch['caption'], padding=True, truncation=True,max_length=params.max_length)
        batch['pixel_values'] = batch['pixel_values'].to(params.device)
        bz = batch["pixel_values"].size(0)
        batch.pop('caption')
        batch.pop('targets')
        for k, v in text_tokens.items(): # add 'input_ids', 'attention_mask' to batch 
            batch[k] = torch.tensor(v).to(params.device)
        batch['return_loss'] = True
        loss = model(**batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), bz)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        # logging
        if global_step % 10 == 0:
            writer.add_scalar("train/loss", scalar_value=loss_meter.avg, global_step=global_step)
            # writer.add_scalar("train/lr", scalar_value=lr_scheduler.get_lr()[0], global_step=global_step)
        global_step += 1
        # torch.cuda.empty_cache()
    return loss_meter, global_step
        
def train(params, model, tokenizer, processor, logf, writer):
    print("Training model")
    if params.dataset == 'COCO':
        train_loader= build_coco_loader(params, option = 'train')
    elif params.dataset == 'ImageNet-dogs':
        train_loader=build_dogs_loader(params, option='train')
    assert train_loader is not None, 'train loader is None'

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    # optimizer = torch.optim.SGD(model.parameters(), params.lr)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=1, factor=0.8
    # )
    lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)
    # lambda1 = lambda epoch: epoch // 30
    # lambda2 = lambda epoch: 0.95 ** epoch
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0002, max_lr=0.002,step_size_up=5,mode="triangular")
    best_loss = float('inf')
    global_step = 0

    ### debug valid_epoch ###
    # with torch.no_grad():
    #     valid_loss, acc = valid_epoch(params, model, tokenizer, processor, global_step)
    #     print(acc)
    ###

    for epoch in range(params.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss, global_step = train_epoch(model, tokenizer, train_loader, optimizer, global_step)
        model.eval()
        with torch.no_grad():
            valid_loss, acc = valid_epoch(params, model, tokenizer, processor, global_step)
        if valid_loss.avg < best_loss:
            params.model_path =  os.path.join(params.save_dir, f"best_{params.lang}.pt") 
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), params.model_path)
            log_write(logf, "Saved Best Model!")
        elif epoch % 4 == 0:
            params.model_path =  os.path.join(params.save_dir, f"{epoch}_{params.lang}.pt") 
            torch.save(model.state_dict(), params.model_path)
            log_write(logf, f"Saved Model at epoch {epoch}")
        lr_scheduler.step(valid_loss.avg)
        log_write(logf, "epoch {} train_loss: {:.4f} val_loss: {:.4f} acc {:.4f}".format(epoch, train_loss.avg, valid_loss.avg, acc))
    lr_scheduler.step()

def valid_epoch(params, model, tokenizer, processor, global_step):
    loss_meter = AverageMeter()
    if params.dataset == 'COCO':
        val_loader= build_coco_loader(params, option = 'val')
    elif params.dataset == 'ImageNet-dogs':
        val_loader=build_dogs_loader(params, option='val')
    assert val_loader is not None, 'val loader is None'

    wordnet_ids = get_dogs_cls()
    labels = labels_from_wordnet_ids(wordnet_ids)
    captions = [f'A photo of a {l}.' for l in labels]
    cls_text_tks = tokenizer(captions, padding=True, truncation=True,max_length=params.max_length)

    tqdm_object = tqdm(val_loader, total=len(val_loader))
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for batch in tqdm_object:
            text_tokens = tokenizer(batch['caption'], padding=True, truncation=True,max_length=params.max_length)
            batch['pixel_values'] = batch['pixel_values'].to(params.device)
            bz = batch["pixel_values"].size(0)
            captions = batch.pop('caption')
            targets = batch.pop('targets')
            for k, v in text_tokens.items():
                batch[k] = torch.tensor(v).to(params.device)
            batch['return_loss'] = True
            output = model(**batch)
            loss = output.loss
            loss_meter.update(loss.item(), bz)
            tqdm_object.set_postfix(valid_loss=loss_meter.avg)

            # eval accuracy
            # inputs = processor(
            #     text=captions, images=batch['pixel_values'], return_tensors="pt", padding=True, truncation=True, max_length=params.max_length
            # )
            for k, v in cls_text_tks.items():
                batch[k] = torch.tensor(v).to(params.device)
            batch['return_loss'] = False
            batch['pixel_values'] = batch['pixel_values'][0:params.batch_size,:]
            cls_text_output = model(**batch)
            pred = torch.argmax(cls_text_output.logits_per_text, dim=0).cpu()
            n_correct = n_correct + count_nonzero(pred == targets)
            n_sample = n_sample + bz

    acc = n_correct/n_sample
    writer.add_scalar("test/loss", scalar_value=loss_meter.avg, global_step=global_step)
    writer.add_scalar("test/acc", scalar_value=acc, global_step=global_step)
    return loss_meter, acc

if __name__ == "__main__":
    params = get_params()
    print("Caption Language: " + params.lang)
    tokenizer = AutoTokenizer.from_pretrained(params.ckpt, do_lower_case=True)
    processor = CLIPProcessor.from_pretrained(params.ckpt, do_lower_case=True)
    model = CLIPModel.from_pretrained(params.ckpt).to(params.device)
    log_dir = 'train_results/logs'
    os.makedirs(log_dir, exist_ok = True)
    logf = open(os.path.join(log_dir, f'{params.dataset}_{params.lang}_{params.unique_id}.out'), 'w')
    writer = SummaryWriter(log_dir= os.path.join(log_dir, params.unique_id))

    if params.resume:
        model.load_state_dict(torch.load(params.model_path))
    if params.is_train:
        train(params, model, tokenizer, processor, logf, writer)
    
