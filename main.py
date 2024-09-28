import logging
import numpy as np
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from apex import amp
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import yaml
# model
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from textModels.multimodal_modeling import MFD
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
# Data process function
from bottom_up_attention.utils.extraction_bbox import load_bottom_model
from processors.glue import glue_processors as processors
from processors.utils import convert_mm_examples_to_features_bert_bottom_vit_8
# metric and result analysis
from result.metric import compute_metrics_mner_boundary_category
from result.metric import during_train_log_mner_boundary_category
from result.metric import finish_train_log_mner_bio
# logging function
from tools.common import init_logger, logger
from tools.common import seed_everything
from tools.progressbar import ProgressBar
import wandb

import processors.utils as utils
import sys
sys.modules['processors.utils_obj'] = utils

dataset_name = "twitter17" # twitter15, twitter17
wandb.init(
    project="MNER-BERT-{}-final".format(dataset_name),
    name="{}".format(dataset_name)
)

MODEL_CLASSES = {
    ## bertModel
    'bert': (BertConfig, BertModel, BertTokenizer),
    'bertweet': (AutoConfig, AutoModel, AutoTokenizer),
    'roberta': (AutoConfig, AutoModel, AutoTokenizer),
}

class ARGS(object):
    def __init__(self, config_file):
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            for key, value in config.items():
                if not hasattr(self, key):
                    setattr(self, key, value)

def load_and_cache_examples(args, task, tokenizer, logger, vit_extractor, vit_model, bottom_model, data_type):
    processor = processors[task]()
    cached_features_file = os.path.join(args.text_data_dir, 'cached_{}_{}_{}_{}'.format(
        args.tagging_schema,
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logging.info("Creating features from dataset file as %s", args.text_data_dir)

        label_list = processor.get_labels(args.tagging_schema)
        label_cl_list = processor.get_cl_labels(args.tagging_schema)
        if data_type == 'train':
            examples = processor.get_train_examples(args.text_data_dir, args.image_data_dir, args.tagging_schema)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.text_data_dir, args.image_data_dir, args.tagging_schema)
        elif data_type == 'test':
            examples = processor.get_test_examples(args.text_data_dir, args.image_data_dir, args.tagging_schema)
        else:
            raise Exception("Invalid data type %s..." % data_type)
        features = convert_mm_examples_to_features_bert_bottom_vit_8(examples=examples, label_list=label_list, tokenizer=tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                    cls_token=tokenizer.cls_token,
                                                    sep_token=tokenizer.sep_token,
                                                    cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    vit_extractor=vit_extractor, vit_model=vit_model,bottom_model=bottom_model,
                                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

        logger.info("Num orig examples = %d", len(examples))
        logger.info("Num split features = %d", len(examples))
        logger.info("Batch size = %d", args.per_gpu_train_batch_size)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_class_ids = torch.tensor([f.class_ids for f in features], dtype=torch.long)
    all_image_mask = torch.tensor([f.image_mask for f in features], dtype=torch.long)
    all_box_dis_position = torch.tensor([f.box_dis_position for f in features], dtype=torch.long)
    all_img_feats = torch.stack([f.img_feat for f in features])
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    # used in evaluation
    all_evaluate_label_ids = [f.evaluate_label_ids for f in features]
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, 
                            all_img_feats, all_class_ids, all_image_mask, all_box_dis_position)

    return dataset, all_evaluate_label_ids


def evaluate(args, model, tokenizer, prefix="", logger=None, epoches=None, vit_extractor=None, vit_model=None,bottom_model=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
    
    results_aspect = {}
    processor = processors[args.task_name]()
    label_list = processor.get_labels(args.tagging_schema)
    num_labels = len(label_list)
    criterion = nn.CrossEntropyLoss()
    
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_evaluate_label_ids = load_and_cache_examples(args, eval_task, tokenizer, logger, vit_extractor, vit_model=vit_model,bottom_model=bottom_model, data_type=prefix)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running  {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        input_ids = None
        pbar = ProgressBar(n_total=len(eval_dataloader),desc = "Evaluating")
        epoch_loss = 0.0
        epoch_bio_loss,epoch_cls_loss = 0.0, 0.0
        for step,batch in enumerate(eval_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],  
                        'labels':         batch[3],
                        'img_feats': batch[4],
                        'img_class_ids': batch[5],
                        'img_mask': batch[6],
                        'img_dis_position': batch[7]}
                aspect_labels = copy.deepcopy(batch[3])
                
                if args.use_bio_class and args.use_cls_class:
                    logits_fusion, loss_bio, loss_cls = model(**inputs, is_train=False)
                    output_aspect= logits_fusion
                    loss_aspect = criterion(output_aspect.view(-1, num_labels), aspect_labels.view(-1))
                    epoch_bio_loss += loss_bio
                    epoch_cls_loss += loss_cls
                else:
                    raise ValueError("error combination")
                
                loss = loss_aspect
                eval_loss += loss.mean().item()
                epoch_loss += loss.item()
            nb_eval_steps += 1
            if input_ids is None:
                input_ids = inputs['input_ids'].detach().cpu().numpy()
            else:
                input_ids = np.append(input_ids, inputs['input_ids'].detach().cpu().numpy(), axis=0)
            if preds is None:
                preds = logits_fusion.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits_fusion.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            eval_loss = eval_loss / nb_eval_steps
            pbar(step,{'loss':loss.item()})
        print(' ')
        logger.info("***** The {} Loss is {} *****".format(prefix, epoch_loss/step))
        wandb.log({'epoch': epoches, '{}_loss'.format(prefix): epoch_loss/step})
        if args.use_bio_class:
            logger.info("***** The {} BIO Class Loss is {} *****".format(prefix, epoch_bio_loss/step))
            wandb.log({'epoch': epoches, '{}_bio_loss'.format(prefix): epoch_bio_loss/step})
        if args.use_cls_class:
            logger.info("***** The {} ClS Class Loss is {} *****".format(prefix, epoch_cls_loss/step))
            wandb.log({'epoch': epoches, '{}_cls_loss'.format(prefix): epoch_cls_loss/step})         

        preds = np.argmax(preds, axis=-1)
        result, pd_predict = compute_metrics_mner_boundary_category(args, preds, out_label_ids, eval_evaluate_label_ids, args.tagging_schema, prefix)
        result['eval_loss'] = eval_loss
        results_aspect.update(result)
    return results_aspect, pd_predict


def train(args, train_dataset, model, tokenizer, logger, vit_extractor, vit_model,bottom_model, timestamp, train_evaluate_label_ids):
    """" Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)

    # prepare optimizer and schedule (linear warmup and decay)
    # 设置不同学习率
    if args.is_diff_lr==True:
        no_decay = ['bias', 'LayerNorm.weight']
        children_names = [name for name, child in model.named_children()]
        bottom_part = ["image_model", "text_model"]
        top_part = [name for name in children_names if name not in bottom_part]
        optimizer_grouped_parameters = [
            {'params': [p for name in bottom_part for n, p in getattr(model, name).named_parameters() if not any(nd in n for nd in no_decay)],'lr': float(args.bottom_lr), 'weight_decay': float(args.weight_decay)},
            {'params': [p for name in bottom_part for n, p in getattr(model, name).named_parameters() if any(nd in n for nd in no_decay)],'lr': float(args.bottom_lr), 'weight_decay': 0.0},
            {'params': [p for name in top_part for n, p in getattr(model, name).named_parameters() if not any(nd in n for nd in no_decay)],'lr': float(args.top_lr), 'weight_decay': float(args.weight_decay)},
            {'params': [p for name in top_part for n, p in getattr(model, name).named_parameters() if any(nd in n for nd in no_decay)],'lr': float(args.top_lr), 'weight_decay': 0.0},
        ]
    else:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': float(args.weight_decay)},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
    if args.is_diff_lr==True:
        optimizer = AdamW(optimizer_grouped_parameters, eps=float(args.adam_epsilon))
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=float(args.learning_rate), eps=float(args.adam_epsilon), weight_decay=float(args.weight_decay))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    seed_everything(args.seed)
    test_result_all = []
    dev_result_all = []
    test_pd_result_all = []
    dev_pd_result_all = []
    processor = processors[args.task_name]()
    label_list = processor.get_labels(args.tagging_schema)
    num_labels = len(label_list)
    for _ in range(int(args.num_train_epochs)):
        loss, epoch_loss = 0.0, 0.0
        epoch_bio_loss, epoch_cls_loss = 0.0, 0.0
        pbar = ProgressBar(n_total=len(train_dataloader), desc="Training")
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels':         batch[3],
                    'img_feats': batch[4],
                    'img_class_ids': batch[5],
                    'img_mask': batch[6],
                    'img_dis_position': batch[7]}
            aspect_labels = copy.deepcopy(batch[3])
            
            if args.use_bio_class and args.use_cls_class:
                logits, loss_bio, loss_cls = model(**inputs, is_train=True)
                output_aspect= logits
                loss_aspect = criterion(output_aspect.view(-1, num_labels), aspect_labels.view(-1))
                epoch_bio_loss += loss_bio
                epoch_cls_loss += loss_cls
                loss = loss_aspect + loss_bio + loss_cls
            else:
                raise ValueError("error combination")

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            epoch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        # logger.info("****** dev metrics ******")
                        metrics, pd_result_dev = evaluate(args, model, tokenizer, prefix="dev", logger=logger, epoches=_, vit_extractor=vit_extractor, vit_model=vit_model,bottom_model=bottom_model)
                        dev_pd_result_all.append(pd_result_dev)
                        dev_result_all.append(metrics)
                        during_train_log_mner_boundary_category(logger=logger, epochs=_, metrics=metrics, prefix="dev", global_step=global_step, wandb=wandb)
                        # logger.info("****** test metrics ******")
                        metrics, pd_result_test = evaluate(args, model, tokenizer, prefix="test", logger=logger, epoches=_, vit_extractor=vit_extractor, vit_model=vit_model,bottom_model=bottom_model)
                        test_pd_result_all.append(pd_result_test)
                        test_result_all.append(metrics)
                        during_train_log_mner_boundary_category(logger=logger, epochs=_, metrics=metrics, prefix="test", global_step=global_step, wandb=wandb)
                    '''
                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'aspect_model.bin'))
                        # model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)
                        #tokenizer.save_vocabulary(vocab_path=output_dir)
                    '''
            pbar(step,{'loss':loss.item()})
        print(" ")
        logger.info("***** The Train {} Epoch Loss is {} *****".format(_, epoch_loss/step))
        wandb.log({'epoch': _, 'train_loss': epoch_loss/step})
        if args.use_bio_class:
            logger.info("***** The Train BIO Class {} Epoch Loss is {} *****".format(_, epoch_bio_loss/step))
            wandb.log({'epoch': _, 'train_bio_loss': epoch_bio_loss/step})
        if args.use_cls_class:
            logger.info("***** The Train CLS Class {} Epoch Loss is {} *****".format(_, epoch_cls_loss/step))
            wandb.log({'epoch': _, 'train_cls_loss': epoch_cls_loss/step})
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    finish_train_log_mner_bio(args, logger, wandb, global_step, timestamp, _, dev_result_all, test_result_all, dev_pd_result_all, test_pd_result_all)
    return global_step, tr_loss / global_step


def main():
    args = ARGS("./config/{}.yaml".format(dataset_name))
    if not args.do_train and not args.do_predict and not args.do_pipeline:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")
    if args.do_train and not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    args.model_type = args.model_type.lower()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    init_logger(log_file=args.output_dir + '/{}-{}-{}-bsz_{}-epoches_{}-warmup_{}.log'.format(timestamp, args.model_type, args.model_architecture,args.per_gpu_train_batch_size,args.num_train_epochs,args.warmup_proportion))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # 参数写进日志
    for k, v in vars(args).items():
        logger.info('{}: {}'.format(k, v))
    seed_everything(args.seed)
    # text model BERT
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.text_model_type]
    config = config_class.from_pretrained(args.text_model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.text_model_name_or_path)
    if args.is_change_text_hidden_dropout:
        config.hidden_dropout_prob = args.text_hidden_dropout
    bert_model = model_class.from_pretrained(args.text_model_name_or_path, from_tf=bool('.ckpt' in args.text_model_name_or_path), config=config)
    processor = processors[args.task_name]()
    label_list = processor.get_labels(args.tagging_schema)
    config.num_labels = len(label_list)
    
    ######
    # image model ViT
    vit_config = ViTConfig.from_pretrained(args.vision_model_name_or_path)
    if args.is_change_image_hidden_dropout:
        vit_config.hidden_dropout_prob = args.image_hidden_dropout
    vit_model = ViTModel.from_pretrained(args.vision_model_name_or_path, config=vit_config)
    vit_extractor = ViTFeatureExtractor.from_pretrained(args.vision_model_name_or_path)
    
    bottom_model = load_bottom_model()
    model = MFD(args, text_config=config, image_config=vit_config, text_model=bert_model, image_model=vit_model, tokenizer=tokenizer)
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    logger.info("Start Time is %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start = datetime.now()
    # Training
    if args.do_train:
        train_dataset,train_evaluate_label_ids = load_and_cache_examples(args, args.task_name, tokenizer, logger, vit_extractor,vit_model,bottom_model,data_type="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, logger, vit_extractor, vit_model,bottom_model,timestamp, train_evaluate_label_ids)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    logger.info("End Time is %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.info("Training complete in: %s", str(datetime.now() - start))
if __name__ == "__main__":
    main()
