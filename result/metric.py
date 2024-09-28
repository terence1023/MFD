import numpy as np
import pandas as pd
from processors.glue import glue_processors as processors
from result.analysis import result_analysis_bio, result_analysis_mner
import os


def ts2bio(ts_tag_sequence):

    new_ts_sequence = []
    n_tags = len(ts_tag_sequence)
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        if ts_tag == 'O' or ts_tag == 'EQ':
            new_ts_sequence.append('O')
        else:
            new_ts_sequence.append(ts_tag)
    return new_ts_sequence


def bio2bioes_ts(ts_tag_sequence):
    n_tags = len(ts_tag_sequence)
    new_ts_sequence = []
    for i in range(n_tags):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 'O' or cur_ts_tag == 'EQ':
            # when meet the EQ label, regard it as O label
            new_ts_sequence.append('O')
        else:
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
            if cur_pos == 'B':
                if (i == n_tags - 1) or (ts_tag_sequence[i+1].split('-')[0] != 'I'):
                    new_ts_sequence.append('S-%s' % cur_sentiment)
                else:
                    new_ts_sequence.append('B-%s' % cur_sentiment)
            elif cur_pos == 'I':
                if (i == n_tags - 1) or (ts_tag_sequence[i + 1].split('-')[0] != 'I' and i != 0 and ts_tag_sequence[i - 1].split('-')[0] != 'O'):
                    new_ts_sequence.append('E-%s' % cur_sentiment)
                else:
                    new_ts_sequence.append('I-%s' % cur_sentiment)
    return new_ts_sequence

def tag2ts(ts_tag_sequence):
    """
    transform ts tag sequence to targeted sentiment
    :param ts_tag_sequence: tag sequence for ts task
    :return:
    """
    n_tags = len(ts_tag_sequence)
    ts_sequence, sentiments = [], []
    beg, end = -1, -1
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        # current position and sentiment
        # tag O and tag EQ will not be counted
        eles = ts_tag.split('-')
        if len(eles) == 2:
            pos, sentiment = eles
        else:
            pos, sentiment = 'O', 'O'
        if sentiment != 'O':
            # current word is a subjective word
            sentiments.append(sentiment)
        if pos == 'S':
            # singleton
            ts_sequence.append((i, i, sentiment))
            sentiments = []
        elif pos == 'B':
            beg = i
            if len(sentiments) > 1:
                # remove the effect of the noisy I-{POS,NEG,NEU}
                sentiments = [sentiments[-1]]
        elif pos == 'E':
            end = i
            # schema1: only the consistent sentiment tags are accepted
            # that is, all of the sentiment tags are the same
            if end > beg > -1 and len(set(sentiments)) == 1:
                ts_sequence.append((beg, end, sentiment))
                sentiments = []
                beg, end = -1, -1
    return ts_sequence



def match(gold_ts_sequence, pred_ts_sequence):
    """
    calculate the number of correctly predicted targeted sentiment
    :param gold_ts_sequence: gold standard targeted sentiment sequence
    :param pred_ts_sequence: predicted targeted sentiment sequence
    :return:
    """
    hit_count, gold_count, pred_count = np.zeros(1), np.zeros(1), np.zeros(1)
    gold_count = len(gold_ts_sequence)
    pred_count = len(pred_ts_sequence)
    for t in pred_ts_sequence:
        if t in gold_ts_sequence:
            hit_count += 1

    return hit_count, gold_count, pred_count

def match_boundary(gold_ts_sequence, pred_ts_sequence):
    """
    calculate the number of correctly predicted targeted sentiment
    :param gold_ts_sequence: gold standard targeted sentiment sequence
    :param pred_ts_sequence: predicted targeted sentiment sequence
    :return:
    """
    hit_count, gold_count, pred_count = np.zeros(1), np.zeros(1), np.zeros(1)
    gold_count = len(gold_ts_sequence)
    pred_count = len(pred_ts_sequence)
    for t in pred_ts_sequence:
        if t in gold_ts_sequence:
            hit_count += 1

    return hit_count, gold_count, pred_count


SMALL_POSITIVE_CONST = 1e-10

def compute_metrics_mner_boundary_category(args, preds, labels, all_evaluate_label_ids, tagging_schema, data_type):
    processor = processors[args.task_name]()
    label_list = processor.get_labels(args.tagging_schema)
    absa_label_vocab = {label: i for i, label in enumerate(label_list)}

    if data_type == 'dev':
        examples = processor.get_dev_examples(args.text_data_dir, args.image_data_dir, args.tagging_schema)
    elif data_type == 'test':
        examples = processor.get_test_examples(args.text_data_dir, args.image_data_dir, args.tagging_schema)
    else:
        raise Exception("Invalid data type %s..." % data_type)

    absa_id2tag = {}
    for k in absa_label_vocab:
        v = absa_label_vocab[k]
        absa_id2tag[v] = k

    pd_predict = pd.DataFrame(columns=["mner_gold_words", "mner_pred_words", "mner_gold_sequence", "mner_pred_sequence", "bio_gold_words", "bio_pred_words", "bio_gold_sequence", "bio_pred_sequence", "text","image_id"])
    # number of true postive, gold standard, predicted targeted sentiment
    n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(1), np.zeros(1), np.zeros(1)
    bio_n_tp_ts, bio_n_gold_ts, bio_n_pred_ts = np.zeros(1), np.zeros(1), np.zeros(1)
    # precision, recall and f1 for aspect-based sentiment analysis
    ts_precision, ts_recall, ts_f1 = np.zeros(1), np.zeros(1), np.zeros(1)
    n_samples = len(all_evaluate_label_ids)
    pred_y, gold_y = [], []
    class_count = np.zeros(3)
    for i in range(n_samples):
        evaluate_label_ids = all_evaluate_label_ids[i]
        if isinstance(preds[i], np.ndarray):
            pred_labels = preds[i][evaluate_label_ids]
        elif isinstance(preds[i], list):
            pred_labels = np.array(preds[i])[evaluate_label_ids]
        gold_labels = labels[i][evaluate_label_ids]
        assert len(pred_labels) == len(gold_labels)
        # here, no EQ tag will be induced
        pred_tags = [absa_id2tag[label] for label in pred_labels]
        gold_tags = [absa_id2tag[label] for label in gold_labels]

        if tagging_schema == "MATE":
            gold_tags = bio2bioes_ts(ts2bio(gold_tags))
            pred_tags = bio2bioes_ts(ts2bio(pred_tags))
            g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=gold_tags), tag2ts(ts_tag_sequence=pred_tags)
        if tagging_schema == "TS":
            gold_tags = bio2bioes_ts(ts2bio(gold_tags))
            pred_tags = bio2bioes_ts(ts2bio(pred_tags))
            g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=gold_tags), tag2ts(ts_tag_sequence=pred_tags)
        if tagging_schema == "MNER":

            gold_tags = bio2bioes_ts(ts2bio(gold_tags))
            pred_tags = bio2bioes_ts(ts2bio(pred_tags))
            g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=gold_tags), tag2ts(ts_tag_sequence=pred_tags)
            bio_g_ts_sequence, bio_p_ts_sequence = [], []
            
            for seq in g_ts_sequence:
                bio_g_ts_sequence.append((seq[0], seq[1], "aspect"))
            for seq in p_ts_sequence:
                bio_p_ts_sequence.append((seq[0], seq[1], "aspect"))

        hit_ts_count, gold_ts_count, pred_ts_count = match(gold_ts_sequence=g_ts_sequence,pred_ts_sequence=p_ts_sequence)
        bio_hit_ts_count, bio_gold_ts_count, bio_pred_ts_count = match_boundary(gold_ts_sequence=bio_g_ts_sequence,pred_ts_sequence=bio_p_ts_sequence)
    
        word_list = examples[i].text_a.split(' ')
        g_ts_word = []
        p_ts_word = []
        bio_g_ts_word = []
        bio_p_ts_word = []
        for seq in g_ts_sequence:
            g_ts_word.append(" ".join(word_list[seq[0]: seq[1]+1]))
        for seq in p_ts_sequence:
            p_ts_word.append(" ".join(word_list[seq[0]: seq[1]+1]))
        # bio
        for seq in bio_g_ts_sequence:
            bio_g_ts_word.append(" ".join(word_list[seq[0]: seq[1]+1]))
        for seq in bio_p_ts_sequence:
            bio_p_ts_word.append(" ".join(word_list[seq[0]: seq[1]+1]))
            
        pd_predict.loc[i, "mner_gold_words"] = g_ts_word
        pd_predict.loc[i, "mner_pred_words"] = p_ts_word
        pd_predict.loc[i, "mner_gold_sequence"] = g_ts_sequence
        pd_predict.loc[i, "mner_pred_sequence"] = p_ts_sequence
        pd_predict.loc[i, "bio_gold_words"] = bio_g_ts_word
        pd_predict.loc[i, "bio_pred_words"] = bio_p_ts_word
        pd_predict.loc[i, "bio_gold_sequence"] = bio_g_ts_sequence
        pd_predict.loc[i, "bio_pred_sequence"] = bio_p_ts_sequence
        pd_predict.loc[i, "text"] = examples[i].text_a
        pd_predict.loc[i, "image_id"] = examples[i].image_path.split('/')[-1]
        
        bio_n_tp_ts += bio_hit_ts_count
        bio_n_gold_ts += bio_gold_ts_count
        bio_n_pred_ts += bio_pred_ts_count

        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count

    mner_precision = float(n_tp_ts)/float(n_pred_ts + SMALL_POSITIVE_CONST)
    mner_recall = float(n_tp_ts)/float(n_gold_ts + SMALL_POSITIVE_CONST)
    mner_f1 = 2*mner_precision*mner_recall/(mner_precision + mner_recall+SMALL_POSITIVE_CONST)
    # bio 
    bio_precision = float(bio_n_tp_ts)/float(bio_n_pred_ts + SMALL_POSITIVE_CONST)
    bio_recall = float(bio_n_tp_ts)/float(bio_n_gold_ts + SMALL_POSITIVE_CONST)
    bio_f1 = 2*bio_precision*bio_recall/(bio_precision + bio_recall+SMALL_POSITIVE_CONST)
    boundary_error = float(n_pred_ts-bio_n_tp_ts)/float(n_pred_ts + SMALL_POSITIVE_CONST) # 边界错误率
    # category
    category_accuracy = float(n_tp_ts)/float(bio_n_tp_ts + SMALL_POSITIVE_CONST)
    category_error = float(bio_n_tp_ts-n_tp_ts)/float(bio_n_tp_ts + SMALL_POSITIVE_CONST)
    result_mner = result_analysis_mner(pd_data=pd_predict)
    result_bio = result_analysis_bio(pd_data=pd_predict)
    
    assert n_pred_ts == bio_n_pred_ts
    assert n_gold_ts == bio_n_gold_ts
    true_entity_sum = n_gold_ts


    scores = {'precision': mner_precision, "recall": mner_recall, "f1": mner_f1,
              'bio_precision': bio_precision, "bio_recall": bio_recall, "bio_f1": bio_f1, 
              "category_accuracy": category_accuracy, "boundary_error": boundary_error, "category_error":category_error, 
              "predict_nums":n_pred_ts, "boundary_right_nums":bio_n_tp_ts, "true_entity_sum":true_entity_sum}
    # scores.update(result)
    scores.update(result_mner)
    scores.update(result_bio)
    return scores, pd_predict





def during_train_log_mner_boundary_category(logger, epochs, metrics, prefix, global_step, wandb):
    logger.info("  The {} Epochs".format(epochs))
    logger.info("per:     , P: {:.4f}, R: {:.4f}, F1: {:.4f} "
        .format(metrics['pre_per'], metrics['rec_per'], metrics['f1_per']))
    logger.info("loc:     , P: {:.4f}, R: {:.4f}, F1: {:.4f} "
        .format(metrics['pre_loc'], metrics['rec_loc'], metrics['f1_loc']))
    logger.info("org:     , P: {:.4f}, R: {:.4f}, F1: {:.4f} "
        .format(metrics['pre_org'], metrics['rec_org'], metrics['f1_org']))
    logger.info("misc:    , P: {:.4f}, R: {:.4f}, F1: {:.4f} "
        .format(metrics['pre_misc'], metrics['rec_misc'], metrics['f1_misc']))
    logger.info("step: {}, P: {:.4f}, R: {:.4f}, F1: {:.4f} "
        .format(global_step, metrics['precision'], metrics['recall'], metrics['f1']))

    logger.info("boundary: , P: {:.4f}, R: {:.4f}, F1: {:.4f} ".format(metrics['pre_aspect'], metrics['rec_aspect'], metrics['f1_aspect']))
    logger.info("boundary: , boundary_error: {:.4f}".format(metrics["boundary_error"]))
    logger.info("category: , ACC: {:.4f}".format(metrics["category_accuracy"]))
    logger.info("category: , category_error: {:.4f}".format(metrics["category_error"]))
    logger.info("Number  : , predict_nums: {:.0f}".format(int(metrics["predict_nums"])))
    logger.info("Number  : , boundary_right_nums: {:.0f}".format(int(metrics["boundary_right_nums"])))
    logger.info("Number  : , true_entity_sum: {:.0f}".format(int(metrics["true_entity_sum"])))
    
    wandb.log({'epoch': epochs, '{}_pre'.format(prefix): metrics['precision'], '{}_rec'.format(prefix): metrics['recall'], '{}_f'.format(prefix):metrics['f1']})
    wandb.log({'epoch': epochs, '{}_bio_pre'.format(prefix): metrics['bio_precision'], '{}_bio_rec'.format(prefix): metrics['bio_recall'], '{}_bio_f'.format(prefix):metrics['bio_f1']})



def finish_train_log_mner_bio(args, logger, wandb, global_step, timestamp, epochs, dev_result_all, test_result_all, dev_pd_result_all, test_pd_result_all):
    best_score = 0.0
    best_index = -1
    for i in range(len(dev_result_all)):
        f1_score = dev_result_all[i]['f1']
        if f1_score > best_score:
            best_index = i
            best_score = f1_score
    dev_result_best = dev_result_all[best_index]
    test_result_best = test_result_all[best_index]
    pd_dev_result_best = dev_pd_result_all[best_index]
    pd_test_result_best = test_pd_result_all[best_index]

    test_best_score = 0.0
    test_best_index = -1
    for i in range(len(test_result_all)):
        f1_score = test_result_all[i]['f1']
        if f1_score > test_best_score:
            test_best_index = i
            test_best_score = f1_score
    all_test_result_best = test_result_all[test_best_index]
    pd_all_test_result_best = test_pd_result_all[test_best_index]

    # 将结果文件写入csv
    dev_path = os.path.join(args.output_dir, "{}_{}_epoches_{}.csv".format(timestamp, "dev", best_index))
    pd_dev_result_best.to_csv(dev_path, index=False)
    test_path = os.path.join(args.output_dir, "{}_{}_epoches_{}.csv".format(timestamp, "dev-test", best_index))
    pd_test_result_best.to_csv(test_path, index=False)

    best_test_path = os.path.join(args.output_dir, "{}_{}_epoches_{}.csv".format(timestamp, "best-test", test_best_index))
    pd_all_test_result_best.to_csv(best_test_path, index=False)


    logger.info("***** the best dev results in {} *****".format(best_index))
    logger.info("mner:   ,P: {:.4f}, R: {:.4f}, F1: {:.4f} "
                .format(dev_result_best['precision'], dev_result_best['recall'], dev_result_best['f1']))
    wandb.log({'epoch': epochs, 'dev_best_pre': dev_result_best['precision'], 'dev_best_rec': dev_result_best['recall'], 'dev_best_f':dev_result_best['f1']})
    # print(" ")
    logger.info("bio:    , P: {:.4f}, R: {:.4f}, F1: {:.4f} "
                .format(dev_result_best['bio_precision'], dev_result_best['bio_recall'], dev_result_best['bio_f1']))
    wandb.log({'epoch': epochs, 'dev_best_pre_bio': dev_result_best['bio_precision'], 'dev_best_rec_bio': dev_result_best['bio_recall'], 'dev_best_f_bio':dev_result_best['bio_f1']})
    logger.info("category: , ACC: {:.4f}".format(dev_result_best["category_accuracy"]))
    wandb.log({'epoch': epochs, 'dev_best_acc_category': dev_result_best["category_accuracy"]})
    
    logger.info("***** the best dev-test results in {} *****".format(best_index))
    logger.info("mner:   , P: {:.4f}, R: {:.4f}, F1: {:.4f} "
                .format(test_result_best['precision'], test_result_best['recall'], test_result_best['f1']))
    wandb.log({'epoch': epochs, 'dev-test_best_pre': test_result_best['precision'], 'dev-test_best_rec': test_result_best['recall'], 'dev-test_best_f':test_result_best['f1']})
    # print(" ")
    logger.info("bio:    , P: {:.4f}, R: {:.4f}, F1: {:.4f} "
                .format(test_result_best['bio_precision'], test_result_best['bio_recall'], test_result_best['bio_f1']))
    wandb.log({'epoch': epochs, 'dev-test_best_pre_bio': test_result_best['bio_precision'], 'dev-test_best_rec_bio': test_result_best['bio_recall'], 'dev-test_best_f_bio':test_result_best['bio_f1']})
    logger.info("category: , ACC: {:.4f}".format(test_result_best["category_accuracy"]))
    wandb.log({'epoch': epochs, 'dev_best_acc_category': test_result_best["category_accuracy"]})