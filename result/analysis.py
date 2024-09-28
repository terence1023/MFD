import numpy as np
import pandas as pd


def result_analysis_mner(pd_data):
    # pd_predict = pd.DataFrame(columns=["mner_gold_words", "mner_pred_words", "mner_gold_sequence", "mner_pred_sequence", "bio_gold_words", "bio_pred_words", "bio_gold_sequence", "bio_pred_sequence", "text","image_id"])
    # absa_label_vocab = {'B-PER':0, 'I-PER':1, 'B-LOC':2, 'I-LOC':3, 'B-ORG':4, 'I-ORG':5, 'B-MISC':6, 'I-MISC':7, 'O':8}
    # calculate four type accuracy
    SMALL_POSITIVE_CONST = 1e-10
    type_list = ["per", "loc", "org", "misc"]
    result = {}
    for first in type_list:
        for second in type_list:
            if first != second:
                result["errorty_{}_pred_{}".format(first, second)] = 0
                result["errorboty_{}_pred_{}".format(first, second)] = 0 # 在边界预测错误的同时，类别也预测错了
    for name in type_list:
        result["gold_{}_count".format(name)] = 0
        result["pred_{}_count".format(name)] = 0
        result["pred_hit_{}_count".format(name)] = 0
    
    result["mner_error_end"] = 0
    result["mner_error_begin"] = 0
    result["mner_error_begin_end"] = 0
    result["mner_not_aspect"] = 0
    result["mner_not_predict"] = 0

    for i in range(len(pd_data)):
        gold_seq = pd_data.loc[i, "mner_gold_sequence"]
        pred_seq = pd_data.loc[i, "mner_pred_sequence"]
        for seq in gold_seq:
            result["gold_{}_count".format(seq[2].lower())] += 1

        for seq in pred_seq:
            result["pred_{}_count".format(seq[2].lower())] += 1
        length = len(gold_seq)
        sample_count = 0
        for seq in pred_seq:
            if seq in gold_seq:
                result["pred_hit_{}_count".format(seq[2].lower())] += 1
                sample_count += 1
            else:
                begin,end,pred_type = seq[0], seq[1], seq[2]
                flag = 0
                for gold in gold_seq:
                    if begin <= gold[0] and end>= gold[0]:
                        flag = 1
                        break
                if flag == 1:
                    # 边界正确，分类错误
                    if begin == gold[0] and end == gold[1]:
                        result["errorty_{}_pred_{}".format(pred_type.lower(), gold[2].lower())] += 1
                    # 三类边界错误
                    elif begin == gold[0] and end != gold[1]:
                        result["mner_error_end"] += 1
                        if pred_type != gold[2]:
                            result["errorboty_{}_pred_{}".format(pred_type.lower(), gold[2].lower())] += 1
                    elif begin != gold[0] and end == gold[1]:
                        result["mner_error_begin"] += 1
                        if pred_type != gold[2]:
                            result["errorboty_{}_pred_{}".format(pred_type.lower(), gold[2].lower())] += 1
                    elif begin != gold[0] and end != gold[1]:
                        result["mner_error_begin_end"] += 1
                        if pred_type != gold[2]:
                            result["errorboty_{}_pred_{}".format(pred_type.lower(), gold[2].lower())] += 1
                    sample_count += 1
                else:
                    result["mner_not_aspect"] += 1
        result["mner_not_predict"] += (length-sample_count)

    for name in type_list:
        result["pre_{}".format(name)] = float(result["pred_hit_{}_count".format(name)])/float(result["pred_{}_count".format(name)] + SMALL_POSITIVE_CONST)
        result["rec_{}".format(name)] = float(result["pred_hit_{}_count".format(name)])/float(result["gold_{}_count".format(name)] + SMALL_POSITIVE_CONST)
        result["f1_{}".format(name)] = 2*result["pre_{}".format(name)]*result["rec_{}".format(name)]/(result["pre_{}".format(name)]+result["rec_{}".format(name)]+SMALL_POSITIVE_CONST)
    
    # 求和
    result["mner_sum_gold_count"] = 0
    result["mner_sum_pred_count"] = 0
    result["mner_sum_pred_hit_count"] = 0
    for name in type_list:
        result["mner_sum_gold_count"] += result["gold_{}_count".format(name)]
        result["mner_sum_pred_count"] += result["pred_{}_count".format(name)]
        result["mner_sum_pred_hit_count"] += result["pred_hit_{}_count".format(name)]

    # 边界正确，但分类错误
    result["mner_sum_errorty"] = 0
    # 边界错误
    result["mner_sum_errorbo"] = result["mner_error_end"] + result["mner_error_begin"] + result["mner_error_begin_end"]
    # 边界错误，分类也错（是上面边界错误的子集）
    result["mner_sum_errorboty"] = 0
    for first in type_list:
        for second in type_list:
            if first != second:
                result["mner_sum_errorty"] += result["errorty_{}_pred_{}".format(first, second)]
                result["mner_sum_errorboty"] += result["errorboty_{}_pred_{}".format(first, second)]
    # 预测结果中的错误
    result["mner_pred_errorty_percent"] = result["mner_sum_errorty"]/(result["mner_sum_pred_count"]-result["mner_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    result["mner_pred_errorbo_percent"] = result["mner_sum_errorbo"]/(result["mner_sum_pred_count"]-result["mner_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    result["mner_pred_errorboty_percent"] = result["mner_sum_errorboty"]/(result["mner_sum_pred_count"]-result["mner_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    result["mner_pred_notaspect_percent"] = result["mner_not_aspect"]/(result["mner_sum_pred_count"]-result["mner_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)

    # 没有被召回的原因
    result["mner_recall_errorty_percent"] = result["mner_sum_errorty"]/(result["mner_sum_gold_count"]-result["mner_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    result["mner_recall_errorbo_percent"] = result["mner_sum_errorbo"]/(result["mner_sum_gold_count"]-result["mner_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    result["mner_recall_errorboty_percent"] = result["mner_sum_errorboty"]/(result["mner_sum_gold_count"]-result["mner_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    result["mner_recall_notpredict_percent"] = result["mner_not_predict"]/(result["mner_sum_gold_count"]-result["mner_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    
    # 计算category F1
    result["category_precent"] = (result["mner_sum_pred_count"]-result["mner_sum_errorty"]-result["mner_not_aspect"]-result["mner_sum_errorboty"])/(result["mner_sum_pred_count"]+SMALL_POSITIVE_CONST)
    result["category_recall"] = (result["mner_sum_pred_count"]-result["mner_sum_errorty"]-result["mner_not_aspect"]-result["mner_sum_errorboty"])/(result["mner_sum_gold_count"]+SMALL_POSITIVE_CONST)
    result["category_f1"] = 2 * result["category_precent"] * result["category_recall"]/(result["category_precent"] + result["category_recall"] + +SMALL_POSITIVE_CONST)
    
    
    return result




def result_analysis_bio(pd_data):
    # pd_predict = pd.DataFrame(columns=["mner_gold_words", "mner_pred_words", "mner_gold_sequence", "mner_pred_sequence", "bio_gold_words", "bio_pred_words", "bio_gold_sequence", "bio_pred_sequence", "text","image_id"])
    # absa_label_vocab = {'B-PER':0, 'I-PER':1, 'B-LOC':2, 'I-LOC':3, 'B-ORG':4, 'I-ORG':5, 'B-MISC':6, 'I-MISC':7, 'O':8}
    # calculate four type accuracy
    SMALL_POSITIVE_CONST = 1e-10
    type_list = ["aspect"]
    result = {}
    for first in type_list:
        for second in type_list:
            if first != second:
                result["errorty_{}_pred_{}".format(first, second)] = 0
                result["errorboty_{}_pred_{}".format(first, second)] = 0 # 在边界预测错误的同时，类别也预测错了
    for name in type_list:
        result["gold_{}_count".format(name)] = 0
        result["pred_{}_count".format(name)] = 0
        result["pred_hit_{}_count".format(name)] = 0
    
    result["bio_error_end"] = 0
    result["bio_error_begin"] = 0
    result["bio_error_begin_end"] = 0
    result["bio_not_aspect"] = 0
    result["bio_not_predict"] = 0

    for i in range(len(pd_data)):
        gold_seq = pd_data.loc[i, "bio_gold_sequence"]
        pred_seq = pd_data.loc[i, "bio_pred_sequence"]
        for seq in gold_seq:
            result["gold_{}_count".format(seq[2].lower())] += 1

        for seq in pred_seq:
            result["pred_{}_count".format(seq[2].lower())] += 1
        length = len(gold_seq)
        sample_count = 0
        for seq in pred_seq:
            if seq in gold_seq:
                result["pred_hit_{}_count".format(seq[2].lower())] += 1
                sample_count += 1
            else:
                begin,end,pred_type = seq[0], seq[1], seq[2]
                flag = 0
                for gold in gold_seq:
                    if begin <= gold[0] and end>= gold[0]:
                        flag = 1
                        break
                if flag == 1:
                    # 边界正确，分类错误
                    if begin == gold[0] and end == gold[1]:
                        result["errorty_{}_pred_{}".format(pred_type.lower(), gold[2].lower())] += 1
                    # 三类边界错误
                    elif begin == gold[0] and end != gold[1]:
                        result["bio_error_end"] += 1
                        if pred_type != gold[2]:
                            result["errorboty_{}_pred_{}".format(pred_type.lower(), gold[2].lower())] += 1
                    elif begin != gold[0] and end == gold[1]:
                        result["bio_error_begin"] += 1
                        if pred_type != gold[2]:
                            result["errorboty_{}_pred_{}".format(pred_type.lower(), gold[2].lower())] += 1
                    elif begin != gold[0] and end != gold[1]:
                        result["bio_error_begin_end"] += 1
                        if pred_type != gold[2]:
                            result["errorboty_{}_pred_{}".format(pred_type.lower(), gold[2].lower())] += 1
                    sample_count += 1
                else:
                    result["bio_not_aspect"] += 1
        result["bio_not_predict"] += (length-sample_count)

    for name in type_list:
        result["pre_{}".format(name)] = float(result["pred_hit_{}_count".format(name)])/float(result["pred_{}_count".format(name)] + SMALL_POSITIVE_CONST)
        result["rec_{}".format(name)] = float(result["pred_hit_{}_count".format(name)])/float(result["gold_{}_count".format(name)] + SMALL_POSITIVE_CONST)
        result["f1_{}".format(name)] = 2*result["pre_{}".format(name)]*result["rec_{}".format(name)]/(result["pre_{}".format(name)]+result["rec_{}".format(name)]+SMALL_POSITIVE_CONST)
    
    # 求和
    result["bio_sum_gold_count"] = 0
    result["bio_sum_pred_count"] = 0
    result["bio_sum_pred_hit_count"] = 0
    for name in type_list:
        result["bio_sum_gold_count"] += result["gold_{}_count".format(name)]
        result["bio_sum_pred_count"] += result["pred_{}_count".format(name)]
        result["bio_sum_pred_hit_count"] += result["pred_hit_{}_count".format(name)]

    # 边界正确，但分类错误
    result["bio_sum_errorty"] = 0
    # 边界错误
    result["bio_sum_errorbo"] = result["bio_error_end"] + result["bio_error_begin"] + result["bio_error_begin_end"]
    # 边界错误，分类也错（是上面边界错误的子集）
    result["bio_sum_errorboty"] = 0
    for first in type_list:
        for second in type_list:
            if first != second:
                result["bio_sum_errorty"] += result["errorty_{}_pred_{}".format(first, second)]
                result["bio_sum_errorboty"] += result["errorboty_{}_pred_{}".format(first, second)]
    # 预测结果中的错误
    result["bio_pred_errorty_percent"] = result["bio_sum_errorty"]/(result["bio_sum_pred_count"]-result["bio_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    result["bio_pred_errorbo_percent"] = result["bio_sum_errorbo"]/(result["bio_sum_pred_count"]-result["bio_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    result["bio_pred_errorboty_percent"] = result["bio_sum_errorboty"]/(result["bio_sum_pred_count"]-result["bio_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    result["bio_pred_notaspect_percent"] = result["bio_not_aspect"]/(result["bio_sum_pred_count"]-result["bio_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)

    # 没有被召回的原因
    result["bio_recall_errorty_percent"] = result["bio_sum_errorty"]/(result["bio_sum_gold_count"]-result["bio_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    result["bio_recall_errorbo_percent"] = result["bio_sum_errorbo"]/(result["bio_sum_gold_count"]-result["bio_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    result["bio_recall_errorboty_percent"] = result["bio_sum_errorboty"]/(result["bio_sum_gold_count"]-result["bio_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    result["bio_recall_notpredict_percent"] = result["bio_not_predict"]/(result["bio_sum_gold_count"]-result["bio_sum_pred_hit_count"]+SMALL_POSITIVE_CONST)
    
    return result
     
     

       