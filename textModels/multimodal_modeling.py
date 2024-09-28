
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import torch
from torch import nn
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 2})
import seaborn as sns
sns.set()


class ADDBertEncoder(nn.Module):
    def __init__(self, config):
        super(ADDBertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new":gelu_new}

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads  # num_attention_heads: 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # hidden_states: self.batch_size, 305, hidden_dim
    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask # attention_mask.shape: self.batch_size, 1, 1, 305

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)]) # self.layer just have one layer:
                                                            # BertLayer:BertAttention, BertIntermediate, BertOutput

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # hidden_states.shape = batch_size, img_len or text_len, hidden_size
        # first_token_tensor = batch_size, hidden_size
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # just change attention_probs name, which is easy to return attention_probs
        # attention_probs = self.dropout(attention_probs)
        attention_probs_dropout = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs_dropout, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_weights = torch.sum(attention_probs, dim=1)/self.num_attention_heads
        return context_layer, attention_weights




class TextOutput(nn.Module):
    def __init__(self, config):
        super(TextOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def bio2bioes(ts_tag_sequence):
    # bio_labels: {0:B, 1:I, 2:O}
    bio_label_map = {0:"B", 1:"I", 2:"O", -100:"No"}
    bioes_label_map = {"B":0, "I":1, "O":2, "E":3, "S":4}
    n_tags = len(ts_tag_sequence)
    new_ts_sequence = torch.ones(n_tags, dtype=torch.int64, device="cuda:0") * -1
    for i in range(n_tags):
        cur_ts_tag = ts_tag_sequence[i]
        # if cur_ts_tag == 'O' or cur_ts_tag == 'No':
        if cur_ts_tag == 2 or cur_ts_tag == -100:
            new_ts_sequence[i] = 2
        else:
            if cur_ts_tag == 0:
                if (i == n_tags - 1) or (ts_tag_sequence[i+1] != 1):
                    new_ts_sequence[i] = 4
                else:
                    new_ts_sequence[i] = 0
            elif cur_ts_tag == 1:
                if i == n_tags - 1:
                    if ts_tag_sequence[i - 1] == 2:
                        new_ts_sequence[i] = 1
                    else:
                        new_ts_sequence[i] = 3
                elif i == 0:
                    new_ts_sequence[i] = 1
                else:
                    if ts_tag_sequence[i - 1] != 2:
                        if ts_tag_sequence[i + 1] != cur_ts_tag:
                            new_ts_sequence[i] = 3
                        else:
                            new_ts_sequence[i] = 1
                    else:
                        new_ts_sequence[i] = 1
    assert n_tags == len(new_ts_sequence)
    return new_ts_sequence


class MFD(nn.Module):
    def __init__(self, args, text_config, image_config, text_model, image_model, tokenizer):
        super(MFD, self).__init__()
        self.args = args
        self.text_model = text_model
        self.image_model = image_model
        self.config = text_config
        self.tokenizer = tokenizer
        self.linear_projection = nn.Sequential(
            nn.Linear(image_config.hidden_size, text_config.hidden_size),
            nn.GELU(),
            nn.Dropout(args.text_hidden_dropout),
            nn.Linear(text_config.hidden_size, text_config.hidden_size),
            nn.GELU(),
            nn.Dropout(args.text_hidden_dropout),
        )
        self.position_embeddings = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros(8, image_config.hidden_size), mean=0.0, std=image_config.initializer_range)
        )
        self.text2img_attention = BertCoAttention(config=text_config)
        self.text2img_attention_ner = BertCoAttention(config=text_config)
        self.text_cls = TextOutput(config=text_config)
        self.text_bio = TextOutput(config=text_config)
        self.text_ner = TextOutput(config=text_config)
        self.ner_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        self.class_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        self.transformer = MultimodalEncoder(config=text_config, layer_number=args.transformer_layer_number)
        if self.args.boundary == "BIO":
            self.bound_label_num = 3
            self.bio_label_embedding = nn.Parameter(torch.empty(self.bound_label_num+1, text_config.hidden_size),requires_grad=True)
            self._init_weights_boundary(self.bio_label_embedding, "BIO")
            self.bio_classifier = nn.Linear(text_config.hidden_size, self.bound_label_num)
        elif self.args.boundary == "BIOES":
            self.bound_label_num = 5
            self.bio_label_embedding = nn.Parameter(torch.empty(self.bound_label_num+1, text_config.hidden_size),requires_grad=True)
            self._init_weights_boundary(self.bio_label_embedding, "BIOES")
            self.bio_classifier = nn.Linear(text_config.hidden_size, self.bound_label_num)
        
        self.class_label_num = 5
        self.class_label_embedding = nn.Parameter(torch.empty(self.class_label_num+1, text_config.hidden_size),requires_grad=True)
        self.class_classifier = nn.Linear(text_config.hidden_size, self.class_label_num)
        self._init_weights_category(self.class_label_embedding)
        self.ner_classifier = nn.Linear(text_config.hidden_size, text_config.num_labels)
        
        init_list = [self.text_cls, self.text_bio, self.text_ner, self.class_classifier, self.bio_classifier, self.ner_classifier, 
                     self.linear_projection, self.text2img_attention, self.text2img_attention_ner, self.transformer]

        for layer in init_list:
            self._init_loop_weights(layers=layer)
        
    def _init_loop_weights(self, layers):
        for key in layers._modules:
            self._init_weights(layers._modules[key])

    def _init_weights_category(self, module):
        """Initialize category the weights"""
        # return ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']
        per_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(['person']))
        loc_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(['location']))
        org_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(['organization']))
        misc_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(['other']))
        outside_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(['outside']))
        none_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(['none']))
        module.data[0] = self.text_model.embeddings.word_embeddings(per_id)
        module.data[1] = self.text_model.embeddings.word_embeddings(loc_id)
        module.data[2] = self.text_model.embeddings.word_embeddings(org_id)
        module.data[3] = self.text_model.embeddings.word_embeddings(misc_id)
        module.data[4] = self.text_model.embeddings.word_embeddings(outside_id)
        module.data[5] = self.text_model.embeddings.word_embeddings(none_id)

    def _init_weights_boundary(self, module, labeling):
        """Initialize boundary the weights"""
        begin_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(['begin']))
        inside_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(['inside']))
        outside_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(['outside']))
        none_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(['none'])) #-100
        if labeling == "BIOES":
            end_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(['end']))
            single_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(['single']))
            module.data[0] = self.text_model.embeddings.word_embeddings(begin_id)
            module.data[1] = self.text_model.embeddings.word_embeddings(inside_id)
            module.data[2] = self.text_model.embeddings.word_embeddings(outside_id)
            module.data[3] = self.text_model.embeddings.word_embeddings(end_id)
            module.data[4] = self.text_model.embeddings.word_embeddings(single_id)
            module.data[5] = self.text_model.embeddings.word_embeddings(none_id)
        elif labeling == "BIO":
            module.data[0] = self.text_model.embeddings.word_embeddings(begin_id)
            module.data[1] = self.text_model.embeddings.word_embeddings(inside_id)
            module.data[2] = self.text_model.embeddings.word_embeddings(outside_id)
            module.data[3] = self.text_model.embeddings.word_embeddings(none_id)
            
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, img_feats, img_class_ids, img_mask, img_dis_position, is_train):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        text_embeds = text_output.last_hidden_state
        image_embeds = img_feats
        image_mask = img_mask
        if self.args.dis_img_sort == True:
            position = copy.deepcopy(img_dis_position)
            image_embeds_dis = copy.deepcopy(img_feats)
            position[img_mask == False] = 0
            for i in range(len(img_feats)):
                image_embeds_dis[i] = img_feats[i][position[i]]
            image_embeds_dis[img_mask == False] = 0.0
            image_embeds = image_embeds_dis
        
        if self.args.use_position == True:
            position_id = torch.arange(img_feats.shape[1]).expand((1, -1))
            position_embeddings = self.position_embeddings[position_id]
            image_embeds += position_embeddings
        
        image_embeds = self.linear_projection(image_embeds)
        text_embeds = self.linear_projection(text_embeds)
        ### Cross Attention
        # text attention image
        extended_img_mask = image_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0
        extended_text_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_text_mask = extended_text_mask.to(dtype=next(self.parameters()).dtype)
        extended_text_mask = (1.0 - extended_text_mask) * -10000.0
        text2img_cross_encoder, text2img_attention_weights = self.text2img_attention(text_embeds,image_embeds,extended_img_mask)
        ### category
        # a*Text2img + Text -> category
        class_labels = copy.deepcopy(labels)
        max_label = class_labels.max() 
        class_labels[abs(labels)<max_label] = class_labels[abs(labels)<max_label]//2
        class_labels[labels==max_label] = self.class_label_num - 1
        CE_criterion_cls = nn.CrossEntropyLoss()
        text_embeds_cls = self.text_cls(text_embeds)
        class_cross_encoder = self.class_weight*text2img_cross_encoder + text_embeds_cls
        logits_class = self.class_classifier(class_cross_encoder)
        output_class = logits_class
        loss_class = CE_criterion_cls(output_class.view(-1, self.class_label_num), class_labels.view(-1))
        ### boundary
        # Text -> boundary
        bio_labels = copy.deepcopy(labels)
        # bio_labels: {0:B, 1:I, 2:O}
        max_label = bio_labels.max()
        bio_labels[abs(labels)<max_label] = bio_labels[abs(labels)<max_label]%2
        bio_labels[labels==max_label] = 2
        
        if self.args.boundary == "BIOES":
            for i in range(len(bio_labels)):
                bio_labels[i] = bio2bioes(bio_labels[i])
            bio_labels[attention_mask==0] = -100
        
        CE_criterion_bio = nn.CrossEntropyLoss()
        text_embeds_bio = self.text_bio(text_embeds)
        bio_cross_encoder = text_embeds_bio
        logits_bio = self.bio_classifier(bio_cross_encoder)
        output_bio = logits_bio
        loss_bio = CE_criterion_bio(output_bio.view(-1, self.bound_label_num), bio_labels.view(-1))
        if self.args.use_diff_text2img:
            text2img_cross_encoder_mner, text2img_attention_weights = self.text2img_attention_ner(text_embeds,image_embeds,extended_img_mask)
        else:
            text2img_cross_encoder_mner = text2img_cross_encoder
        if is_train:
            tmp_bio_labels = copy.deepcopy(bio_labels)
            tmp_bio_labels[attention_mask==0] = self.bound_label_num
            bio_embedding = self.bio_label_embedding[tmp_bio_labels]
            
            tmp_class_labels = copy.deepcopy(class_labels)
            tmp_class_labels[attention_mask==0] = self.class_label_num
            class_embedding = self.class_label_embedding[tmp_class_labels]
            
            text_embeds_ner = self.text_ner(text_embeds)
            ner_text_embeds = self.ner_weight*text2img_cross_encoder_mner + text_embeds_ner + bio_embedding + class_embedding
            ## transformer
            ner_text_embeds = self.transformer(ner_text_embeds, extended_text_mask)[0]
            logits = self.ner_classifier(ner_text_embeds)
        else:
            tmp_output_bio = copy.deepcopy(output_bio)
            tmp_output_bio = torch.argmax(tmp_output_bio, dim=2)
            tmp_output_bio[attention_mask==0] = self.bound_label_num
            bio_embedding = self.bio_label_embedding[tmp_output_bio]
            
            tmp_output_class = copy.deepcopy(output_class)
            tmp_output_class = torch.argmax(tmp_output_class, dim=2)
            tmp_output_class[attention_mask==0] = self.class_label_num
            class_embedding = self.class_label_embedding[tmp_output_class]

            text_embeds_ner = self.text_ner(text_embeds)
            ner_text_embeds = self.ner_weight*text2img_cross_encoder_mner + text_embeds_ner + bio_embedding + class_embedding
            ## transformer
            ner_text_embeds = self.transformer(ner_text_embeds, extended_text_mask)[0]
            logits = self.ner_classifier(ner_text_embeds)
            
        return logits, loss_bio, loss_class


