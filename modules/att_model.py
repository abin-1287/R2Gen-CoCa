# absolute_import确保绝对导入的语义被使用
from __future__ import absolute_import
# division确保除法操作的默认行为是浮点除法
from __future__ import division
# print_function确保print函数的使用方式与Python 3的行为一致
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

import modules.utils as utils
from modules.caption_model import CaptionModel

# 对输入的序列数据进行排序，并进行打包操作以便于后续处理
def sort_pack_padded_sequence(input, lengths):
    # 使用torch.sort函数对lengths进行降序排列，并返回排序后的长度数组和对应的索引数组。
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    # 使用input和排序后的长度数组构造一个打包后的序列数据tmp，并设置batch_first=True表示数据的第一个维度是批次大小。
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    # 创建一个与indices相同的索引数组inv_ix，然后根据排序后的索引数组重新排列inv_ix，以便与打包后的序列数据对应。
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix

# 对打包后的序列数据进行填充和排序还原
def pad_unsort_packed_sequence(input, inv_ix):
    # 使用pad_packed_sequence函数对打包后的序列数据input进行填充操作，将其还原为原始的序列数据tmp，并设置batch_first=True表示数据的第一个维度是批次大小。
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    # 根据重新排列后的索引数组inv_ix，对填充后的序列数据tmp进行排序还原，得到最终结果。
    tmp = tmp[inv_ix]
    return tmp

# 对给定的模块进行打包操作，并根据需要进行填充和排序还原。
def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        # 使用sort_pack_padded_sequence函数对att_feats和att_masks进行打包和排序操作，得到打包后的序列数据packed和重新排列后的索引数组inv_ix。
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        # 使用pad_unsort_packed_sequence函数对处理后的序列数据和附加信息进行填充和排序还原，得到最终结果。
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        # 使用module对打包后的序列数据的第一个元素进行处理，并使用packed[1]作为附加信息创建一个PackedSequence对象。
        return module(att_feats)


class AttModel(CaptionModel):
    def __init__(self, args, tokenizer):
        super(AttModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.idx2token) # 词汇表的大小，由分词器中的标记数量决定。
        self.input_encoding_size = args.d_model # 输入编码的大小（d_model）
        self.rnn_size = args.d_ff # RNN层的大小（d_ff）
        self.num_layers = args.num_layers # RNN层的数量
        self.drop_prob_lm = args.drop_prob_lm # 在语言模型中使用的丢弃概率
        self.max_seq_length = args.max_seq_length # 生成的字幕的最大序列长度
        self.att_feat_size = args.d_vf # 注意力特征的大小（d_vf）
        self.att_hid_size = args.d_model # 注意力隐藏层的大小（d_model）

        self.bos_idx = args.bos_idx # 序列的起始标记索引
        self.eos_idx = args.eos_idx # 序列的结束标记索引
        self.pad_idx = args.pad_idx # 序列的填充标记索引

        self.use_bn = args.use_bn # 是否使用批归一化（batch normalization）

        self.embed = lambda x: x # 接受输入x并返回x本身，即不对输入进行任何嵌入操作
        self.fc_embed = lambda x: x # 与embed相似
#       是一个nn.Sequential模块，它由一系列层组成，包括：
#           可选的批归一化层（如果use_bn为真）。
#           线性层，将注意力特征的大小（att_feat_size）映射到输入编码的大小（input_encoding_size）。
#           ReLU激活函数。
#           Dropout层，以drop_prob_lm的概率随机丢弃输入。
#           可选的批归一化层（如果use_bn等于2）。
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.input_encoding_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))

    # 裁剪att_feats（注意力特征）和att_masks（注意力掩码）的长度，使其不超过最大长度
    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    # 用于准备输入特征
    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        # 调用clip_att函数来裁剪att_feats和att_masks的长度
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        # 对fc_feats（全连接特征）进行嵌入操作
        fc_feats = self.fc_embed(fc_feats)
        # 对att_feats（注意力特征）进行封装操作
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # 将注意力特征投影为p_att_feats，以减少内存和计算开销
        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    # 用于获取生成序列的对数概率和当前状态
    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    
    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10) # 获取束搜索的束大小，默认为10
        group_size = opt.get('group_size', 1) # 获取每个样本生成的候选序列数量，默认为1
        sample_n = opt.get('sample_n', 10) # 获取每个时间步保留的样本数量，默认为10
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0) # 批次大小

        # 准备特征
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        # 保存生成序列的张量，初始化为 pad_idx
        seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
        # 保存序列的对数概率的张量，初始化为0
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity

        # 保存每个样本的完成的束搜索结果的列表
        self.done_beams = [[] for _ in range(batch_size)]

        # 初始化隐藏状态
        state = self.init_hidden(batch_size)

        # first step, feed bos !!
        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        # 创建包含开始符号的张量
        logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
        # 复制特征以匹配束搜索的候选序列数量
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
                                                                                  [p_fc_feats, p_att_feats,
                                                                                   pp_att_feats, p_att_masks]
                                                                                  )
         # 执行束搜索
        self.done_beams = self.beam_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt)
        for k in range(batch_size):
            if sample_n == beam_size:
                # 如果 sample_n 等于 beam_size，每个束是一个样本
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    # 将每个束的序列填充到 seq 张量中
                    seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
                    # 将每个束的对数概率填充到 seqLogprobs 张量中
                    seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                # 否则选择每个样本中得分最高的束的序列和对数概率进行填充
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                # 将得分最高的束的序列填充到 seq 张量中
                seq[k, :seq_len] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                # 将得分最高的束的对数概率填充到 seqLogprobs 张量中
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs

    def _sample(self, fc_feats, att_feats, att_masks=None):
        opt = self.args.__dict__ # 获取参数选项
        sample_method = opt.get('sample_method', 'greedy') # 获取采样方法，默认为贪婪采样
        beam_size = opt.get('beam_size', 1) # 获取束搜索的束大小，默认为1
        temperature = opt.get('temperature', 1.0) # 获取温度参数，默认为1.0
        sample_n = int(opt.get('sample_n', 1)) # 获取每个时间步停留的样本数量，默认为1
        group_size = opt.get('group_size', 1) # 获取每个样本生成的候选序列数量，默认为1
        output_logsoftmax = opt.get('output_logsoftmax', 1) # 获取是否输出对数softmax，默认为1
        decoding_constraint = opt.get('decoding_constraint', 0) # 获取解码约束，默认为0
        block_trigrams = opt.get('block_trigrams', 0) # 获取是否阻塞三元组，默认为0
        # 如果使用束搜索且采样方法为贪婪或束搜索，则调用束搜索采样函数
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,
                                                                                      [p_fc_feats, p_att_feats,
                                                                                       pp_att_feats, p_att_masks]
                                                                                      )

        trigrams = []  # will be a list of batch_size dictionaries

        seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size + 1)
        for t in range(self.max_seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state,
                                                      output_logsoftmax=output_logsoftmax)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t - 3:t - 1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:  # initialize
                        trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t - 2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.max_seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx  # This allows eos_idx not being overwritten to 0
                logprobs = logprobs * unfinished.unsqueeze(1).float()
                unfinished = unfinished * (it != self.eos_idx)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _diverse_sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams_table = [[] for _ in range(group_size)]  # will be a list of batch_size dictionaries

        seq_table = [fc_feats.new_full((batch_size, self.max_seq_length), self.pad_idx, dtype=torch.long) for _ in
                     range(group_size)]
        seqLogprobs_table = [fc_feats.new_zeros(batch_size, self.max_seq_length) for _ in range(group_size)]
        state_table = [self.init_hidden(batch_size) for _ in range(group_size)]

        for tt in range(self.max_seq_length + group_size):
            for divm in range(group_size):
                t = tt - divm
                seq = seq_table[divm]
                seqLogprobs = seqLogprobs_table[divm]
                trigrams = trigrams_table[divm]
                if t >= 0 and t <= self.max_seq_length - 1:
                    if t == 0:  # input <bos>
                        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
                    else:
                        it = seq[:, t - 1]  # changed

                    logprobs, state_table[divm] = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats,
                                                                          p_att_masks, state_table[divm])  # changed
                    logprobs = F.log_softmax(logprobs / temperature, dim=-1)

                    # Add diversity
                    if divm > 0:
                        unaug_logprobs = logprobs.clone()
                        for prev_choice in range(divm):
                            prev_decisions = seq_table[prev_choice][:, t]
                            logprobs[:, prev_decisions] = logprobs[:, prev_decisions] - diversity_lambda

                    if decoding_constraint and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                        logprobs = logprobs + tmp

                    # Mess with trigrams
                    if block_trigrams and t >= 3:
                        # Store trigram generated at last step
                        prev_two_batch = seq[:, t - 3:t - 1]
                        for i in range(batch_size):  # = seq.size(0)
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            current = seq[i][t - 1]
                            if t == 3:  # initialize
                                trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                            elif t > 3:
                                if prev_two in trigrams[i]:  # add to list
                                    trigrams[i][prev_two].append(current)
                                else:  # create list
                                    trigrams[i][prev_two] = [current]
                        # Block used trigrams at next step
                        prev_two_batch = seq[:, t - 2:t]
                        mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                        for i in range(batch_size):
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            if prev_two in trigrams[i]:
                                for j in trigrams[i][prev_two]:
                                    mask[i, j] += 1
                        # Apply mask to log probs
                        # logprobs = logprobs - (mask * 1e9)
                        alpha = 2.0  # = 4
                        logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

                    it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, 1)

                    # stop when all finished
                    if t == 0:
                        unfinished = it != self.eos_idx
                    else:
                        unfinished = seq[:, t - 1] != self.pad_idx & seq[:, t - 1] != self.eos_idx
                        it[~unfinished] = self.pad_idx
                        unfinished = unfinished & (it != self.eos_idx)  # changed
                    seq[:, t] = it
                    seqLogprobs[:, t] = sampleLogprobs.view(-1)

        return torch.stack(seq_table, 1).reshape(batch_size * group_size, -1), torch.stack(seqLogprobs_table,
                                                                                           1).reshape(
            batch_size * group_size, -1)
