import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
from torch.utils.tensorboard import SummaryWriter
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import pudb

# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")

# TensorBoard看graph意义真的不大，还不如print(nn.Module)看的清楚
tb_writer = SummaryWriter()


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        # 
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    # tgt的长度是逐渐增长的，因为一个个输出的词汇会往里面append
    # tgt: [bs, 已生成词汇数量（包含最开始的start_symbol）]
    # self.tgt_embed(tgt): [bs, tgt.size(1), 512]
    # memory: [bs, 128(句长), 512]
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # 512 -> vocab size
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 
        return log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        # 
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 
        # x: [1, sl, 512]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # output: [bs, sl, 512]
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        # 
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # 
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        # 
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # 
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        # 
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        # x仅包含目前已经生成的长度，所以比memory的size小很多
        # 比如x刚开始只有一个start_symbol，shape是: [bs, 1, 512], 而memory: [bs, sl, 512]
        # pudb.set_trace()
        m = memory  # memory来自于encoder的输出
        # 第一步是self attention，也就是已经decode输出的部分自我之间的attention
        # 注意self attention的输入的shape，不要求是大小都固定，比如都为[bs, sl, 512], 可以是[bs, 任意大小, 512]
        # 因为attention的Q*K^T的结果（我叫做attention matrix），是可以大也可以小的，因为query和key的shape大小可以不同！
        #   - 当然key和value的shape必须相同！(必须都是一样的行数，也就是表达一样个数的words)
        #   - query代表的words数量可以<=key,value代表的word数量
        # 是的，attention matrix从来没有说必须是方阵！可以是任何长宽
        # attention matrix的第i行表示第query中第i个word应该怎么受到key和value代表的所有的word的线性组合来影响（也就是叫做被attend）
        # 所以这样就很容易理解：
        #   1. 当query的size小于key,value的size的情况。比如query就只有一行，key value是k行，那么attention matrix的大小也是1行k列的，
        #      代表query这一个word应该由这key,value的k个word做这样的线性组合来attend
        #   2. mask的使用：
        #      - 如果是self attend，那么attention matrix是方阵。mask应该为左下一半都是1（包含对角线）的方阵，意思表示每一个word都只能被
        #      - 它自身和前面的word来attend，这样是符合decoder自回归的使用方式（也就是只有encoder会提前看到后面的word，而decoder只能看到自己
        #        已经生成的word，看不到自己未来即将生成的word（这不废话吗））
        #        你可以让attention matrix * 这个mask矩阵，结果就是attention matrix右上部分(不包括对角线)都变成-1e9,这样当attention matrix
        #        对每列进行softmax操作的时候，被mask的部分就会得到一个基本上等于0的值，也就是清除了对应word参加attend的影响
        #      - decoder输出的query和encoder的key&value attend（也叫做cross attend），那么mask并不一定是方阵，但应该是一个左边若干列为1，右边
        #        若干列为0的样式，然后我们用mask_fill来把attention matrix中mask为0的位置写上-1e9, 这样的目的是只让key&value句子当中前面几个word
        #        可以参加attend，因为句子后面的word都是placeholder而已（为0），是没有真正的word的
        #        实际操作的时候，因为tensor.mask_fill的输入mask只要可以broadcast到tensor就可以了，所以实际上这里的mask shape并不是attention matrix
        #        的shape([bs,8,sl,sl])，而是[bs,1,1,sl]，也就是最后一维中下标i表示第i个word是placeholder，应该忽略
        #        所以，这里面代码，如果是self attention, query和key&value都代表是k个word, 那么，mask是[bs,1,query_word_count,query_word_count], attention matrix
        #        shape为[bs,8,query_word_count,query_word_count]
        #        如果是cross attention，mask是[bs,1,1,sl], attention matrix shape为[bs,8,query_word_count,sl/8]
        #        这两种shape都可以broadcast到各自的attention matrix. 前者相当于是每个attention matrix（即最后两维）每个行列都被[query_word_count, query_word_count]的mask处理了一遍，
        #        也就是一种2维mask处理（三角形mask）
        #        后者相当于是每个attention matrix的每个最后一维都被一个[sl]的mask处理了一遍，也就是是一个一维mask，每行的mask处理都完全一样
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # self attention on decoder's output
        # x经常是小于m的size的，因为x(query)代表的word数量要小于m(key&value)代表的word数量
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))  # cross attention between encoder's output and decoder's output
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # 64
    d_k = query.size(-1)
    # [bs, 8, sl, sl]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        print(f'scores shape: {scores.shape}, mask shape: {mask.shape}')
        scores = scores.masked_fill(mask == 0, -1e9)
    # [bs, 8, sl, sl]
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: [bs, 8, sl, sl]
    # value: [bs, 8, sl, 64]
    # matmul结果: [bs, 8, sl, 64]
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # 
        assert d_model % h == 0
        # We assume d_v always equals d_k
        # self.d_k: 64
        self.d_k = d_model // h
        # self.h: 8
        self.h = h
        # d_model: 512
        # Each linear layer: input 512, output 512
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    # query, key, value: [bs, sl, 512]
    # 但注意在做cross attention（decoding阶段）时，有可能query（从decoder出来，大小可能还很小因为才刚生成很少的word）
    # 的size要小于key和value的size，比如
    # query: [bs, 1, 512]
    # key: [bs, sl, 512]
    # value: [bs, sl, 512]
    def forward(self, query, key, value, mask=None):
        # pudb.set_trace()
        # 
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        # nbatches: 1
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value from [bs, sl, 512],
        # for each,
        # after linear [bs, sl, 512]
        # after view (reshape): [bs, sl, 8, 64]
        # after transpose (switch sl dim (index1) and h dim(index 2), updated to : [bs, 8, sl, 64]
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))  # 注意这里zip出了3个pair，也就是说只用了前三个512x512 linear layer
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # x: [bs, 8, sl, 64]：这个是attention输出
        # self.attn: [bs, 8, sl, sl]: 这个叫做attention matrix（这里返回没啥用）
        # 在最后两个dim (sl, 64) 中做attention。这下等于做了8份独立的self attention
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        # 把8个独立attention的结果，拼成一起，最终拼回到[bs, sl, 512]，也就是和最早的输入大小完全一致
        # transpose只是为了操作方便，并不影响model的学习能力
        # 我也承认这里的做法并不是唯一可行的做法
        x = (
            x.transpose(1, 2)  # x from [bs, 8, sl, 64] to [bs, sl, 8, 64]
            .contiguous()  # make a deep copy of Tensor
            .view(nbatches, -1, self.h * self.d_k)  # reshape into [bs, sl, 512]
        )
        del query
        del key
        del value
        # output: [bs, sl, 512]
        return self.linears[-1](x)  # 这里用上了第四个512x512的linear layer


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    # d_model: 512
    # d_ff: 2048
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # 
        # in: 512, out: 2048
        self.w_1 = nn.Linear(d_model, d_ff)
        # in: 2048, out: 512
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 
        # src词汇表或者tgt词汇表的大小 -> 512
        self.lut = nn.Embedding(vocab, d_model)
        # nn.Embedding进行forward的时候，
        # 输入shape是[batch_size, 句子长度(固定，内容不够加mask)]，每个值都为单词的id num（也包括填充的dummy id）
        # 输出shape是[batch_size, 句子长度(同上), embedding_dim size]，每个值都为embedding的float值
        # 所以nn.Embedding自己内部其实做了one-hot encoding以及linear forward等操作
        self.d_model = d_model

    # output: [bs, sl, 512]
    def forward(self, x):
        # 
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    # d_model: 512
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 

        # Compute the positional encodings once in log space.
        # pe: [5000, 512]
        pe = torch.zeros(max_len, d_model)
        # [5000, 1]
        position = torch.arange(0, max_len).unsqueeze(1)
        # [512]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # becomes [bs, 5000, 512]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    # output: [bs, sl, 512]
    def forward(self, x):
        # 
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(
        src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    # src_vocab = 8315
    # tgt_vocab = 6384
    # pudb.set_trace()
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    # 
    attn = MultiHeadedAttention(h, d_model)
    # 
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 
    position = PositionalEncoding(d_model, dropout)
    # 
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # pudb.set_trace()
    return model


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
        data_iter,
        model,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    # pudb.set_trace()
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        # 
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
                self.criterion(
                    x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
                )
                / norm
        )
        return sloss.data * norm, sloss


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # src: [bs, sl]
    # src_mask: [bs, 1, sl]
    # pudb.set_trace()
    # memory: [bs, sl, 512]
    memory = model.encode(src, src_mask)
    print(model.encoder)
    # pudb.set_trace()
    # tb_writer.add_graph(model.encoder, [model.src_embed(src), src_mask])
    # ys存的就是长度逐渐变长的prediction结果
    # 最开始ys只有一个句首的start_symbol(数字0)，然后先predict一个词（无中生有），放到ys，然后ys里面有两个word了
    # 然后ys
    # pudb.set_trace()
    # ys初始为[bs,1]. ys存的就是prediction输出结果
    # 过一个循环，ys变成[bs, 2], 再过一个循环变成[bs, 3]...
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        # TODO: out的shape好像有问题
        # ys_mask: [ys.size(1), ys.size(1)]，或者说是[ys的词数量,ys的词数量]
        ys_mask = subsequent_mask(ys.size(1))
        pudb.set_trace()
        # mask是一个size递增的方阵，其左下角（包括对角线）都是true，其余都是false。这个就是作为attention mask用，
        # 防止和还没有输出位置的词汇产生attention
        # src_mask: [bs, 1, sl]
        # out: [bs, ys词数量, 512]
        # 注意这个decode的输出有多少个word取决于其输入的ys有多少个word！
        out = model.decode(
            memory, src_mask, ys, ys_mask.type_as(src.data)
        )
        # 选输出的最后一个词的feature vector来放入generator，也就是说out[:,-1]的shape一定为[bs, 512]
        prob = model.generator(out[:, -1])
        # prob: [bs, 6384]
        _, next_word = torch.max(prob, dim=1)  # next_word: [1]
        next_word = next_word.data[0]  # 把next_word里面唯一的一个值拿出来
        ys = torch.cat(
            # ys最后一个dim的大小+1，也就是[bs,1]=>[bs,2], [bs,2]=>[bs,3]这样
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys
    # 注意，这里 有一个重大的提示。你看到，因为我们在tgt中，还是ys中，第一个词永远都是start_symbol(0)，所以这就造成
    # decoder里面self attention中，如果我们mask确实是左下角都为1的阵且对角线的值也为1，看似好像意思是说，
    # 第k行的词(k从1开始算)，是可以用1行~k行词加权出来的. 但其实因为tgt和ys都手工塞入了start_symbol为第一个词，
    # 所以第k行的词的attend过程，本质上最多用到前面k-1个词。而第k行的输出，直接会送给generator来告诉我最终的第k个输出(k从1算)
    # 所以，第一个词的inference（也就是ys中下标为1的那个词，下标为0为start_symbol），根本就没让decoder用上任何有意义的词汇(只用了start_symbol)
    #
    # 这样，如果我们在training阶段，直接把整个src和tgt都喂给encoder+decoder网络，等于说上来就给ys写满了，其长度为tgt.length
    # 这样大小的ys进去上面，就不用走循环了，而是一步就输出了tgt.length这么多行，也就是直接输出了tgt.length这么多word，一次inference就完成了
    # 整句话的输出。 但由于ys[0]是start_symbol，用上述说的左下角阵为mask，你会看到整个decoder都从来没有 **用第k个词去生成第k个词** 。
    # 第k个词的生成，decoder网络中能参与的只有它前面的tgt的词汇 (encoder当然是src的所有词汇)
    # 所以，看似好像tgt都给你了，让你去学target，好像可以抄近路，但由于每个词的输出，都必须是他前面的词的线性组合(attend)产生，所以这个位置的
    # gt的词，反而没法参与这个线性组合！所以这样做training才是不可能作弊的，能训练出来内容的。而且这样训练速度可以很快
    # 真心可以有大量并行训练！

    # 但是在使用model的时候，因为只有src，没有tgt，所以就不得不用ys从start_symbol开始，一次inference生成一个词了。所以实际使用时候并行能力
    # 要差不少。

    # 这里也是给你一个对于transformer的一个重要理解：你发现，整个transformer网络，只有在attention阶段，能让多个词的feature进行一定的交叉融合
    # 其他的所有地方，比如multihead attention后面的feed forward，还是最开始的word embedding，还是QKV生成的linear以及attention之后的linear
    # (恢复成512宽度，也就是生成QKV之外的第四个linear)，还是最后的generator，他们都是只有一个word自身的内部线性变换，然后变换完了，还是仅仅
    # 来表示自己。对比CNN网络，如果我们把pixel当成是word，这就相当于，整个网络，除了attention部分是有类似于conv2d的多pixel feature融合，其他
    # 的地方就tmd全都是1x1的卷积。仅有1x1卷积CNN网络结果，就相当于是没有attention的结果。所以convolution和attention是完全可以类比的东西


def load_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


spacy_de, spacy_en = load_tokenizers()


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)


def collate_batch(
        batch,
        src_pipeline,
        tgt_pipeline,
        src_vocab,
        tgt_vocab,
        device,
        max_padding=128,
        pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=12000,
        max_padding=128,
        is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


def train_worker(
        gpu,
        ngpus_per_node,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        config,
        is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        pudb.set_trace()
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    from the_annotated_transformer import train_worker

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


def load_trained_model():
    global vocab_src, vocab_tgt, spacy_de, spacy_en
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model


def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


def check_outputs(
        valid_dataloader,
        model,
        vocab_src,
        vocab_tgt,
        n_examples=15,
        pad_idx=2,
        eos_string="</s>",
):
    pudb.set_trace()
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        # pudb.set_trace()
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
                " ".join(
                    [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
                ).split(eos_string, 1)[0]
                + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(n_examples=5):
    global vocab_src, vocab_tgt, spacy_de, spacy_en

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


load_trained_model()
run_model_example(1)
# pudb.set_trace()

tb_writer.flush()
tb_writer.close()
