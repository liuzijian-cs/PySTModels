import torch
import torch.nn as nn
from model.iTransformer.layers.AttentionLayers import FullAttention, MultiHeadAttention
from model.iTransformer.layers.AttentionEncDec import Encoder, EncoderLayer
from model.iTransformer.layers.Embedding import InvertedEmbedding


class Model(nn.Module):
    def __init__(self, args):
        
        super(Model, self).__init__()
        # variables
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.use_norm = False  # TODO，暂未在argparse中配置
        self.output_attention = args.output_attention
        # Embedding
        self.inverted_embedding = InvertedEmbedding(args.seq_len, args.d_model)
        # Encoder-Only Architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # model, d_model, d_ffn, dropout=0.1, activation="relu"
                    attention_model=MultiHeadAttention(
                        # model, d_model, n_head, dropout=0.1, output_attention=False, d_key=None, d_value=None
                        attention_model=FullAttention(  # mask=False, scale=None, dropout=0.1, output_attention=False
                            mask=False,
                            scale=None,
                            dropout=args.attn_dropout,
                            output_attention= args.output_attention
                        ),
                        d_model=args.d_model,
                        n_head=args.n_heads,
                        dropout=args.attn_dropout,
                        output_attention=args.output_attention,
                        d_key=None,
                        d_value=None
                    ),
                    d_model=args.d_model,
                    d_ffn=args.d_ff,
                    dropout=args.attn_dropout,
                    activation="relu",
                    output_attention=args.output_attention
                ) for _ in range(args.enc_layers)
            ],
            d_model=args.d_model
        )
        self.projector = nn.Linear(args.d_model, args.pred_len, bias=True)  # Output projection layer

    def forward(self, x, x_mask=None):  # Encoder-Only
        _, _, N = x.shape
        x_embedded = self.inverted_embedding(x, x_mask)
        enc_out, attention_weight = self.encoder(x_embedded, x_mask)
        dec_out = self.projector(enc_out).transpose(2, 1)
        return dec_out, attention_weight


if __name__ == '__main__':
    class Args:
        seq_len = 96
        pred_len = 12
        d_model = 1024
        n_heads = 8
        attn_dropout = 0.1
        d_ff = 2048
        enc_layers = 4
        output_attention = True

    print("testing...")
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(args).to(device)
    x = torch.rand(32, args.seq_len, args.d_model).to(device)
    output, attention_weights = model(x)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights[0].shape)
