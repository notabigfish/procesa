import torch
import torch.nn as nn
from models.builder import MODELS
from models import GraphModel20

@MODELS.register_module()
class GraphModel23(GraphModel20):
    def __init__(self, **kwargs):
        super(GraphModel23, self).__init__(**kwargs)

        self.embed_dim = 21
        self.w_s = nn.Embedding(self.embed_dim, self.embed_dim)
        del self.fc
        self.fc = nn.Linear(self.in_features + self.embed_dim, self.hidden_features)


    def extract_feat(self, graphs, seq_num):
        onehot_embeds = self.w_s(seq_num)

        # MLP extract features
        graphs.ndata['h'] = self.fc(torch.cat((graphs.ndata['x'], onehot_embeds), dim=1))
        mlp_feat, mask, bs = self.split_node_feat(graphs.ndata['h'], graphs)

        # GCN extract features
        graphs.ndata['h'] = self.ln1(self.relu(self.conv1(graphs)))
        graphs.ndata['h'] = self.ln2(self.relu(self.conv2(graphs)))
        gcn_feat, mask, bs = self.split_node_feat(graphs.ndata['h'], graphs)

        # attention
        queries = self.q_layer(mlp_feat)
        keys = self.k_layer(gcn_feat)
        values = self.v_layer(mlp_feat)
        feat = self.multihead_attn_layer(
            queries.transpose(0, 1),
            keys.transpose(0, 1),
            values.transpose(0, 1))[0].transpose(0, 1)  # q, k, v
        
        # last block
        feat = self.linear1(feat)
        feat = self.dropout(feat)
        feat = self.ln3(feat + values)
        feat = self.pooling(feat, mask)  # 8,1024
        output = self.fc_final(feat)
        
        graphs.ndata.pop('h')
        
        return output, None, None

    def forward_train(self, graphs, labels, sequences, seq_num):
        output, f_rep, h0 = self.extract_feat(graphs, seq_num)
        losses = dict()
        loss_reg = self.loss_reg(labels, output)  # labels: y_true, outputs: y_pred
        losses.update(loss_reg)
        if self.loss_triplet is not None:
            # h0: f_anchor
            loss_triplet = self.loss_triplet(f_rep, sequences, h0)
            losses.update(loss_triplet)
        return losses

    def forward_test(self, graphs, seq_num):
        output, _, _ = self.extract_feat(graphs, seq_num)
        return output
        
    def forward(self, batch, return_loss=True):
        _, sequences, graphs, labels, seq_num = batch
        graphs = graphs.to('cuda:0')
        labels = labels.to('cuda:0')
        seq_num = seq_num.to('cuda:0')
        if return_loss:
            return self.forward_train(graphs, labels, sequences, seq_num)
        else:
            return self.forward_test(graphs, seq_num), labels