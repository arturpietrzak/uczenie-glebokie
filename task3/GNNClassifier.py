import torch
from torch.nn import Linear
from torch_geometric.nn import TransformerConv, GCNConv, global_mean_pool


class GNNClassifier(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, hidden_output_dim, transformer_heads=1, graph_layer_type="gcn", predictor="linear"):
        super().__init__()
        self.predictor_type = predictor

        if graph_layer_type == "gcn":
            transformer_heads = 1
            self.conv1 = GCNConv(input_dim, embedding_dim)
            self.conv2 = GCNConv(embedding_dim, embedding_dim)
        elif graph_layer_type == "transformer":
            self.conv1 = TransformerConv(input_dim, embedding_dim, heads=transformer_heads)
            self.conv2 = TransformerConv(transformer_heads * embedding_dim, embedding_dim, heads=transformer_heads)

        self.embed = global_mean_pool

        if predictor == "linear":
            self.predictor = Linear(transformer_heads * embedding_dim, output_dim)
        elif predictor == "nonlinear":
            self.hidden_layer = Linear(transformer_heads * embedding_dim, hidden_output_dim)
            self.predictor = Linear(hidden_output_dim, output_dim)

    def forward(self, x, edge_index, batch, return_embeddings=False):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        embed = self.embed(x, batch)
        x = embed

        if self.predictor_type == "nonlinear":
            x = self.hidden_layer(x)
            x = x.relu()

        output = self.predictor(x)

        if return_embeddings:
            return output, embed  # Return both the output and embeddings for analysis
        return output

    def from_embeddings(self, input):
        x = input

        if self.predictor_type == "nonlinear":
            x = self.hidden_layer(x)
            x = x.relu()
        output = self.predictor(x)
        return output
