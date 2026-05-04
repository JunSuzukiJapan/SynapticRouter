import logging
logging.basicConfig(level=logging.INFO)

class Router(nn.Module):
    def __init__(self, dim, num_synapse, k=2):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(dim, num_synapse)

    def forward(self, x):
        logits = self.linear(x)              # (B, N)
        topk_val, topk_idx = torch.topk(logits, self.k, dim=-1)
        weights = F.softmax(topk_val, dim=-1)
        return topk_idx, weights

    def route(self, inputs):
        logging.info(f"Routing inputs: {inputs}")
        logging.info(f"Load distribution: {self.load_distribution}")
        return outputs