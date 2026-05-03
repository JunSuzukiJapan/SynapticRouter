class SynapticModel(nn.Module):
    def __init__(self, dim, num_synapse, k=2):
        super().__init__()
        self.synapses = nn.ModuleList([
            Synapse(dim) for _ in range(num_synapse)
        ])
        self.controller = Controller(dim, num_synapse, k)

    def forward(self, x):
        # x: (B, dim)

        idx, weights = self.controller(x)

        outputs = []

        for i in range(self.controller.k):
            syn_idx = idx[:, i]

            out = torch.stack([
                self.synapses[j](x[b])
                for b, j in enumerate(syn_idx)
            ])

            outputs.append(out * weights[:, i:i+1])

        y = sum(outputs)
        return y