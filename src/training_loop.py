model = SynapticModel(dim=64, num_synapse=16, k=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1000):
    x = torch.randn(32, 64)
    target = torch.randn(32, 64)

    y = model(x)
    loss = F.mse_loss(y, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(loss.item())