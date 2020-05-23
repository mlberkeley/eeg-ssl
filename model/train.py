def train(model, train_loader, optimizer, epoch):
  model.train()
  
  train_losses = []
  for pair in train_loader:
    x = pair[0].cuda().contiguous().float()
    labels = pair[1].cuda().float()

    loss = model.loss(x, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
  return train_losses

def eval_acc(model, test_loader):
  with torch.no_grad():
    model.eval()
    total_correct = 0.0
    for pair in test_loader:
      x = pair[0].cuda().contiguous().float()
      labels = pair[1].cuda().float()
      out = F.sigmoid(model(x))
      out_np = out.cpu().numpy()
      # technically puts more weight on last batch if it's not divisible by batch size, but whatever
      total_correct += np.sum(np.round(out_np) == labels.cpu().numpy())
  return total_correct / len(test_loader.dataset)

def train_epochs(model, train_loader, train_args):
  epochs, lr, betas = train_args['epochs'], train_args['lr'], train_args['betas']
  optimizer = optim.Adam(model.parameters(), lr=lr)

  train_losses = []
  eval_accuracies = []
  for epoch in range(epochs):
    model.train()
    new_train_losses = train(model, train_loader, optimizer, epoch)
    train_losses.extend(new_train_losses)
    eval_accuracies.append(eval_acc(model, train_loader))

    print(f'Epoch {epoch}, Train loss {np.mean(new_train_losses):.4f}, Eval Acc: {eval_accuracies[-1]:.4f}')

  return train_losses

def train_rp(pairs_np, labels_np):
  """
  Args:
    pairs_np: (bs, 2, n_channels, T)

  """
  labels_np_processed = (labels_np+1.0)/2.0 # {-1,1} --> {0,1}
  dataset = data.TensorDataset(torch.tensor(pairs_np), torch.tensor(labels_np_processed))
  data_loader = data.DataLoader(dataset, batch_size=32)
  model = Relative_Positioning(n_channels, n_timepoints).cuda()
  train_losses = train_epochs(model, data_loader, dict(epochs=10, lr=1e-3,
                                                       betas=(0.9,0.999)))
  return train_losses