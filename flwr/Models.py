## the "DataLoader" function needs to be changed as follow to use LSTM and CNN:
def load_datasets(input, labels):

  Xtrain_raw = [x[0: val] for x in input]
  Xval_raw = [x[val: test] for x in input]
  Xtest_raw = [x[test: ] for x in input]

  Ytrain_raw = [x[0: val] for x in labels]
  Yval_raw = [x[val: test] for x in labels]
  Ytest_raw = [x[test: ] for x in labels]

  for i in range(30):
    ds_train = GetLoader(Xtrain_raw[i], Ytrain_raw[i])
    trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True))
    ds_val = GetLoader(Xval_raw[i], Yval_raw[i])
    valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE, drop_last=True))
    ds_test= GetLoader(Xtest_raw[i], Ytest_raw[i])
    testloaders.append(DataLoader(ds_test, batch_size=BATCH_SIZE, drop_last=True))

  return trainloaders, valloaders, testloaders

##LSTM:
## To use LSTM, we need to add "drop_last=True" in Dataloader
# make sure below: net = LSTM().to(DEVICE) 

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1, batch_size=32):
      super().__init__()
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.output_size = output_size
      self.num_directions = 1 # 单向LSTM
      self.batch_size = batch_size
      self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
      self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
      h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(DEVICE)
      c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(DEVICE)
      seq_len = input_seq.shape[1]
      input_seq = input_seq.view(self.batch_size, seq_len, 1)
      
      output, _ = self.lstm(input_seq, (h_0, c_0)) # output(32, 48, 32)
      output = output.contiguous().view(self.batch_size * seq_len, self.hidden_size)  # (32 * 48, 32)


      pred = self.linear(output)  # (32*48, 1)
      pred = pred.view(self.batch_size, seq_len, -1) #(32,48,1)
      pred = pred[:, -1, :]  # (32, 1)
      return pred


criterion = nn.MSELoss() 
net = LSTM().to(DEVICE)

##CNN:
## 1D-CNN in pytorch is very different from keras.
# they will consider the batch_size first by default.
# this structure is completely different from what I build in TFF
# the performance cannot be guaranteed

class CNN(nn.Module):
  def __init__(self):
     super().__init__()
     self.conv1 = nn.Conv1d(1, 6, 2)
     self.act = nn.ReLU(inplace=True)
     self.fc1 = nn.Linear(32*47*6,64)
     self.fc2 = nn.Linear(64,1)

  def forward(self,x):
    x = x.view(32, 1, 48)
    x = self.conv1(x)
    x = self.act(x)
    x = x.view(-1)
    x = self.fc1(x)
    x = self.act(x)
    out = self.fc2(x)
    return out

criterion = nn.MSELoss() 
net = CNN().to(DEVICE)


##WaveNet: UNFINISHED YET
class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res
# detail 
class WaveNet(nn.Module):
    def __init__(self, inch=1, kernel_size=2): #inch=8
        super().__init__()
        self.wave_block1 = Wave_Block(inch, 16, 12, kernel_size)
        self.wave_block2 = Wave_Block(16, 32, 8, kernel_size)
        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):

        #x = x.permute(0, 2, 1)
        x = x.view(32, 1, 48)

        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)

        x = self.wave_block4(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x

criterion = nn.MSELoss() 
net = WaveNet().to(DEVICE)

