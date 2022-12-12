# torch-rnn
pytorch rnn


```
import torch

batch_size = 2
sesquence = 10
input_size = 5000
num_layers = 1
hidden_size = 4

input = torch.randn(batch_size, sesquence, input_size)
h0 = torch.randn(num_layers, batch_size, hidden_size)

rnn = torch.nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)

for name , param in rnn.named_parameters():
    print(name, param.shape)

print(f'input : {input.shape}  //  h0 : {h0.shape}')
output, hn = rnn(input, h0)
print(f'output : {output.shape}  //  hn : {hn.shape}')
```


```
weight_ih_l0 torch.Size([4, 5000])
weight_hh_l0 torch.Size([4, 4])
bias_ih_l0 torch.Size([4])
bias_hh_l0 torch.Size([4])
input : torch.Size([2, 10, 5000])  //  h0 : torch.Size([1, 2, 4])
output : torch.Size([2, 10, 4])  //  hn : torch.Size([1, 2, 4])

```









--------------


```
import torch

batch_size = 3
sesquence = 5
input_size = 7
num_layers = 2
hidden_size = 4

input = torch.randn(sesquence, batch_size, input_size)
h0 = torch.randn(num_layers, batch_size, hidden_size)

rnn = torch.nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = False)

for name , param in rnn.named_parameters():
    print(name, param.shape)

print(f'input : {input.shape}  //  h0 : {h0.shape}')
output, hn = rnn(input, h0)
print(f'output : {output.shape}  //  hn : {hn.shape}')

```




```
weight_ih_l0 torch.Size([4, 7])
weight_hh_l0 torch.Size([4, 4])
bias_ih_l0 torch.Size([4])
bias_hh_l0 torch.Size([4])
weight_ih_l1 torch.Size([4, 4])
weight_hh_l1 torch.Size([4, 4])
bias_ih_l1 torch.Size([4])
bias_hh_l1 torch.Size([4])
input : torch.Size([5, 3, 7])  //  h0 : torch.Size([2, 3, 4])
output : torch.Size([5, 3, 4])  //  hn : torch.Size([2, 3, 4])


```


---------------



```
import torch

batch_size = 3
sesquence = 5000
input_size = 7
num_layers = 2
hidden_size = 4

input = torch.randn(sesquence, batch_size, input_size)
h0 = torch.randn(num_layers, batch_size, hidden_size)

rnn = torch.nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = False)

for name , param in rnn.named_parameters():
    print(name, param.shape)

print(f'input : {input.shape}  //  h0 : {h0.shape}')
output, hn = rnn(input, h0)
print(f'output : {output.shape}  //  hn : {hn.shape}')
```

```
weight_ih_l0 torch.Size([400, 7])
weight_hh_l0 torch.Size([400, 400])
bias_ih_l0 torch.Size([400])
bias_hh_l0 torch.Size([400])
weight_ih_l1 torch.Size([400, 400])
weight_hh_l1 torch.Size([400, 400])
bias_ih_l1 torch.Size([400])
bias_hh_l1 torch.Size([400])
input : torch.Size([5000, 3, 7])  //  h0 : torch.Size([2, 3, 400])
output : torch.Size([5000, 3, 400])  //  hn : torch.Size([2, 3, 400])
```
