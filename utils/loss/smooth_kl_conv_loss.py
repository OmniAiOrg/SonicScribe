from typing import Optional
import torch
from torch import Tensor

class SmoothKLConvLoss(torch.nn.Module):
    r'''
    input: [N, d, C]
    target: [N, d]
    '''
    def __init__(self, kernel:list[float], eps=1e-7, log_target: bool = True) -> None:
        super().__init__()
        kernel_size = len(kernel)
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.deconv = torch.nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding=padding)
        self.deconv.weight.requires_grad = False
        self.deconv.weight.data = torch.tensor([[kernel]])
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='none', log_target=log_target)
        self.eps = eps
        
    def forward(self, input: Tensor, target: Tensor, start:int, end:int) -> Tensor:
        timestamp_mask = (target.ge(start) & target.le(end)).float()
        # print('timestamp_mask', timestamp_mask)
        p_target = torch.nn.functional.one_hot(target, num_classes = input.shape[-1]) # [N, d] -> [N, d, C]
        p_target = p_target[..., start:end+1]
        p_target_shape = p_target.shape
        # print('p_target_shape', p_target_shape)
        p_target = torch.reshape(p_target, (-1, 1, p_target_shape[-1]))
        p_target = self.deconv(p_target.float())
        p_target = torch.reshape(p_target, p_target_shape)
        p_target = self.log_softmax(p_target)
        p_input = self.log_softmax(input[..., start:end+1])
        # print('input', input)
        # print('target', p_target)
        naive_kl_loss = self.kl(p_input, p_target)
        # print('naive_kl_loss', naive_kl_loss)
        return (torch.mul(timestamp_mask.unsqueeze(-1), naive_kl_loss)).sum() / (timestamp_mask.sum()+self.eps)
    
    
if __name__ == '__main__':
    kernel = [1.,2.,3.,4.,3.,2.,1.]
    loss = SmoothKLConvLoss(kernel)
    print('softmax kernel', loss.softmax(torch.tensor(kernel)))
    d = 2
    N = 2
    input = torch.zeros(N, d, 5) # [N, d, C]
    input[0, 0, 1] = 1.
    input[1, 0, 2] = 1.
    input[1, 1, 2] = 1.
    print('input', input)
    target = torch.zeros(N, d, dtype=torch.long)
    target[0, 0] = 1
    target[1, 0] = 2
    print('target', target)
    output = loss(input, target, 1, 5)
    print(output)
    # output.backward()
    
    exit()
    kernel_size = 3
    assert kernel_size % 2 == 1
    padding = kernel_size // 2
    deconv = torch.nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding=padding)
    print('deconv.weight.requires_grad', deconv.weight.requires_grad)
    deconv.weight.requires_grad = False
    deconv.weight.data = torch.tensor([[[1., 2., 1.]]])#.repeat(5, 1, 1)
    # deconv.weight.data = torch.ones(5,1,3)
    print('deconv.weight.data', deconv.weight.data)
    y = torch.zeros(2,5,10)
    y[0, 0, [5,7]] = 1
    y[0, 2, [1,4]] = 1
    print('y', y)
    y = torch.reshape(y, (2 * 5, 1, 10))
    x=deconv(y)
    x = torch.reshape(x, (2, 5, 10))
    print('x', x)
    