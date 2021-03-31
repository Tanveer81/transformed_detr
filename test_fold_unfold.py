import torch
import torch.nn.functional as f
torch.manual_seed(0)

B, C, H, W = 2, 768, 32, 32
k = 8 # kernel size
s = 8 # stride
# unfold and fold
nn_unfold = torch.nn.Unfold(kernel_size=k,stride=s)
nn_fold = torch.nn.Fold((H,W),kernel_size=k,stride=s)
# Let's assume that our batch has only 1 sample, for simplicity
x  = torch.arange(0, B*H*W*C).view(B, C, H, W).double()
print("original image : ",x)
print("original image shape : ",x.shape)
# im2 = torch.arange(0, H*W).view(1, C, H, W)
# x = torch.cat((im, im2), dim=0).type(torch.float32) # shape = [2, 1, 9, 9]

N = x.shape[0]
unfold_patches = nn_unfold(x)
l = unfold_patches.shape[-1]
rearranged_unfolded_patches = unfold_patches.permute(0,2,1).reshape(B*l,C,k*k).permute(0,2,1)
print("unfolded image : ", rearranged_unfolded_patches)
print("unfolded shape : ", rearranged_unfolded_patches.shape)
prepare_fold = rearranged_unfolded_patches.view(B,l,k*k,C).permute(0,3,2,1).view(B,k*k*C,l)
fold_patches = nn_fold(prepare_fold)
print("folded image : ", fold_patches)
print("folded image shape : ", fold_patches.shape)
assert torch.any(fold_patches == x)