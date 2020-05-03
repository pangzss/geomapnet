import torch
def CX_sim(x, y, h=0.5):
    """Computes contextual loss between x and y.
    
    Args:
      x: features of shape (N, C, H, W).
      y: features of shape (N, C, H, W).
      
    Returns:
      cx_loss = contextual loss between x and y (Eq (1) in the paper)
    """
    assert x.size() == y.size()
    N, C, H, W = x.size()   # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).
    
    y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)
    x_centered = x - y_mu
    y_centered = y - y_mu
    x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
    y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

    # The equation at the bottom of page 6 in the paper
    # Vectorized computation of cosine similarity for each pair of x_i and y_j
    x_normalized = x_normalized.reshape(N, C, -1)                                # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)                                # (N, C, H*W)
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)           # (N, H*W, H*W)
    
    d = (1 - cosine_sim)/2                                 # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data 
    d_min, idces = torch.min(d, dim=2, keepdim=True)        # (N, H*W, 1)
    
    # Eq (2)
    d_tilde = d / (d_min + 1e-5)
    
    # Eq(3)
    w = torch.exp((1 - d_tilde) / h)

    # Eq(4)
    cx_ij = w / torch.sum(w, dim=2, keepdim=True)       # (N, H*W, H*W)

    # Eq (1)
    cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
    return cx
  
def CX_sim_NNDR(x, y, h=0.5):
  """Computes contextual loss between x and y.
  
  Args:
    x: features of shape (N, C, H, W).
    y: features of shape (N, C, H, W).
    
  Returns:
    cx_loss = contextual loss between x and y (Eq (1) in the paper)
  """
  assert x.size() == y.size()
  N, C, H, W = x.size()   # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

  y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)
  x_centered = x - y_mu
  y_centered = y - y_mu
  x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
  y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

  # The equation at the bottom of page 6 in the paper
  # Vectorized computation of cosine similarity for each pair of x_i and y_j
  x_normalized = x_normalized.reshape(N, C, -1)                                # (N, C, H*W)
  y_normalized = y_normalized.reshape(N, C, -1)                                # (N, C, H*W)
  cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)           # (N, H*W, H*W)
  
  d = (1 - cosine_sim)/2                                 # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data 
  '''
  d_min = torch.zeros_like(d)
  for i in range(H*W):
    mask = torch.zeros_like(d,dtype=bool)
    mask[:,:,i] = True
    d_i = d[~mask].reshape(N,H*W,H*W-1)
    d_min[:,:,i],_ = torch.min(d_i,dim=2)
  '''
  d_min, idces = torch.min(d, dim=2, keepdim=True)        # (N, H*W, 1)
  d_min = torch.repeat_interleave(d_min,H*W,dim=2)
  d_2nd_min = torch.topk(d,k=2,dim=2,largest=False)[0][:,:,1]
  mask = torch.zeros_like(d,dtype=bool)
  mask.scatter_(2,idces,True)
  d_min[mask] = d_2nd_min.flatten()

  d_tilde = d / (d_min+1e-5)

  w = torch.exp((1-d_tilde)/h)
  # Eq(4)
  cx_ij = w / torch.sum(w,dim=2,keepdim=True)
  # Eq (1)
  cx = torch.mean(torch.diagonal(cx_ij, offset=0,dim1=1,dim2=2), dim=1)  # (N, )
  return cx
def CX_sim_dii(x, y, h=0.5):
  """Computes contextual loss between x and y.
  
  Args:
    x: features of shape (N, C, H, W).
    y: features of shape (N, C, H, W).
    
  Returns:
    cx_loss = contextual loss between x and y (Eq (1) in the paper)
  """
  assert x.size() == y.size()
  N, C, H, W = x.size()   # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

  y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)
  x_centered = x - y_mu
  y_centered = y - y_mu
  x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
  y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

  # The equation at the bottom of page 6 in the paper
  # Vectorized computation of cosine similarity for each pair of x_i and y_j
  x_normalized = x_normalized.reshape(N, C, -1)                                # (N, C, H*W)
  y_normalized = y_normalized.reshape(N, C, -1)                                # (N, C, H*W)
  cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)           # (N, H*W, H*W)
  
  d = (1 - cosine_sim)/2                                 # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data 
  '''
  d_min = torch.zeros_like(d)
  for i in range(H*W):
    mask = torch.zeros_like(d,dtype=bool)
    mask[:,:,i] = True
    d_i = d[~mask].reshape(N,H*W,H*W-1)
    d_min[:,:,i],_ = torch.min(d_i,dim=2)
  '''
  d_ii = torch.diagonal(d, offset=0,dim1=1,dim2=2)

  mask = ~torch.eye(H*W,dtype=bool)
  d_ij = d[:,mask].reshape(N,H*W,H*W-1)
  d_min,_ = torch.min(d_ij,dim=2)

  d_ii_tilde = d_ii / (d_min+1e-5)
  loss = torch.log(torch.mean(d_ii_tilde,dim=1)+1)
  
  return loss

if __name__ == '__main__':
   a = torch.randn(5,2,5,5)
   b = a 

   print(CX_sim(a,b))
   print(CX_sim_dii(a,b))
    