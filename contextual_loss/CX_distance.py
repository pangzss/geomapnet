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
    d_min, _ = torch.min(d, dim=2, keepdim=True)        # (N, H*W, 1)

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
  d_2nd_min = torch.topk(d,k=2,dim=2,largest=False)[0][:,:,1]
  d_2nd_min = d_2nd_min[:,:,None]
  
  w = d_2nd_min / (d + 1e-5)

  # Eq(4)
  cx_ij = torch.nn.functional.softmax(w,dim=2)


  # Eq (1)
  cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
  return cx

if __name__ == '__main__':
    seed = 0
    a = torch.tensor([4,20,60,70])

    topk = torch.topk(a,k=2,largest=False)[0]
    # cx 
    a_min = topk[0]
    a_tilde = a/(a_min+1e-5)
    w1 = torch.exp((1 - a_tilde) / 0.5)
    w1 = w1/torch.sum(w1)
    
    # NNDR
    a_2nd_min = topk[1]
    w2 = a_2nd_min / (a + 1e-5)
    w2 = torch.nn.functional.softmax(w2)

    