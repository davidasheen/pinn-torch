import numpy as np
import torch

def del_scalar(f,X):
    "Computes the gradient of scalar function f with respect to Cartesian coordinates X."
    return torch.autograd.grad(f.sum(),X,create_graph=True)

def del_scalar_cyl(f,R):
    "Computes the gradient of scalar function f with respect to cylindrical coordinates R."
    gr = torch.autograd.grad(f.sum(),X,create_graph=True)
    
    return 

def del_vector(F,X):
    """Computes the gradients of the components of vector function F with respect to Cartesian coordinates X
    In three dimensions, this will return dFx/dx, dFy/dy, and dFz/dz, where Fx is the x-component of F.
    Note that the sum of these partials is the divergence of the vector F
    """
    return tuple(del_scalar(f,x)[0] for f,x in zip(F,X))

def div_vector(F,X):
    """Computes the divergence of vector function F with respect to Cartesian coordinates X
    This function calls del_vector(F,X) and takes the sum of its components
    """
    grads = del_vector(F,X)
    nabl = torch.stack(grads,dim=1).sum(dim=1) #dim 0 is the batch size, dim 1 is the number of dimensions
    return nabl

def curl_vector(F,X):
    """Computes the curl of vector function F
    This function computes curl only for 2 and 3 dimensions (curl is defined for higher dimensions but why)
    """
    
    #Compute the jacobian of F
    jac = torch.stack(tuple(del_scalar(f,X) for f in F),dim=1)
    
    #compute number of spatial dimensions, which is the extent of dimension 1
    dims = F.shape(1)
    
    #3 dimensions
    if dims == 3:
        torch.stack([j[2, 1] - j[1, 2], j[0, 2] - j[2, 0], j[1, 0] - j[0, 1]])

    
    #2 dimensions
    if dims == 2:
        pass