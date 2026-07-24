import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def is_iterable(obj):
    '''Determine whether the argument obj is iterable and return True if true, False if false
    '''
    try:
        iter(obj)
        return True
    except TypeError:
        return False
    
    return

class PhysicsHead(nn.Module):
    def __init__(
        self,
        num_spatial,
        layers
    ):
        super(PhysicsHead, self).__init__()
        
        self.num_spatial = num_spatial
        self.stack = nn.Sequential(*layers)
        
    def forward(self,spatial,t):
        if self.num_spatial > 1:
            inputs = spatial + [t]
        else:
            inputs = [spatial,t]
        inputs = torch.cat(inputs,axis=1)
        
        return self.stack(inputs)
    
class PhysicsHeadSteady(nn.Module):
    def __init__(
        self,
        num_spatial,
        layers
    ):
        super(PhysicsHeadSteady, self).__init__()
        
        self.num_spatial = num_spatial
        self.stack = nn.Sequential(*layers)
        
    def forward(self,spatial):
#         if self.num_spatial > 1:
#             inputs = spatial + [t]
#         else:
#             inputs = [spatial,t]
        inputs = torch.cat(spatial,axis=1)
        
        return self.stack(inputs)
        

class LinearSkip(nn.Module):
    '''Fully connected layer with a tanh activation and a residual connection
    
    This module initializes with linear fully-connected layer and a tanh activation.
    
    This module has the same signature as nn.Linear() and all keyword arguments are passed to nn.Linear().
    '''
    def __init__(self,in_features,out_features,**kwargs):
        super(LinearSkip, self).__init__()
        
        self.linear = nn.Linear(in_features,out_features,**kwargs)
        self.act = nn.Tanh()
        
    def forward(self,x):        
        out = self.linear(x)
        out = self.act(out)
        out = out + x
        
        return out

class RadialBasisFunction(nn.Module):
    '''A radial basis function layer
    '''
    def __init__(self,num_centers=11,rbf_width=None,domain_sizes=[],):
        '''
        Create the RBF layer. If 
        
        Inputs
        ------
        num_centers: int or iterable of ints of same length as domain_sizes, 
        rbf_wdith: int or iterable of ints of same length as domain_sizes
        domain_sizes: iterable of floats 
        '''
        super(RadialBasisFunction, self).__init__()
        
        #Compute number of dimensions
        num_dims = len(domain_sizes)
        
        #Check to see whether we have a single number of centers for all dimensions or if each dimension has its own
        if not is_iterable(num_centers):
            num_centers = [num_centers]*num_dims #duplicate the single num_centers value num_dims times
        
        #Check to see if we will be creating the RBF widths
        create_rbf_widths = False
        if is_iterable(rbf_width): 
            #if rbf_width is iterable, that means the user already provded rbf_widths for each dimension
            #all we have to do is make rbf_widths a torch tensor
            rbf_widths = rbf_width
        else:
            if rbf_width is None: #rbf_width wasn't provided, compute rbf_width for each dimension
                create_rbf_widths = True
            else: #in this case, a single RBF width was provided, so repeat it num_dims times
                rbf_widths = [rbf_width]*num_dims
                
        #Determine the centers for the RBF kernels
        #These will be linearly spaced over the domain with a number of elements determined by num_centers
        rbf_centers = []
        if create_rbf_widths: rbf_widths = []
        for domain_size,ncenters in zip(domain_sizes,num_centers):
            this_centers = np.linspace(0,domain_size,num=ncenters)# generate the set of center coordinates
            rbf_centers += [this_centers]
            #if we are creating rbf_widths, the width of the kernel function should be equal to the spacing
            #between the rbf centers
            if create_rbf_widths: rbf_widths += [4*(this_centers[1] - this_centers[0])]
                
        #use numpy.meshgrid to create a mesh from the RBF centers, stack the mesh into a single array, and reshape
        rbf_centers = np.stack(
            np.meshgrid(*rbf_centers), #use np.meshgrid to turn the center coordinates into a grid
            axis=-1 #stack along the last dimension
        ).reshape(-1,num_dims) #reshape to shape (num_centers**num_dims,num_dims)
        
        #make rbf_centers into a torch tensor
        self.rbf_centers = torch.tensor(
            rbf_centers,
            device=device
        ).float().unsqueeze(0) #make torch tensor and add singleton leading dimension 
        # shape is now (1,num_centers**num_dims,num_dims)
        #self.rbf_centers = nn.Parameter(self.rbf_centers)
        
        #Make rbf_widths into a torch tensor
        rbf_widths = torch.tensor(
            rbf_widths,
            device=device
        ).float().unsqueeze(0).unsqueeze(0) #add two leading singleton dimensions so this will be (1,1,num_dims)
        
        self.epsilon = 0.5*(1/rbf_widths)**2
        
    def forward(self,x):
        xu = x.unsqueeze(1) #add singleton second dimension so this is (batch_size, 1, num_dims)
        zeta = torch.sum(-self.epsilon*(xu-self.rbf_centers)**2,dim=-1) #radial basis function
        return torch.exp(zeta)
    
class ResidualFullyConnected(nn.Module):
    '''Creates a fully-connected neural network with residual skip connections
    '''
    def __init__(
        self,
        num_inputs,num_layers,nodes_per_layer,
        activation=nn.Tanh,
    ):
        super(ResidualFullyConnected, self).__init__()
        
        #Define the nodes in each layer
        layers_nodes = [num_inputs] + num_layers*[nodes_per_layer]
        
        #empty layers list (torch modulelist)
        self.layers_list = []#nn.ModuleList()
        
        #first layer, linear o tanh - this is different because layer 1 has no skip connection
        last_nodes = layers_nodes[0] #input nodes
        this_nodes = layers_nodes[1]
        self.layers_list.append(nn.Linear(last_nodes,this_nodes))
        self.layers_list.append(activation())
        last_nodes=this_nodes
        
        #now do a bunch of skip connections
        for this_nodes in layers_nodes[2:-1]: #skip the first element because input
            
            self.layers_list.append(LinearSkip(last_nodes,this_nodes)) #
            
            #self.layers_list.append(nn.Linear(last_nodes,this_nodes))
            #self.layers_list.append(nn.Tanh())
            last_nodes = this_nodes
            
            #layers_list += [this_layer]
        #layers_list.pop() #remove the last activation
        this_nodes = layers_nodes[-1]
        self.layers_list.append(nn.Linear(last_nodes,this_nodes))
        
        self.resnn = nn.Sequential(*self.layers_list)
        
    def forward(self,x):
        return self.resnn(x)
    
class PhysicsLayer(nn.Module):
    '''A fully-connected linear output layer for physics-informed neural networks
    This module splits the output into three segments:
       1. A purely linear layer (e.g displacement, velocity field, etc)
       2. A softmax layer for variables that must sum to 1 (e.g. mole fraction, mass fraction)
       3. An ELU layer for variables that must be positive but otherwise unbounded (concentration, surface population)
    The number of output variables in each segment is specified at instantiation. If any are zero, that segment is not used./
       
    '''
    def __init__(
        self,
        num_inputs,
        num_linear_outputs=1,
        num_fraction_outputs=0,
        num_forced_positive_outputs=0,
    ):
        
        super(PhysicsLayer, self).__init__()
        
        self.use_linear = num_linear_outputs > 0
        self.use_fraction = num_fraction_outputs > 0
        self.use_positive = num_forced_positive_outputs > 0
        if self.use_linear:
            self.linear_segment = nn.Linear(num_inputs,num_linear_outputs,)
        if self.use_fraction:
            self.fraction_segment = nn.Linear(num_inputs,num_fraction_outputs,)
        if self.use_positive:
            self.positive_segment = nn.Linear(num_inputs,num_forced_positive_outputs,)
        self.n_var = num_linear_outputs + num_fraction_outputs + num_forced_positive_outputs
        
    def forward(self,x):
        
        outputs = []
        if self.use_linear:
            outputs += [self.linear_segment(x)]
        if self.use_fraction:
            outputs += [nn.Softmax(dim=1)(self.fraction_segment(x))]
        if self.use_positive:
            outputs += [nn.ELU()(self.positive_segment(x))]
        outputs = torch.cat(outputs,dim=1)
        
        return outputs