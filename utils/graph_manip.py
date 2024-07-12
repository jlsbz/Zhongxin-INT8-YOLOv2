#########################
# SOME GRAPH TECHNIQUES #
#########################
# Experiment with iterator pattern:
# NOTE: taken from https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern
import torch
from torch import fx
from torch.fx.node import Node
from typing import Dict, Union, Tuple, Any

class GraphIteratorStorage:
    """
    A general Iterator over the graph. This class takes a `GraphModule`,
    and a callable `storage` representing a function that will store some
    attribute for each node when the `propagate` method is called.

    Its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments, e.g. an example input tensor.
    As each operation executes, the GraphIteratorStorage class stores
    away the result of the callable for the output values of each operation on
    the attributes of the operation's `Node`. For example,
    one could use a callable `store_shaped_dtype()` where:

    ```
    def store_shape_dtype(result):
        if isinstance(result, torch.Tensor):
            node.shape = result.shape
            node.dtype = result.dtype
    ```
    This would store the `shape` and `dtype` of each operation on
    its respective `Node`, for the given input to `propagate`.
    """
    def __init__(self, mod, storage):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.storage = storage

    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        # Each of these `result` variables correpsonds to the output of the node in question
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                module = self.mod.get_submodule(node.target) #  this is robust to "shared" nodes.
                result = module(*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to the `storage` function.
            # you can delete this line and this `propagate` function becomes
            # a generic GraphModule interpreter
            self.storage(node, result)

            # Store the output activation in `env` for the given node
            env[node.name] = result

        #return load_arg(self.graph.result)

def store_shape_dtype(node, result):
    """
    Function that takes in the current node, and the tensor it is operating on (`result`)
    and stores the shape and dtype of `result` on the node as attributes.
    """
    if isinstance(result, torch.Tensor):
        node.shape = result.shape
        node.dtype = result.dtype
# NOTE: I just discovered that they have the `Interpreter` class, which accomplishes the same thing:
# https://pytorch.org/docs/stable/fx.html#torch.fx.Interpreter
# GraphIteratorStorage(fx_model, store_shape_dtype).propagate(example_inputs[0])

#print("\nPrinting size and data types:")
#for node in fx_model.graph.nodes:
    #print(node.name, node.shape, node.dtype)



###########################################
### Fusing Bn in ConvBnReLU into ConvReLU #
###########################################
from torch.ao.nn.intrinsic.qat.modules.conv_fused import ConvBnReLU2d, ConvReLU2d, ConvBn2d
from torch.ao.nn.qat import Conv2d

def fuse_conv_bn_relu_eval(conv: Union[ConvBnReLU2d, ConvBn2d]) -> Union[ConvReLU2d, Conv2d]:
    """
    Given a quantizable ConvBnReLU2d Module returns a quantizable ConvReLU2d
    module such that the BatchNorm has been fused into the Conv, in inference mode.
    Given a ConvBn2d, it does the same to produce a Conv2d.
    One could also use `torch.nn.utils.fuse_conv_bn_eval` to produce a Conv, and then quantize that as desired.
    """
    assert(not (conv.training or conv.bn.training)), "Fusion only for eval!"
    qconfig = conv.qconfig
    if type(conv) is ConvBnReLU2d:
        new_conv = ConvReLU2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                            conv.stride, conv.padding, conv.dilation,
                            conv.groups, conv.bias is not None,
                            conv.padding_mode, qconfig=qconfig)
    elif type(conv) is ConvBn2d:
        new_conv = Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                            conv.stride, conv.padding, conv.dilation,
                            conv.groups, conv.bias is not None,
                            conv.padding_mode, qconfig=qconfig)


    new_conv.weight, new_conv.bias = \
        fuse_conv_bn_weights(conv.weight, conv.bias,
                             conv.bn.running_mean, conv.bn.running_var, conv.bn.eps, conv.bn.weight, conv.bn.bias)

    return new_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """
    Helper function for fusing a Conv and BatchNorm into a single weight/bias tensor pair.
    """
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)

# Graph manipulation functions for fusing Convs and BatchNorms
def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`) 
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    """
    Helper function for having `new_module` take the place of `node` in a dict of modules.
    """
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    #modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)
    
def convbn_to_conv(fx_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Iterates through the graph nodes, and:
    - where it finds a ConvBnReLU it replaces it with ConvReLU
    - where it finds a ConvBn it replaces it with Conv

    This function works in-place on `fx_model`.

    Inputs:
    fx_model (torch.fx.GraphModule): a graph module, that we want to perform transformations on

    Output:
    (torch.fx.GraphModule): a model where we have swapped out the 2d ConvBn/ConvBnReLU for Conv/ConvReLU, and
                            fused the Bns into the Convs.
    """
    modules = dict(fx_model.named_modules())

    for node in fx_model.graph.nodes:
        # If the operation the node is doing is to call a module
        if node.op == 'call_module':
            # The current node
            orig = fx_model.get_submodule(node.target)
            if type(orig) in [ConvBnReLU2d, ConvBn2d]:
                # Produce a fused Bn equivalent.
                fused_conv = fuse_conv_bn_relu_eval(orig)
                # This updates `modules` so that `fused_conv` takes the place of what was represented by `node`
                replace_node_module(node, modules, fused_conv)
    
    return fx_model