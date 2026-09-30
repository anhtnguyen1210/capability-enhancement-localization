"""Zero selected query-head outputs immediately before their output projection."""
from collections import defaultdict

def attach_fixed_head_hooks(layers, head_pairs, num_heads, head_dim):
    """Validate query-head coordinates and fail closed on projection shape drift."""
    groups=defaultdict(list)
    for layer,head in head_pairs:
        if type(layer) is not int or type(head) is not int or not 0<=layer<len(layers) or not 0<=head<num_heads:
            raise ValueError('Head coordinates are outside the model geometry')
        if head in groups[layer]:
            raise ValueError('Duplicate head coordinate')
        groups[layer].append(head)
    activity={str(layer):{'calls':0,'nonzero_before':0} for layer in groups}
    handles=[]
    for layer,heads in groups.items():
        def hook(module,args,layer=layer,heads=tuple(heads)):
            x=args[0]
            if x.shape[-1] != num_heads*head_dim:
                raise ValueError('Attention output shape differs from query-head geometry')
            y=x.clone()
            for head in heads:
                sl=slice(head*head_dim,(head+1)*head_dim)
                # Record intervention activity without changing unrelated channels.
                if activity[str(layer)]['calls']==0:
                    activity[str(layer)]['nonzero_before']+=int(x[...,sl].count_nonzero().item())
                y[...,sl]=0
            activity[str(layer)]['calls']+=1
            return (y,*args[1:])
        handles.append(layers[layer].self_attn.o_proj.register_forward_pre_hook(hook))
    return handles,activity
