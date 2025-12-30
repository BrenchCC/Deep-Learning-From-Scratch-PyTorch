import logging

import torch
from graphviz import Digraph

# Global logger configuration
logger = logging.getLogger("ComputationGraphViz")

def build_dot(var, params=None):
    """
    Constructs an Autograd computation graph starting from a scalar Tensor.
    
    Args:
        var (torch.Tensor): The output tensor (usually loss) to backtrace from.
        params (dict, optional): A dictionary mapping names to Tensors for labeling parameters.
    """
    if params is not None:
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    dot = Digraph(format = "png", graph_attr = {"rankdir": "LR"})
    seen = set()

    def size_to_str(size):
        return "(" + ", ".join(map(str, size)) + ")"

    def add_node(fn):
        if fn in seen:
            return
        seen.add(fn)

        # Current Node (Function or Tensor)
        if torch.is_tensor(fn):
            uid = str(id(fn))
            name = param_map.get(id(fn), "")
            label = f"{name}\nTensor {size_to_str(fn.size())}"
            dot.node(uid, label, shape = "ellipse", color = "lightblue")
        else:
            uid = str(id(fn))
            # Use the class name of the grad_fn (e.g., AddBackward0)
            dot.node(uid, type(fn).__name__, shape = "box")

        # Recursively visit parents (next_functions)
        if hasattr(fn, "next_functions"):
            for u, _ in fn.next_functions:
                if u is not None:
                    dot.edge(str(id(u)), uid)
                    add_node(u)

        # Leaf Tensor (Parameters that accumulated gradients)
        if hasattr(fn, "variable"):
            var = fn.variable
            uid_var = str(id(var))
            name = param_map.get(id(var), "")
            label = f"{name}\nParameter {size_to_str(var.size())}"
            dot.node(uid_var, label, shape = "ellipse", color = "orange")
            dot.edge(uid_var, uid)

    # Start recursion from the grad_fn of the target variable
    if var.grad_fn:
        add_node(var.grad_fn)
    else:
        logger.warning("The provided variable has no grad_fn. Ensure requires_grad=True for inputs.")
        
    return dot

if __name__ == "__main__":
    # Configure logging as per preference
    logging.basicConfig(
        level = logging.INFO, 
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        handlers = [logging.StreamHandler()]
    )

    logger.info("Starting computation graph generation...")

    # 1. Define input data (batch_size=1, feature_dim=3)
    x = torch.randn(1, 3)

    # 2. Define learnable parameters (Weights and Bias)
    # Using explicit requires_grad = True to track operations
    W = torch.randn(3, 1, requires_grad = True)
    b = torch.randn(1, 1, requires_grad = True)

    logger.info(f"Initialized parameters W: {W.shape}, b: {b.shape}")

    # 3. Perform Forward Pass: y = xW + b
    matmul_out = torch.matmul(x, W)
    y_pred = matmul_out + b

    # 4. Compute Loss (MSE): loss = mean((y - target)^2)
    y_target = torch.randn(1, 1)
    loss = (y_pred - y_target).pow(2).mean()

    logger.info(f"Calculated loss: {loss.item()}")

    # 5. Define parameter mapping for visualization labels
    # Use spaces around = as preferred
    model_params = {
        "Weight_Matrix": W,
        "Bias_Vector": b
    }

    # 6. Build the dot object
    # Calling the function with named arguments
    dot_graph = build_dot(var = loss, params = model_params)

    # 7. Render the graph
    # Generates 'torch_viz_output.png' in the current directory
    output_filename = "torch_viz_output"
    try:
        dot_graph.render(output_filename, view = False, cleanup = True)
        logger.info(f"Graph successfully rendered to {output_filename}.png")
    except Exception as e:
        logger.error(f"Failed to render graph. Ensure Graphviz is installed. Error: {e}")