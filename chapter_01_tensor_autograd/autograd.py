import os
import sys
import math
import logging

import torch

sys.path.append(os.getcwd())
from utils import setup_seed, get_device, Timer

logger = logging.getLogger("Chapter01_Tensor_Autograd")

class SimpleScalarGraph:
    """
    【Simple Example】
    演示标量复合函数的计算图与反向传播。
    
    Formula:
        a = x * y
        b = sin(x)
        z = a + b
        
    Gradients (Manual Derivation):
        dz/dx = dz/da * da/dx + dz/db * db/dx
              = 1 * y + 1 * cos(x)
        dz/dy = dz/da * da/dy
              = 1 * x
    """
    
    def __init__(self, x_val: float, y_val: float, device: torch.device):
        self.device = device
        # 叶子节点 (Leaf Nodes) 需要梯度
        self.x = torch.tensor(
            x_val, 
            requires_grad = True, 
            device = self.device, 
            dtype = torch.float32
        )
        self.y = torch.tensor(
            y_val, 
            requires_grad = True, 
            device = self.device, 
            dtype = torch.float32
        )
        
    def forward(self):
        """前向传播构建计算图"""
        # PyTorch 会自动构建图：x, y -> a, b -> z
        # retain_grad() 用于非叶子节点，方便我们调试观察中间梯度
        self.a = self.x * self.y
        self.a.retain_grad() 
        
        self.b = torch.sin(self.x)
        self.b.retain_grad()
        
        self.z = self.a + self.b
        self.z.retain_grad()
        return self.z
    
    def manual_backward(self):
        """
        手动实现链式法则 (Chain Rule)
        用于验证我们对 calculus 的理解
        """
        with torch.no_grad():
            # 1. Calculate numerical values for gradients
            # dz/dx = y + cos(x)
            grad_x_manual = self.y + torch.cos(self.x)
            
            # dz/dy = x
            grad_y_manual = self.x
            
            return grad_x_manual, grad_y_manual

    def verify(self):
        """对比 PyTorch Autograd 和 手动推导的结果"""
        # 1. Run Forward
        _ = self.forward()
        
        # 2. Run PyTorch Autograd
        # backward() 从根节点 z 开始反向传播
        self.z.backward()
        
        # 3. Run Manual Backward
        grad_x_man, grad_y_man = self.manual_backward()
        
        # 4. Compare
        # 使用 allclose 处理浮点数精度误差
        x_match = torch.allclose(self.x.grad, grad_x_man, atol = 1e-6)
        y_match = torch.allclose(self.y.grad, grad_y_man, atol = 1e-6)
        
        logger.info(f"[Scalar Graph] Grad X Match: {x_match} | PyTorch: {self.x.grad:.4f}, Manual: {grad_x_man:.4f}")
        logger.info(f"[Scalar Graph] Grad Y Match: {y_match} | PyTorch: {self.y.grad:.4f}, Manual: {grad_y_man:.4f}")


class ComplexMatrixGraph:
    """
    【Complex Example】
    演示全连接层 (Linear Layer) 的矩阵求导 (Vector-Jacobian Product)。
    这是理解 LLM 训练效率和手写算子 (Custom Function) 的基石。
    
    Formula:
        Y = X @ W + b
        Loss = Sum(Y * Upstream_Grad)  (模拟后续层传回来的梯度)
        
    Dimensions:
        X: (Batch, In_Dim)
        W: (In_Dim, Out_Dim)
        b: (Out_Dim)
        Y: (Batch, Out_Dim)
    """
    
    def __init__(
        self, 
        batch_size: int, 
        in_dim: int, 
        out_dim: int, 
        device: torch.device
    ):
        self.device = device
        
        # 初始化权重，模拟 nn.Linear
        self.X = torch.randn(
            batch_size, 
            in_dim, 
            requires_grad = True, 
            device = device
        )
        self.W = torch.randn(
            in_dim, 
            out_dim, 
            requires_grad = True, 
            device = device
        )
        self.b = torch.randn(
            out_dim, 
            requires_grad = True, 
            device = device
        )
        
        # 模拟“上一层”传回来的梯度 (Upstream Gradient dL/dY)
        # 在真实的训练中，这是 Loss 对 Y 的导数
        self.upstream_grad = torch.randn(
            batch_size, 
            out_dim, 
            device = device
        )

    def forward(self):
        """前向计算: Y = XW + b"""
        self.Y = self.X @ self.W + self.b
        return self.Y
    
    def manual_backward(self):
        """
        手动实现矩阵微积分 (VJP)
        Key concept: Transpose is the backward of Matmul.
        """
        with torch.no_grad():
            # 1. dL/dW = X^T @ (dL/dY)
            # Shape: (In, Batch) @ (Batch, Out) -> (In, Out)
            grad_W_manual = self.X.T @ self.upstream_grad
            
            # 2. dL/dX = (dL/dY) @ W^T
            # Shape: (Batch, Out) @ (Out, In) -> (Batch, In)
            grad_X_manual = self.upstream_grad @ self.W.T
            
            # 3. dL/db = Sum(dL/dY, dim=0)
            # Bias 是广播加法，反向传播时需要对 Batch 维度求和
            grad_b_manual = self.upstream_grad.sum(dim = 0)
            
            return grad_X_manual, grad_W_manual, grad_b_manual

    def verify(self):
        """验证矩阵求导逻辑"""
        # 1. Forward
        Y = self.forward()
        
        # 2. PyTorch Autograd
        # 这里我们不定义具体的 Loss 函数，而是直接从 Y 开始反向传播
        # 传入 gradient 参数等于告诉 PyTorch: "已知 dL/dY = upstream_grad，请继续往前传"
        Y.backward(gradient = self.upstream_grad)
        
        # 3. Manual Backward
        grad_X_man, grad_W_man, grad_b_man = self.manual_backward()
        
        # 4. Compare
        X_match = torch.allclose(self.X.grad, grad_X_man, atol = 1e-5)
        W_match = torch.allclose(self.W.grad, grad_W_man, atol = 1e-5)
        b_match = torch.allclose(self.b.grad, grad_b_man, atol = 1e-5)
        
        logger.info(f"[Matrix Graph] Grad X Match: {X_match}")
        logger.info(f"[Matrix Graph] Grad W Match: {W_match}")
        logger.info(f"[Matrix Graph] Grad b Match: {b_match}")
        
        # 简单的维度检查打印，帮助理解 VJP
        logger.info(f"Shape Mismatch Debug: W.grad {self.W.grad.shape} vs Manual {grad_W_man.shape}")


def main():
    # 1. Global Logger Configuration
    # 严格遵守：只在 main 中配置 basicConfig
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    
    # 2. Setup Environment
    setup_seed(42)
    device = get_device()
    logger.info(f"Running Chapter 01 on device: {device}")
    
    # 3. Run Simple Scalar Example
    logger.info("--- Starting Scalar Graph Example ---")
    scalar_graph = SimpleScalarGraph(x_val = 2.0, y_val = 3.0, device = device)
    scalar_graph.verify()
    
    # 4. Run Complex Matrix Example
    logger.info("--- Starting Matrix Graph Example (Linear Layer) ---")
    # 使用 Timer 监测反向传播验证过程
    with Timer("Matrix Autograd Verification"):
        matrix_graph = ComplexMatrixGraph(
            batch_size = 32,
            in_dim = 128,
            out_dim = 64,
            device = device
        )
        matrix_graph.verify()
        
    logger.info("Chapter 01: Computational Graph & Autograd - Completed.")

if __name__ == "__main__":
    main()