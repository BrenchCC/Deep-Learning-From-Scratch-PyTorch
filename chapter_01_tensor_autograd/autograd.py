import os
import sys
import math
import logging

import torch

sys.path.append(os.getcwd())
from utils import setup_seed, get_device, Timer, build_dot

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
        logger.info(f"[Matrix Graph] Batch Size: {self.X.shape[0]}")
        logger.info(f"[Matrix Graph] In_Dim: {self.X.shape[1]}")
        logger.info(f"[Matrix Graph] Out_Dim: {self.W.shape[1]}")

        logger.info(f"[Matrix Graph] X shape: {self.X.shape}")
        logger.info(f"[Matrix Graph] X: {self.X}")
        logger.info(f"[Matrix Graph] W shape: {self.W.shape}")
        logger.info(f"[Matrix Graph] W: {self.W}")
        logger.info(f"[Matrix Graph] b shape: {self.b.shape}")
        logger.info(f"[Matrix Graph] b: {self.b}")
        logger.info(f"[Matrix Graph] Upstream Grad: {self.upstream_grad}")
        Y = self.forward()
        logger.info(f"[Matrix Graph] Forward Output Shape: {Y.shape}")
        logger.info(f"[Matrix Graph] Y: {Y}")
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

class FourLayerNetWithLoss:
    """
    四层网络 + MSE Loss
    用于完整演示 forward → loss → backward → 梯度验证
    
    Architecture:
        h1 = W1 x + b1
        h2 = ReLU(h1)
        h3 = W2 h2 + b2
        ŷ  = W3 h3 + b3
        L  = (ŷ - y)^2
    """

    def __init__(self, device: torch.device):
        self.device = device

        # 固定一个 2-d 输入，方便手推和验证
        self.x = torch.tensor(
            [1.0, 2.0],
            device = device
        )
        self.y = torch.tensor(
            0.5,
            device = device
        )

        # 权重参数
        self.w1 = torch.randn(
            2, 2,
            requires_grad = True,
            device = device
        )
        self.b1 = torch.randn(
            2,
            requires_grad = True,
            device = device
        )

        self.w2 = torch.randn(
            2, 2,
            requires_grad = True,
            device = device
        )
        self.b2 = torch.randn(
            2,
            requires_grad = True,
            device = device
        )

        self.w3 = torch.randn(
            1, 2,
            requires_grad = True,
            device = device
        )
        self.b3 = torch.randn(
            1,
            requires_grad = True,
            device = device
        )

    def forward(self):
        """前向传播：构建完整计算图"""
        self.h1 = self.w1 @ self.x + self.b1
        self.h1.retain_grad()

        self.h2 = torch.relu(self.h1)
        self.h2.retain_grad()
        
        self.h3 = self.w2 @ self.h2 + self.b2
        self.h3.retain_grad()

        self.y_pred = self.w3 @ self.h3 + self.b3
        self.y_pred.retain_grad()

        self.loss = (self.y_pred - self.y) ** 2
        self.loss.retain_grad()

        return self.loss

    def manual_backward(self):
        """
        手写反向传播（严格对应数学推导）
        """
        with torch.no_grad():
            # dL/dy
            dl_dypred = 2 * (self.y_pred - self.y)

            # -------- Layer 4 --------
            dl_dw3 = dl_dypred * self.h3.unsqueeze(0)
            dl_db3 = dl_dypred  
            dl_dh3 = self.w3.t() * dl_dypred

            # -------- Layer 3 --------
            dl_dw2 = dl_dh3 @ self.h2.unsqueeze(0)
            dl_db2 = dl_dh3
            dl_dh2 = self.w2.t() @ dl_dh3

            # -------- Layer 2 --------
            relu_mask = (self.h1 > 0).float().unsqueeze(1)
            dl_dh1 = dl_dh2 * relu_mask
            
            # -------- Layer 1 --------
            dl_dw1 = dl_dh1 @ self.x.unsqueeze(0)
            dl_db1 = dl_dh1

            return {
                "w1": dl_dw1,
                "b1": dl_db1.view_as(self.b1),
                "w2": dl_dw2,
                "b2": dl_db2.view_as(self.b2),
                "w3": dl_dw3,
                "b3": dl_db3.view_as(self.b3),
            }
    
    def verify(self):
        """ PyTorch Autograd 与手写梯度"""
        loss = self.forward()
        dot = build_dot(
            loss,
            params={
                "w1": self.w1,
                "b1": self.b1,
                "w2": self.w2,
                "b2": self.b2,
                "w3": self.w3,
                "b3": self.b3,
            },
        )
        dot.render("chapter_01_tensor_autograd/four_layer_autograd_graph", cleanup=True)

        loss.backward()

        manual_grads = self.manual_backward()

        for name, param in [
            ("w1", self.w1),
            ("b1", self.b1),
            ("w2", self.w2),
            ("b2", self.b2),
            ("w3", self.w3),
            ("b3", self.b3),
        ]:
            match = torch.allclose(param.grad, manual_grads[name], atol=1e-5)
            logger.info(
                f"[Grad Check] {name}: match={match} | "
                f"autograd={param.grad.flatten()[:3]} | "
                f"manual={manual_grads[name].flatten()[:3]}"
            )


def main():
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
        
    # 5. Run a Four-Layer Network Example
    logger.info("--- Starting Four-Layer Network Example with Loss Function ---")
    four_layer_graph = FourLayerNetWithLoss(device = device)
    four_layer_graph.verify()
    
    logger.info("Chapter 01: Computational Graph & Autograd - Completed.")

if __name__ == "__main__":
    main()