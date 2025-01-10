import torch
import timm
import math
from torch import nn
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.parameter import Parameter

class _MultiLoRA_qkv_timm(nn.Module):
    """Multiple LoRA adapters for QKV attention in ViT"""
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_qs: list,
        linear_b_qs: list,
        linear_a_vs: list,
        linear_b_vs: list,
    ):
        super().__init__()
        self.qkv = qkv
        # Store multiple LoRA adapters
        for i in range(len(linear_a_qs)):
            setattr(self, f'linear_a_q_{i}', linear_a_qs[i])
            setattr(self, f'linear_b_q_{i}', linear_b_qs[i])
            setattr(self, f'linear_a_v_{i}', linear_a_vs[i])
            setattr(self, f'linear_b_v_{i}', linear_b_vs[i])
        self.dim = qkv.in_features
        self.active_adapter = 0  # Track which adapter is currently active

    def switch_adapter(self, adapter_id: int):
        """Switch to a different LoRA adapter"""
        self.active_adapter = adapter_id

    def forward(self, x):   
        qkv = self.qkv(x)  # B,N,3*org_C
        # Use the currently active adapter
        linear_a_q = getattr(self, f'linear_a_q_{self.active_adapter}')
        linear_b_q = getattr(self, f'linear_b_q_{self.active_adapter}')
        linear_a_v = getattr(self, f'linear_a_v_{self.active_adapter}')
        linear_b_v = getattr(self, f'linear_b_v_{self.active_adapter}')
        
        new_q = linear_b_q(linear_a_q(x))
        new_v = linear_b_v(linear_a_v(x))
        qkv[:, :, :self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv

class MultiLoRAViT(nn.Module):
    def __init__(self, adapter_config: dict, r: int, pretrained=True, layer=-1):
        super().__init__()
        self.layer = layer
        self.adapter_config = adapter_config
        self.num_adapters = len(adapter_config)
        self.adapter_names = list(adapter_config.keys())
        
        # Initialize base ViT model
        self.model = timm.create_model('vit_small_patch16_224', num_classes=0, pretrained=pretrained)
        
        # Freeze base model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize multiple classifier heads (one per adapter)
        self.heads = nn.ModuleDict({
            name: nn.Linear(self.model.embed_dim, num_classes) 
            for name, num_classes in adapter_config.items()
        })
        
        # Initialize LoRA adapters
        self.w_As = []  # Store all A matrices
        self.w_Bs = []  # Store all B matrices
        
        # Perform LoRA surgery on attention blocks
        for t_layer_i, blk in enumerate(self.model.blocks):
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            
            # Create multiple adapters per attention block
            w_a_linear_qs = []
            w_b_linear_qs = []
            w_a_linear_vs = []
            w_b_linear_vs = []
            
            for _ in range(self.num_adapters):
                # Q adapter
                w_a_q = nn.Linear(dim, r, bias=False)
                w_b_q = nn.Linear(r, dim, bias=False)
                # V adapter
                w_a_v = nn.Linear(dim, r, bias=False)
                w_b_v = nn.Linear(r, dim, bias=False)
                
                w_a_linear_qs.append(w_a_q)
                w_b_linear_qs.append(w_b_q)
                w_a_linear_vs.append(w_a_v)
                w_b_linear_vs.append(w_b_v)
                
                self.w_As.extend([w_a_q, w_a_v])
                self.w_Bs.extend([w_b_q, w_b_v])
            
            # Replace QKV with multi-adapter version
            blk.attn.qkv = _MultiLoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_qs,
                w_b_linear_qs,
                w_a_linear_vs,
                w_b_linear_vs,
            )
        
        self.reset_parameters()
        self.active_adapter = self.adapter_names[0]

    def switch_adapter(self, adapter_name: str):
        """Switch to a different LoRA adapter"""
        assert adapter_name in self.adapter_config, f"Unknown adapter: {adapter_name}"
        self.active_adapter = adapter_name
        # Switch adapter in all attention blocks
        adapter_id = self.adapter_names.index(adapter_name)
        for blk in self.model.blocks:
            blk.attn.qkv.switch_adapter(adapter_id)

    def reset_parameters(self):
        """Initialize LoRA parameters"""
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x):
        x = self.model(x)
        x = self.heads[self.active_adapter](x)
        return x

    def save_model(self, epoch: int, exp: str, adapter_name: str = None):
        """Save model parameters for one or all adapters"""
        if adapter_name is not None:
            # Save single adapter
            filename = f'{exp}/lora_adapter_{adapter_name}.safetensors'
            self._save_adapter(filename, adapter_name, epoch)
        else:
            # Save all adapters
            for name in self.adapter_names:
                filename = f'{exp}/lora_adapter_{name}.safetensors'
                self._save_adapter(filename, name, epoch)

    def _save_adapter(self, filename: str, adapter_name: str, epoch: int):
        """Helper method to save a single adapter's parameters"""
        adapter_id = self.adapter_names.index(adapter_name)
        # Save LoRA parameters for this adapter
        a_tensors = {}
        b_tensors = {}
        
        for i in range(len(self.w_As)):
            if i // 2 % self.num_adapters == adapter_id:
                a_tensors[f"w_a_{i:03d}"] = self.w_As[i].weight
                b_tensors[f"w_b_{i:03d}"] = self.w_Bs[i].weight
        
        # Save classifier head
        head_tensors = {
            f"head_{adapter_name}": self.heads[adapter_name].weight
        }
        
        merged_dict = {
            **a_tensors, 
            **b_tensors, 
            **head_tensors,
            'epoch': torch.tensor(epoch)
        }
        save_file(merged_dict, filename)

    def load_model(self, exp: str, adapter_name: str = None):
        """Load model parameters for one or all adapters"""
        if adapter_name is not None:
            # Load single adapter
            filename = f'{exp}/lora_adapter_{adapter_name}.safetensors'
            return self._load_adapter(filename, adapter_name)
        else:
            # Load all adapters
            epochs = []
            for name in self.adapter_names:
                filename = f'{exp}/lora_adapter_{name}.safetensors'
                epochs.append(self._load_adapter(filename, name))
            return epochs

    def _load_adapter(self, filename: str, adapter_name: str):
        """Helper method to load a single adapter's parameters"""
        adapter_id = self.adapter_names.index(adapter_name)
        with safe_open(filename, framework="pt") as f:
            # Load LoRA parameters
            for i in range(len(self.w_As)):
                if i // 2 % self.num_adapters == adapter_id:
                    saved_key = f"w_a_{i:03d}"
                    self.w_As[i].weight = Parameter(f.get_tensor(saved_key))
                    
                    saved_key = f"w_b_{i:03d}"
                    self.w_Bs[i].weight = Parameter(f.get_tensor(saved_key))
            
            # Load classifier head
            self.heads[adapter_name].weight = Parameter(f.get_tensor(f"head_{adapter_name}"))
            
            epoch = f.get_tensor('epoch')
            return epoch.item()