import brt
import torch
import torch.nn as nn
import copy
from brt.jit import make_jit_kernel
from transformers.activations import ACT2FN

class BRTFusedExperts(nn.Module):
    def __init__(self, expert, capacities, num_local_experts=1):
        super(BRTFusedExperts, self).__init__()
        self.deepspeed_experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
        hidden_size = expert.fc1.in_features
        intermediate_size = expert.fc1.out_features
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts

        wi_denses = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=True, dtype=torch.float16)
            for _ in range(num_local_experts)
        ])

        wo_denses = nn.ModuleList([
            nn.Linear(intermediate_size, hidden_size, bias=True, dtype=torch.float16)
            for _ in range(num_local_experts)
        ])

        sample_inputs = [torch.randn((i, hidden_size), device = "meta", dtype=torch.float16) for i in capacities]
        self.fused_wi = make_jit_kernel(
            nn.ModuleList(wi_denses),
            sample_inputs,
            "forward",
            opt_level="homo_fuse"
        )
        sample_inputs = [torch.randn((i, intermediate_size), device = "meta", dtype=torch.float16) for i in capacities]
        self.fused_wo = make_jit_kernel(
            nn.ModuleList(wo_denses),
            sample_inputs,
            "forward",
            opt_level="homo_fuse"
        )

    def forward(self, inputs: torch.Tensor):
        torch.cuda.current_stream().synchronize()
        origin_shape = inputs.shape
        branch_capacities = torch.tensor([inputs.shape[2] for _ in range(self.num_local_experts)], device="cuda", dtype=torch.int32)
        inputs = torch.flatten(inputs, start_dim=0, end_dim=2)
        wi_out = torch.empty(
            (inputs.shape[0], self.intermediate_size), device=inputs.device, dtype=torch.float16
        )
        self.fused_wi(
            shared_inputs=[wi_out, inputs],
            standalone_inputs=self.fused_wi_standalone_inputs,
            capacities=branch_capacities,
        )
        wo_out = torch.empty(
            (inputs.shape[0], self.hidden_size), device=inputs.device, dtype=torch.float16
        )
        torch.cuda.current_stream().synchronize()
        self.fused_wo(
            shared_inputs=[wo_out, wi_out],
            standalone_inputs=self.fused_wo_standalone_inputs,
            capacities=branch_capacities,
        )

        wo_out = torch.reshape(wo_out, origin_shape)
        return wo_out
    
    def initialize_fused_expert(self):
        self.fused_wi_standalone_inputs = []
        self.fused_wo_standalone_inputs = []
        for expert in self.deepspeed_experts:
            self.fused_wi_standalone_inputs.extend([expert.fc1.weight])
            self.fused_wi_standalone_inputs.extend([expert.fc1.bias])
            self.fused_wo_standalone_inputs.extend([expert.fc2.weight])
            self.fused_wo_standalone_inputs.extend([expert.fc2.bias])




