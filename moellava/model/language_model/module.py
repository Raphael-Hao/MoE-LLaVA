import brt
import typing
import torch
import torch.nn as nn
from .experts import  BRTFusedExperts
from deepspeed.moe.sharded_moe import TopKGate
from deepspeed.moe.mappings import drop_tokens
from deepspeed.moe.sharded_moe import _AllToAll, einsum
from brt.router import GatherRouter, ScatterRouter
from .phi.modeling_phi import PhiDecoderLayer

if typing.TYPE_CHECKING:
    Base = nn.Module[torch.Tensor]
else:
    Base = nn.Module

class BRTMOELayer(Base):
    def __init__(self,
                 gate: nn.Module,
                 experts: nn.Module,
                 ep_group_name,
                 ep_size,
                 num_local_experts: int,
                 use_tutel: bool = False) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.wall_clock_breakdown = False
    
    def forward(self, *input: torch.Tensor, **kwargs: typing.Any) -> torch.Tensor:
        d_model = input[0].shape[-1]
        reshaped_input = input[0].reshape(-1, d_model)
        self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input, input[1])
        dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)

        dispatched_input = drop_tokens(dispatched_input, dim=1)
        dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)
        dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)

        expert_output = self.experts(dispatched_input)
        expert_output = _AllToAll.apply(self.ep_group, expert_output)
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)
        combined_output = einsum("sec,ecm->sm", combine_weights.type_as(expert_output), expert_output)
        a = combined_output.reshape(input[0].shape)
        return a
    
    def initialize_fused_expert(self):
        self.experts.initialize_fused_expert()




class BRTMOE(nn.Module):
    def __init__(
        self,
        hidden_size,
        expert,
        capacities,
        num_experts=1,
        ep_size=1,
        k=1,
        capacity_factor=1.,
        eval_capacity_factor=1.,
        min_capacity=4,
        use_residual=False,
        noisy_gate_policy: typing.Optional[str] = None,
        drop_tokens: bool = True,
        use_rts=True,
        use_tutel: bool = False,
        enable_expert_tensor_parallelism: bool = False,
    ):  
        super(BRTMOE, self).__init__()

        self.use_residual = use_residual
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        assert num_experts % ep_size == 0, f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        self.ep_size = ep_size
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size
        experts = BRTFusedExperts(expert, capacities, self.num_local_experts )

        self.deepspeed_moe = BRTMOELayer(TopKGate(hidden_size, num_experts, k, capacity_factor, eval_capacity_factor,
                                               min_capacity, noisy_gate_policy, drop_tokens, use_rts),
                                      experts,
                                      self.expert_group_name,
                                      self.ep_size,
                                      self.num_local_experts,
                                      use_tutel=use_tutel)
        if self.use_residual:
            self.mlp = expert
            # coefficient is used for weighted sum of the output of expert and mlp
            self.coefficient = torch.nn.Linear(hidden_size, 2)
        


    
    def forward(self, hidden_states, used_tokens=None):
        output = self.deepspeed_moe(hidden_states, used_tokens)
        if self.use_residual:
            # Residual MoE
            output_mlp = self.mlp(hidden_states)
            if type(output_mlp) is tuple:
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = torch.nn.functional.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output


    def initialize_fused_expert(self):
        self.deepspeed_moe.initialize_fused_expert()