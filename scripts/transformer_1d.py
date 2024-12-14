import torch
from torch import nn

from typing import Any, Dict, Tuple, Optional
from diffusers.models.modeling_utils import ModelMixin, LegacyModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config, ConfigMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import deprecate, is_torch_version, logging
from transformer_block import BasicTransformerBlock
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DiffuserTransformerPolicyOutput(Transformer2DModelOutput):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class DiffuserTransformerValueOutput(Transformer2DModelOutput):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Transformer1DModel(ModelMixin, ConfigMixin):
    """

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_train_timesteps ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """
    _supports_gradient_checkpointing = False
    _no_split_modules = ["BasicTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        num_layers: int = 1,
        dropout: float = 0.0,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_zero",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        interpolation_scale: float = None,
        positional_embeddings: str = "sinusoidal",
        num_positional_embeddings: int = 1000,
        ff_inner_mult: int = 2,
        **kwargs
    ):
        super().__init__()
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.ff_inner_dim = self.inner_dim * self.config.ff_inner_mult
        self.transformer_blocks = nn.ModuleList(
            [   
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    ff_inner_dim=self.ff_inner_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=self.config.norm_type,
                    positional_embeddings=self.config.positional_embeddings if i == 0 else None,
                    num_positional_embeddings=self.config.num_positional_embeddings if i == 0 else None,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for i in range(self.config.num_layers)
            ]
        )
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
            """
            The [`Transformer2DModel`] forward method.

            Args:
                hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.Tensor` of shape `(batch size, channel, height, width)` if continuous):
                    Input `hidden_states`.
                timestep ( `torch.LongTensor`, *optional*):
                    Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
                attention_mask ( `torch.Tensor`, *optional*):
                    An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                    is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                    negative values to the attention scores corresponding to "discard" tokens.
            """

            if attention_mask is not None and attention_mask.ndim == 2:
                # assume that mask is expressed as:
                #   (1 = keep,      0 = discard)
                # convert mask into a bias that can be added to attention scores:
                #       (keep = +0,     discard = -10000.0)
                attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # 2. Blocks
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    timestep=timestep,
                )
            # hidden states are here, do our own stuff with hidden state            
            return hidden_states



class DiffuserTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False
    _no_split_modules = ["BasicTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 1024 // 8,
        num_layers: int = 1,
        dropout: float = 0.0,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        upcast_attention: bool = False,
        norm_type: str = "ada_joker_norm_zero",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        interpolation_scale: float = None,
        positional_embeddings: str = "sinusoidal",
        num_positional_embeddings: int = 1000,
        ff_inner_mult: int = 2,
        state_dim: int = 1,
        action_dim: int = 1  
    ):
        super().__init__()
        self.transformer = Transformer1DModel(**self.config)
        self.embed_state = nn.Linear(self.config.state_dim, self.transformer.inner_dim)
        self.embed_action = nn.Linear(self.config.action_dim, self.transformer.inner_dim)
        self.project_state =  nn.Linear(self.transformer.inner_dim, self.config.state_dim)
        self.project_action =  nn.Linear(self.transformer.inner_dim, self.config.action_dim)
        # NOTE, we are not using Fourier positional embeddings, and the 256 thing

    def forward(
        self,
        # history_states, # B x 1 x s 
        # history_actions, # B x 1 x a
        sample_states: torch.Tensor, # b x H x s 
        sample_actions: torch.Tensor, # b x H x s 
        timestep: torch.LongTensor,
        # attention_mask: Optional[torch.Tensor] = None,
    ):
        # history_state_embeds = self.embed_state(history_states)
        # history_actions_embeds = self.embed_action(history_actions)
        # combined_states = torch.cat([history_states, sample_states])
        # combined_actions = torch.cat([history_actions, sample_actions])

        assert sample_states.shape[1] == sample_actions.shape[1], 'state action mistmatch'

        state_embeds = self.embed_state(sample_states)
        action_embeds = self.embed_action(sample_actions)

        horizon = sample_states.shape[1]
        batch_size = sample_states.shape[0]

        combined_embeds  = torch.stack(
            (state_embeds, action_embeds), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*horizon, self.transformer.inner_dim)

        hidden_embeds = self.transformer(combined_embeds, timestep)
        hidden_embeds = hidden_embeds.reshape(batch_size, horizon, 2, self.transformer.inner_dim).permute(0, 2, 1, 3)
        
        out_state_embeds = hidden_embeds[:, 0]
        out_action_embeds = hidden_embeds[:, 1]
        
        out_states = self.project_state(out_state_embeds)
        out_actions = self.project_action(out_action_embeds)

        return (out_states, out_actions)
    

class DiffuserTransformerPolicy(nn.Module):
     
    def __init__(self, diffuser):
        super().__init__()
        self.diffuser = diffuser
        self.action_dim = self.diffuser.config.action_dim
        self.state_dim = self.diffuser.config.state_dim
        self.dtype = self.diffuser.dtype
        self.device = self.diffuser.device
        
    def forward(self, sample, timestep):
        if sample.shape[1] == self.action_dim + self.state_dim: 
            sample = sample.permute(0, 2, 1)

        assert sample.shape[2] == self.action_dim + self.state_dim

        sample_actions = sample[:,:, :self.action_dim]
        sample_states = sample[:,:, self.action_dim:]
        out_states, out_actions = self.diffuser(sample_states, sample_actions, timestep)
        out_sample = torch.cat([out_actions, out_states], dim=-1)
        out_sample = out_sample.permute(0, 2, 1)
        return DiffuserTransformerPolicyOutput(sample=out_sample)
    


class ValueTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False
    _no_split_modules = ["BasicTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 1024 // 8,
        num_layers: int = 1,
        dropout: float = 0.0,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        upcast_attention: bool = False,
        norm_type: str = "ada_joker_norm_zero",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        interpolation_scale: float = None,
        positional_embeddings: str = "sinusoidal",
        num_positional_embeddings: int = 1000,
        # horizon: int = 500,
        ff_inner_mult: int = 2,
        state_dim: int = 1,
        action_dim: int = 1  
    ):
        super().__init__()
        # num_positional_embeddings = horizon * 2 + 1
        self.transformer = Transformer1DModel(**self.config)
        self.embed_state = nn.Linear(self.config.state_dim, self.transformer.inner_dim)
        self.embed_action = nn.Linear(self.config.action_dim, self.transformer.inner_dim)

        # NOTE, we are not using Fourier positional embeddings, and the 256 thing
        self.value_embedding = nn.Embedding(num_embeddings=1, embedding_dim=self.transformer.inner_dim)
        self.value_head = nn.Linear(self.transformer.inner_dim, 1)
    
    def forward(self, sample, timestep, return_dict=True):
        if sample.shape[1] == self.config.action_dim + self.config.state_dim: 
            sample = sample.permute(0, 2, 1)

        assert sample.shape[2] == self.config.action_dim + self.config.state_dim

        sample_actions = sample[:,:, :self.config.action_dim]
        sample_states = sample[:,:, self.config.action_dim:]
        value = self.forward_divided(sample_states, sample_actions, timestep)
        if return_dict:
            return DiffuserTransformerValueOutput(sample=value)
        return (value,)
    
    def forward_divided(
        self,
        # history_states, # B x 1 x s 
        # history_actions, # B x 1 x a
        sample_states: torch.Tensor, # b x H x s 
        sample_actions: torch.Tensor, # b x H x s 
        timestep: torch.LongTensor,
        # attention_mask: Optional[torch.Tensor] = None,
    ):
        # history_state_embeds = self.embed_state(history_states)
        # history_actions_embeds = self.embed_action(history_actions)
        # combined_states = torch.cat([history_states, sample_states])
        # combined_actions = torch.cat([history_actions, sample_actions])

        assert sample_states.shape[1] == sample_actions.shape[1], 'state action mistmatch'

        state_embeds = self.embed_state(sample_states)
        action_embeds = self.embed_action(sample_actions)

        horizon = sample_states.shape[1]
        batch_size = sample_states.shape[0]

        combined_embeds  = torch.stack(
            (state_embeds, action_embeds), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*horizon, self.transformer.inner_dim)
        
        value_token = self.value_embedding(torch.zeros(batch_size, 1, device=self.device, dtype=int))
        combined_embeds = torch.cat([combined_embeds, value_token], dim=1)

        hidden_embeds = self.transformer(combined_embeds, timestep)
        hidden_value_embed = hidden_embeds[:,-1,:] # B, inner_dim
        value = self.value_head(hidden_value_embed)
        return value
    
class ActionProposalTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False
    _no_split_modules = ["BasicTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 1024 // 8,
        num_layers: int = 1,
        dropout: float = 0.0,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        upcast_attention: bool = False,
        norm_type: str = "ada_joker_norm_zero",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        interpolation_scale: float = None,
        positional_embeddings: str = "sinusoidal",
        num_positional_embeddings: int = 1000,
        ff_inner_mult: int = 2,
        state_dim: int = 1,
        action_dim: int = 1  
    ):
        super().__init__()
        self.transformer = Transformer1DModel(**self.config)
        self.embed_state = nn.Linear(self.config.state_dim, self.transformer.inner_dim)
        self.embed_action = nn.Linear(self.config.action_dim, self.transformer.inner_dim)
        self.project_action =  nn.Linear(self.transformer.inner_dim, self.config.action_dim)
        # NOTE, we are not using Fourier positional embeddings, and the 256 thing

    def forward(
        self,
        # history_states, # B x 1 x s 
        # history_actions, # B x 1 x a
        initial_state: torch.Tensor, # b x 1 x s 
        sample_actions: torch.Tensor, # b x H x s 
        timestep: torch.LongTensor,
        # attention_mask: Optional[torch.Tensor] = None,
    ):
        # history_state_embeds = self.embed_state(history_states)
        # history_actions_embeds = self.embed_action(history_actions)
        # combined_states = torch.cat([history_states, sample_states])
        # combined_actions = torch.cat([history_actions, sample_actions])

        # assert initial_state.shape[1] == sample_actions.shape[1], 'state action mistmatch'
        if len(initial_state.shape) == 2:
            initial_state = initial_state.unsqueeze(1)
        assert initial_state.shape[1] == 1, 'only want one state'

        state_embed = self.embed_state(initial_state)
        action_embeds = self.embed_action(sample_actions)

        horizon = sample_actions.shape[1]
        batch_size = sample_actions.shape[0]

        combined_embeds  = torch.stack(
            (state_embed, action_embeds), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, horizon + 1, self.transformer.inner_dim)

        hidden_embeds = self.transformer(combined_embeds, timestep)
        out_action_embeds = hidden_embeds[:, 1:]        
        
        out_actions = self.project_action(out_action_embeds)

        return out_actions
    
class DynamicsTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False
    _no_split_modules = ["BasicTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 1024 // 8,
        num_layers: int = 1,
        dropout: float = 0.0,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        upcast_attention: bool = False,
        norm_type: str = "ada_joker_norm_zero",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        interpolation_scale: float = None,
        positional_embeddings: str = "sinusoidal",
        num_positional_embeddings: int = 1000,
        ff_inner_mult: int = 2,
        state_dim: int = 1,
        action_dim: int = 1  
    ):
        super().__init__()
        self.transformer = Transformer1DModel(**self.config)
        self.embed_state = nn.Linear(self.config.state_dim, self.transformer.inner_dim)
        self.embed_action = nn.Linear(self.config.action_dim, self.transformer.inner_dim)
        self.project_state =  nn.Linear(self.transformer.inner_dim, self.config.state_dim)
        # NOTE, we are not using Fourier positional embeddings, and the 256 thing

    def forward(
        self,
        # history_states, # B x 1 x s 
        # history_actions, # B x 1 x a
        sample_states: torch.Tensor, # b x H x s 
        sample_actions: torch.Tensor, # b x H x s 
        timestep: torch.LongTensor,
        # attention_mask: Optional[torch.Tensor] = None,
    ):
        assert sample_states.shape[1] == sample_actions.shape[1], 'state action mistmatch'

        state_embeds = self.embed_state(sample_states)
        action_embeds = self.embed_action(sample_actions)

        horizon = sample_states.shape[1]
        batch_size = sample_states.shape[0]

        combined_embeds  = torch.stack(
            (state_embeds, action_embeds), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*horizon, self.transformer.inner_dim)

        hidden_embeds = self.transformer(combined_embeds, timestep)
        hidden_embeds = hidden_embeds.reshape(batch_size, horizon, 2, self.transformer.inner_dim).permute(0, 2, 1, 3)
        
        out_state_embeds = hidden_embeds[:, 0]
        
        out_states = self.project_state(out_state_embeds)
        return out_states
    
model_type_to_class = {
    "diffusion_transformer": DiffuserTransformer,
    "action_transformer": ActionProposalTransformer,
    "dynamics_transformer": DynamicsTransformer,
}
