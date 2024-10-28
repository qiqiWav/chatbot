"""Gemma2Audio model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING

# CONFIG_MAPPING = {
#     "gemma2_audio_encoder": "Gemma2AudioEncoderConfig",
#     "gemma2_audio": "Gemma2AudioConfig",
#     "gemma2": "Gemma2Config",
# }

logger = logging.get_logger(__name__)


class Gemma2AudioEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Gemma2AudioEncoder`]. It is used to instantiate a
    Gemma2-Audio audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the Gemma2-Audio
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `Gemma2AudioProcessor` class.
        encoder_layers (`int`, *optional*, defaults to 32):
            Number of encoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        d_model (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.

    Example:

    ```python
    >>> from transformers import Gemma2AudioEncoderConfig, Gemma2AudioEncoder

    >>> # Initializing a Gemma2AudioEncoderConfig
    >>> configuration = Gemma2AudioEncoderConfig()

    >>> # Initializing a Gemma2AudioEncoder (with random weights)
    >>> model = Gemma2AudioEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gemma2_audio_encoder"

    def __init__(
        self,
        num_mel_bins=80,
        encoder_layers=24,
        encoder_attention_heads=16,
        encoder_ffn_dim=4096,
        encoder_layerdrop=0.0,
        d_model=1024,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        scale_embedding=False,
        init_std=0.02,
        max_source_positions=1500,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.num_hidden_layers = encoder_layers
        self.init_std = init_std
        self.scale_embedding = (
            scale_embedding  # scale factor will be sqrt(d_model) if True
        )
        self.max_source_positions = max_source_positions
        
        


# We need to define two configurations! One for the audio encoder and one for the full model
class Gemma2AudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Gemma2AudioForConditionalGeneration`]. It is used to instantiate an
    Gemma2-Audio model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Gemma2-Audio.

    e.g. [Qwen/Gemma2-Audio-7B](https://huggingface.co/Qwen/Gemma2-Audio-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the audio backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        audio_token_index (`int`, *optional*, defaults to 151646):
            The image token index to encode the image prompt.

    Example:

    ```python
    >>> from transformers import Gemma2AudioForConditionalGeneration, Gemma2AudioConfig, Gemma2AudioEncoderConfig, Gemma2Config

    >>> # Initializing a Gemma2AudioEncoder config
    >>> audio_config = Gemma2AudioEncoderConfig()

    >>> # Initializing a Gemma2 config
    >>> text_config = Gemma2Config()

    >>> # Initializing a Gemma2Audio configuration
    >>> configuration = Gemma2AudioConfig(audio_config, text_config)

    >>> # Initializing a model from the gemma2-audio style configuration
    >>> model = Gemma2AudioForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gemma2_audio"
    is_composition = False

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_index=151646,
        hidden_size=1024,   # especially for the deepspeed training, alias for d_model in this case
        **kwargs,
    ):
        self.audio_token_index = audio_token_index
        self.hidden_size = hidden_size
        if isinstance(audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"]
                if "model_type" in audio_config
                else "gemma2_audio_encoder"
            )
            audio_config = Gemma2AudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Gemma2AudioEncoderConfig(
                d_model=1024,
                encoder_attention_heads=16,
                encoder_ffn_dim=4096,
                encoder_layerdrop=0.0,
                encoder_layers=24,
                num_mel_bins=80,
                max_source_positions=1500,
                scale_embedding=False,
                activation_function="gelu",
            )

        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config["model_type"] = (
                text_config["model_type"] if "model_type" in text_config else "gemma2"
            )
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["gemma2"]()

        self.text_config = text_config

        super().__init__(**kwargs)