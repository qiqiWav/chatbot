"""
Processor class for Gemma2Audio.
"""

from typing import List, Optional, Union

import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
)


class Gemma2AudioProcessor(ProcessorMixin):
    r"""
    Constructs a Gemma2Audio processor which wraps a Gemma2Audio feature extractor and a Gemma2Audio tokenizer into a single processor.

    [`Gemma2AudioProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`Gemma2TokenizerFast`]. See the
    [`~Gemma2AudioProcessor.__call__`] and [`~Gemma2AudioProcessor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The feature extractor is a required input.
        tokenizer ([`Gemma2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the default chat template
                is used.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None):
        if chat_template is None:
            chat_template = self.default_chat_template
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        audios: Union[np.ndarray, List[np.ndarray]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Gemma2TokenizerFast's [`~Gemma2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audios` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audios (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audios to be prepared. Each audio can be a NumPy array.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            sampling_rate (`int`, defaults to 16000):
                The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        """

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")
        inputs = self.tokenizer(text, padding=padding, **kwargs)
        
        # print("Alex debug here")
        if audios is not None:
            # print(kwargs)
            audio_inputs = self.feature_extractor(
                audios,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                padding="max_length",
                **kwargs,
            )
            
            # print(self.feature_extractor)
            # print(f"The audio input now is {audio_inputs}")
            # print(f"The input feature shape is {audio_inputs['input_features'].shape}")
            # print(f"The input feature attention mask shape is {audio_inputs['attention_mask'].shape}")
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename attention_mask to prevent conflicts later on
            inputs.update(audio_inputs)
        return BatchFeature(data={**inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Gemma2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Gemma2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + ["feature_attention_mask"]
            )
        )
        
    # We may not need this default chat template since we have the chat template along with the tokenizer.
    @property
    def default_chat_template(self):
        """
        This default chat template is modified for the gemma tokenizer to handle audio inputs. It formats inputs in the form of a chat history. For each message in the chat history:

        - The template outputs the role of the speaker followed by the content of the message.
        - The content can be a string or a list of content elements, which may include both text and audio.
        - If a content element is an audio, the template outputs `<|audio_bos|><|AUDIO|><|audio_eos|>` to represent the audio input.

        Example:

        ```python
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://example.com/audio1.mp3"},
                {"type": "text", "text": "What's that sound?"},
            ]},
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://example.com/audio2.wav"},
                {"type": "text", "text": "How about this one?"},
            ]},
        ]

        result = template.render(messages=messages, add_generation_prompt=True)
        ```
        """
        return (
            "{{ bos_token }}"
            "{% if messages[0]['role'] == 'system' %}"
                "{{ raise_exception('System role not supported') }}"
            "{% endif %}"
            "{% set audio_count = namespace(value=0) %}"
            "{% for message in messages %}"
                "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
                    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
                "{% endif %}"
                "{% if message['role'] == 'assistant' %}"
                    "{% set role = 'model' %}"
                "{% else %}"
                    "{% set role = message['role'] %}"
                "{% endif %}"
                "{{ '<start_of_turn>' + role + '\n' }}"
                "{% if message['content'] is string %}"
                    "{{ message['content'] | trim }}"
                "{% else %}"
                    "{% for content in message['content'] %}"
                        "{% if 'audio' in content or 'audio_url' in content %}"
                            "{% set audio_count.value = audio_count.value + 1 %}"
                            "Audio {{ audio_count.value }}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
                        "{% elif 'text' in content %}"
                            "{{ content['text'] | trim }}"
                        "{% endif %}"
                    "{% endfor %}"
                "{% endif %}"
                "{{ '<end_of_turn>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{{ '<start_of_turn>model\n' }}"
            "{% endif %}"
        )