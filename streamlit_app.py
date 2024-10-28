import streamlit as st
import torch
import librosa
import tempfile
import os
from nexa_omni_audio.processing_gemma2_audio import Gemma2AudioProcessor
from nexa_omni_audio.modeling_gemma2_audio import Gemma2AudioForConditionalGeneration

@st.cache_resource
def load_model():
    """Load model and processor with caching"""
    try:
        model = Gemma2AudioForConditionalGeneration.from_pretrained(
            "nexaAIDev/nano-omini-instruct",
            device_map="cuda",
            torch_dtype=torch.bfloat16
        )
        processor = Gemma2AudioProcessor.from_pretrained(
            "nexaAIDev/gemma2-2b-audio-whisper-medium-full-model-it"
        )
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "right"
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def process_audio(audio_data, sample_rate, task_type, model, processor):
    """Process audio using the loaded model"""
    try:
        # Resample if needed
        if sample_rate != processor.feature_extractor.sampling_rate:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=processor.feature_extractor.sampling_rate
            )

        # Prepare prompt based on task
        prompts = {
            "Transcribe": '<bos><start_of_turn>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n<|transcribe|><end_of_turn>\n<start_of_turn>model\n',
            "Summarize": '<bos><start_of_turn>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\nSummarize the key points of this audio<end_of_turn>\n<start_of_turn>model\n',
            "Meeting Notes": '<bos><start_of_turn>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\nList the actionable items from this meeting<end_of_turn>\n<start_of_turn>model\n',
            "Lecture Notes": '<bos><start_of_turn>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\nWhat are the main topics and key points?<end_of_turn>\n<start_of_turn>model\n'
        }

        # Process inputs
        inputs = processor(
            text=[prompts[task_type]],
            audios=[audio_data],
            return_tensors="pt",
            sampling_rate=processor.feature_extractor.sampling_rate,
            padding=True
        )
        inputs = inputs.to('cuda')

        # Initialize generation
        cur_ids = inputs['input_ids']
        cur_attention_mask = inputs['attention_mask']
        input_features = inputs['input_features']
        feature_attention_mask = inputs['feature_attention_mask']
        
        generated_text = []
        
        # Generation loop
        with st.status("Generating response...") as status:
            for i in range(50):
                out = model(
                    cur_ids,
                    attention_mask=cur_attention_mask,
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    use_cache=False
                )
                
                next_token = out.logits[:, -1].argmax(dim=-1)
                next_token = next_token.unsqueeze(0)
                next_word = processor.decode(next_token.squeeze())
                
                generated_text.append(next_word)
                status.update(label=f"Token {i+1}/50: {next_word}")
                
                cur_ids = torch.cat([cur_ids, next_token], dim=-1)
                cur_attention_mask = torch.cat([cur_attention_mask, torch.ones_like(next_token)], dim=-1)
                
                if next_word == "<end_of_turn>":
                    break
        
        return " ".join(generated_text)

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def main():
    st.title("Audio Processing Assistant")
    
    # Load model
    with st.spinner("Loading model..."):
        model, processor = load_model()
        if model is None or processor is None:
            st.error("Failed to load model. Please check the model repository and your access permissions.")
            return
    
    # Task selection
    task_type = st.sidebar.selectbox(
        "Select Task",
        ["Transcribe", "Summarize", "Meeting Notes", "Lecture Notes"],
        help="Choose the type of processing you want to perform on the audio"
    )
    
    # File uploader
    audio_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'ogg'])
    
    if audio_file:
        # Display audio player
        st.audio(audio_file)
        
        # Process button
        if st.button("Process Audio"):
            # Load audio
            with st.spinner("Loading audio file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    audio_data, sample_rate = librosa.load(tmp_file.name)
                    os.unlink(tmp_file.name)
            
            # Process audio
            with st.spinner("Processing audio..."):
                result = process_audio(audio_data, sample_rate, task_type, model, processor)
                if result:
                    st.subheader("Results:")
                    st.write(result)
    
    # Add sidebar info
    st.sidebar.markdown("""
    ### Task Types:
    - **Transcribe**: Convert audio to text
    - **Summarize**: Generate a concise summary
    - **Meeting Notes**: Extract actionable items
    - **Lecture Notes**: List main topics and key points
    """)

    # Display CUDA info
    if torch.cuda.is_available():
        st.sidebar.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.warning("No GPU detected. Running on CPU.")

if __name__ == "__main__":
    main()