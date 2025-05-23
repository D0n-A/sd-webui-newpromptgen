import os
import html
import torch
import transformers
import gradio as gr
import logging 
import asyncio 
import functools 
import anyio 
import cachetools 
from cachetools import LRUCache 

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple 

from modules import shared, generation_parameters_copypaste, scripts, script_callbacks, devices, ui

logger = logging.getLogger('promptgen')

class Model:
    name = None
    model = None
    tokenizer = None

model_list = [
    "AUTOMATIC/promptgen-lexart",
    "AUTOMATIC/promptgen-majinai-safe",
    "AUTOMATIC/promptgen-majinai-unsafe",
    "succinctly/text2image-prompt-generator",
    "microsoft/Promptist",
    "RamAnanth1/distilgpt2-sd-prompts",
    "Gustavosta/MagicPrompt-Stable-Diffusion",
    "FredZhang7/distilgpt2-stable-diffusion-v2"
]
available_models = []
current = Model() 
current_device_name = None 
base_dir = scripts.basedir()

prompt_cache: Optional[LRUCache] = None
cache_initialized_with_size: Optional[int] = None 

class PromptGenerationRequest(BaseModel):
    model_name: str = Field(..., description="Name of the Hugging Face model to use.")
    text: str = Field(..., description="Input text to generate prompts from.")
    # batch_count is now the number of times to call generate for the same text, each call producing num_return_sequences
    batch_count: int = Field(1, gt=0, description="Number of times to repeat the generation process for the given input text.") 
    # batch_size is removed in favor of num_return_sequences for clarity in API
    # For UI, the old "batch_size" slider will be repurposed or removed.
    # The new "num_return_sequences_slider_ui" will control this.
    num_return_sequences: int = Field(default=1, gt=0, description="Number of alternative prompts to generate from the input text per batch_count iteration.")
    min_length: int = Field(20, gt=0, description="Minimum length of generated prompts.")
    max_length: int = Field(150, gt=0, description="Maximum length of generated prompts.")
    num_beams: int = Field(1, ge=1, description="Number of beams for beam search.")
    temperature: float = Field(1.0, gt=0, description="Sampling temperature (must be > 0).")
    repetition_penalty: float = Field(1.0, ge=1.0, description="Repetition penalty.")
    length_penalty: float = Field(1.0, description="Length penalty.")
    sampling_mode: str = Field("Top K", enum=["Top K", "Top P"], description="Sampling mode: 'Top K' or 'Top P'.")
    top_k: int = Field(50, ge=0, description="Top K for sampling.") 
    top_p: float = Field(1.0, ge=0, le=1.0, description="Top P for sampling.") 

class PromptGenerationResponse(BaseModel):
    prompts: List[str]
    model_name: str
    input_text: str
    detail: Optional[str] = None

def _initialize_cache(): # As before
    global prompt_cache, cache_initialized_with_size
    if not hasattr(shared.opts, 'data'): logger.debug("PromptGen Cache: shared.opts.data not available yet."); return
    enable_cache = shared.opts.data.get("promptgen_enable_cache", True)
    cache_size = int(shared.opts.data.get("promptgen_cache_size", 100))
    if not enable_cache:
        if prompt_cache is not None: logger.info("PromptGen Cache: Disabling cache.")
        if prompt_cache is not None: prompt_cache.clear() 
        prompt_cache = None; cache_initialized_with_size = None
        return
    if cache_size <= 0:
        if prompt_cache is not None: logger.info("PromptGen Cache: Disabling cache (size <= 0).")
        if prompt_cache is not None: prompt_cache.clear()
        prompt_cache = None; cache_initialized_with_size = None
        return
    if prompt_cache is None or cache_initialized_with_size != cache_size:
        logger.info(f"PromptGen Cache: Initializing LRU cache size {cache_size}.")
        prompt_cache = LRUCache(maxsize=cache_size); cache_initialized_with_size = cache_size

def _get_device_logic() -> torch.device: # As before
    global current_device_name
    selected_device_option = shared.opts.data.get("promptgen_device", "gpu") 
    final_device_obj = devices.cpu # Default to CPU
    if selected_device_option == 'gpu' and torch.cuda.is_available():
        if current_device_name != 'gpu': logger.info("PromptGen: Setting device to GPU.")
        current_device_name = 'gpu'; final_device_obj = devices.device 
    elif selected_device_option == 'gpu': # GPU selected but not available
        if current_device_name != 'cpu': logger.warning("PromptGen: GPU selected, but not available. Falling back to CPU.")
        current_device_name = 'cpu'
    else: # CPU selected
        if current_device_name != 'cpu': logger.info("PromptGen: Setting device to CPU.")
        current_device_name = 'cpu'
    logger.info(f"PromptGen: Device determined: {current_device_name}")
    return final_device_obj

def list_available_models(): # As before
    available_models.clear()
    model_names_str = shared.opts.data.get("promptgen_names", (', ').join(model_list))
    for name in [x.strip() for x in model_names_str.split(",")]:
        if not name: continue
        available_models.append(name)

async def _perform_generation( 
    model_name: str, batch_count: int, num_return_sequences: int, text: str, # Changed batch_size to num_return_sequences
    min_length: int, max_length: int, num_beams: int, temperature: float,
    repetition_penalty: float, length_penalty: float, sampling_mode: str,
    top_k: int, top_p: float
) -> Tuple[Optional[List[str]], Optional[str]]:
    global current, prompt_cache
    loop = asyncio.get_event_loop()
    _initialize_cache() 
    cache_key = None
    actual_model_name_for_status = model_name 

    if prompt_cache is not None:
        cache_key_params = ( model_name, text, batch_count, num_return_sequences, # Use num_return_sequences in cache key
                             min_length, max_length, num_beams, temperature, 
                             repetition_penalty, length_penalty, sampling_mode, top_k, top_p )
        try: cache_key = hash(cache_key_params) 
        except TypeError as e: logger.warning(f"PromptGen Cache: Unhashable type in cache key: {e}. Disabling cache for this request."); prompt_cache = None 
    
    if prompt_cache is not None and cache_key is not None:
        cached_result = prompt_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"PromptGen Cache: Cache hit for key {str(cache_key_params)[:100]}...") 
            return cached_result[0], cached_result[1] 
        else: logger.debug(f"PromptGen Cache: Cache miss for key {str(cache_key_params)[:100]}...")

    logger.debug(f"PromptGen Async: _perform_generation: model='{model_name}', num_sequences='{num_return_sequences}'...")

    if not model_name or model_name == 'None':
        if current.name is not None: logger.info(f"PromptGen Async: Unloading model '{current.name}'."); current.model = None; current.tokenizer = None; current.name = None; await loop.run_in_executor(None, devices.torch_gc) 
        return None, "PromptGen: No model selected."

    if current.name != model_name or current.model is None or current.tokenizer is None:
        logger.info(f'PromptGen Async: Loading model: {model_name}')
        if current.name is not None: logger.info(f"PromptGen Async: Clearing previous model '{current.name}'."); current.model = None; current.tokenizer = None; current.name = None; await loop.run_in_executor(None, devices.torch_gc) 
        try:
            current.tokenizer = await loop.run_in_executor(None, transformers.AutoTokenizer.from_pretrained, model_name)
            current.model = await loop.run_in_executor(None, transformers.AutoModelForCausalLM.from_pretrained, model_name)
            current.name = model_name; logger.info(f'PromptGen Async: Model "{model_name}" loaded.')
        except Exception as e: error_msg = f'Error loading model "{model_name}": {e}'; logger.error(f"PromptGen Async: {error_msg}", exc_info=True); current.name = None; await loop.run_in_executor(None, devices.torch_gc); return None, f"PromptGen: {error_msg}"
    
    actual_model_name_for_status = current.name
    active_device = await loop.run_in_executor(None, _get_device_logic) 
    try: await loop.run_in_executor(None, current.model.to, active_device)
    except Exception as e: error_msg = f"Error moving model '{actual_model_name_for_status}' to device '{active_device}': {e}"; logger.error(f"PromptGen Async: {error_msg}", exc_info=True); return None, f"PromptGen: {error_msg}"

    try:
        input_processing_result = await loop.run_in_executor(None, current.tokenizer, text, {"return_tensors": "pt"})
        input_ids = input_processing_result.input_ids
        if input_ids.shape[1] == 0: input_ids = torch.tensor([[current.tokenizer.bos_token_id]], dtype=torch.long) 
        input_ids = await loop.run_in_executor(None, input_ids.to, active_device)
    except Exception as e: error_msg = f"Error tokenizing input text: {e}"; logger.error(f"PromptGen Async: {error_msg}", exc_info=True); return None, f"PromptGen: {error_msg}"

    logger.info(f'PromptGen Async: Generating text model="{actual_model_name_for_status}", batches={batch_count}, sequences_per_batch={num_return_sequences} on device="{active_device}"...')
    all_generated_texts: List[str] = []
    generation_error_message: Optional[str] = None
    try:
        for i in range(batch_count): 
            logger.debug(f"PromptGen Async: Generating for user batch {i+1} of {batch_count}...")
            # Input_ids should be shape [1, seq_len] for num_return_sequences to work as expected.
            # If input_ids is already batched (e.g. from tokenizer for multiple inputs), num_return_sequences might behave differently.
            # Assuming input_ids here is for a single base prompt.
            generate_call = functools.partial(current.model.generate, input_ids, do_sample=True, temperature=max(float(temperature), 1e-6), repetition_penalty=float(repetition_penalty), length_penalty=float(length_penalty), top_p=float(top_p) if sampling_mode == 'Top P' else None, top_k=int(top_k) if sampling_mode == 'Top K' else 0, num_beams=int(num_beams), min_length=int(min_length), max_length=int(max_length), pad_token_id=current.tokenizer.pad_token_id or current.tokenizer.eos_token_id, num_return_sequences=num_return_sequences )
            outputs = await loop.run_in_executor(None, generate_call)
            decode_call = functools.partial(current.tokenizer.batch_decode, outputs, skip_special_tokens=True)
            generated_texts_this_batch = await loop.run_in_executor(None, decode_call)
            all_generated_texts.extend(generated_texts_this_batch)
            logger.debug(f"PromptGen Async: Batch {i+1} generated {len(generated_texts_this_batch)} texts.")
        logger.info(f"PromptGen Async: Successfully generated {len(all_generated_texts)} total prompts for model '{actual_model_name_for_status}'.")
    except Exception as e: generation_error_message = f"Error during text generation: {e}"; logger.error(f"PromptGen Async: {generation_error_message}", exc_info=True)
    
    if prompt_cache is not None and cache_key is not None and generation_error_message is None:
        logger.debug(f"PromptGen Cache: Storing result for key {str(cache_key_params)[:100]}...")
        prompt_cache[cache_key] = (all_generated_texts, None) 
    return all_generated_texts, generation_error_message

def model_selection_changed_ui(model_name: str): # As before
    status_message = ""
    if model_name == "None" or not model_name: logger.info(f'PromptGen UI: Selection "None".'); status_message = "Model unloaded."
    else: logger.info(f'PromptGen UI: Selection "{model_name}".'); status_message = f"Selected: {model_name}." + (f" (Currently loaded)" if current.name == model_name and current.model is not None else " (Will load on generation.)")
    return gr.update(value=status_message)

def _perform_generation_sync_for_ui(*args, **kwargs): # As before
    try: return anyio.from_thread.run(_perform_generation, *args, **kwargs)
    except Exception as e: logger.error(f"PromptGen Sync UI Error: {e}", exc_info=True); return None, f"PromptGen: Async bridge error: {e}"

# generate_ui now takes num_return_sequences_from_slider
def generate_ui(id_task, model_name_from_dropdown, batch_count, num_return_sequences_from_slider, text, min_length, max_length, num_beams, temperature, repetition_penalty, length_penalty, sampling_mode, top_k, top_p):
    logger.info(f"PromptGen UI: Request. Model: {model_name_from_dropdown}, Sequences: {num_return_sequences_from_slider}, Text: '{text[:50]}...'")
    status_updates = {} 
    if not model_name_from_dropdown or model_name_from_dropdown == "None":
        err_msg = "PromptGen: No model selected."; logger.error(err_msg)
        status_updates[res_html_ui] = gr.update(value=""); status_updates[res_info_html_ui] = gr.update(value=err_msg); status_updates[model_status_display_ui] = gr.update(value="Error: No model selected.")
        return tuple(status_updates.get(comp, gr.update()) for comp in [res_html_ui, res_info_html_ui, model_status_display_ui])

    preliminary_status = f"Processing with {model_name_from_dropdown}..."
    if current.name != model_name_from_dropdown: preliminary_status = f"Loading {model_name_from_dropdown}..."
    
    shared.state.begin('promptgen'); shared.state.job_count = batch_count 
    logger.debug(f"PromptGen UI: Calling sync wrapper with model: {model_name_from_dropdown}, sequences: {num_return_sequences_from_slider}")

    # Pass num_return_sequences_from_slider as num_return_sequences
    generated_texts, err_msg = _perform_generation_sync_for_ui( model_name_from_dropdown, batch_count, num_return_sequences_from_slider, text, min_length, max_length, num_beams, temperature, repetition_penalty, length_penalty, sampling_mode, top_k, top_p )
    
    final_model_status_text = ""
    if err_msg: 
        logger.error(f"PromptGen UI Error: {err_msg}"); shared.state.textinfo = err_msg 
        final_model_status_text = f"Error: {err_msg[:100]}"; status_updates[res_html_ui] = gr.update(value=""); status_updates[res_info_html_ui] = gr.update(value=err_msg)
    elif not generated_texts: 
        no_text_msg = "No text generated."; logger.info("PromptGen UI: No text generated.")
        shared.state.textinfo = no_text_msg; final_model_status_text = f"{current.name or model_name_from_dropdown} Ready. (No prompts)"
        status_updates[res_html_ui] = gr.update(value=""); status_updates[res_info_html_ui] = gr.update(value=no_text_msg)
    else:
        markup = '<table><tbody>'; total_prompts_displayed = 0
        # The total number of texts is batch_count * num_return_sequences_from_slider
        # The UI loops `batch_count` times, and each "group" in UI is `num_return_sequences_from_slider` texts.
        for i in range(batch_count): 
            shared.state.textinfo = f"Displaying results for batch {i + 1} of {batch_count}"
            start_idx = i * num_return_sequences_from_slider; end_idx = start_idx + num_return_sequences_from_slider
            texts_for_this_ui_batch = generated_texts[start_idx:end_idx]
            for text_item in texts_for_this_ui_batch:
                total_prompts_displayed += 1; escaped_text = html.escape(text_item)
                p_id = f'promptgen_res_{total_prompts_displayed}'
                markup += f"<tr><td><div class=\"prompt gr-box gr-text-input\"><p id='{p_id}'>{escaped_text}</p></div></td><td class=\"sendto\"><a class='gr-button gr-button-lg gr-button-secondary' onclick=\"promptGenSend(gradioApp().getElementById('{p_id}').textContent, 'txt2img')\">to Txt2Img</a><a class='gr-button gr-button-lg gr-button-secondary' onclick=\"promptGenSend(gradioApp().getElementById('{p_id}').textContent, 'img2img')\">to Img2Img</a></td></tr>"
            shared.state.nextjob() 
        markup += '</tbody></table>'
        logger.info(f"PromptGen UI: Displayed {total_prompts_displayed} prompts.")
        final_model_status_text = f"{current.name or model_name_from_dropdown} Ready."
        status_updates[res_html_ui] = gr.update(value=markup); status_updates[res_info_html_ui] = gr.update(value="")
    status_updates[model_status_display_ui] = gr.update(value=final_model_status_text)
    shared.state.end()
    return status_updates[res_html_ui], status_updates[res_info_html_ui], status_updates[model_status_display_ui]

def find_prompts(fields): return [f[0] for f in fields if f[1] == "Prompt"][0], [f[0] for f in fields if f[1] == "Negative prompt"][0] or gr.update()
def send_prompts(text): params = generation_parameters_copypaste.parse_generation_parameters(text); return params.get("Prompt", ""), params.get("Negative prompt", "") or gr.update()

res_html_ui = None; res_info_html_ui = None; model_status_display_ui = None 

def add_tab():
    global res_html_ui, res_info_html_ui, model_status_display_ui
    list_available_models() 
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row(): 
            prompt_input_ui = gr.Textbox(label="Prompt", elem_id="promptgen_prompt_input_ui", show_label=False, lines=2, placeholder="Beginning of the prompt...", container=False, scale=80)
            submit_button_ui = gr.Button('Generate', elem_id="promptgen_generate_button_ui", variant='primary', scale=10)
        with gr.Row(): model_status_display_ui = gr.Textbox(label="Model Status", interactive=False, value="Select model and generate.")
        with gr.Row(elem_id="promptgen_main_ui"):
            with gr.Column(variant="compact"):
                # ... (other UI elements as before, ensure elem_ids are unique if needed) ...
                model_selection_dd_ui = gr.Dropdown(label="Model", elem_id="promptgen_model_selection_dd_ui", value=available_models[0] if available_models else "None", choices=["None"] + available_models)
                num_return_sequences_slider_ui = gr.Slider(label="Number of Sequences per Input", elem_id="promptgen_num_return_sequences_slider_ui", value=lambda: shared.opts.data.get("promptgen_num_return_sequences", 1), minimum=1, maximum=20, step=1)
                batch_count_slider_ui = gr.Slider(label="Batch Count (Iterations)", elem_id="promptgen_batch_count_slider_ui", value=lambda: shared.opts.data.get("promptgen_batch_count", 1), minimum=1, maximum=100, step=1) # Renamed for clarity
                # The old "batch_size_slider_ui" is effectively replaced by num_return_sequences_slider_ui for generation.
                # If batch_size had a different meaning (e.g. parallel processing of distinct inputs), it would need different handling.
                # For now, it's removed to avoid confusion with num_return_sequences.
                with gr.Row(): sampling_mode_radio_ui = gr.Radio(label="Sampling mode", elem_id="promptgen_sampling_mode_radio_ui", value="Top K", choices=["Top K", "Top P"]); top_k_slider_ui = gr.Slider(label="Top K", elem_id="promptgen_top_k_slider_ui", value=lambda: shared.opts.data.get("promptgen_top_k", 50), minimum=0, maximum=100, step=1); top_p_slider_ui = gr.Slider(label="Top P", elem_id="promptgen_top_p_slider_ui", value=lambda: shared.opts.data.get("promptgen_top_p", 1.0), minimum=0.0, maximum=1.0, step=0.01)
                with gr.Row(): num_beams_slider_ui = gr.Slider(label="Num beams", elem_id="promptgen_num_beams_slider_ui", value=lambda: shared.opts.data.get("promptgen_num_beams", 1), minimum=1, maximum=8, step=1); temperature_slider_ui = gr.Slider(label="Temperature", elem_id="promptgen_temperature_slider_ui", value=lambda: shared.opts.data.get("promptgen_temperature", 1.0), minimum=0.01, maximum=4.0, step=0.01); repetition_penalty_slider_ui = gr.Slider(label="Repetition penalty", elem_id="promptgen_repetition_penalty_slider_ui", value=lambda: shared.opts.data.get("promptgen_repetition_penalty", 1.0), minimum=1.0, maximum=4.0, step=0.01)
                with gr.Row(): length_penalty_slider_ui = gr.Slider(label="Length preference", elem_id="promptgen_length_preference_slider_ui", value=lambda: shared.opts.data.get("promptgen_length_penalty", 1.0), minimum=-10.0, maximum=10.0, step=0.1); min_length_slider_ui = gr.Slider(label="Min length", elem_id="promptgen_min_length_slider_ui", value=lambda: shared.opts.data.get("promptgen_min_length", 20), minimum=1, maximum=400, step=1); max_length_slider_ui = gr.Slider(label="Max length", elem_id="promptgen_max_length_slider_ui", value=lambda: shared.opts.data.get("promptgen_max_length", 150), minimum=1, maximum=400, step=1)
                with open(os.path.join(base_dir, "explanation.html"), encoding="utf8") as file: footer = file.read(); gr.HTML(footer)
            with gr.Column(): res_html_ui = gr.HTML(); res_info_html_ui = gr.HTML()
        
        gradio_ui_inputs = [ model_selection_dd_ui, batch_count_slider_ui, num_return_sequences_slider_ui, prompt_input_ui, min_length_slider_ui, max_length_slider_ui, num_beams_slider_ui, temperature_slider_ui, repetition_penalty_slider_ui, length_penalty_slider_ui, sampling_mode_radio_ui, top_k_slider_ui, top_p_slider_ui ]
        submit_button_ui.click( fn=ui.wrap_gradio_gpu_call(generate_ui, extra_outputs=['', '', '']), _js="promptGenSubmit", inputs=gradio_ui_inputs, outputs=[res_html_ui, res_info_html_ui, model_status_display_ui] )
        model_selection_dd_ui.change(fn=model_selection_changed_ui, inputs=[model_selection_dd_ui], outputs=[model_status_display_ui]) 
        # Hidden buttons for send to txt2img/img2img functionality
        selected_text_ui = gr.TextArea(elem_id='promptgen_selected_text_ui', visible=False) # Keep these if JS relies on them
        send_to_txt2img_button_ui = gr.Button(elem_id='promptgen_send_to_txt2img_button_ui', visible=False)
        send_to_img2img_button_ui = gr.Button(elem_id='promptgen_send_to_img2img_button_ui', visible=False)
        send_to_txt2img_button_ui.click(fn=send_prompts, inputs=[selected_text_ui], outputs=find_prompts(ui.txt2img_paste_fields))
        send_to_img2img_button_ui.click(fn=send_prompts, inputs=[selected_text_ui], outputs=find_prompts(ui.img2img_paste_fields))

    return [(tab, "PromptGen", "promptgen_tab")]

def on_ui_settings():
    section = ("promptgen", "PromptGen")
    shared.opts.add_option("promptgen_names", shared.OptionInfo((', ').join(model_list), "PromptGen Hugging Face model names", section=section))
    shared.opts.add_option("promptgen_device", shared.OptionInfo("gpu", "Device for PromptGen", gr.Radio, {"choices": ["gpu", "cpu"]}, section=section))
    shared.opts.add_option("promptgen_enable_cache", shared.OptionInfo(True, "Enable caching", gr.Checkbox, section=section))
    shared.opts.add_option("promptgen_cache_size", shared.OptionInfo(100, "Cache size", gr.Number, {"precision": 0}, section=section))
    shared.opts.add_option("promptgen_num_return_sequences", shared.OptionInfo(1, "Default sequences per input (UI)", gr.Number, {"precision": 0, "minimum": 1}, section=section)) # New setting
    # ... (other options as before) ...
    shared.opts.add_option("promptgen_top_k", shared.OptionInfo(50, "Default Top K (UI)", gr.Slider, {"minimum":0, "maximum":100, "step":1}, section=section))
    shared.opts.add_option("promptgen_top_p", shared.OptionInfo(1.0, "Default Top P (UI)", gr.Slider, {"minimum":0.0, "maximum":1.0, "step":0.01}, section=section))
    shared.opts.add_option("promptgen_num_beams", shared.OptionInfo(1, "Default Num beams (UI)", gr.Slider, {"minimum":1, "maximum":8, "step":1}, section=section))
    shared.opts.add_option("promptgen_temperature", shared.OptionInfo(1.0, "Default Temperature (UI)", gr.Slider, {"minimum":0.01, "maximum":4.0, "step":0.01}, section=section))
    shared.opts.add_option("promptgen_repetition_penalty", shared.OptionInfo(1.0, "Default Repetition penalty (UI)", gr.Slider, {"minimum":1.0, "maximum":4.0, "step":0.01}, section=section))
    shared.opts.add_option("promptgen_length_penalty", shared.OptionInfo(1.0, "Default Length penalty (UI)", gr.Slider, {"minimum":-10.0, "maximum":10.0, "step":0.1}, section=section))
    shared.opts.add_option("promptgen_min_length", shared.OptionInfo(20, "Default Min length (UI)", gr.Slider, {"minimum":1, "maximum":400, "step":1}, section=section))
    shared.opts.add_option("promptgen_max_length", shared.OptionInfo(150, "Default Max length (UI)", gr.Slider, {"minimum":1, "maximum":400, "step":1}, section=section))
    shared.opts.add_option("promptgen_batch_count", shared.OptionInfo(1, "Default Batch count (Iterations in UI)", gr.Slider, {"minimum":1, "maximum":100, "step":1}, section=section))
    # The old "promptgen_batch_size" setting is removed as its role is now taken by "promptgen_num_return_sequences" for clarity.

def on_unload(): global current; logger.info("PromptGen: Unloading."); if current.name: logger.info(f"Clearing model '{current.name}'."); current.model=None; current.tokenizer=None; current.name=None; devices.torch_gc() 

async def api_generate_prompts_endpoint(request: PromptGenerationRequest) -> PromptGenerationResponse: 
    logger.info(f"PromptGen API: Request model='{request.model_name}', sequences='{request.num_return_sequences}', text='{request.text[:50]}...'")
    logger.debug(f"PromptGen API: Full request: {request.dict()}")
    try:
        generated_texts, err = await _perform_generation( model_name=request.model_name, batch_count=request.batch_count, num_return_sequences=request.num_return_sequences, text=request.text, min_length=request.min_length, max_length=request.max_length, num_beams=request.num_beams, temperature=request.temperature, repetition_penalty=request.repetition_penalty, length_penalty=request.length_penalty, sampling_mode=request.sampling_mode, top_k=request.top_k, top_p=request.top_p )
        if err: logger.error(f"API Error: {err}"); raise HTTPException(status_code=500, detail=err)
        if not generated_texts: return PromptGenerationResponse(prompts=[], model_name=request.model_name, input_text=request.text, detail="No prompts generated.")
        return PromptGenerationResponse(prompts=generated_texts, model_name=request.model_name, input_text=request.text)
    except HTTPException as e: logger.error(f"API HTTPException: {e.detail}"); raise
    except Exception as e: error_detail = f"API Unexpected error: {e}"; logger.error(error_detail, exc_info=True); raise HTTPException(status_code=500, detail=error_detail)

def add_api_routes_on_app_started(app: FastAPI): logger.info("PromptGen: Registering API route."); app.add_api_route( "/promptgen/v1/generate", api_generate_prompts_endpoint, methods=["POST"], response_model=PromptGenerationResponse, summary="Generate Prompts", tags=["Prompt Generation (PromptGen Extension)"] ); logger.info("PromptGen: API route registered.")
def on_app_started_callback(app): _initialize_cache(); add_api_routes_on_app_started(app)

script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_script_unloaded(on_unload)
script_callbacks.on_app_started(on_app_started_callback) 
logger.info("PromptGen Script: Initialized.")
