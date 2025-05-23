import os
import html
import torch
import transformers
import gradio as gr
import logging # Added for logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple # For type hints

from modules import shared, generation_parameters_copypaste, scripts, script_callbacks, devices, ui

# Initialize logger for this script
logger = logging.getLogger('promptgen')
# Configure logger if necessary (e.g., set level, add handler).
# Assuming basic configuration is handled by the main application.
# If running standalone or need specific output, uncomment and configure:
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
current = Model() # Global variable for model state. This is not ideal for concurrent API calls.
current_device_name = None # Global variable for device name tracking.
base_dir = scripts.basedir()


# --- Pydantic Models for API ---
class PromptGenerationRequest(BaseModel):
    model_name: str = Field(..., description="Name of the Hugging Face model to use.")
    text: str = Field(..., description="Input text to generate prompts from.")
    batch_count: int = Field(1, gt=0, description="Number of batches to generate.")
    batch_size: int = Field(1, gt=0, description="Number of prompts per batch.")
    min_length: int = Field(20, gt=0, description="Minimum length of generated prompts.")
    max_length: int = Field(150, gt=0, description="Maximum length of generated prompts.")
    num_beams: int = Field(1, ge=1, description="Number of beams for beam search.")
    temperature: float = Field(1.0, gt=0, description="Sampling temperature (must be > 0).")
    repetition_penalty: float = Field(1.0, ge=1.0, description="Repetition penalty.")
    length_penalty: float = Field(1.0, description="Length penalty.")
    sampling_mode: str = Field("Top K", enum=["Top K", "Top P"], description="Sampling mode: 'Top K' or 'Top P'.")
    top_k: int = Field(50, ge=0, description="Top K for sampling.") # Default from HF
    top_p: float = Field(1.0, ge=0, le=1.0, description="Top P for sampling.") # Default from HF

class PromptGenerationResponse(BaseModel):
    prompts: List[str]
    model_name: str
    input_text: str
    detail: Optional[str] = None


# --- Core Logic ---
def _get_device_logic() -> torch.device:
    global current_device_name
    selected_device_option = shared.opts.data.get("promptgen_device", "gpu") 

    determined_device_type = 'unknown'
    final_device_obj = None

    if selected_device_option == 'gpu':
        if torch.cuda.is_available():
            if current_device_name != 'gpu':
                logger.info("PromptGen: Setting device to GPU.")
            current_device_name = 'gpu'
            determined_device_type = 'gpu'
            final_device_obj = devices.device 
        else:
            if current_device_name != 'cpu':
                logger.warning("PromptGen: GPU selected or default, but not available. Falling back to CPU.")
            current_device_name = 'cpu'
            determined_device_type = 'cpu (fallback)'
            final_device_obj = devices.cpu
    else: # 'cpu'
        if current_device_name != 'cpu':
            logger.info("PromptGen: Setting device to CPU.")
        current_device_name = 'cpu'
        determined_device_type = 'cpu'
        final_device_obj = devices.cpu
    
    logger.info(f"PromptGen: Device determined: {determined_device_type} (using object: {final_device_obj})")
    return final_device_obj


def list_available_models():
    available_models.clear()
    model_names_str = shared.opts.data.get("promptgen_names", (', ').join(model_list))
    for name in [x.strip() for x in model_names_str.split(",")]:
        if not name:
            continue
        available_models.append(name)


def _perform_generation(
    model_name: str,
    batch_count: int,
    batch_size: int,
    text: str,
    min_length: int,
    max_length: int,
    num_beams: int,
    temperature: float,
    repetition_penalty: float,
    length_penalty: float,
    sampling_mode: str,
    top_k: int,
    top_p: float
) -> Tuple[Optional[List[str]], Optional[str]]:
    global current 

    logger.debug(
        f"PromptGen: _perform_generation called with params: model_name='{model_name}', text='{text[:50]}...', "
        f"batch_count={batch_count}, batch_size={batch_size}, min_length={min_length}, max_length={max_length}, "
        f"num_beams={num_beams}, temperature={temperature}, repetition_penalty={repetition_penalty}, "
        f"length_penalty={length_penalty}, sampling_mode='{sampling_mode}', top_k={top_k}, top_p={top_p}"
    )

    if not model_name or model_name == 'None':
        if current.name is not None: 
            logger.info(f"PromptGen: Unloading model '{current.name}' due to 'None' selection.")
            current.model = None
            current.tokenizer = None
            current.name = None
            devices.torch_gc()
        return None, "PromptGen: No model selected."

    if current.name != model_name or current.model is None or current.tokenizer is None:
        logger.info(f'PromptGen: Attempting to load model: {model_name}')
        if current.name is not None:
            logger.info(f"PromptGen: Clearing previous model '{current.name}' before loading '{model_name}'.")
            current.model = None
            current.tokenizer = None
            current.name = None
            devices.torch_gc()
        try:
            current.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            current.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
            current.name = model_name
            logger.info(f'PromptGen: Model "{model_name}" loaded successfully.')
        except Exception as e:
            error_msg = f'Error loading model "{model_name}": {str(e)}'
            logger.error(f"PromptGen: {error_msg}", exc_info=True)
            current.name = None 
            devices.torch_gc()
            return None, f"PromptGen: {error_msg}"


    active_device = _get_device_logic()
    try:
        logger.debug(f"PromptGen: Moving model '{current.name}' to device '{active_device}'.")
        current.model.to(active_device)
    except Exception as e:
        error_msg = f"Error moving model '{current.name}' to device '{active_device}': {str(e)}"
        logger.error(f"PromptGen: {error_msg}", exc_info=True)
        return None, f"PromptGen: {error_msg}"

    try:
        input_ids = current.tokenizer(text, return_tensors="pt").input_ids
        if input_ids.shape[1] == 0:
            logger.debug("PromptGen: Input text tokenized to empty sequence, using BOS token.")
            input_ids = torch.tensor([[current.tokenizer.bos_token_id]], dtype=torch.long)
        input_ids = input_ids.to(active_device)
    except Exception as e:
        error_msg = f"Error tokenizing input text: {str(e)}"
        logger.error(f"PromptGen: {error_msg}", exc_info=True)
        return None, f"PromptGen: {error_msg}"

    logger.info(f'PromptGen: Generating text with model="{model_name}" (batch_count={batch_count}, prompts_per_batch_call={batch_size}) on device="{active_device}"...')
    all_generated_texts: List[str] = []
    try:
        for i in range(batch_count): 
            logger.debug(f"PromptGen: Generating for user batch {i+1} of {batch_count}...")
            current_call_input_ids = input_ids.repeat((batch_size, 1))
            
            outputs = current.model.generate(
                current_call_input_ids,
                do_sample=True,
                temperature=max(float(temperature), 1e-6),
                repetition_penalty=float(repetition_penalty),
                length_penalty=float(length_penalty),
                top_p=float(top_p) if sampling_mode == 'Top P' else None,
                top_k=int(top_k) if sampling_mode == 'Top K' else 0,
                num_beams=int(num_beams),
                min_length=int(min_length),
                max_length=int(max_length),
                pad_token_id=current.tokenizer.pad_token_id or current.tokenizer.eos_token_id
            )
            generated_texts_this_batch = current.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_generated_texts.extend(generated_texts_this_batch)
            logger.debug(f"PromptGen: Finished generation for user batch {i+1}, {len(generated_texts_this_batch)} texts added.")
        logger.info(f"PromptGen: Successfully generated {len(all_generated_texts)} total prompts for model '{model_name}'.")
    except Exception as e:
        error_msg = f"Error during text generation with model '{model_name}': {str(e)}"
        logger.error(f"PromptGen: {error_msg}", exc_info=True)
        return all_generated_texts, f"PromptGen: {error_msg}" 

    return all_generated_texts, None


# --- Gradio UI Functions ---
def model_selection_changed_ui(model_name: str):
    if model_name == "None":
        if current.name is not None and current.name != "None":
            logger.info(f'PromptGen UI: Selection changed to "None". Model "{current.name}" will be unloaded on next generation call or if cleared.')
    else:
        logger.info(f'PromptGen UI: Selection changed to "{model_name}". Model will be loaded on next generation call.')


def generate_ui(id_task, model_name, batch_count, batch_size, text, min_length, max_length, num_beams, temperature, repetition_penalty, length_penalty, sampling_mode, top_k, top_p):
    logger.info(f"PromptGen UI: Received generation request. Model: {model_name}, Text: '{text[:50]}...'")
    shared.state.begin('promptgen')
    shared.state.job_count = batch_count 

    generated_texts, err_msg = _perform_generation(
        model_name, batch_count, batch_size, text,
        min_length, max_length, num_beams, temperature,
        repetition_penalty, length_penalty, sampling_mode, top_k, top_p
    )

    if err_msg:
        logger.error(f"PromptGen UI: Error during generation for UI: {err_msg}")
        shared.state.textinfo = err_msg 
        shared.state.end()
        return '', err_msg 

    if not generated_texts:
        no_text_msg = "No text generated."
        logger.info("PromptGen UI: No text generated, no specific error message.")
        shared.state.textinfo = no_text_msg
        shared.state.end()
        return '', no_text_msg

    markup = '<table><tbody>'
    total_prompts_displayed = 0
    for i in range(batch_count): 
        shared.state.textinfo = f"Displaying results for user batch {i + 1} of {batch_count}"
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        texts_for_this_ui_batch = generated_texts[start_idx:end_idx]

        for text_item in texts_for_this_ui_batch:
            total_prompts_displayed += 1
            escaped_text = html.escape(text_item)
            p_id = f'promptgen_res_{total_prompts_displayed}'
            markup += f"""
                <tr>
                    <td><div class="prompt gr-box gr-text-input"><p id='{p_id}'>{escaped_text}</p></div></td>
                    <td class="sendto">
                        <a class='gr-button gr-button-lg gr-button-secondary' onclick="promptGenSend(gradioApp().getElementById('{p_id}').textContent, 'txt2img')">to Txt2Img</a>
                        <a class='gr-button gr-button-lg gr-button-secondary' onclick="promptGenSend(gradioApp().getElementById('{p_id}').textContent, 'img2img')">to Img2Img</a>
                    </td>
                </tr>
                """
        shared.state.nextjob() 

    markup += '</tbody></table>'
    logger.info(f"PromptGen UI: Successfully displayed {total_prompts_displayed} prompts.")
    shared.state.end()
    return markup, '' 


def find_prompts(fields):
    field_prompt = [x for x in fields if x[1] == "Prompt"][0]
    field_negative_prompt = [x for x in fields if x[1] == "Negative prompt"][0]
    return [field_prompt[0], field_negative_prompt[0]]


def send_prompts(text):
    params = generation_parameters_copypaste.parse_generation_parameters(text)
    negative_prompt = params.get("Negative prompt", "")
    return params.get("Prompt", ""), negative_prompt or gr.update()


def add_tab():
    list_available_models() 
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row():
            with gr.Column(scale=80):
                prompt_input_ui = gr.Textbox(label="Prompt", elem_id="promptgen_prompt_input_ui", show_label=False, lines=2, placeholder="Beginning of the prompt (press Ctrl+Enter or Alt+Enter to generate)", container=False)
            with gr.Column(scale=10):
                submit_button_ui = gr.Button('Generate', elem_id="promptgen_generate_button_ui", variant='primary')
        with gr.Row(elem_id="promptgen_main_ui"):
            with gr.Column(variant="compact"):
                selected_text_ui = gr.TextArea(elem_id='promptgen_selected_text_ui', visible=False)
                send_to_txt2img_button_ui = gr.Button(elem_id='promptgen_send_to_txt2img_button_ui', visible=False)
                send_to_img2img_button_ui = gr.Button(elem_id='promptgen_send_to_img2img_button_ui', visible=False)
                with gr.Row():
                    model_selection_dd_ui = gr.Dropdown(label="Model", elem_id="promptgen_model_selection_dd_ui", value=available_models[0] if available_models else "None", choices=["None"] + available_models)
                with gr.Row():
                    sampling_mode_radio_ui = gr.Radio(label="Sampling mode", elem_id="promptgen_sampling_mode_radio_ui", value="Top K", choices=["Top K", "Top P"])
                    top_k_slider_ui = gr.Slider(label="Top K", elem_id="promptgen_top_k_slider_ui", value=lambda: shared.opts.data.get("promptgen_top_k", 50), minimum=0, maximum=100, step=1)
                    top_p_slider_ui = gr.Slider(label="Top P", elem_id="promptgen_top_p_slider_ui", value=lambda: shared.opts.data.get("promptgen_top_p", 1.0), minimum=0.0, maximum=1.0, step=0.01)
                with gr.Row():
                    num_beams_slider_ui = gr.Slider(label="Number of beams", elem_id="promptgen_num_beams_slider_ui", value=lambda: shared.opts.data.get("promptgen_num_beams", 1), minimum=1, maximum=8, step=1)
                    temperature_slider_ui = gr.Slider(label="Temperature", elem_id="promptgen_temperature_slider_ui", value=lambda: shared.opts.data.get("promptgen_temperature", 1.0), minimum=0.01, maximum=4.0, step=0.01)
                    repetition_penalty_slider_ui = gr.Slider(label="Repetition penalty", elem_id="promptgen_repetition_penalty_slider_ui", value=lambda: shared.opts.data.get("promptgen_repetition_penalty", 1.0), minimum=1.0, maximum=4.0, step=0.01)
                with gr.Row():
                    length_penalty_slider_ui = gr.Slider(label="Length preference", elem_id="promptgen_length_preference_slider_ui", value=lambda: shared.opts.data.get("promptgen_length_penalty", 1.0), minimum=-10.0, maximum=10.0, step=0.1)
                    min_length_slider_ui = gr.Slider(label="Min length", elem_id="promptgen_min_length_slider_ui", value=lambda: shared.opts.data.get("promptgen_min_length", 20), minimum=1, maximum=400, step=1)
                    max_length_slider_ui = gr.Slider(label="Max length", elem_id="promptgen_max_length_slider_ui", value=lambda: shared.opts.data.get("promptgen_max_length", 150), minimum=1, maximum=400, step=1)
                with gr.Row():
                    batch_count_slider_ui = gr.Slider(label="Batch count (UI loops)", elem_id="promptgen_batch_count_slider_ui", value=lambda: shared.opts.data.get("promptgen_batch_count", 1), minimum=1, maximum=100, step=1)
                    batch_size_slider_ui = gr.Slider(label="Prompts per batch (API call)", elem_id="promptgen_batch_size_slider_ui", value=lambda: shared.opts.data.get("promptgen_batch_size", 1), minimum=1, maximum=100, step=1)
                with open(os.path.join(base_dir, "explanation.html"), encoding="utf8") as file:
                    footer = file.read()
                    gr.HTML(footer)
            with gr.Column():
                with gr.Group(elem_id="promptgen_results_column_ui"):
                    res_html_ui = gr.HTML()
                    res_info_html_ui = gr.HTML()
        
        gradio_ui_inputs = [
            model_selection_dd_ui, batch_count_slider_ui, batch_size_slider_ui, prompt_input_ui,
            min_length_slider_ui, max_length_slider_ui, num_beams_slider_ui, temperature_slider_ui,
            repetition_penalty_slider_ui, length_penalty_slider_ui, sampling_mode_radio_ui,
            top_k_slider_ui, top_p_slider_ui
        ]
        
        submit_button_ui.click(
            fn=ui.wrap_gradio_gpu_call(generate_ui, extra_outputs=['']),
            _js="promptGenSubmit",
            inputs=gradio_ui_inputs,
            outputs=[res_html_ui, res_info_html_ui]
        )
        model_selection_dd_ui.change(fn=model_selection_changed_ui, inputs=[model_selection_dd_ui], outputs=[])
        send_to_txt2img_button_ui.click(fn=send_prompts, inputs=[selected_text_ui], outputs=find_prompts(ui.txt2img_paste_fields))
        send_to_img2img_button_ui.click(fn=send_prompts, inputs=[selected_text_ui], outputs=find_prompts(ui.img2img_paste_fields))
    return [(tab, "PromptGen", "promptgen_tab")]


# --- WebUI Settings ---
def on_ui_settings():
    section = ("promptgen", "PromptGen")
    default_models = (', ').join(model_list)
    shared.opts.add_option("promptgen_names", shared.OptionInfo(default_models, "PromptGen Hugging Face model names (comma-separated)", section=section))
    shared.opts.add_option("promptgen_device", shared.OptionInfo("gpu", "Device for PromptGen (gpu or cpu)", gr.Radio, {"choices": ["gpu", "cpu"]}, section=section))
    
    shared.opts.add_option("promptgen_top_k", shared.OptionInfo(50, "Default Top K for PromptGen UI", gr.Slider, {"minimum":0, "maximum":100, "step":1}, section=section))
    shared.opts.add_option("promptgen_top_p", shared.OptionInfo(1.0, "Default Top P for PromptGen UI", gr.Slider, {"minimum":0.0, "maximum":1.0, "step":0.01}, section=section))
    shared.opts.add_option("promptgen_num_beams", shared.OptionInfo(1, "Default Number of beams for PromptGen UI", gr.Slider, {"minimum":1, "maximum":8, "step":1}, section=section))
    shared.opts.add_option("promptgen_temperature", shared.OptionInfo(1.0, "Default Temperature for PromptGen UI", gr.Slider, {"minimum":0.01, "maximum":4.0, "step":0.01}, section=section))
    shared.opts.add_option("promptgen_repetition_penalty", shared.OptionInfo(1.0, "Default Repetition penalty for PromptGen UI", gr.Slider, {"minimum":1.0, "maximum":4.0, "step":0.01}, section=section))
    shared.opts.add_option("promptgen_length_penalty", shared.OptionInfo(1.0, "Default Length penalty for PromptGen UI", gr.Slider, {"minimum":-10.0, "maximum":10.0, "step":0.1}, section=section))
    shared.opts.add_option("promptgen_min_length", shared.OptionInfo(20, "Default Min prompt length for PromptGen UI", gr.Slider, {"minimum":1, "maximum":400, "step":1}, section=section))
    shared.opts.add_option("promptgen_max_length", shared.OptionInfo(150, "Default Max prompt length for PromptGen UI", gr.Slider, {"minimum":1, "maximum":400, "step":1}, section=section))
    shared.opts.add_option("promptgen_batch_count", shared.OptionInfo(1, "Default Batch count (UI loops) for PromptGen UI", gr.Slider, {"minimum":1, "maximum":100, "step":1}, section=section))
    shared.opts.add_option("promptgen_batch_size", shared.OptionInfo(1, "Default Prompts per batch (API call) for PromptGen UI", gr.Slider, {"minimum":1, "maximum":100, "step":1}, section=section))


def on_unload():
    global current
    logger.info("PromptGen: Unloading script and attempting to clear model from memory.")
    if current.name is not None:
        logger.info(f"PromptGen: Clearing model '{current.name}'.")
        current.model = None
        current.tokenizer = None
        current.name = None
        devices.torch_gc() 


# --- FastAPI Endpoint Registration ---
async def api_generate_prompts_endpoint(request: PromptGenerationRequest) -> PromptGenerationResponse:
    logger.info(f"PromptGen API: Request received for model '{request.model_name}' with text snippet '{request.text[:50]}...'")
    logger.debug(f"PromptGen API: Full request details: {request.dict()}")

    try:
        generated_texts, err = _perform_generation(
            model_name=request.model_name,
            batch_count=request.batch_count,
            batch_size=request.batch_size,
            text=request.text,
            min_length=request.min_length,
            max_length=request.max_length,
            num_beams=request.num_beams,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            length_penalty=request.length_penalty,
            sampling_mode=request.sampling_mode,
            top_k=request.top_k,
            top_p=request.top_p
        )

        if err:
            logger.error(f"PromptGen API: Error during generation for request (model: {request.model_name}): {err}")
            raise HTTPException(status_code=500, detail=err) # Using 500 for any backend processing error as per subtask.
        
        if not generated_texts: 
            logger.info("PromptGen API: No text generated, but no explicit error from _perform_generation.")
            return PromptGenerationResponse(prompts=[], model_name=request.model_name, input_text=request.text, detail="No prompts were generated by the model for the given input.")

        logger.debug(f"PromptGen API: Successfully generated {len(generated_texts)} prompts. Returning successful response.")
        return PromptGenerationResponse(prompts=generated_texts, model_name=request.model_name, input_text=request.text)

    except HTTPException as http_exc: # Re-raise HTTPException directly
        logger.error(f"PromptGen API: HTTPException occurred: Status {http_exc.status_code}, Detail: {http_exc.detail}")
        raise
    except Exception as e: # Catch any other unexpected errors
        error_detail = f"An unexpected error occurred in PromptGen API endpoint: {str(e)}"
        logger.error(error_detail, exc_info=True) # Log full traceback for unexpected errors
        raise HTTPException(status_code=500, detail=error_detail)


def add_api_routes_on_app_started(app: FastAPI):
    logger.info("PromptGen: Registering API route /promptgen/v1/generate")
    app.add_api_route(
        path="/promptgen/v1/generate",
        endpoint=api_generate_prompts_endpoint, 
        methods=["POST"],
        response_model=PromptGenerationResponse, 
        summary="Generate Prompts with PromptGen",
        description="Generates creative prompts based on an initial text input and various generation parameters using a specified Hugging Face model accessible to the server.",
        tags=["Prompt Generation (PromptGen Extension)"] 
    )
    logger.info("PromptGen: API route /promptgen/v1/generate registered.")

# --- Script Callbacks Registration ---
script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_script_unloaded(on_unload)
script_callbacks.on_app_started(add_api_routes_on_app_started) 
logger.info("PromptGen Script: Initialization complete. All callbacks registered.")
