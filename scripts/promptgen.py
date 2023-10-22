import os
import html
import torch
import transformers
import gradio as gr
from modules import shared, generation_parameters_copypaste, scripts, script_callbacks, devices, ui


class Model:
    name = None
    model = None
    tokenizer = None


model_list = "AUTOMATIC/promptgen-lexart, AUTOMATIC/promptgen-majinai-safe, AUTOMATIC/promptgen-majinai-unsafe, succinctly/text2image-prompt-generator, microsoft/Promptist, RamAnanth1/distilgpt2-sd-prompts, Gustavosta/MagicPrompt-Stable-Diffusion, FredZhang7/distilgpt2-stable-diffusion-v2"
available_models = []
current = Model()
base_dir = scripts.basedir()


def list_available_models():
    available_models.clear()
    for name in [x.strip() for x in shared.opts.promptgen_names.split(",")]:
        if not name:
            continue
        available_models.append(name)


def generate_batch(input_ids, min_length, max_length, num_beams, temperature, repetition_penalty, length_penalty, sampling_mode, top_k, top_p):
    # https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.inference._text_generation.TextGenerationParameters
    # https://huggingface.co/docs/transformers/main/en/generation_strategies#text-generation-strategies
    outputs = current.model.generate(
        input_ids,
        do_sample=True,
        temperature=max(float(temperature), 1e-6),
        repetition_penalty=float(repetition_penalty),
        length_penalty=length_penalty,
        top_p=float(top_p) if sampling_mode == 'Top P' else None,
        top_k=int(top_k) if sampling_mode == 'Top K' else None,
        num_beams=int(num_beams),
        min_length=min_length,
        max_length=max_length,
        pad_token_id=current.tokenizer.pad_token_id or current.tokenizer.eos_token_id
    )
    texts = current.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return texts


def model_selection_changed(model_name):
    if model_name == "None":
        current.tokenizer = None
        current.model = None
        current.name = None
        devices.torch_gc()


def generate(id_task, model_name, batch_count, batch_size, text, *args): # pylint: disable=unused-argument
    shared.state.begin('promptgen')
    shared.state.job_count = batch_count
    if current.name != model_name:
        current.tokenizer = None
        current.model = None
        current.name = None
        if model_name != 'None':
            shared.log.info(f'PromptGen: loading model={model_name}')
            current.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            current.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
            current.name = model_name
    if current.model is None:
        msg = 'PromptGen: no model available'
        shared.log.error(msg)
        shared.state.end()
        return '', msg
    if current.tokenizer is None:
        msg = 'PromptGen: no tokenizer available'
        shared.log.error(msg)
        shared.state.end()
        return '', msg
    current.model.to(devices.device)
    shared.state.textinfo = ""
    input_ids = current.tokenizer(text, return_tensors="pt").input_ids
    if input_ids.shape[1] == 0:
        input_ids = torch.asarray([[current.tokenizer.bos_token_id]], dtype=torch.long)
    input_ids = input_ids.to(devices.device)
    input_ids = input_ids.repeat((batch_size, 1))
    markup = '<table><tbody>'
    index = 0
    shared.log.info(f'PromptGen: model={model_name} batch={batch_count}x{batch_size}')
    for _i in range(batch_count):
        texts = generate_batch(input_ids, *args)
        shared.state.nextjob()
        for generated_text in texts:
            index += 1
            markup += f"""
                <tr>
                    <td><div class="prompt gr-box gr-text-input"><p id='promptgen_res_{index}'>{html.escape(generated_text)}</p></div></td>
                    <td class="sendto">
                        <a class='gr-button gr-button-lg gr-button-secondary' onclick="promptGenSend(gradioApp().getElementById('promptgen_res_{index}').textContent, 'txt2img')">text</a>
                        <a class='gr-button gr-button-lg gr-button-secondary' onclick="promptGenSend(gradioApp().getElementById('promptgen_res_{index}').textContent, 'txt2img')">image</a>
                    </td>
                </tr>
                """
    markup += '</tbody></table>'
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
                prompt = gr.Textbox(label="Prompt", elem_id="promptgen_prompt", show_label=False, lines=2, placeholder="Beginning of the prompt (press Ctrl+Enter or Alt+Enter to generate)").style(container=False)
            with gr.Column(scale=10):
                submit = gr.Button('Generate', elem_id="promptgen_generate", variant='primary')
        with gr.Row(elem_id="promptgen_main"):
            with gr.Column(variant="compact"):
                selected_text = gr.TextArea(elem_id='promptgen_selected_text', visible=False)
                send_to_txt2img = gr.Button(elem_id='promptgen_send_to_txt2img', visible=False)
                send_to_img2img = gr.Button(elem_id='promptgen_send_to_img2img', visible=False)
                with gr.Row():
                    model_selection = gr.Dropdown(label="Model", elem_id="promptgen_model", value="None", choices=["None"] + available_models)
                with gr.Row():
                    sampling_mode = gr.Radio(label="Sampling mode", elem_id="promptgen_sampling_mode", value="Top K", choices=["Top K", "Top P"])
                    top_k = gr.Slider(label="Top K", elem_id="promptgen_top_k", value=12, minimum=1, maximum=50, step=1)
                    top_p = gr.Slider(label="Top P", elem_id="promptgen_top_p", value=0.15, minimum=0, maximum=1, step=0.001)
                with gr.Row():
                    num_beams = gr.Slider(label="Number of beams", elem_id="promptgen_num_beams", value=1, minimum=1, maximum=8, step=1)
                    temperature = gr.Slider(label="Temperature", elem_id="promptgen_temperature", value=1, minimum=0, maximum=4, step=0.01)
                    repetition_penalty = gr.Slider(label="Repetition penalty", elem_id="promptgen_repetition_penalty", value=2, minimum=1, maximum=4, step=0.01)
                with gr.Row():
                    length_penalty = gr.Slider(label="Length preference", elem_id="promptgen_length_preference", value=1, minimum=-10, maximum=10, step=0.1)
                    min_length = gr.Slider(label="Min length", elem_id="promptgen_min_length", value=20, minimum=1, maximum=400, step=1)
                    max_length = gr.Slider(label="Max length", elem_id="promptgen_max_length", value=150, minimum=1, maximum=400, step=1)
                with gr.Row():
                    batch_count = gr.Slider(label="Batch count", elem_id="promptgen_batch_count", value=1, minimum=1, maximum=100, step=1)
                    batch_size = gr.Slider(label="Batch size", elem_id="promptgen_batch_size", value=10, minimum=1, maximum=100, step=1)
                with open(os.path.join(base_dir, "explanation.html"), encoding="utf8") as file:
                    footer = file.read()
                    gr.HTML(footer)
            with gr.Column():
                with gr.Group(elem_id="promptgen_results_column"):
                    res = gr.HTML()
                    res_info = gr.HTML()
        submit.click(fn=ui.wrap_gradio_gpu_call(generate, extra_outputs=['']), _js="promptGenSubmit",
            inputs=[model_selection, model_selection, batch_count, batch_size, prompt, min_length, max_length, num_beams, temperature, repetition_penalty, length_penalty, sampling_mode, top_k, top_p],
            outputs=[res, res_info]
        )
        model_selection.change(fn=model_selection_changed, inputs=[model_selection], outputs=[])
        send_to_txt2img.click(fn=send_prompts, inputs=[selected_text], outputs=find_prompts(ui.txt2img_paste_fields))
        send_to_img2img.click(fn=send_prompts, inputs=[selected_text], outputs=find_prompts(ui.img2img_paste_fields))
    return [(tab, "PromptGen", "promptgen")]


def on_ui_settings():
    section = ("promptgen", "PromptGen")
    shared.opts.add_option("promptgen_names", shared.OptionInfo(model_list, "PromptGen Hugginface models", section=section))


def on_unload():
    current.model = None
    current.tokenizer = None


script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_script_unloaded(on_unload)
