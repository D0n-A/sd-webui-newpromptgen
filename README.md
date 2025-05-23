# PromptGen

Update version of this extension:
<https://github.com/AUTOMATIC1111/stable-diffusion-webui-promptgen>

based on the fork below:
<https://github.com/vladmandic/sd-extension-promptgen>

## API Usage

This extension provides an HTTP API for programmatic prompt generation, integrated with the Stable Diffusion WebUI.

### Enabling the API

The API functionality is part of the main Stable Diffusion WebUI. To enable it, you typically need to launch the WebUI with the `--api` command-line argument. Consult the WebUI's documentation for the most up-to-date instructions on enabling API access.

### Endpoint: `POST /promptgen/v1/generate`

This endpoint accepts a JSON payload to generate prompts based on the provided parameters.

### Request Body

The request body must be a JSON object. Key parameters include:

-   `model_name` (string, **required**): The Hugging Face model name to be used for generation (e.g., `"AUTOMATIC/promptgen-lexart"`).
-   `text` (string, **required**): The initial text or keywords to base the prompt generation on.
-   `batch_count` (integer, optional, default: `1`): Number of times to repeat the generation process for the given input text.
-   `num_return_sequences` (integer, optional, default: `1`): Number of prompt variations to generate for each input text per batch_count iteration. Total prompts generated will be `batch_count * num_return_sequences`.
-   `min_length` (integer, optional, default: `20`): Minimum length of the generated prompts.
-   `max_length` (integer, optional, default: `150`): Maximum length of the generated prompts.
-   `temperature` (float, optional, default: `1.0`): Controls randomness. Higher values mean more random, lower values mean more deterministic. Must be > 0.
-   `repetition_penalty` (float, optional, default: `1.0`): Penalizes repeated tokens. 1.0 means no penalty.
-   `length_penalty` (float, optional, default: `1.0`): Influences the length of generated text. >1.0 encourages longer, <1.0 shorter.
-   `num_beams` (integer, optional, default: `1`): Number of beams for beam search. `1` means no beam search.
-   `sampling_mode` (string, optional, default: `"Top K"`): Choose between `"Top K"` or `"Top P"` sampling.
-   `top_k` (integer, optional, default: `50`): Filters the vocabulary to the K most likely next tokens. Used if `sampling_mode` is "Top K".
-   `top_p` (float, optional, default: `1.0`): Filters the vocabulary to the smallest set of tokens whose cumulative probability exceeds P. Used if `sampling_mode` is "Top P".

**Example JSON Request:**

```json
{
  "model_name": "AUTOMATIC/promptgen-lexart",
  "text": "A cat exploring a mysterious forest",
  "num_return_sequences": 2,
  "max_length": 75,
  "temperature": 1.2,
  "repetition_penalty": 1.5
}
```

### Response Body

-   **On Success (HTTP 200):**
    A JSON object with the following structure:
    -   `prompts` (list of strings): The list of generated prompts.
    -   `model_name` (string): The Hugging Face model name that was used.
    -   `input_text` (string): The original input text provided in the request.
    -   `detail` (string, optional): Typically `null` on success, but may contain informational messages.

    **Example JSON Success Response:**
    ```json
    {
      "prompts": [
        "A curious calico cat, with wide green eyes, cautiously steps into a mysterious, moonlit forest, ancient trees with glowing fungi, detailed digital painting, artstation.",
        "Mysterious forest ambiance, a small cat silhouette against towering, gnarled trees, volumetric lighting, fog, fantasy art by Brom and Frank Frazetta."
      ],
      "model_name": "AUTOMATIC/promptgen-lexart",
      "input_text": "A cat exploring a mysterious forest",
      "detail": null
    }
    ```

-   **On Error (e.g., HTTP 400, HTTP 500):**
    A JSON object, typically containing:
    -   `detail` (string): A message describing the error (e.g., model loading failure, invalid parameter).

    **Example JSON Error Response (HTTP 500):**
    ```json
    {
      "detail": "PromptGen: Error loading model \"AUTOMATIC/non_existent_model\": Model Hub CHTTP Error (Request ID: ...): 404 Client Error for url: https://huggingface.co/api/models/AUTOMATIC/non_existent_model. (Not Found)"
    }
    ```

### Example Usage

#### curl

```bash
curl -X POST http://127.0.0.1:7860/promptgen/v1/generate \
-H "Content-Type: application/json" \
-d '{
  "model_name": "AUTOMATIC/promptgen-lexart",
  "text": "A futuristic cityscape at sunset",
  "num_return_sequences": 1,
  "max_length": 80,
  "temperature": 1.1
}'
```

#### Python (`requests` library)

```python
import requests
import json

api_url = "http://127.0.0.1:7860/promptgen/v1/generate" # Ensure this is your WebUI's URL

payload = {
  "model_name": "AUTOMATIC/promptgen-lexart", # Or any other model configured in PromptGen
  "text": "A secret agent in a cyberpunk city",
  "batch_count": 1,
  "num_return_sequences": 2, # Generate 2 prompt variations
  "max_length": 70,
  "temperature": 1.15,
  "repetition_penalty": 1.2,
  "sampling_mode": "Top K",
  "top_k": 40
}

try:
    response = requests.post(api_url, json=payload)
    response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
    
    data = response.json()
    print("Generated Prompts:")
    for i, prompt_text in enumerate(data.get("prompts", [])):
        print(f"{i+1}. {prompt_text}")
    
    print(f"\nModel Used: {data.get('model_name')}")
    print(f"Input Text: {data.get('input_text')}")
    if data.get('detail'):
        print(f"Details: {data.get('detail')}")

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
    try:
        print(f"Error details: {response.json().get('detail')}")
    except json.JSONDecodeError:
        print(f"Raw error response: {response.text}")
except requests.exceptions.RequestException as req_err:
    print(f"Request exception occurred: {req_err}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

## Recent Improvements

-   **HTTP API:** Added a robust HTTP API for programmatic prompt generation (see API Usage section for details).
-   **Logging:** Implemented comprehensive server-side Python logging (`promptgen` logger) for easier debugging, monitoring of model loading, generation process, and API requests.
-   **Error Handling:** Enhanced error handling for model loading (tokenizer and model), device placement, tokenization, and text generation stages. Errors are now more specific and also reported clearly in the API responses and UI.
-   **Refactoring:** The core prompt generation logic was refactored into an internal `_perform_generation` function, used by both the Gradio UI and the new FastAPI endpoint.
-   **Unit Tests:** Added unit tests for the core `_perform_generation` logic, covering various scenarios including successful generation, error conditions, and device handling.
-   **UI Settings:** Default values for UI sliders are now configurable via the WebUI's main settings page under the "PromptGen" section.
-   **Asynchronous API:** Implemented asynchronous API operations using FastAPI and asyncio, making the API more responsive, especially under concurrent loads.
-   **Prompt Caching:** Added LRU (Least Recently Used) caching for generated prompts to significantly speed up responses for repeated requests. Cache can be enabled/disabled and its size configured in PromptGen's settings.
-   **UI Model Status Display:** Enhanced the user interface with a dedicated model status display, providing clearer feedback on model loading, readiness, and errors.
-   **`num_return_sequences` Integration:** Integrated `num_return_sequences` parameter (API and UI) for more direct and efficient generation of multiple prompt variations per input, aligning better with Hugging Face model capabilities.
