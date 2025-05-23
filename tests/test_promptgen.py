import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import sys
import os

# Adjust sys.path to allow importing from the 'scripts' directory
# This assumes the tests are run from the repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to be tested
from scripts import promptgen 

# Import torch within the test file if needed for type hints or specific torch objects
import torch # Added for dtype=promptgen.torch.long

class TestPromptGenCore(unittest.TestCase):

    def setUp(self):
        # Reset global 'current' model state before each test
        promptgen.current.name = None
        promptgen.current.model = None
        promptgen.current.tokenizer = None
        promptgen.current_device_name = None # Reset the tracked device name

        # Template for mock shared.opts. Specific values should be set in each test.
        self.mock_opts_template = MagicMock()
        self.mock_opts_template.promptgen_device = 'gpu' 
        self.mock_opts_template.promptgen_names = "mock/model1,mock/model2"
        # Add other default opts attributes used by the script if any
        self.mock_opts_template.data = { # Mock for shared.opts.data.get()
            "promptgen_device": "gpu",
            "promptgen_names": "mock/model1,mock/model2"
        }


        # Template for mock devices module. Specific attributes like .device or .cpu should be set in tests.
        self.mock_devices_template = MagicMock()
        self.mock_devices_template.device = 'cuda:test_setup' 
        self.mock_devices_template.cpu = 'cpu:test_setup'    

        # Patch the logger for all tests in this class
        self.patcher_logger = patch('scripts.promptgen.logger')
        self.mock_logger = self.patcher_logger.start()
        self.addCleanup(self.patcher_logger.stop) # Stop patcher after test methods


    # --- Test Cases for _perform_generation ---

    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices') # Mock the entire devices module
    @patch('scripts.promptgen.shared')  # Mock the entire shared module
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True)
    def test_successful_generation(self, mock_cuda_is_available, mock_shared, mock_devices, 
                                   mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        # Configure shared.opts for this specific test
        mock_shared.opts = self.mock_opts_template 
        mock_shared.opts.data = {"promptgen_device": "gpu"} # Ensure device selection logic uses this

        # Configure devices module mock for this specific test
        mock_devices.device = 'cuda:test_success' # Specific device string for this test
        mock_devices.cpu = 'cpu:test_success'

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.bos_token_id = 12345
        mock_tokenizer_instance.pad_token_id = 0
        # Simulate tokenizer call: tokenizer(text, return_tensors="pt")
        # It returns a dict with 'input_ids', which is a tensor mock
        mock_input_ids_tensor = MagicMock(shape=(1,4)) # Simulate tensor shape
        mock_input_ids_tensor.to = MagicMock(return_value=mock_input_ids_tensor) # Mock .to(device)
        mock_tokenizer_instance.return_value = {"input_ids": mock_input_ids_tensor} 
        mock_tokenizer_instance.batch_decode.return_value = ["Generated prompt 1", "Generated prompt 2"]
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = "mock_generated_ids_tensor" 
        mock_model_instance.to = MagicMock() 
        mock_model_from_pretrained.return_value = mock_model_instance
        
        model_name = "mock/model1"
        text = "Test prompt"
        batch_count = 1
        batch_size = 2 
        min_length, max_length, num_beams, temperature, repetition_penalty, length_penalty, sampling_mode, top_k, top_p = 20, 50, 1, 1.0, 1.0, 1.0, "Top K", 50, 1.0

        generated_texts, error_msg = promptgen._perform_generation(
            model_name, batch_count, batch_size, text,
            min_length, max_length, num_beams, temperature,
            repetition_penalty, length_penalty, sampling_mode, top_k, top_p
        )

        self.assertIsNone(error_msg, f"Error message was returned: {error_msg}")
        self.assertEqual(generated_texts, ["Generated prompt 1", "Generated prompt 2"])
        mock_tokenizer_from_pretrained.assert_called_once_with(model_name)
        mock_model_from_pretrained.assert_called_once_with(model_name)
        mock_model_instance.to.assert_called_with('cuda:test_success')
        mock_input_ids_tensor.to.assert_called_with('cuda:test_success')
        mock_model_instance.generate.assert_called_once()
        mock_tokenizer_instance.batch_decode.assert_called_once_with("mock_generated_ids_tensor", skip_special_tokens=True)
        self.assertEqual(promptgen.current.name, model_name)


    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True)
    def test_model_already_loaded(self, mock_cuda_is_available, mock_shared, mock_devices, 
                                  mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_shared.opts = self.mock_opts_template
        mock_shared.opts.data = {"promptgen_device": "gpu"}
        mock_devices.device = 'cuda:test_loaded'

        model_name = "mock/model1"
        # Pre-load the model into 'current'
        promptgen.current.name = model_name
        promptgen.current.tokenizer = MagicMock()
        promptgen.current.model = MagicMock()
        promptgen.current.model.to = MagicMock() 
        
        # Configure generate and batch_decode for the preloaded model
        promptgen.current.model.generate.return_value = "mock_ids_loaded"
        promptgen.current.tokenizer.batch_decode.return_value = ["Already loaded result"]
        # Simulate tokenizer call for the preloaded tokenizer
        mock_loaded_input_ids = MagicMock(shape=(1,4))
        mock_loaded_input_ids.to = MagicMock(return_value=mock_loaded_input_ids)
        promptgen.current.tokenizer.return_value = {"input_ids": mock_loaded_input_ids}


        text = "Test prompt"
        batch_count, batch_size = 1, 1
        min_length, max_length, num_beams, temperature, repetition_penalty, length_penalty, sampling_mode, top_k, top_p = 20, 50, 1, 1.0, 1.0, 1.0, "Top K", 50, 1.0

        generated_texts, error_msg = promptgen._perform_generation(
            model_name, batch_count, batch_size, text,
            min_length, max_length, num_beams, temperature,
            repetition_penalty, length_penalty, sampling_mode, top_k, top_p
        )

        self.assertIsNone(error_msg)
        self.assertEqual(generated_texts, ["Already loaded result"])
        mock_tokenizer_from_pretrained.assert_not_called()
        mock_model_from_pretrained.assert_not_called()
        promptgen.current.model.to.assert_called_with('cuda:test_loaded') 
        mock_loaded_input_ids.to.assert_called_with('cuda:test_loaded')


    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained', side_effect=Exception("Tokenizer Load Error"))
    @patch('scripts.promptgen.devices') # Still need to mock devices for _get_device_logic if it's reached
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True) 
    def test_model_loading_tokenizer_error(self, mock_cuda_is_available, mock_shared, mock_devices, mock_tokenizer_error):
        mock_shared.opts = self.mock_opts_template
        mock_shared.opts.data = {"promptgen_device": "gpu"} # Needed by _get_device_logic if called

        model_name = "mock/failing_tokenizer_model"
        text = "Test prompt"
        batch_count, batch_size = 1,1
        min_length, max_length, num_beams, temperature, repetition_penalty, length_penalty, sampling_mode, top_k, top_p = 20, 50, 1, 1.0, 1.0, 1.0, "Top K", 50, 1.0

        generated_texts, error_msg = promptgen._perform_generation(
            model_name, batch_count, batch_size, text,
            min_length, max_length, num_beams, temperature,
            repetition_penalty, length_penalty, sampling_mode, top_k, top_p
        )
        
        self.assertIsNone(generated_texts)
        self.assertIsNotNone(error_msg)
        self.assertIn("Error loading model", error_msg)
        self.assertIn("Tokenizer Load Error", error_msg)
        self.assertIn(model_name, error_msg)
        self.assertIsNone(promptgen.current.name) 


    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained', side_effect=Exception("Model Load Error"))
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained') 
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True)
    def test_model_loading_model_error(self, mock_cuda_is_available, mock_shared, mock_devices, mock_tokenizer_from_pretrained, mock_model_error):
        mock_shared.opts = self.mock_opts_template
        mock_shared.opts.data = {"promptgen_device": "gpu"}
        mock_tokenizer_from_pretrained.return_value = MagicMock() 

        model_name = "mock/failing_model_load"
        text = "Test prompt"
        batch_count, batch_size = 1,1
        min_length, max_length, num_beams, temperature, repetition_penalty, length_penalty, sampling_mode, top_k, top_p = 20, 50, 1, 1.0, 1.0, 1.0, "Top K", 50, 1.0

        generated_texts, error_msg = promptgen._perform_generation(
            model_name, batch_count, batch_size, text,
            min_length, max_length, num_beams, temperature,
            repetition_penalty, length_penalty, sampling_mode, top_k, top_p
        )

        self.assertIsNone(generated_texts)
        self.assertIsNotNone(error_msg)
        self.assertIn("Error loading model", error_msg)
        self.assertIn("Model Load Error", error_msg)
        self.assertIn(model_name, error_msg)
        self.assertIsNone(promptgen.current.name)


    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True)
    def test_generation_error(self, mock_cuda_is_available, mock_shared, mock_devices, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_shared.opts = self.mock_opts_template
        mock_shared.opts.data = {"promptgen_device": "gpu"}
        mock_devices.device = 'cuda:test_gen_err'

        mock_tokenizer_instance = MagicMock()
        mock_input_ids_gen_err = MagicMock(shape=(1,4))
        mock_input_ids_gen_err.to = MagicMock(return_value=mock_input_ids_gen_err)
        mock_tokenizer_instance.return_value = {"input_ids": mock_input_ids_gen_err}
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_instance.generate.side_effect = Exception("Generate Func Error")
        mock_model_instance.to = MagicMock()
        mock_model_from_pretrained.return_value = mock_model_instance
        
        model_name = "mock/generation_fail_model"
        text = "Test prompt"
        batch_count, batch_size = 1,1
        min_length, max_length, num_beams, temperature, repetition_penalty, length_penalty, sampling_mode, top_k, top_p = 20, 50, 1, 1.0, 1.0, 1.0, "Top K", 50, 1.0

        generated_texts, error_msg = promptgen._perform_generation(
            model_name, batch_count, batch_size, text,
            min_length, max_length, num_beams, temperature,
            repetition_penalty, length_penalty, sampling_mode, top_k, top_p
        )
        
        self.assertEqual(generated_texts, []) 
        self.assertIsNotNone(error_msg)
        self.assertIn("Error during text generation", error_msg)
        self.assertIn("Generate Func Error", error_msg)
        self.assertIn(model_name, error_msg)


    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices') 
    @patch('scripts.promptgen.shared') 
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True) 
    def test_device_selection_gpu(self, mock_cuda_is_available, mock_shared, mock_devices, mock_tokenizer, mock_model):
        mock_opts_instance = MagicMock()
        mock_opts_instance.promptgen_device = 'gpu' # User wants GPU
        # Configure shared.opts.data for _get_device_logic
        type(mock_shared).opts = PropertyMock(return_value=MagicMock(data={'promptgen_device': 'gpu'}))


        mock_devices.device = 'cuda_mock_direct' 
        
        mock_tokenizer.return_value = MagicMock(return_value = {"input_ids": MagicMock(shape=(1,4), to=MagicMock())})
        mock_model_instance = MagicMock()
        mock_model_instance.to = MagicMock()
        mock_model.return_value = mock_model_instance

        promptgen._perform_generation("mock/model", 1, 1, "text", 10,20,1,1,1,1,"Top K",10,1)
        mock_model_instance.to.assert_called_with('cuda_mock_direct')


    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=False) 
    def test_device_selection_cpu_fallback(self, mock_cuda_is_available, mock_shared, mock_devices, mock_tokenizer, mock_model):
        type(mock_shared).opts = PropertyMock(return_value=MagicMock(data={'promptgen_device': 'gpu'})) # User wants GPU

        mock_devices.cpu = 'cpu_mock_fallback' 
        
        mock_tokenizer.return_value = MagicMock(return_value = {"input_ids": MagicMock(shape=(1,4), to=MagicMock())})
        mock_model_instance = MagicMock()
        mock_model_instance.to = MagicMock()
        mock_model.return_value = mock_model_instance

        promptgen._perform_generation("mock/model", 1, 1, "text", 10,20,1,1,1,1,"Top K",10,1)
        mock_model_instance.to.assert_called_with('cpu_mock_fallback')


    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available') 
    def test_device_selection_cpu_explicit(self, mock_cuda_is_available, mock_shared, mock_devices, mock_tokenizer, mock_model):
        type(mock_shared).opts = PropertyMock(return_value=MagicMock(data={'promptgen_device': 'cpu'})) # User wants CPU

        mock_devices.cpu = 'cpu_mock_explicit'
        
        mock_tokenizer.return_value = MagicMock(return_value = {"input_ids": MagicMock(shape=(1,4), to=MagicMock())})
        mock_model_instance = MagicMock()
        mock_model_instance.to = MagicMock()
        mock_model.return_value = mock_model_instance

        promptgen._perform_generation("mock/model", 1, 1, "text", 10,20,1,1,1,1,"Top K",10,1)
        mock_model_instance.to.assert_called_with('cpu_mock_explicit')
        mock_cuda_is_available.assert_not_called() 


    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True)
    @patch('scripts.promptgen.torch.tensor') 
    def test_empty_input_text(self, mock_torch_tensor, mock_cuda_is_available, mock_shared, mock_devices, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        type(mock_shared).opts = PropertyMock(return_value=MagicMock(data={'promptgen_device': 'gpu'}))
        mock_devices.device = 'cuda:test_empty'

        mock_tokenizer_instance = MagicMock()
        empty_input_ids_mock = MagicMock(shape=(1,0)) 
        mock_tokenizer_instance.return_value = {"input_ids": empty_input_ids_mock}
        mock_tokenizer_instance.bos_token_id = 12345 
        mock_tokenizer_instance.batch_decode.return_value = ["Prompt from BOS"]
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.to = MagicMock()
        mock_model_from_pretrained.return_value = mock_model_instance

        mock_bos_tensor = MagicMock()
        mock_bos_tensor.to = MagicMock(return_value=mock_bos_tensor) 
        mock_torch_tensor.return_value = mock_bos_tensor


        model_name = "mock/model_empty_input"
        text = "" 
        batch_count, batch_size = 1,1
        min_length, max_length, num_beams, temperature, repetition_penalty, length_penalty, sampling_mode, top_k, top_p = 10,20,1,1,1,1,"Top K",10,1

        generated_texts, error_msg = promptgen._perform_generation(
            model_name, batch_count, batch_size, text,
            min_length, max_length, num_beams, temperature,
            repetition_penalty, length_penalty, sampling_mode, top_k, top_p
        )

        self.assertIsNone(error_msg)
        self.assertEqual(generated_texts, ["Prompt from BOS"])
        mock_torch_tensor.assert_called_once_with([[mock_tokenizer_instance.bos_token_id]], dtype=torch.long)
        mock_bos_tensor.to.assert_called_with('cuda:test_empty')

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# To run from repo root: python -m unittest tests.test_promptgen
# (Ensure scripts/__init__.py exists)
