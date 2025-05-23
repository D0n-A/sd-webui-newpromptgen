import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import sys
import os
import asyncio 
import cachetools 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts import promptgen 
import torch 

class TestPromptGenCore(unittest.IsolatedAsyncioTestCase): 

    async def asyncSetUp(self): 
        promptgen.current.name = None
        promptgen.current.model = None
        promptgen.current.tokenizer = None
        promptgen.current_device_name = None 
        promptgen.prompt_cache = None 
        promptgen.cache_initialized_with_size = None

        self.mock_opts_template = MagicMock()
        self.mock_opts_template.data = { 
            "promptgen_device": "gpu",
            "promptgen_names": "mock/model1,mock/model2",
            "promptgen_enable_cache": True,
            "promptgen_cache_size": 10,
            "promptgen_num_return_sequences": 1 # Default for tests
        }

        self.mock_devices_template = MagicMock()
        self.mock_devices_template.device = 'cuda:test_setup' 
        self.mock_devices_template.cpu = 'cpu:test_setup'    

        self.patcher_logger = patch('scripts.promptgen.logger')
        self.mock_logger = self.patcher_logger.start()
        self.addCleanup(self.patcher_logger.stop) 

        self.patcher_init_cache = patch('scripts.promptgen._initialize_cache', MagicMock())
        self.mock_init_cache = self.patcher_init_cache.start()
        self.addCleanup(self.patcher_init_cache.stop)

    # Default parameters for _perform_generation, now including num_return_sequences
    def get_default_gen_params(self, **kwargs):
        defaults = {
            "model_name": "mock/model1",
            "batch_count": 1,
            "num_return_sequences": 1, # Default num_return_sequences
            "text": "Test prompt",
            "min_length": 20, "max_length": 50, "num_beams": 1, 
            "temperature": 1.0, "repetition_penalty": 1.0, "length_penalty": 1.0, 
            "sampling_mode": "Top K", "top_k": 50, "top_p": 1.0
        }
        defaults.update(kwargs)
        return defaults

    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices') 
    @patch('scripts.promptgen.shared')  
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True)
    async def test_successful_generation(self, mock_cuda_is_available, mock_shared, mock_devices, 
                                   mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_shared.opts = MagicMock() 
        mock_shared.opts.data = { "promptgen_device": "gpu", "promptgen_enable_cache": False, "promptgen_cache_size": 0 }
        mock_devices.device = 'cuda:test_success'; mock_devices.cpu = 'cpu:test_success'

        mock_tokenizer_instance = MagicMock(bos_token_id=12345, pad_token_id=0)
        mock_input_ids_tensor = MagicMock(shape=(1,4)); mock_input_ids_tensor.to = MagicMock(return_value=mock_input_ids_tensor) 
        mock_tokenizer_instance.return_value = {"input_ids": mock_input_ids_tensor} 
        mock_tokenizer_instance.batch_decode.return_value = ["Generated prompt 1"] # Adjusted for num_return_sequences=1
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock(generate=MagicMock(return_value="mock_ids_tensor"), to=MagicMock())
        mock_model_from_pretrained.return_value = mock_model_instance
        
        params = self.get_default_gen_params()

        generated_texts, error_msg = await promptgen._perform_generation(**params)

        self.assertIsNone(error_msg, f"Error: {error_msg}")
        self.assertEqual(generated_texts, ["Generated prompt 1"])
        mock_tokenizer_from_pretrained.assert_called_once_with(params["model_name"])
        mock_model_from_pretrained.assert_called_once_with(params["model_name"])
        mock_model_instance.to.assert_called_with('cuda:test_success')
        mock_input_ids_tensor.to.assert_called_with('cuda:test_success')
        mock_model_instance.generate.assert_called_once()
        self.assertEqual(mock_model_instance.generate.call_args[1]['num_return_sequences'], params["num_return_sequences"])
        mock_tokenizer_instance.batch_decode.assert_called_once_with("mock_ids_tensor", skip_special_tokens=True)
        self.assertEqual(promptgen.current.name, params["model_name"])

    # --- New Tests for num_return_sequences ---
    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True)
    async def test_num_return_sequences_passed_to_model(self, mock_cuda, mock_shared, mock_devices, mock_tokenizer, mock_model):
        mock_shared.opts = MagicMock(); mock_shared.opts.data = {"promptgen_device": "gpu", "promptgen_enable_cache": False}
        mock_devices.device = 'cuda:num_seq_test'
        
        mock_tokenizer_instance = MagicMock(bos_token_id=1, pad_token_id=0, return_value={"input_ids": MagicMock(shape=(1,1), to=MagicMock())}, batch_decode=MagicMock(return_value=["seq1", "seq2"]))
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock(to=MagicMock())
        mock_model_instance.generate = MagicMock(return_value="mock_output_ids") # Mock the generate method itself
        mock_model.return_value = mock_model_instance

        test_num_sequences = 3
        params = self.get_default_gen_params(num_return_sequences=test_num_sequences)
        
        await promptgen._perform_generation(**params)

        mock_model_instance.generate.assert_called_once()
        # Check if num_return_sequences was passed correctly to model.generate
        called_kwargs = mock_model_instance.generate.call_args[1]
        self.assertEqual(called_kwargs.get('num_return_sequences'), test_num_sequences)

    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True)
    async def test_cache_key_includes_num_return_sequences(self, mock_cuda, mock_shared, mock_devices, mock_tokenizer, mock_model):
        mock_shared.opts = MagicMock()
        mock_shared.opts.data = {"promptgen_enable_cache": True, "promptgen_cache_size": 10, "promptgen_device": "gpu"}
        self.patcher_init_cache.stop(); promptgen._initialize_cache(); self.addCleanup(self.patcher_init_cache.start)

        mock_devices.device = 'cuda:cache_num_seq'
        mock_tokenizer_instance = MagicMock(bos_token_id=1, pad_token_id=0, return_value={"input_ids": MagicMock(shape=(1,1), to=MagicMock())})
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_instance = MagicMock(to=MagicMock())
        mock_model_instance.generate = MagicMock() # Mock generate directly
        mock_model.return_value = mock_model_instance

        params1 = self.get_default_gen_params(num_return_sequences=1, text="unique_text_for_num_seq_cache_test1")
        mock_model_instance.generate.return_value = "ids_seq1"; mock_tokenizer_instance.batch_decode.return_value = ["res_seq1"]
        await promptgen._perform_generation(**params1)
        mock_model_instance.generate.assert_called_once()
        
        params2 = self.get_default_gen_params(num_return_sequences=2, text="unique_text_for_num_seq_cache_test1") # Same text, different num_return_sequences
        mock_model_instance.generate.return_value = "ids_seq2"; mock_tokenizer_instance.batch_decode.return_value = ["res_seq2a", "res_seq2b"]
        await promptgen._perform_generation(**params2)
        self.assertEqual(mock_model_instance.generate.call_count, 2) # Should be called again due to different num_return_sequences

    # --- Ensure other tests are updated to use get_default_gen_params or pass num_return_sequences ---
    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True)
    async def test_model_already_loaded(self, mock_cuda, mock_shared, mock_devices, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_shared.opts = MagicMock(); mock_shared.opts.data = {"promptgen_device": "gpu", "promptgen_enable_cache": False}
        mock_devices.device = 'cuda:test_loaded'

        params = self.get_default_gen_params(model_name="mock/model_loaded_test")
        
        promptgen.current.name = params["model_name"]
        promptgen.current.tokenizer = MagicMock(return_value={"input_ids": MagicMock(shape=(1,4), to=MagicMock())}, batch_decode=MagicMock(return_value=["Already loaded result"]))
        promptgen.current.model = MagicMock(to=MagicMock(), generate=MagicMock(return_value="mock_ids_loaded"))
        
        generated_texts, error_msg = await promptgen._perform_generation(**params)

        self.assertIsNone(error_msg)
        self.assertEqual(generated_texts, ["Already loaded result"])
        mock_tokenizer_from_pretrained.assert_not_called()
        mock_model_from_pretrained.assert_not_called()
        promptgen.current.model.to.assert_called_with('cuda:test_loaded') 

    # (Other tests like error handling, device selection, empty input text should also be updated
    # by passing parameters using self.get_default_gen_params() to ensure num_return_sequences is included)

    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained', side_effect=Exception("Tokenizer Load Error"))
    @patch('scripts.promptgen.devices') 
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True) 
    async def test_model_loading_tokenizer_error(self, mock_cuda, mock_shared, mock_devices, mock_tokenizer_error):
        mock_shared.opts = MagicMock(); mock_shared.opts.data = {"promptgen_device": "gpu", "promptgen_enable_cache": False}
        params = self.get_default_gen_params(model_name="mock/failing_tokenizer_model")
        generated_texts, error_msg = await promptgen._perform_generation(**params)
        self.assertIsNone(generated_texts); self.assertIsNotNone(error_msg)
        self.assertIn("Error loading model", error_msg); self.assertIn("Tokenizer Load Error", error_msg)

    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained', side_effect=Exception("Model Load Error"))
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained') 
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True)
    async def test_model_loading_model_error(self, mock_cuda, mock_shared, mock_devices, mock_tokenizer, mock_model_error):
        mock_shared.opts = MagicMock(); mock_shared.opts.data = {"promptgen_device": "gpu", "promptgen_enable_cache": False}
        mock_tokenizer.return_value = MagicMock() 
        params = self.get_default_gen_params(model_name="mock/failing_model_load")
        generated_texts, error_msg = await promptgen._perform_generation(**params)
        self.assertIsNone(generated_texts); self.assertIsNotNone(error_msg)
        self.assertIn("Error loading model", error_msg); self.assertIn("Model Load Error", error_msg)

    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True)
    async def test_generation_error(self, mock_cuda, mock_shared, mock_devices, mock_tokenizer, mock_model):
        mock_shared.opts = MagicMock(); mock_shared.opts.data = {"promptgen_device": "gpu", "promptgen_enable_cache": False}
        mock_devices.device = 'cuda:test_gen_err'
        mock_tokenizer.return_value = MagicMock(return_value={"input_ids": MagicMock(shape=(1,4), to=MagicMock())})
        mock_model_instance = MagicMock(to=MagicMock(), generate=MagicMock(side_effect=Exception("Generate Func Error")))
        mock_model.return_value = mock_model_instance
        params = self.get_default_gen_params(model_name="mock/generation_fail_model")
        generated_texts, error_msg = await promptgen._perform_generation(**params)
        self.assertEqual(generated_texts, []); self.assertIsNotNone(error_msg)
        self.assertIn("Error during text generation", error_msg); self.assertIn("Generate Func Error", error_msg)

    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices') 
    @patch('scripts.promptgen.shared') 
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True) 
    async def test_device_selection_gpu(self, mock_cuda, mock_shared, mock_devices, mock_tokenizer, mock_model):
        type(mock_shared).opts = PropertyMock(return_value=MagicMock(data={'promptgen_device': 'gpu', 'promptgen_enable_cache': False}))
        mock_devices.device = 'cuda_mock_direct' 
        mock_tokenizer.return_value = MagicMock(return_value = {"input_ids": MagicMock(shape=(1,4), to=MagicMock())})
        mock_model_instance = MagicMock(to=MagicMock()); mock_model.return_value = mock_model_instance
        params = self.get_default_gen_params()
        await promptgen._perform_generation(**params)
        mock_model_instance.to.assert_called_with('cuda_mock_direct')

    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=False) 
    async def test_device_selection_cpu_fallback(self, mock_cuda, mock_shared, mock_devices, mock_tokenizer, mock_model):
        type(mock_shared).opts = PropertyMock(return_value=MagicMock(data={'promptgen_device': 'gpu', 'promptgen_enable_cache': False})) 
        mock_devices.cpu = 'cpu_mock_fallback' 
        mock_tokenizer.return_value = MagicMock(return_value = {"input_ids": MagicMock(shape=(1,4), to=MagicMock())})
        mock_model_instance = MagicMock(to=MagicMock()); mock_model.return_value = mock_model_instance
        params = self.get_default_gen_params()
        await promptgen._perform_generation(**params)
        mock_model_instance.to.assert_called_with('cpu_mock_fallback')

    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available') 
    async def test_device_selection_cpu_explicit(self, mock_cuda, mock_shared, mock_devices, mock_tokenizer, mock_model):
        type(mock_shared).opts = PropertyMock(return_value=MagicMock(data={'promptgen_device': 'cpu', 'promptgen_enable_cache': False})) 
        mock_devices.cpu = 'cpu_mock_explicit'
        mock_tokenizer.return_value = MagicMock(return_value = {"input_ids": MagicMock(shape=(1,4), to=MagicMock())})
        mock_model_instance = MagicMock(to=MagicMock()); mock_model.return_value = mock_model_instance
        params = self.get_default_gen_params()
        await promptgen._perform_generation(**params)
        mock_model_instance.to.assert_called_with('cpu_mock_explicit')
        mock_cuda.assert_not_called() 

    @patch('scripts.promptgen.transformers.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.promptgen.transformers.AutoTokenizer.from_pretrained')
    @patch('scripts.promptgen.devices')
    @patch('scripts.promptgen.shared')
    @patch('scripts.promptgen.torch.cuda.is_available', return_value=True)
    @patch('scripts.promptgen.torch.tensor') 
    async def test_empty_input_text(self, mock_torch_tensor, mock_cuda, mock_shared, mock_devices, mock_tokenizer, mock_model):
        type(mock_shared).opts = PropertyMock(return_value=MagicMock(data={'promptgen_device': 'gpu', 'promptgen_enable_cache': False}))
        mock_devices.device = 'cuda:test_empty'
        mock_tokenizer_instance = MagicMock(return_value={"input_ids": MagicMock(shape=(1,0))}, bos_token_id=12345, batch_decode=MagicMock(return_value=["Prompt from BOS"]))
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model.return_value = MagicMock(to=MagicMock())
        mock_bos_tensor = MagicMock(to=MagicMock(return_value=MagicMock())); mock_torch_tensor.return_value = mock_bos_tensor
        params = self.get_default_gen_params(text="")
        generated_texts, error_msg = await promptgen._perform_generation(**params)
        self.assertIsNone(error_msg); self.assertEqual(generated_texts, ["Prompt from BOS"])
        mock_torch_tensor.assert_called_once_with([[mock_tokenizer_instance.bos_token_id]], dtype=torch.long)
        mock_bos_tensor.to.assert_called_with('cuda:test_empty')

if __name__ == '__main__':
    pass
