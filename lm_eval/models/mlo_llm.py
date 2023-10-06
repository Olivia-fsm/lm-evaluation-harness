import torch
import transformers
from typing import Optional, Union
from lm_eval.base import BaseLM
from .base_mlo import GPTBase
import tiktoken


def _get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype

class ModelConf:
    def __init__(self, config):
        self.vocab_size = config['vocab_size']
        self.dropout = config['dropout']
        self.n_head = config['n_head']
        self.n_layer = config['n_layer']
        self.n_embd = config['n_embd']
        self.sequence_length = config['sequence_length']
        self.bias = config['bias']
        self.ckpt_path=config['ckpt_path']
        
def load_checkpoint(checkpoint_path, 
                    model_config:ModelConf, 
                    device='cpu',
                    train=False,):
    print("\nLoading MLO-LLM checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()} # distributed code appends '_orig_mod.' to param name
    model = GPTBase(model_config)
    model.load_state_dict(state_dict, strict=False) # olivia-add: strict=False, but do not understand why params fail to match
    model.to(device)
    if train:
        model.train()
    else:
        model.eval()
    return model

class MLOLM(BaseLM):

    _DEFAULT_MAX_LENGTH = 512

    def __init__(
        self,
        device="cuda",
        config=None,
        low_cpu_mem_usage=None,
        subfolder=None,
        batch_size=1,
        max_batch_size=512,
        max_length=None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
    ):
        super().__init__()
        self._device = torch.device(device)
        print(f"Using device '{device}'")
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.config = ModelConf(config=config)
        self.model_path = self.config.ckpt_path
        
        # Initialize new model and tokenizer instances
        self.model = load_checkpoint(checkpoint_path=self.model_path,
                                     model_config=config,
                                     device=self.device,
                                     train=False,).to('cuda')

        self.vocab_size = self.tokenizer.n_vocab

        # Validate batch_size
        assert type(batch_size)==str or type(batch_size)==int, f"batch_size must be str or int but got {type(batch_size)}"

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_length = self.config.sequence_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eot_token

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("sequence_length", "n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, allowed_special={"<|endoftext|>"})

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            if len(inps.shape)<2:
                return self.model(inps[:-1],
                                targets=inps[1:],
                                get_logits=True)["logits"]
            else:
                return self.model(inps[:, :-1],
                                targets=inps[:, 1:],
                                get_logits=True)["logits"]

    def _model_generate(self, context, max_length,):
        if type(context)==str:
            context = self.tok_encode(context)
        generation_kwargs = {"max_new_tokens": max_length}
        gen_ids = self.model.generate(context, **generation_kwargs)
        return self.tokenizer.decode(gen_ids)


# for backwards compatibility
# GPT2LM = MLOLM
