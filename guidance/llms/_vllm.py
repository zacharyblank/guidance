from ._llm import LLM, LLMSession, SyncSession
from vllm.vllm import LLM as _vLLM, SamplingParams
from vllm.logits_processors import LogitsProcessor
import torch
import regex
from typing import Dict

class VLLM(LLM):
    llm_name: str = "vLLM"

    def __init__(self, model=None, tokenizer=None, caching=True, token_healing=True, acceleration=True,
                 temperature=0.0, device="cuda", **kwargs):
        super().__init__()

        # TODO: Handle model and tokenizer objects instead of only strings
        self.model_obj = _vLLM(model=model, **kwargs)
        self.tokenizer = self.model_obj.get_tokenizer()
        self.device = device

        print(self.tokenizer.vocab_size)

    def new_string_builder(self, starting_ids=None):
        return VLLMStringBuilder(self.tokenizer, starting_ids)

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string, **kwargs)
        
    def decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)
    
    def id_to_token(self, id):
        return self.tokenizer.convert_ids_to_tokens([id])[0]
    
    def token_to_id(self, token):
        return self.tokenizer.convert_tokens_to_ids([token])[0]

    def end_of_text(self):
        return self.tokenizer.eos_token

    def session(self, asynchronous=False):
        if asynchronous:
            return VLLMSession(self)
        else:
            return SyncSession(VLLMSession(self))

class VLLMSession(LLMSession):
    def __init__(self, llm):
        super().__init__(llm)
        
        self._past_key_values = None
        self._prefix_cache = []

    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None,
                       top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=False,
                       cache_seed=0, caching=None, **generate_kwargs):

        encoded = self.llm.encode(prompt)
        encoded = torch.tensor([encoded for _ in range(n)])
        if self.llm.device is not None:
            encoded = encoded.to(self.llm.device)
        input_ids = encoded#["input_ids"]
        coded_prompt = self.llm.decode(input_ids[0])

        # Set the stop patterns
        if stop is not None:
            if isinstance(stop, str):
                stop_regex = [regex.escape(stop)]
            else:
                stop_regex = [regex.escape(s) for s in stop]
        if isinstance(stop_regex, str):
            stop_regex = [stop_regex]
        if stop_regex is None:
            stop_regex = []
        stop_regex.append(regex.escape(self.llm.tokenizer.eos_token)) # make sure the end of sequence token is always included

        # Set the Logits Processors
        logit_processors = []

        # setup logit biasing
        if logit_bias is not None:
            print("Call the bias processor")
            # processors.append(BiasLogitsProcessor(self.llm, model_config.vocab_size, logit_bias))
            
        if pattern is not None:
            logit_processors.append(RegexLogitsProcessor(pattern, stop_regex, self.llm, self.llm.tokenizer.vocab_size, temperature == 0, len(coded_prompt), self.llm.tokenizer.eos_token_id))

        sampling_params = SamplingParams(
            n=n,
            # presence_penalty=request.presence_penalty,
            # frequency_penalty=request.frequency_penalty,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            max_tokens=max_tokens,
            # best_of=request.best_of,
            # top_k=request.top_k,
            # ignore_eos=request.ignore_eos,
            # use_beam_search=request.use_beam_search,
            logits_processors=logit_processors
        )

        outputs = self.llm.model_obj.generate(prompt, sampling_params)
        response = []

        for output in outputs[0].outputs:
            response.append({
                "text": output.text.strip()
            })

        out = {"choices": response}

        return out

class VLLMStringBuilder():
    """This deals with the complexity of building up a string from tokens bit by bit."""
    def __init__(self, tokenizer, starting_ids=None):
        self.tokenizer = tokenizer
        self.token_strings = []
        self._joint_string = ""
        if starting_ids is not None:
            self.extend(starting_ids)

    def extend(self, new_ids):
        new_token_strings = self.tokenizer.convert_ids_to_tokens(new_ids)
        self.token_strings.extend(new_token_strings)
        new_str = self.tokenizer.convert_tokens_to_string(self.token_strings)
        diff_str = new_str[len(self._joint_string):]
        self._joint_string = new_str
        return diff_str

    def append_id(self, new_id):
        new_token_strings = self.tokenizer.convert_ids_to_tokens([new_id])
        new_str = self.tokenizer.convert_tokens_to_string(new_token_strings)
        return (str(self) + new_str).strip()

    def pop(self):
        """Remove the last token from the string and return text it removed."""
        self.token_strings.pop()
        new_str = self.tokenizer.convert_tokens_to_string(self.token_strings)
        diff_str = self._joint_string[len(new_str):]
        self._joint_string = new_str
        return diff_str

    def __str__(self):
        return self._joint_string

    def __len__(self):
        return len(self._joint_string)

class RegexLogitsProcessor(LogitsProcessor):
        
    def __init__(self, pattern, stop_regex, llm, vocab_size, is_greedy, prefix_length, eos_token_id, max_consider=500000):
        if isinstance(stop_regex, str):
            stop_regex = [stop_regex]
        self.pattern_no_stop = regex.compile(pattern)
        self.pattern = regex.compile(pattern + "(" + "|".join(stop_regex) + ")?")
        self.llm = llm
        self.is_greedy = is_greedy
        self.prefix_length = prefix_length
        self.max_consider = max_consider
        self.bias_vector = torch.zeros(vocab_size)
        self.current_strings = None
        self.current_length = 0
        self.forced_chars = 0
        self.eos_token_id = eos_token_id
        self.vocab = {v: k for k, v in self.llm.model_obj.get_tokenizer().vocab.items()}

    def __call__(self, logits: torch.tensor, input_ids: Dict[int, int]) -> torch.tensor:
        # extend our current strings
        if self.current_strings is None:
            self.current_strings = [self.llm.new_string_builder() for i in range(len(input_ids))]
        for i in range(len(self.current_strings)):
            self.current_strings[i].extend(input_ids[i][self.current_length:])

        def regex_match_partial(token_id, logit, vocab, current_string, pattern):
            token = str(vocab.get(token_id))

            proposed_string = current_string + token
            match = pattern.fullmatch(proposed_string, partial=True)
            if match:
                return logit

            return float('-inf')

        biases = [regex_match_partial(token_id, logit, self.vocab, str(self.current_strings[0]), self.pattern) for token_id, logit in enumerate(logits[0])]
        biases = torch.tensor(biases, device=logits.device).unsqueeze(0)

        return biases