from ._llm import LLM, LLMSession, SyncSession
from vllm.vllm import LLM as _vLLM, SamplingParams
from vllm.logits_processors import LogitsProcessor
import torch
import regex
import queue
from typing import Dict

class VLLM(LLM):
    llm_name: str = "vLLM"

    def __init__(self, model=None, tokenizer=None, caching=True, token_healing=True, acceleration=True,
                 temperature=0.0, device="cuda", **kwargs):
        super().__init__()

        # TODO: Handle model and tokenizer objects instead of only strings
        self.model_obj = _vLLM(model=model, **kwargs)
        self.tokenizer = self.model_obj.get_tokenizer()
        self.temperature = temperature
        self.device = device

        self._prefix_ids = [self.tokenizer.bos_token_id, 100] # token ids that we use to decode tokens after a prefix
        self._prefix_str = self.tokenizer.decode(self._prefix_ids, fragment=False)


    def new_string_builder(self, starting_ids=None):
        return VLLMStringBuilder(self.tokenizer, starting_ids)

    def encode(self, string, fragment=True, **kwargs):
        if fragment:
            string = self._prefix_str + string

        if "return_tensors" in kwargs:
            out = self.tokenizer(string, **kwargs)
        else:
            out = self.tokenizer.encode(string, **kwargs)
        
        # remove the start token when we are encoding a suffix
        if fragment:
            if out[1] == self.tokenizer.bos_token_id: # sometime the tokenizer adds an extra start token
                out = out[3:]
            else:
                out = out[2:]
        
        return out
            
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

        # fill in defaults
        if temperature is None:
            temperature = self.llm.temperature
        # if token_healing is None:
        #     token_healing = self.llm.token_healing

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
            print(f"Call the bias processor on {logit_bias}")
            logit_processors.append(BiasLogitsProcessor(logit_bias))
            
        if pattern is not None:
            print("Call the regex")
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
        
        print(f"Outputs: {out}")

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
    
class BiasLogitsProcessor(LogitsProcessor):
    def __init__(self, biases):
        self.biases = biases

        print(f"Biases: {self.biases}")

        if not biases:
            return

        self.keys = torch.tensor(list(self.biases.keys()), dtype=torch.long)
        self.values = torch.tensor(list(self.biases.values()), dtype=torch.long)

    def __call__(self, logits: torch.tensor, input_ids: Dict[int, int]) -> torch.tensor:
        if not self.biases:
            return logits

        values = self.values.to(logits.device)
        keys = self.keys.to(logits.device)

        update_factors = torch.where(values >= 0, 1 + (values / 100), 1 / (1 - (values / 100)))
        logits[0, keys] *= update_factors

        logits = torch.zeros(logits.shape)

        logits[0][6219] = 23

        print(logits[0][6212:6222])

        return logits
    
class VLLMStreamer():
    def __init__(self, input_ids, stop_regex, last_token_str, coded_prompt, llm, max_new_tokens, lobprobs, timeout=None):
        self.timeout = timeout
        self.input_ids = input_ids
        self.stop_regex = stop_regex
        self.logprobs = lobprobs
        self.last_token_str = last_token_str
        self.llm = llm
        self.max_total_tokens = max_new_tokens + len(input_ids[0])
        coded_prompt = coded_prompt[:len(coded_prompt)-len(last_token_str)] # strip off the last token which will be regenerated
        self.str_pos = [len(coded_prompt) + len(self.last_token_str) for i in range(len(self.input_ids))]
        self.out_queue = queue.Queue()
        self.sequence_pos = [len(self.input_ids[0]) for i in range(len(self.input_ids))]
        self.generated_sequence = [[] for i in range(len(self.input_ids))]
        self.display_logprobs = [[] for i in range(len(self.input_ids))]
        self.generated_string = [coded_prompt for i in range(len(self.input_ids))]
        self.prefix_cache = []

    def put(self, token_obj):

        import torch
        if isinstance(token_obj, torch.Tensor):
            new_tokens = token_obj
        else:
            new_tokens = token_obj['sequences']
        

        if isinstance(new_tokens, torch.Tensor):
            new_tokens = new_tokens.cpu()
        
        # if we are given a single sequence, then make it a batch of size 1
        if len(new_tokens.shape) == 1:
            new_tokens = new_tokens.unsqueeze(0)
        
        
        # extract the scores if we are given them (and format them to be the same shape as the tokens)
        if self.logprobs:
            assert len(new_tokens) == 1, "logprobs are not supported for batched generation right now in guidance.llms.Transformers"
            new_scores = [torch.nn.functional.log_softmax(x, dim=-1).cpu() for x in token_obj['scores']]
            len_diff = len(new_tokens[0]) - len(new_scores)
            if len_diff > 0:
                new_scores = [None for i in range(len_diff)] + new_scores
            new_scores = [new_scores]
        
        out = {"choices": [None for i in range(len(self.input_ids))]}
        put_data = False
        for i in range(len(self.input_ids)):
            self.generated_sequence[i].extend(list(new_tokens[i]))
            
            # save logprobs if needed
            if self.logprobs:
                for scores in new_scores[i]:
                    if scores is None:
                        self.display_logprobs[i].append(None)
                    else:
                        top_inds = scores[0].argsort(descending=True)[:self.logprobs] # TODO: verify the [0] is always correct
                        self.display_logprobs[i].append({self.llm.id_to_token(j): float(scores[0][j]) for j in top_inds})

            if self.sequence_pos[i] < len(self.generated_sequence[i]):
                display_tokens = list(self.generated_sequence[i][self.sequence_pos[i]:])
                val = self.llm.decode(display_tokens)#[self.llm._prefix_token_id] + display_tokens)[len(self.llm._prefix_token):]
                self.generated_string[i] += val
                
                if self.str_pos[i] < len(self.generated_string[i]):
                    val = self.generated_string[i][self.str_pos[i]:]
                    finish_reason = None
                    
                    # check why we stopped
                    stop_pos = len(val) + 1
                    if len(self.generated_sequence[i]) >= self.max_total_tokens:
                        finish_reason = "length"
                    elif self.generated_sequence[i][-1] == self.llm._tokenizer.eos_token_id:
                        finish_reason = "endoftext"
                        stop_pos = len(val) - len(self.llm.decode([self.llm._tokenizer.eos_token_id]))

                    # trim off the stop regex matches if needed
                    found_partial = False
                    stop_text = None
                    if self.stop_regex is not None:# and (finish_reason is None or len(self.input_ids) > 1):
                        stop_regex_obj = [regex.compile(s) for s in self.stop_regex]
                        for s in stop_regex_obj:
                            m = s.search(val, partial=True)
                            if m:
                                span = m.span()
                                if span[1] > span[0]:
                                    if m.partial: # we might be starting a stop sequence, so we can't emit anything yet
                                        found_partial = True
                                        break
                                    else:
                                        stop_text = val[span[0]:span[1]]
                                        stop_pos = min(span[0], stop_pos)
                                        break

                    # record the reason we stopped (if we have stopped)
                    if stop_pos <= len(val):
                        finish_reason = "stop"
                    
                    # emit the data if we are not potentially in the middle of a stop sequence
                    if not found_partial or finish_reason is not None:
                        out["choices"][i] = {
                            "text": val[:stop_pos],
                            "finish_reason": finish_reason,
                            "stop_text": stop_text,
                            "logprobs": {
                                "token_healing_prefix": self.last_token_str,
                                "top_logprobs": self.display_logprobs[i][self.sequence_pos[i]:]
                            }
                        }
                        self.str_pos[i] = len(self.generated_string[i])
                        put_data = True
                self.sequence_pos[i] = len(self.generated_sequence[i])
        
        if put_data:
            self.out_queue.put(out)

    def end(self):

        # make sure we have flushed all of the data
        for i in range(len(self.input_ids)):
            assert self.str_pos[i] >= len(self.generated_string[i]), "Not all data was flushed, this means generation stopped for an unknown reason!"
        
        self.out_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.out_queue.get(timeout=self.timeout)
        if value is None:
            raise StopIteration()
        else:
            return value