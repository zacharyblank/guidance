import os
import time
import collections
import regex
import pygtrie
import queue
import torch
from typing import Dict

import threading
import logging
from vllm import LLM as _vLLM, SamplingParams
from vllm.logits_processors import LogitsProcessor
from ._llm import LLM, LLMSession, SyncSession

class VLLM(LLM):
    """ A HuggingFace transformers language model with Guidance support.
    """

    cache = LLM._open_cache("_vllm.diskcache")

    def __init__(self, model=None, tokenizer=None, caching=True, token_healing=True, acceleration=True, \
                 temperature=0.0, device=None, **kwargs):
        super().__init__()

        # fill in default model value
        if model is None:
            model = os.environ.get("TRANSFORMERS_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser('~/.transformers_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        self.model_obj, self._tokenizer = self._model_and_tokenizer(model, tokenizer, **kwargs)
        self._generate_call = self.model_obj.generate

        self.model_name = model
        self.caching = caching
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.token_healing = token_healing
        self.acceleration = acceleration
        if device is not None: # set the device if requested
            self.model_obj = self.model_obj.to(device)
        self.device = "cuda" # TODO: Make this dynamic

        self._prefix_ids = [self._tokenizer.bos_token_id, 100] # token ids that we use to decode tokens after a prefix
        self._prefix_str = self._tokenizer.decode(self._prefix_ids, fragment=False)

        self._token_prefix_map = self._build_token_prefix_map(model)

    def prefix_matches(self, prefix):
        """ Return the list of tokens that match the given prefix.
        """
        return [v for arr in self._token_prefix_map.values(prefix=prefix) for v in arr]

    def new_string_builder(self, starting_ids=None):
        return VLLMStringBuilder(self._tokenizer, starting_ids)

    def encode(self, string, fragment=True, **kwargs):

        if fragment:
            string = self._prefix_str + string

        if "return_tensors" in kwargs:
            out = self._tokenizer(string, **kwargs)
        else:
            out = self._tokenizer.encode(string, **kwargs)
        
        # remove the start token when we are encoding a suffix
        if fragment:
            if out[1] == self._tokenizer.bos_token_id: # sometime the tokenizer adds an extra start token
                out = out[3:]
            else:
                out = out[2:]
        
        return out
    
    def id_to_token(self, id):
        return self._tokenizer.convert_ids_to_tokens([id])[0]
    
    def token_to_id(self, token):
        return self._tokenizer.convert_tokens_to_ids([token])[0]
    
    # def role_start(self, role):
    #     """ The starting role tag for chat models.

    #     #TODO Right now this just assumes the StableLM syntax, but this should be expanded later.
    #     """
    #     return "<|"+role.upper()+"|>"
    
    # def role_end(self, role=None):
    #     return ""

    def end_of_text(self):
        return self._tokenizer.eos_token

    @staticmethod
    def role_start(role):
        raise NotImplementedError("In order to use chat role tags you need to use a chat-specific subclass of Transformers for your LLM from guidance.transformers.*!")
    
    def decode(self, tokens, fragment=True, **kwargs):

        # if the last token is the end of string token, or the first is a start of string we remove it because it cause odd spacing decoding of fragments
        add_eos = ""
        add_bos = ""
        if fragment:
            if len(tokens) > 0 and tokens[-1] == self._tokenizer.eos_token_id:
                add_eos = self._tokenizer.eos_token
                tokens = tokens[:-1]
            if len(tokens) > 0 and tokens[0] == self._tokenizer.bos_token_id:
                add_bos = self._tokenizer.bos_token
                tokens = tokens[1:]
        
        # Decode the string corresponding to a single suffix token.
        # Note that we need to decode after the start token for sentence-piece tokenizers so that white space is preserved
        if fragment:
            return add_bos + self._tokenizer.decode(self._prefix_ids + list(tokens))[len(self._prefix_str):] + add_eos
        else:
            return add_bos + self._tokenizer.decode(tokens, **kwargs) + add_eos

    def _build_token_prefix_map(self, model_name):
        """ Build a map from token to index.
        """
        token_map = pygtrie.CharTrie()
        for i in range(self._tokenizer.vocab_size):
            s = self.decode([i])
            if s in token_map:
                token_map[s].append(i) # handle duplicate token encodings... (GPT2 BPE has this oddly enough)
            else:
                token_map[s] = [i]

        return token_map

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):

        # make sure transformers is installed
        try:
            import transformers
        except:
            raise Exception("Please install transformers with `pip install transformers` in order to use guidance.llms.Transformers!")

        # intantiate the model and tokenizer if needed
        if isinstance(model, str):
            if tokenizer is None:
                tokenizer = transformers.AutoTokenizer.from_pretrained(model, **kwargs)
            model = _vLLM(model=model, **kwargs)
        
        assert tokenizer is not None, "You must give a tokenizer object when you provide a model object (as opposed to just a model name)!"
            
        return model, tokenizer

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
    
    # def __enter__(self):

    #     # we only need decorators if we are using token acceleration
    #     if self.llm.acceleration:

    #         # decorate the prep step to preserve the initial past key values we have passed
    #         def prep_step_decorator(method):
    #             def decorate_prep_step(input_ids, **kwargs):

    #                 # if we are extending the input ids with the cached tokens then
    #                 # don't pass past key values to the input prep step, otherwise it
    #                 # would delete all but the last input_ids, and we have already removed
    #                 # the correct prefix from the input_ids (which is not always all but the last one)
    #                 if len(self._prefix_cache) > 0:
                        
    #                     kwargs["past"] = None
    #                     input_ids = input_ids[:,len(self._prefix_cache):]
    #                     # if "attention_mask" in kwargs:
    #                     #     kwargs["attention_mask"] = kwargs["attention_mask"][:,len(self._prefix_cache):]
    #                     model_kwargs = method(input_ids, **kwargs)

    #                     # provide the past key values for the actual model call
    #                     model_kwargs["past_key_values"] = self._past_key_values
    #                     if "position_ids" in model_kwargs: # models like OPT update the position ids internally
    #                         model_kwargs["position_ids"] = model_kwargs["position_ids"][:,len(self._prefix_cache):] # and update position ids

    #                     # we only need to do this first time, after that the past key values will
    #                     # be up until the last token, just like transformer models normally expect
    #                     # so we can clear our cache and let transformers cache like normal
    #                     self._prefix_cache = [] # this will get refilled once the generate call is done
                    
    #                     return model_kwargs
    #                 else:
    #                     return method(input_ids, **kwargs)
    #             decorate_prep_step.__func__ = method.__func__ # make us still look like a bound method
    #             return decorate_prep_step
    #         if getattr(self.llm.model_obj, "_orig_prepare_method", None) is None:
    #             self.llm.model_obj._orig_prepare_method = self.llm.model_obj.prepare_inputs_for_generation
    #         self.llm.model_obj.prepare_inputs_for_generation = prep_step_decorator(self.llm.model_obj._orig_prepare_method)

    #         # decorate the update step to save the past key values
    #         def update_step_decorator(method):
    #             def decorate_update_step(outputs, *args, **kwargs):

    #                 # save the past key values
    #                 self._past_key_values = getattr(outputs, "past_key_values", None)

    #                 return method(outputs, *args, **kwargs)
    #             return decorate_update_step
    #         if getattr(self.llm.model_obj, "_orig_update_method", None) is None:
    #             self.llm.model_obj._orig_update_method = self.llm.model_obj._update_model_kwargs_for_generation
    #         self.llm.model_obj._update_model_kwargs_for_generation = update_step_decorator(self.llm.model_obj._orig_update_method)

    #     return self

    # def __call__(self, *args, **kwargs):
    #     return self.__call__(*args, **kwargs)
    
    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None, top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=False, cache_seed=0, caching=None):
        """ Generate a completion of the given prompt.
        """
        
        # fill in defaults
        if not temperature:
            temperature = self.llm.temperature
        if token_healing is None:
            token_healing = self.llm.token_healing

        # generate the cache key
        key = self._cache_key(locals())

        # set the stop patterns
        if stop is not None:
            if isinstance(stop, str):
                stop_regex = [regex.escape(stop)]
            else:
                stop_regex = [regex.escape(s) for s in stop]
        if isinstance(stop_regex, str):
            stop_regex = [stop_regex]
        if stop_regex is None:
            stop_regex = []
        stop_regex.append(regex.escape(self.llm._tokenizer.eos_token)) # make sure the end of sequence token is always included

        # handle caching
        in_cache = key in self.llm.cache
        not_caching = (caching is not True and not self.llm.caching) or caching is False
        if not in_cache or not_caching:
            import transformers

            assert prompt != "", "You must provide a non-zero length prompt to the Transformers language model!"

            # encode the prompt
            encoded = self.llm.encode([prompt for _ in range(n)], return_tensors="pt", fragment=False)
            if self.llm.device is not None:
                encoded = encoded.to(self.llm.device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            model_config = self.llm.model_obj.llm_engine.model_config

            # ensure that we are extending a common sequence batch (our token healing assumes this right now)
            assert (input_ids[0,-1] == input_ids[:,-1]).all(), "The current token healing implementation assumes that batches are reps of the same sequence!"

            last_token_str = ""
            logprobs = 0
            processors = []
            stoppers = []

            # save what the prompt looks like when coded and then decoded (this captures added start tokens, etc.)
            coded_prompt = self.llm.decode(input_ids[0])

            # setup token healing
            print(f"Token Healing: {token_healing}")
            if True: #if token_healing:
                # pop off the last token since we will regen it
                last_token_id = input_ids[0][-1]
                last_token_str = self.llm._tokenizer.decode([last_token_id])
                healer = TokenHealingLogitsProcessor(self.llm, self.llm._tokenizer.vocab_size, last_token_str)
                if healer.should_bias:
                    input_ids = input_ids[:,:-1]
                    attention_mask = attention_mask[:,:-1]
                    max_tokens += 1 # add one for the token we regen for token healing
                    processors.append(healer)
                else:
                    last_token_str = ""

            # setup logit biasing
            if logit_bias is not None:
                processors.append(BiasLogitsProcessor(logit_bias))
                logprobs = len(logit_bias)
                print(f"num lgo probs: {logprobs}")

            # make sure we don't run off the end of the model
            # TODO: Calculate

            # find how much of the prompt is cached
            prefix_match_len = 0
            for token in input_ids[0]:
                if prefix_match_len >= len(self._prefix_cache) or token != self._prefix_cache[prefix_match_len]:
                    break
                else:
                    prefix_match_len += 1

            # we always need to run the model on at least one token so transformers is happy
            if prefix_match_len == len(input_ids[0]):
                prefix_match_len -= 1

            # trim the cache to what we can use
            if prefix_match_len < len(self._prefix_cache): # prefix_match_len > 0 and 
                self._past_key_values = tuple((key[:,:,:prefix_match_len,:],value[:,:,:prefix_match_len,:]) for key,value in self._past_key_values) # TODO: this is specific to the GPT2 tensor layout
                self._prefix_cache = self._prefix_cache[:prefix_match_len]

            # add support for pattern guidance
            if pattern is not None:
                # processors.append(RegexLogitsProcessor(pattern, stop_regex, self.llm.decode, model_config.vocab_size, temperature == 0, len(coded_prompt), self.llm._tokenizer.eos_token_id))
                processors.append(RegexLogitsProcessor(pattern, stop_regex, self.llm, self.llm._tokenizer.vocab_size, temperature == 0, len(coded_prompt), self.llm._tokenizer.eos_token_id))

            if stop_regex is not None:
                stoppers.append(RegexStoppingCriteria(stop_regex, self.llm.decode, len(coded_prompt)))

            sampling_params = SamplingParams(
                n=n,
                # presence_penalty=request.presence_penalty,
                # frequency_penalty=request.frequency_penalty,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                max_tokens=max_tokens,
                logprobs=1,
                # best_of=request.best_of,
                # top_k=request.top_k,
                # ignore_eos=request.ignore_eos,
                # use_beam_search=request.use_beam_search,
                logits_processors=processors
            )

            print(sampling_params)

            # # the args for the transformers generate call
            # generate_args = dict(
            #     inputs=input_ids,
            #     attention_mask=attention_mask,
            #     # position_ids=position_ids,
            #     temperature=temperature,
            #     max_new_tokens=max_tokens,
            #     top_p=top_p,
            #     pad_token_id=model_config.pad_token_id if model_config.pad_token_id is not None else self.llm._tokenizer.eos_token_id,
            #     logits_processor=transformers.LogitsProcessorList(processors),
            #     stopping_criteria=transformers.StoppingCriteriaList(stoppers),
            #     # past_key_values=self._past_key_values,
            #     output_scores=logprobs is not None and logprobs > 0,
            #     return_dict_in_generate=True
            # )

            # # override the model config for do_sample when the temperature requires it
            # do_sample = getattr(self.llm.model_obj.config, "do_sample", None)
            # if do_sample is True and temperature == 0:
            #     generate_args["do_sample"] = False
            # elif do_sample is False and temperature > 0:
            #     generate_args["do_sample"] = True

            outputs = self.llm.model_obj.generate(prompt, sampling_params)
            

            response = []
            for output in outputs[0].outputs:

                for idx, logprob in enumerate(output.logprobs):
                    token_id = next(iter(logprob.keys()))
                    token = self.llm._tokenizer.convert_ids_to_tokens(token_id)
                    output.logprobs[idx][token] = logprob.pop(token_id)

                response.append({
                    "text": output.text,
                    "logprobs": {
                        "token_healing_prefix": last_token_str,
                        "top_logprobs": output.logprobs
                    }
                })
            
            self.llm.cache[key] = {"choices": response}

        return self.llm.cache[key]
    
    def _update_prefix_cache(self, streamer):
        # note what we now have cached and ready for our next call in this session
        if self._past_key_values and len(streamer.generated_sequence) == 1:
            self._prefix_cache = streamer.generated_sequence[0][:self._past_key_values[0][0].shape[-2]] # self._past_key_values is already saved, this just aligns with it

    def _stream_then_save(self, streamer, key, thread):
        list_out = []
        for out in streamer:
            list_out.append(out)
            yield out
        thread.join() # clean up the thread
        self.llm.cache[key] = list_out
        self._update_prefix_cache(streamer)
        self._last_computed_key = key

    def __exit__(self, exc_type, exc_value, traceback):
        """ Restore the model to its original state by removing monkey patches.
        """
        if getattr(self.llm.model_obj, "_orig_prepare_method", None) is not None:
            self.llm.model_obj.prepare_inputs_for_generation = self.llm.model_obj._orig_prepare_method
            del self.llm.model_obj._orig_prepare_method
        if getattr(self.llm.model_obj, "_orig_update_method", None) is not None:
            self.llm.model_obj._update_model_kwargs_for_generation = self.llm.model_obj._orig_update_method
            del self.llm.model_obj._orig_update_method
        return False



class TokenHealingLogitsProcessor(LogitsProcessor):
    """ Token healing.

    When we tokenize the prompt the last token we get is not the last token we would
    have gotten if the prompt + generation was concatented and then tokenized. This
    is not good because it does not align with the pretraining of the model, so
    we "heal" this boundary by backing up one token and then forcing the first token
    generated to start with the prefix of the last token in the prompt. This could
    result in the same token as the end of the prompt, or another longer one.
    """

    def __init__(self, model, vocab_size, last_token_str, bias_value=50.):
        """ Build a new TokenHealingLogitsProcessor.

        Note that bias value is in score space (log-odds normally) and should be
        enough to ensure those tokens are the only ones used. But not so high
        as to destroy numerical precision.
        """
        import torch

        try:
            allowed_first_tokens = model.prefix_matches(last_token_str)
            assert len(allowed_first_tokens) > 0, "Error in token healing map! No match found for: `"+last_token_str+"`"
        except KeyError:
            # this must be a special token outside the vocab, so we assume it does not have any valid extensions
            allowed_first_tokens = []
        
        # if we have multiple possible completions past the last token, then biasing is needed
        if len(allowed_first_tokens) > 1:
            self.first_token_mask = torch.zeros(50272) # TODO: Fix this hard coded value
            self.first_token_mask.scatter_(0, torch.tensor(allowed_first_tokens), bias_value)
            if model.device is not None:
                self.first_token_mask = self.first_token_mask.to(model.device)
            self.should_bias = True
        
        # otherwise we have nothing to do (the last token is already unique)
        else:
            self.should_bias = False

    def __call__(self, logits, input_ids):

        # we only bias the first token generated
        if not self.should_bias:
            return logits
        self.should_bias = False
        
        # make only allowed tokens possible
        return logits + self.first_token_mask
    
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

        return logits

class RegexStoppingCriteria():
    def __init__(self, stop_pattern, decode, prefix_length):
        if isinstance(stop_pattern, str):
            self.stop_patterns = [regex.compile(stop_pattern)]
        else:
            self.stop_patterns = [regex.compile(pattern) for pattern in stop_pattern]
        self.prefix_length = prefix_length
        self.decode = decode
        self.current_strings = None
        self.current_length = 0

    def __call__(self, input_ids, scores, **kwargs):

        # extend our current strings
        if self.current_strings is None:
            self.current_strings = ["" for _ in range(len(input_ids))]
        for i in range(len(self.current_strings)):
            self.current_strings[i] += self.decode(input_ids[i][self.current_length:])
        
        # trim off the prefix string so we don't look for stop matches in the prompt
        if self.current_length == 0:
            for i in range(len(self.current_strings)):
                self.current_strings[i] = self.current_strings[i][self.prefix_length:]
        
        self.current_length = len(input_ids[0])
        
        # check if all of the strings match a stop string (and hence we can stop the batch inference)
        all_done = True
        for i in range(len(self.current_strings)):
            found = False
            for s in self.stop_patterns:
                if s.search(self.current_strings[i]):
                    found = True
            if not found:
                all_done = False
                break
        
        return all_done

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
        return (str(self) + new_str)

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