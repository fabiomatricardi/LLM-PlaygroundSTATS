# LLM-PlaygroundSTATS
A Gradio Playground with LLMs (llama.cpp or Transformers) with CPU/RAM usage statistics

## General Information
This repo hosts python file with GRADIO UI
the tested LLMs are
- Flan-T5-base - pytorch/Transformers
- Dolphi2.6-Phi2 GGUF - llama.cpp
- Phi1.5 GGUF - llama.cpp

## UI enhanchements
- like/dislike buttons to evaluate the output
- A comment section to explain the results of the tuning paraemters and issues on the prompt
- Temperature, Repetition Penalty and Max generation lenght sliders
- CLEAR field Button
- CPU statistic plot
- RAM statistic plot

---

screeshots examples
<img src="https://github.com/fabiomatricardi/LLM-PlaygroundSTATS/raw/main/Dolphin2.6-Phi2_PlayGround.png" width=900>


### Test with OpenOrca Phi 1.5
Tested run only on CPU with Transformers library
- no mask attenction
- trust remote code
#### original repo
https://huggingface.co/Open-Orca/oo-phi-1_5


```python

oophi = './openorcaPhi1_5/'
tokenizer = AutoTokenizer.from_pretrained(oophi,trust_remote_code=True,)
llm = AutoModelForCausalLM.from_pretrained(oophi,
                                             trust_remote_code=True,
                                             device_map='cpu',
                                             torch_dtype=torch.float32)


    prefix = "<|im_start|>"
    suffix = "<|im_end|>\n"
    sys_format = prefix + "system\n" + a + suffix
    user_format = prefix + "user\n" + b + suffix
    assistant_format = prefix + "assistant\n"
    prompt = sys_format + user_format + assistant_format


    inputs = tokenizer([prompt], return_tensors="pt", return_attention_mask=False)
    streamer = TextIteratorStreamer(tokenizer)

    generation_kwargs = dict(inputs, streamer=streamer, max_length = max_new_tokens, 
                        temperature=temperature,
                        #top_p=top_p,
                        repetition_penalty = repeat_penalty,
                        eos_token_id=tokenizer.eos_token_id, 
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=True,
                        use_cache=True,) #pad_token_id=tokenizer.eos_token_id
    thread = Thread(target=llm.generate, kwargs=generation_kwargs)

```


screeshots examples
<img src="https://github.com/fabiomatricardi/LLM-PlaygroundSTATS/raw/main/Dolphin2.6-Phi2_PlayGround.png" width=900>


#### Supporting links
- ICON from https://github.com/Lightning-AI/lit-llama
- PLOTLY tutorial https://plotly.com/python/text-and-annotations/
- COLOR codes from https://html-color.codes/gold/chart
