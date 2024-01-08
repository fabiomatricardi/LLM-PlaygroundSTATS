"""
Download the model google/flan-t5-base
-------------------------------------
https://huggingface.co/Open-Orca/oo-phi-1_5
Hugging Face repo: Open-Orca/oo-phi-1_5

Unreleased, untested, unfinished beta.

We've trained Microsoft Research's phi-1.5, 1.3B parameter model with the same OpenOrca dataset 
as we used with our OpenOrcaxOpenChat-Preview2-13B model.

This model doesn't dramatically improve on the base model's general task performance, 
but the instruction tuning has made the model reliably handle the ChatML prompt format.
2.8 Gb HD

pip install einops

"""
import gradio as gr
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, GenerationConfig
from transformers import pipeline
import torch
import datetime
from threading import Thread
import psutil # to get the SYSTEM MONITOR CPU/RAM stats
import pandas as pd # to visualize the SYSTEM MONITOR CPU/RAM stats

#MODEL SETTINGS also for DISPLAY
initial_RAM = psutil.virtual_memory()[2]
initial_CPU = psutil.cpu_percent() 
print(f"initial memory usage {initial_RAM}")
print(f"initial CPU usage {initial_CPU}")
import plotly.express as px
plot_end = 1
data = pd.DataFrame.from_dict({"x": [0], "y": [initial_RAM],"y1":[initial_CPU]}) 




liked = 2
convHistory = ''
convHistory = ''
mrepo = 'Open-Orca/oo-phi-1_5'
modelfile = "OpenOraca-Phi1_5"
modeltitle = "OpenOrca-Phi-1.5"
modelparameters = '1.3 B'
model_is_sys = True
modelicon = '🐋'
imagefile = 'https://cdn-lfs.huggingface.co/repos/e6/e0/e6e08b2cd954361f60d9e5774df5d1aa3a7f9249499a93e87271dfec47d24386/1bad47383dd7983065d7674007aac5334f278ae7741d58d48511c16294431273?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27OpenOrcaLogo.png%3B+filename%3D%22OpenOrcaLogo.png%22%3B&response-content-type=image%2Fpng&Expires=1704963258&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwNDk2MzI1OH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy9lNi9lMC9lNmUwOGIyY2Q5NTQzNjFmNjBkOWU1Nzc0ZGY1ZDFhYTNhN2Y5MjQ5NDk5YTkzZTg3MjcxZGZlYzQ3ZDI0Mzg2LzFiYWQ0NzM4M2RkNzk4MzA2NWQ3Njc0MDA3YWFjNTMzNGYyNzhhZTc3NDFkNThkNDg1MTFjMTYyOTQ0MzEyNzM%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=Alb7kfjJqwQUL7SzGd-YAXoPHTYavxTN4dD8VnVjzvX2OWn3ghylNwKcRMpW-7tZGhXoTxssGjbeQeZ6mdrKZg9Fjai95G9apiQApzitjYfAZutTOvAzWFJQVd3afsp3rCLaMO4HU7fgfCkOIvnu4sjGwwxexXGiJs63sthZIKHSYqtgBCokY-TP%7EL5faP1-Dwv0dhkFzKqJNAe4Ip%7EWJdC09i2MPP9avzgohD%7E-DpY1CZdB0LlmCDHrUwhsUblWlYzpv6oeSd8gVZdIAHxf3GSy0IQqTbhil-aWUHHTPVrvNAzDr0MtMVJnHnjwEVO4MO5vjVjBK334RTZ0piVThg__&Key-Pair-Id=KVTP0A1DKRTAX'
repetitionpenalty = 1.2
contextlength=2048
stoptoken = '<|endoftext|>'
logfile = f'{modeltitle}_logs.txt'
print(f"loading model {modelfile}...")
stt = datetime.datetime.now()
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
oophi = './openorcaPhi1_5/'
tokenizer = AutoTokenizer.from_pretrained(oophi,trust_remote_code=True,)
llm = AutoModelForCausalLM.from_pretrained(oophi,
                                             trust_remote_code=True,
                                             device_map='cpu',
                                             torch_dtype=torch.float32)
print(tokenizer.eos_token_id)
print(tokenizer.bos_token_id)

dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")

########## FUnCTIOn TO WRITe lOGFIle ######################
def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

######## FUNCTION FOR PLOTTING CPU RAM % ################

def get_plot(period=1):
    global plot_end
    global data
    w = 300
    h = 150
    # NEW DATA FOR THE DATAFRAME
    x = plot_end
    y = psutil.virtual_memory()[2]
    y1 = psutil.cpu_percent()
    new_record = pd.DataFrame([{'x':x, 'y':y, 'y1':y1}])
    data = pd.concat([data, new_record], ignore_index=True)
    # TO HIDE ALL PLOTLY OPTION BAR
    modebars = ["autoScale2d", "autoscale", "editInChartStudio", "editinchartstudio", "hoverCompareCartesian", "hovercompare", "lasso", "lasso2d", "orbitRotation", "orbitrotation", "pan", "pan2d", "pan3d", "reset", "resetCameraDefault3d", "resetCameraLastSave3d", "resetGeo", "resetSankeyGroup", "resetScale2d", "resetViewMapbox", "resetViews", "resetcameradefault", "resetcameralastsave", "resetsankeygroup", "resetscale", "resetview", "resetviews", "select", "select2d", "sendDataToCloud", "senddatatocloud", "tableRotation", "tablerotation", "toImage", "toggleHover", "toggleSpikelines", "togglehover", "togglespikelines", "toimage", "zoom", "zoom2d", "zoom3d", "zoomIn2d", "zoomInGeo", "zoomInMapbox", "zoomOut2d", "zoomOutGeo", "zoomOutMapbox", "zoomin", "zoomout"]
    # RAM LINE CHART
    fig = px.area(data, x="x", y='y',height=h,line_shape='spline',range_y=[0,100]) #, width=300
    fig.update_traces(line_color='#6495ed', line_width=2)
    fig.update_layout(annotations=[], overwrite=True)
    fig.update_xaxes(visible=False) #, fixedrange=False
    fig.add_annotation(text=f"<b>{y} %</b>",
                  xref="paper", yref="paper",
                  x=0.3, y=0.12, showarrow=False,
                  font=dict(
                        family="Balto, sans-serif",
                        size=30,
                        color="#ffe02e"  #
                        ),
                  align="center",)
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(t=1,l=1,b=1,r=1),
        modebar_remove=modebars
    )
    # CPU LINE CHART
    fig2 = px.area(data, x="x", y='y1',line_shape='spline',height=h,range_y=[0,100])  #, width=300 #line_shape='spline'
    fig2.update_traces(line_color='#ff5757', line_width=2)
    fig2.update_layout(annotations=[], overwrite=True)
    fig2.update_xaxes(visible=False) #, fixedrange=True
    #fig.update_yaxes(visible=False, fixedrange=True)
    # strip down the rest of the plot
    fig2.add_annotation(text=f"<b>{y1} %</b>",
                  xref="paper", yref="paper",
                  x=0.3, y=0.12, showarrow=False,
                  font=dict(
                        family="Balto, sans-serif",
                        size=30,
                        color="#ad9300"  ##ad9300
                        ),
                  align="center",)
    fig2.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        modebar_remove=modebars
    )
    plot_end += 1 
    return fig, fig2



"""
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant

f"<|im_start|>system\n{a}<|im_end|>\n<|im_start|>user\n{b}<|im_end|>\n<|im_start|>assistant"
"""
def combine(a, b, c, d,e,f):
    global convHistory
    import datetime
    temperature = c
    max_new_tokens = d
    repeat_penalty = f
    top_p = e
    prefix = "<|im_start|>"
    suffix = "<|im_end|>\n"
    sys_format = prefix + "system\n" + a + suffix
    user_format = prefix + "user\n" + b + suffix
    assistant_format = prefix + "assistant\n"
    prompt = sys_format + user_format + assistant_format    
    #prompt = f"Q: {b}\nA:"
    start = datetime.datetime.now()
    generation = ""
    delta = ""
    prompt_tokens = f"Prompt Tokens: {len(tokenizer.tokenize(prompt))}"
    ptt = len(tokenizer.tokenize(prompt))
    generated_text = ""
    answer_tokens = ''
    total_tokens = ''   
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
    thread.start()
    #generated_text = ""
    for new_text in streamer:
        generation += new_text

        answer_tokens = f"Out Tkns: {len(tokenizer.tokenize(generation))}"
        total_tokens = f"Total Tkns: {ptt + len(tokenizer.tokenize(generation))}"
        delta = datetime.datetime.now() - start
        seconds = delta.total_seconds()
        speed = (ptt + len(tokenizer.tokenize(generation)))/seconds
        textspeed = f"Gen.Speed: {speed} t/s"        
        yield generation, delta, prompt_tokens, answer_tokens, total_tokens, textspeed
    timestamp = datetime.datetime.now()
    textspeed = f"Gen.Speed: {speed} t/s"
    logger = f"""time: {timestamp}\n Temp: {temperature} - MaxNewTokens: {max_new_tokens} - RepPenalty: {repeat_penalty}  Top_P: {top_p}  \nPROMPT: \n{prompt}\n{modeltitle}_{modelparameters}: {generation}\nGenerated in {delta}\nPromptTokens: {prompt_tokens}   Output Tokens: {answer_tokens}  Total Tokens: {total_tokens}  Speed: {speed}\n---"""
    writehistory(logger)
    convHistory = convHistory + prompt + "\n" + generation + "\n"
    print(convHistory)
    return generation, delta, prompt_tokens, answer_tokens, total_tokens, textspeed   
    #return generation, delta


# MAIN GRADIO INTERFACE
with gr.Blocks(theme='Medguy/base2') as demo:   #theme=gr.themes.Glass()  #theme='remilia/Ghostly'
    #TITLE SECTION
    with gr.Row(variant='compact'):
            with gr.Column(scale=3):            
                gr.Image(value=imagefile, 
                        show_label = False, width = 160,
                        show_download_button = False, container = False,)     #height = 300         
            with gr.Column(scale=10):
                gr.HTML("<center>"
                + "<h3>Prompt Engineering Playground!</h3>"
                + f"<h1>{modelicon} {modeltitle} - {modelparameters} parameters - {contextlength} context window</h1></center>")  
                with gr.Row():
                        with gr.Column(min_width=80):
                            gentime = gr.Textbox(value="", placeholder="Generation Time:", min_width=50, show_label=False)                          
                        with gr.Column(min_width=80):
                            prompttokens = gr.Textbox(value="", placeholder="Prompt Tkn:", min_width=50, show_label=False)
                        with gr.Column(min_width=80):
                            outputokens = gr.Textbox(value="", placeholder="Output Tkn:", min_width=50, show_label=False)            
                        with gr.Column(min_width=80):
                            totaltokens = gr.Textbox(value="", placeholder="Total Tokens:", min_width=50, show_label=False)   
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            - **Prompt Template**: None
            - **Context Lenght**: {contextlength} tokens
            - **LLM Engine**: Transformers Pytorch
            - **Model**: {modelicon} {modeltitle}
            - **Log File**: {logfile}
            """)
        with gr.Column(scale=2):
            plot = gr.Plot(label="RAM usage")
        with gr.Column(scale=2):
            plot2 = gr.Plot(label="CPU usage")


    # INTERACTIVE INFOGRAPHIC SECTION
    

    # PLAYGROUND INTERFACE SECTION
    with gr.Row():
        with gr.Column(scale=1):
            #gr.Markdown(
            #f"""### Tunning Parameters""")
            temp = gr.Slider(label="Temperature",minimum=0.0, maximum=1.0, step=0.01, value=0.1)
            top_p = gr.Slider(label="Top_P",minimum=0.0, maximum=1.0, step=0.01, value=0.8, visible=False)
            repPen = gr.Slider(label="Repetition Penalty",minimum=0.0, maximum=4.0, step=0.01, value=1.2)
            max_len = gr.Slider(label="Maximum output lenght", minimum=10,maximum=(contextlength-150),step=2, value=512)          

            txt_Messagestat = gr.Textbox(value="", placeholder="SYS STATUS:", lines = 1, interactive=False, show_label=False)              
            txt_likedStatus = gr.Textbox(value="", placeholder="Liked status: none", lines = 1, interactive=False, show_label=False)
            txt_speed = gr.Textbox(value="", placeholder="Gen.Speed: none", lines = 1, interactive=False, show_label=False) 
            clear_btn = gr.Button(value=f"🗑️ Clear Input", variant='primary')
            #CPU_usage = gr.Textbox(value="", placeholder="RAM:", lines = 1, interactive=False, show_label=False)
            #plot = gr.Plot(show_label=False)
            #plot2 = gr.Plot(show_label=False)

        with gr.Column(scale=4):
            txt = gr.Textbox(label="System Prompt", lines=1, interactive = model_is_sys, value = 'You are an advanced and helpful AI assistant.', visible=False)
            txt_2 = gr.Textbox(label="User Prompt", lines=5, show_copy_button=True)
            with gr.Row():
                btn = gr.Button(value=f"{modelicon} Generate", variant='primary', scale=3)
                btnlike = gr.Button(value=f"👍 GOOD", variant='secondary', scale=1)
                btndislike = gr.Button(value=f"🤮 BAD", variant='secondary', scale=1)
                submitnotes = gr.Button(value=f"💾 SAVE NOTES", variant='secondary', scale=2)
            txt_3 = gr.Textbox(value="", label="Output", lines = 8, show_copy_button=True)
            txt_notes = gr.Textbox(value="", label="Generation Notes", lines = 2, show_copy_button=True)
                
            def likeGen():
                #set like/dislike and clear the previous Notes
                global liked
                liked = f"👍 GOOD"
                resetnotes = ""
                return liked
            def dislikeGen():
                #set like/dislike and clear the previous Notes
                global liked
                liked = f"🤮 BAD"
                resetnotes = ""
                return liked
            def savenotes(vote,text):
                logging = f"### NOTES AND COMMENTS TO GENERATION\nGeneration Quality: {vote}\nGeneration notes: {text}\n---\n\n"
                writehistory(logging)
                message = "Notes Successfully saved"
                print(logging)
                print(message)
                return message
            def clearInput(): #Clear the Input TextArea
                message = ""
                resetnotes = ""
                reset_output = ""
                return message, resetnotes, reset_output

            btn.click(combine, inputs=[txt, txt_2,temp,max_len,top_p,repPen], outputs=[txt_3,gentime,prompttokens,outputokens,totaltokens,txt_speed])
            btnlike.click(likeGen, inputs=[], outputs=[txt_likedStatus])
            btndislike.click(dislikeGen, inputs=[], outputs=[txt_likedStatus])
            submitnotes.click(savenotes, inputs=[txt_likedStatus,txt_notes], outputs=[txt_Messagestat])
            clear_btn.click(clearInput, inputs=[], outputs=[txt_2,txt_notes,txt_3])
            dep = demo.load(get_plot, None, [plot,plot2], every=2)


if __name__ == "__main__":
    demo.launch(inbrowser=True)

#psutil.cpu_percent()