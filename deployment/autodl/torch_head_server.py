#!/usr/bin/env python3
"""Author-documented PyTorch/PEFT decision head, with bounded local batching.

Implements bare-v1 Choice (2..16 options) only. No chat generation or images.
See https://huggingface.co/autotrust/JEV-27B#plain-transformers--peft
"""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import queue
import threading
import time

MODEL='autotrust/JEV-27B'
REVISION='14b425269ed24452e0a5ad5a8f3e51d139647a88'
FOLDER=Path('/root/autodl-tmp/JEV-27B')
MAX_BATCH=8
MAX_LENGTH=8192
TASKS=queue.Queue(maxsize=64)


def prompt(body):
    if body.get('kind')!='choice' or body.get('thinking','off')!='off':raise ValueError('Only System 1 Choice is enabled')
    options=body.get('options')
    if not isinstance(options,list) or not 2<=len(options)<=16 or len(set(options))!=len(options):
        raise ValueError('Two to sixteen unique options required')
    if any(not isinstance(v,str) or '\n' in v for v in options):raise ValueError('Single-line string options required')
    if not isinstance(body.get('state'),str) or not isinstance(body.get('question'),str):raise ValueError('Text-only state and question required')
    lines=[f'{chr(65+i)}) {v}' for i,v in enumerate(options)]
    return '[kind] choice\n[state] '+body['state']+'\n[question] '+body['question']+'\n[options]\n'+'\n'.join(lines)+'\n[decision]:'


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(FOLDER,dtype=torch.bfloat16,device_map='cuda',local_files_only=True)
    model=PeftModel.from_pretrained(base,FOLDER/'adapter',local_files_only=True).merge_and_unload().eval()
    head=load_file(str(FOLDER/'head.safetensors'));W=head['proj.weight'].float().cuda();b=head['proj.bias'].float().cuda()
    cfg=json.loads((FOLDER/'judge_config.json').read_text());temp=json.loads((FOLDER/'calibration.json').read_text())['per_kind']
    lo,hi=cfg['slots']['ranges']['choice']
    if hi-lo!=16 or W.shape!=(24,5120) or cfg.get('softcap') is not None:raise ValueError('Unexpected published head configuration')
    info=dict(model=MODEL,protocol='jev27-bare-v1',huggingface_revision=REVISION,
        implementation='pytorch-peft-direct-head-v1',max_options=16,max_batch=MAX_BATCH,max_model_len=MAX_LENGTH,
        temperatures=temp,dtype='bfloat16 backbone, float32 decision head',padding='left, masked cumulative positions',
        adapter='adapter (in-memory merge)',source='published plain transformers + peft recipe')

    def infer():
        while True:
            tasks=[TASKS.get()];until=time.monotonic()+.010
            while len(tasks)<MAX_BATCH:
                try:tasks.append(TASKS.get(timeout=max(0,until-time.monotonic())))
                except queue.Empty:break
            try:
                enc=tokenizer([t['prompt'] for t in tasks],padding=True,return_tensors='pt',add_special_tokens=False)
                if enc['input_ids'].shape[1]>MAX_LENGTH:raise ValueError('Prompt exceeds context limit; no truncation')
                enc={k:v.cuda() for k,v in enc.items()}
                positions=(enc['attention_mask'].cumsum(-1)-1).clamp(min=0)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden=model.model(**enc,position_ids=positions,use_cache=False).last_hidden_state[:,-1].float()
                logits=(hidden@W.T+b)/temp['choice']
                for t,z in zip(tasks,logits):
                    options=t['body']['options'];p=torch.softmax(z[lo:lo+len(options)],0).tolist();winner=max(range(len(p)),key=p.__getitem__)
                    t['result']=dict(model=MODEL,protocol='jev27-bare-v1',kind='choice',effective_kind='choice',adaptation='native',
                        options=options,probabilities=p,choice_index=winner,choice=options[winner],num_model_requests=1,
                        elapsed_seconds=time.monotonic()-t['start'],implementation=info['implementation'],image_used=False,
                        huggingface_revision=REVISION)
            except Exception as e:
                for t in tasks:t['error']=type(e).__name__+': '+str(e)[:500]
                print('BATCH_ERROR',type(e).__name__,str(e)[:500],flush=True)
            finally:
                for t in tasks:t['done'].set();TASKS.task_done()

    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,value):
            body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            self.reply(200,info) if self.path in ('/health','/v1/decide/info') else self.reply(404,{'error':'Not found'})
        def do_POST(self):
            if self.path!='/v1/decide':return self.reply(404,{'error':'Not found'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=100000:raise ValueError('Request size out of bounds')
                body=json.loads(self.rfile.read(length));text=prompt(body)
                task=dict(body=body,prompt=text,done=threading.Event(),start=time.monotonic())
                TASKS.put_nowait(task)
            except queue.Full:return self.reply(503,{'error':'Inference queue full'})
            except (ValueError,TypeError,KeyError) as e:return self.reply(400,{'error':str(e)})
            if not task['done'].wait(180):return self.reply(504,{'error':'Inference timeout'})
            self.reply(500,{'error':task['error']}) if 'error' in task else self.reply(200,task['result'])
        def log_message(self,*args):pass
    threading.Thread(target=infer,daemon=True).start()
    print('JEV_TORCH_HEAD_READY',json.dumps(info),flush=True)
    ThreadingHTTPServer(('127.0.0.1',8000),Handler).serve_forever()


if __name__=='__main__':main()
