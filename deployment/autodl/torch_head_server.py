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
MAX_BATCH=1
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
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    tokenizer=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(FOLDER,dtype=torch.float16,device_map='cuda',local_files_only=True)
    model=PeftModel.from_pretrained(base,FOLDER/'adapter',local_files_only=True).merge_and_unload().eval()
    # Keep residual additions in FP32 across the deep decoder. Linear/attention
    # operations use FP16 autocast for more mantissa precision. The publisher
    # checkpoint is unchanged; this is a separately recorded inference dtype.
    model.model.embed_tokens.register_forward_hook(lambda module,args,output:output.float())
    head=load_file(str(FOLDER/'head.safetensors'));W=head['proj.weight'].float().cuda();b=head['proj.bias'].float().cuda()
    cfg=json.loads((FOLDER/'judge_config.json').read_text());temp=json.loads((FOLDER/'calibration.json').read_text())['per_kind']
    lo,hi=cfg['slots']['ranges']['choice']
    if hi-lo!=16 or W.shape!=(24,5120) or cfg.get('softcap') is not None:raise ValueError('Unexpected published head configuration')
    info=dict(model=MODEL,protocol='jev27-bare-v1',huggingface_revision=REVISION,
        implementation='pytorch-peft-group-v3',max_options=16,max_questions=7,max_model_len=MAX_LENGTH,
        temperatures=temp,dtype='float16 weights/matmuls; float32 residual stream and decision head',padding='right suffix padding; read last valid token',
        adapter='adapter (in-memory merge)',source='published plain transformers + peft recipe',
        prefix_reuse='within one seven-goal slate only',prefix_alignment_tokens=64,fp32_residuals=True,
        bf16_reduced_precision_reduction=False,fp16_reduced_precision_reduction=False)

    def hidden_for(task):
        if not task['group']:
            ids=tokenizer(task['prompts'][0],return_tensors='pt',add_special_tokens=False)['input_ids'].cuda()
            if ids.shape[1]>MAX_LENGTH:raise ValueError('Prompt exceeds context limit; no truncation')
            h=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state[:,-1].float()
            return h,1
        sequences=tokenizer(task['prompts'],add_special_tokens=False)['input_ids']
        if max(map(len,sequences))>MAX_LENGTH:raise ValueError('Prompt exceeds context limit; no truncation')
        cut=0
        for values in zip(*sequences):
            if len(set(values))!=1:break
            cut+=1
        cut=min(cut,min(map(len,sequences))-1)
        # FLA gated-delta chunking uses 64-token blocks. Preserve those block
        # boundaries across the prefix/suffix split to reduce arithmetic drift.
        if cut>=64:cut=(cut//64)*64
        if cut<=0:raise ValueError('Missing shared token prefix')
        prefix_ids=torch.tensor([sequences[0][:cut]],device='cuda')
        prefix=model.model(input_ids=prefix_ids,attention_mask=torch.ones_like(prefix_ids),use_cache=True)
        branch=prefix.past_key_values
        if branch is None:raise ValueError('Backbone did not return reusable cache')
        # LinearAttentionLayer exposes beam reordering, but not the generic
        # repeat method. Selecting row zero m times copies both recurrent and
        # convolution states as well as attention K/V, without shared writes.
        m=len(sequences);branch.reorder_cache(torch.zeros(m,dtype=torch.long,device='cuda'))
        lengths=[len(s)-cut for s in sequences];width=max(lengths)
        suffix=torch.full((m,width),tokenizer.pad_token_id,dtype=torch.long,device='cuda')
        mask=torch.zeros_like(suffix)
        for i,s in enumerate(sequences):
            suffix[i,:lengths[i]]=torch.tensor(s[cut:],device='cuda');mask[i,:lengths[i]]=1
        attention=torch.cat((torch.ones((m,cut),dtype=mask.dtype,device='cuda'),mask),1)
        h=model.model(input_ids=suffix,attention_mask=attention,
            position_ids=torch.arange(cut,cut+width,device='cuda')[None,:].expand(m,-1),past_key_values=branch,
            use_cache=True).last_hidden_state
        # Future right-padding cannot affect a valid causal position. Discard
        # every suffix cache; no goal's continuation is reused by another goal.
        return h[torch.arange(m,device='cuda'),torch.tensor(lengths,device='cuda')-1].float(),2

    def infer():
        while True:
            tasks=[TASKS.get()];until=time.monotonic()+.010
            while len(tasks)<MAX_BATCH:
                try:tasks.append(TASKS.get(timeout=max(0,until-time.monotonic())))
                except queue.Empty:break
            try:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    hidden,passes=hidden_for(tasks[0])
                logits=(hidden@W.T+b)/temp['choice']
                if not torch.isfinite(logits).all():raise ValueError('Non-finite decision logits; refusing score production')
                t=tasks[0];results=[]
                for j,z in enumerate(logits):
                    options=t['body']['options'];p=torch.softmax(z[lo:lo+len(options)],0).tolist();winner=max(range(len(p)),key=p.__getitem__)
                    result=dict(model=MODEL,protocol='jev27-bare-v1',kind='choice',effective_kind='choice',adaptation='native',
                        options=options,probabilities=p,choice_index=winner,choice=options[winner],
                        elapsed_seconds=time.monotonic()-t['start'],implementation=info['implementation'],image_used=False,
                        huggingface_revision=REVISION)
                    if t['group']:result.update(question_index=j,group_shared_forwards=passes)
                    else:result['num_model_requests']=passes
                    results.append(result)
                t['result']=(dict(model=MODEL,protocol='jev27-bare-v1',implementation=info['implementation'],
                    results=results,model_forward_calls=passes,question_count=len(results),
                    elapsed_seconds=time.monotonic()-t['start']) if t['group'] else results[0])
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
            if self.path not in ('/v1/decide','/v1/decide_many'):return self.reply(404,{'error':'Not found'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=100000:raise ValueError('Request size out of bounds')
                body=json.loads(self.rfile.read(length));group=self.path.endswith('_many')
                questions=body.get('questions') if group else [body.get('question')]
                if not isinstance(questions,list) or not 1<=len(questions)<=7 or any(not isinstance(q,str) for q in questions):
                    raise ValueError('One to seven text questions required')
                texts=[prompt(dict(body,question=q)) for q in questions]
                task=dict(body=body,prompts=texts,group=group,done=threading.Event(),start=time.monotonic())
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
