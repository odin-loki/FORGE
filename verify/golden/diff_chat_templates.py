#!/usr/bin/env python3
"""Fetch each vendor tokenizer_config.json chat_template from huggingface.co (public repos or ungated mirrors, see REPOS) and
compare with the T[...] texts in tests/data/chat_template/make_goldens.py: byte equality, then equality with \n escapes
normalised, then rendering under the same jinja2 environment on 12 scenarios (no tools).  Writes the fetched templates to
./hf_templates/.  Needs jinja2 only.  Network: huggingface.co.
"""
import ast, json, urllib.request, os, itertools, re
from jinja2.exceptions import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "data", "chat_template", "make_goldens.py"),encoding="utf-8").read()
T={}
for node in ast.parse(src).body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Subscript) and getattr(node.targets[0].value,"id","")=="T":
        T[ast.literal_eval(node.targets[0].slice)]=ast.literal_eval(node.value)
REPOS={
 "chatml":["teknium/OpenHermes-2.5-Mistral-7B"],
 "qwen2_5":["Qwen/Qwen2.5-7B-Instruct"],
 "qwen3":["Qwen/Qwen3-8B","Qwen/Qwen3-0.6B"],
 "llama2":["unsloth/llama-2-7b-chat"],
 "llama3":["NousResearch/Meta-Llama-3-8B-Instruct"],
 "llama3_1":["unsloth/Llama-3.1-8B-Instruct"],
 "mistral_v1":["mistralai/Mistral-7B-Instruct-v0.1","mistralai/Mistral-7B-Instruct-v0.2"],
 "mistral_v3":["unsloth/mistral-7b-instruct-v0.3"],
 "gemma":["unsloth/gemma-7b-it","unsloth/gemma-2b-it"],
 "phi3":["microsoft/Phi-3-mini-4k-instruct"],
 "deepseek":["deepseek-ai/deepseek-llm-7b-chat"],
 "deepseek_r1":["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B","deepseek-ai/DeepSeek-R1-Distill-Llama-8B"],
 "zephyr":["HuggingFaceH4/zephyr-7b-beta"],
}
def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
def raise_exception(m): raise TemplateError(m)
def env():
    e=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=[loopcontrols])
    e.filters["tojson"]=tojson; e.globals["raise_exception"]=raise_exception
    e.globals["strftime_now"]=lambda f: "2026-01-01"
    return e
E=env()
U={"role":"user","content":"Hi there"}; A={"role":"assistant","content":"Hello!"}; S={"role":"system","content":"Be brief."}
U2={"role":"user","content":"  second  "}
SCEN=[[U],[S,U],[U,A,U2],[S,U,A,U2],[U,A],[S,U,A]]
def render(t,msgs,agp,bos,eos):
    try: return E.from_string(t).render(messages=msgs,add_generation_prompt=agp,bos_token=bos,eos_token=eos)
    except TemplateError as e: return "ERR:"+str(e)[:40]
    except Exception as e: return "ERR:"+type(e).__name__
def norm(s): return s.replace("\n","\n")
os.makedirs("hf_templates",exist_ok=True)
for k,repos in REPOS.items():
    got=None
    for r in repos:
        try:
            d=urllib.request.urlopen(urllib.request.Request("https://huggingface.co/%s/resolve/main/tokenizer_config.json"%r,headers={"User-Agent":"curl/8"}),timeout=30).read()
            cfg=json.loads(d); ct=cfg.get("chat_template")
            if isinstance(ct,list): ct=next(x["template"] for x in ct if x["name"]=="default")
            if ct: got=(r,ct,cfg.get("bos_token"),cfg.get("eos_token")); break
        except Exception as e: pass
    if not got: print("NOFETCH",k,repos); continue
    r,ct,bos,eos=got
    bos=bos if isinstance(bos,str) else (bos or {}).get("content","<s>"); eos=eos if isinstance(eos,str) else (eos or {}).get("content","</s>")
    open("hf_templates/%s.jinja"%k,"w",encoding="utf-8").write(ct)
    f=T[k]
    if f==ct: print("IDENTICAL(bytes)",k,r); continue
    if norm(f)==norm(ct): print("IDENTICAL(modulo \n escape vs real LF)",k,r); continue
    diffs=0; n=0; ex=None
    for m in SCEN:
        for agp in (False,True):
            n+=1; a=render(f,m,agp,bos,eos); b=render(ct,m,agp,bos,eos)
            if a!=b:
                diffs+=1
                if ex is None: ex=(m,agp,a,b)
    if diffs==0: print("TEXT-DIFFERS but renders equal on %d scenarios"%n,k,r,"(lens %d/%d)"%(len(f),len(ct)))
    else:
        print("RENDER-DIFFERS %d/%d scenarios"%(diffs,n),k,r,"(lens %d/%d)"%(len(f),len(ct)))
        m,agp,a,b=ex; print("    scenario roles",[x['role'] for x in m],"agp",agp); print("    forge:",repr(a[:160])); print("    hf   :",repr(b[:160]))
