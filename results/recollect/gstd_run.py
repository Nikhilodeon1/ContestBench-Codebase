import os,time,json,hashlib,sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,os.getcwd())
from dotenv import load_dotenv; load_dotenv(dotenv_path=os.path.join(os.getcwd(),".env"))
from google import genai; from google.genai import types
from contestbench.eval.batch_api import gemini_keys
keys=gemini_keys(); st={"i":0}
d=Path("results/recollect/gstd"); od=Path("results/recollect/gstd_out"); od.mkdir(exist_ok=True)
prov=open("results/recollect/provenance.jsonl","a",encoding="utf-8")
def call(prompt):
    for _ in range(len(keys)*2):
        i=st["i"]
        try:
            c=genai.Client(api_key=keys[i])
            r=c.models.generate_content(model="gemini-flash-latest",contents=prompt,
                config=types.GenerateContentConfig(temperature=0,max_output_tokens=4000))
            return r.text
        except Exception as e:
            if any(s in str(e) for s in ("429","RESOURCE_EXHAUSTED","404")): st["i"]=(i+1)%len(keys); continue
            print("err",str(e)[:60],flush=True); return None
    return None
for p in sorted(d.glob("batch_*.txt")):
    ci=p.stem.split("_")[-1]; out=od/f"gemstd_{ci}.txt"
    if out.exists() and out.stat().st_size>50: continue
    prompt=p.read_text(encoding="utf-8"); t=call(prompt)
    if t:
        out.write_text(t,encoding="utf-8")
        prov.write(json.dumps({"label":"gemini:standard","provider":"gemini","model":"gemini-flash-latest","thinking":False,"temperature":0,"prompt_chunk":int(ci),"prompt_hash":hashlib.sha256(prompt.encode()).hexdigest()[:16],"n_answers":t.count('"confidence"'),"chunk_size":25,"timestamp":datetime.now(timezone.utc).isoformat()})+"\n"); prov.flush()
        print(f"gemstd_{ci}: {t.count(chr(34)+'confidence')}",flush=True)
    time.sleep(1)
print("GSTD DONE",flush=True)
