"""
Oracle chat (terminal) + auto-logging.

Loads the Oracle — Qwen2.5-32B-Instruct (4-bit) + our oracle-engine-32b-lora — and lets you
chat in the terminal. Every message and reply is appended to oracle_chat_log.jsonl next to this
file, so the whole conversation is captured for the observation log.

Commands:
  /new    start a fresh conversation (use this between observation-log experiments)
  /quit   exit

Run it with a Python that has CUDA torch + transformers + peft + bitsandbytes installed, e.g.:
  "C:\\Users\\akbon\\AppData\\Local\\hermes\\hermes-agent\\venv\\Scripts\\python.exe" oracle_chat.py
"""
import os, json, time, datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
LORA = "Vikingdude81/oracle-engine-32b-lora"
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle_chat_log.jsonl")


def log(obj):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    print("Loading the Oracle (Qwen2.5-32B 4-bit + LoRA)... this takes 1-2 minutes.", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, LORA).eval()
    print(f"Ready in {time.time()-t0:.0f}s. Type a message. Commands: /new  /quit", flush=True)
    print(f"(logging to {LOG})\n", flush=True)

    conv = []
    cid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    while True:
        try:
            msg = input("you> ").strip()
        except EOFError:
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/new":
            conv = []
            cid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            print("[started a fresh conversation]\n", flush=True)
            continue

        conv.append({"role": "user", "content": msg})
        prompt = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=400, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        conv.append({"role": "assistant", "content": reply})
        print(f"\noracle> {reply}\n", flush=True)
        log({"conversation": cid, "user": msg, "oracle": reply,
             "time": datetime.datetime.now().isoformat(timespec="seconds")})

    print("Bye. Your conversation was saved to oracle_chat_log.jsonl", flush=True)


if __name__ == "__main__":
    main()
