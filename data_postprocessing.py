import argparse, json, re, sys, pathlib
from typing import Iterable

ASSIST_INLINE_RE = re.compile(
    r'\[?\|assistant\|\][:]*\s*', flags=re.IGNORECASE
)
ASSIST_LINE_RE = re.compile(
    r'^\s*assistant[:\s]*$', flags=re.IGNORECASE | re.MULTILINE
)

THINK_RE  = re.compile(r'<think>.*?(?:</think>|$)', flags=re.DOTALL | re.IGNORECASE)

def strip_prompts(text: str) -> str:
    m = ASSIST_INLINE_RE.search(text)
    if not m:
        m = ASSIST_LINE_RE.search(text)
    return text[m.end():] if m else text  

def postprocess(
    raw: str,
    strip_prompts: bool = True,
    strip_think: bool = False,
    extra_regex: Iterable[str] = (),
) -> str:
    out = raw
    if strip_prompts:
        out = strip_prompts_func(out)  
    if strip_think:
        out = THINK_RE.sub('', out)
    for pattern in extra_regex:
        out = re.sub(pattern, '', out, flags=re.DOTALL)
    return out.strip()

strip_prompts_func = strip_prompts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--strip-prompts", action="store_true", default=True)
    ap.add_argument("--no-strip-prompts", dest="strip_prompts",
                    action="store_false")
    ap.add_argument("--strip-think", action="store_true")
    ap.add_argument("--extra-regex", action="append", default=[])
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--new-field", default=None)
    ap.add_argument("--backup", action="store_true")

    args = ap.parse_args()

    inp_path = pathlib.Path(args.input)
    if args.inplace and not args.output:
        args.output = str(inp_path) 

    if args.backup and args.inplace:
        backup = inp_path.with_suffix(inp_path.suffix + ".bak")
        backup.write_bytes(inp_path.read_bytes())

    fin  = inp_path.open(encoding="utf-8")
    fout = (open(args.output, "w", encoding="utf-8")
            if args.output else sys.stdout)

    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        raw_ans = obj.get("model_answer", "")
        cleaned = postprocess(
            raw_ans,
            strip_prompts=args.strip_prompts,
            strip_think = args.strip_think,
            extra_regex = args.extra_regex,
        )
        if args.new_field:
            obj[args.new_field] = cleaned
        else:
            obj["model_answer"] = cleaned
        json.dump(obj, fout, ensure_ascii=False)
        fout.write("\n")

    fin.close()
    if fout is not sys.stdout:
        fout.close()

if __name__ == "__main__":
    main()
