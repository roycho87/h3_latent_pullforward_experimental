"""Sequential Wildcard Processor.

Same __file__ syntax and the same wildcards folder as Mikey's Wildcard Processor
(ComfyUI/user/default/wildcards or ComfyUI/wildcards), but nothing is random:
the seed is a line number. Seed 1 -> line 1, seed 2 -> line 2, ... (wraps around
at the end of the file). Every file is read independently, so two lists with the
same number of lines always stay in step.

Syntax
  __name__        line N of name.txt                      (N from the seed)
  __sub/name__    same, from a sub folder
  __+name__       the NEXT line  (N+1)
  __-name__       the PREVIOUS line (N-1)
  3$$__name__     3 lines in a row starting at N, joined with ", "
  {a|b|c}         sequential too: picks option N (wrapping)
  Lines inside a wildcard file may contain more __wildcards__ / {a|b}.
Blank lines and lines starting with # are skipped when counting lines.
"""
import os
import re

import folder_paths

WC_RE = re.compile(r'((\d+)\$\$)?__(\+|-)?((?:[^|_]+_)*[^|_]+)__')
BRACE_RE = re.compile(r'{([^{}]*)}')


def _wildcard_dir():
    p = os.path.join(folder_paths.get_user_directory(), 'wildcards')
    if not os.path.isdir(p):
        p = os.path.join(os.path.dirname(folder_paths.__file__), 'wildcards')
    return p


def _read_lines(name):
    base = _wildcard_dir()
    path = os.path.join(base, *name.split('/')) + '.txt'
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.read().splitlines()]
    return [l for l in lines if l and not l.startswith('#')]


class SequentialWildcardProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "placeholder": "Prompt text with __wildcards__"}),
            "seed": ("INT", {"default": 1, "min": -0xffffffffffff, "max": 0xffffffffffffffff,
                             "tooltip": "Line number to use. With first_line = 1, seed 1 is the first line."}),
            "first_line": ("INT", {"default": 1, "min": 0, "max": 1,
                                   "tooltip": "1 = seed 1 is the first line. 0 = seed 0 is the first line."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "log")
    FUNCTION = "process"
    CATEGORY = "Mikey/Text"

    def process(self, prompt, seed, first_line):
        n = seed - first_line          # zero-based line index
        log = []
        text = prompt
        for _ in range(10):            # resolve wildcards found inside wildcard files
            new = self._pass(text, n, log)
            if new == text:
                break
            text = new
        return (text, "\n".join(log))

    def _pass(self, text, n, log):
        def wc(m):
            count = int(m.group(2)) if m.group(2) else 1
            mod, name = m.group(3), m.group(4)
            lines = _read_lines(name)
            if not lines:
                log.append(f"{name}: file not found or empty")
                return m.group(0)
            start = n + (1 if mod == '+' else -1 if mod == '-' else 0)
            picked = [lines[(start + i) % len(lines)] for i in range(count)]
            log.append(f"{name}: line {start % len(lines) + 1} of {len(lines)}")
            return ", ".join(picked)

        def br(m):
            opts = m.group(1).split('|')
            return opts[n % len(opts)].strip()

        text = WC_RE.sub(wc, text)
        while BRACE_RE.search(text):
            text = BRACE_RE.sub(br, text)
        return text


NODE_CLASS_MAPPINGS = {"SequentialWildcardProcessor": SequentialWildcardProcessor}
NODE_DISPLAY_NAME_MAPPINGS = {"SequentialWildcardProcessor": "Sequential Wildcard Processor"}
