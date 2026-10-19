# ComfyUI Sequential Wildcards

A wildcard processor that walks a list **in order** instead of picking at random. Clip 1 gets line 1, clip 2 gets line 2.

Modelled on the Wildcard Processor in [bash-j/mikey_nodes](https://github.com/bash-j/mikey_nodes), and it reads the same
`ComfyUI/wildcards` folder.

> **EXPERIMENTAL.** Lightly tested. Check the `log` output before trusting the line numbers.

## Install
Copy this folder into `ComfyUI/custom_nodes/` as its own folder and restart ComfyUI.
The node appears as **Sequential Wildcard Processor**.

## Inputs
| Input | Meaning |
|---|---|
| `prompt` | Prompt text with `__filename__` tags |
| `seed` | Which line to use. Wire it to the same counter as your shot number |
| `first_line` | `1` = seed 1 is the first line. `0` = seed 0 is the first line |

## Outputs
`prompt` (the text with the lines substituted) and `log` (which line it used, per tag).

## Notes
* It wraps around at the end of the file, so a 19-line list repeats from line 1 on clip 20.
* Several tags in one prompt all step together, so keep your lists the same length if they need to stay in step.
* Whether seed 1 gives line 1 depends on when your frontend increments the counter. Run one clip and read `log`.
