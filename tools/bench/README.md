# Benchmarks

`bench.sh` times `mandelbrot --headless` on a fixed set of views with
[hyperfine](https://github.com/sharkdp/hyperfine). That's the whole real
pipeline: the reference orbit on the CPU, then the GPU render, readback and PNG
encode. It builds the headless-only binary (`--no-default-features`) itself.

```sh
tools/bench/bench.sh --rev HEAD            # did my uncommitted edits help? (vs last commit)
tools/bench/bench.sh --rev main seahorse-1e300   # one scenario vs another branch
tools/bench/bench.sh --quick               # all scenarios, 960×540, 3 runs
tools/bench/bench.sh                       # all scenarios, 1920×1080, 5 runs
tools/bench/bench.sh --list
```

`--rev REF` checks REF out into `.worktrees/<sha>` and builds it there, with its
own target dir, so your `target/` stays warm. Each scenario then runs both
binaries back to back in one hyperfine call. The line to read is
`working tree ran 1.23 ± 0.04 times faster than main (…)`. A ratio within about
2σ of 1.00 is noise.

Results (hyperfine JSON and markdown per scenario) go to
`results/<sha>[-dirty]/`, or `results/<rev>-vs-<sha>/` with `--rev`.
Both `results/` and `.worktrees/` are git-ignored. To clean up:
`rm -rf tools/bench/.worktrees && git worktree prune`.

## Scenarios

| name | mostly measures |
|---|---|
| `shallow` | GPU, plain f32 path, f64 orbit fast path |
| `seahorse-1e30` | GPU f32 perturbation from a `Big` orbit |
| `seahorse-1e100` | the `DEEP` shader pipeline |
| `elephant-1e100-aa-de` | 2×2 antialiasing + DE shading |
| `burning-ship` | a non-holomorphic kind |
| `3d` | the raymarched 3D export path |
| `anim-zoom` | 24-frame zoom (⅓ size): the parallel orbit, GPU and PNG pipeline |
| `seahorse-1e300` *(slow)* | ~270k iterations per pixel on the deep path, ~40 s per run even with `--quick`. It's GPU-bound: the orbit itself is under a second. Only runs when named or with `--all` |

Views come from `tools/deep-zoom/LOCATIONS.md`. Each wall time includes process
start, device creation and shader compilation, so a small change to a GPU-bound
scenario shows up diluted.

## Tips

- Keep the machine idle and on AC power. Other GPU apps (browsers, video) add a lot of noise.
- Use `--quick` while you iterate and a full run to confirm.
- Timings are only comparable on the same machine and GPU.
