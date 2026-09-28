#!/usr/bin/env bash
# Benchmark `mandelbrot --headless` on a fixed set of scenarios with hyperfine.
#
# Usage:
#   bench.sh [--quick] [--runs N] [--rev REF] [--all] [--list] [scenario...]
#
#   (default)   time the working tree; results go to results/<sha>[-dirty]/
#   --rev REF   also build REF (in a git worktree) and time both binaries side
#               by side; hyperfine prints "X ± σ times faster" per scenario
#   --quick     960×540 and 3 runs instead of 1920×1080 and 5
#   --runs N    timed runs per command (after 1 warmup run)
#   --all       include the slow scenarios
#   --list      print the scenario names and exit
#
# With no scenario names, all but the slow ones run. See README.md.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(git -C "$here" rev-parse --show-toplevel)

# Views from tools/deep-zoom/LOCATIONS.md.
SEAHORSE_40='-0.77568376800905379745347652613832924487504096622022,0.13646736829469012473375311880735014411233827361594'
SEAHORSE_101='-0.77568376800905379746948350393474104572824650461579342765769586746577714101553375163935420167637213972439444483,0.13646736829469012473327440961784876519777211159246218448277041663416833530840062457397927487282583495378053495'
SEAHORSE_300='-0.775683768009053797469483503934741045728246504615800394794952001400666017222819589894883159368782255028252296689050972508942991790804637379375064948276511044167168394677127466362472192356953918438738697548775738756167744421444715134058363508346750264411321139485834882343906172015818621107808449401958194265462,0.136467368294690124733274409617848765197772111592467696407800195400835665974815500875448611244178656625706823059759836009902469392551108149827753781145338163746214967880988693412636771323356061132825037923307975359146666033691080243216637006179374407866237575096209504710178751614036006531433101633993532313930'
ELEPHANT_100='0.2920738619144539179997175176922057340804018914331633820622644536373359023331882946705674169275530423410817910,0.01577091721139011808246902823033833768260048863232968822435568049949556139801802963232184866659058960379114747'

# name -> headless args (no size or --export-path; those are added per run).
# Order matters for display, hence the separate list. SLOW ones (tens of
# seconds per run) only run when named or with --all.
SCENARIOS=(shallow seahorse-1e30 seahorse-1e100 elephant-1e100-aa-de burning-ship 3d anim-zoom)
SLOW=(seahorse-1e300)
declare -A ARGS=(
    [shallow]="--view=-0.75,0,1.5"
    [seahorse-1e30]="--view=$SEAHORSE_40,1e-30"
    [seahorse-1e100]="--view=$SEAHORSE_101,3.4e-101"
    [seahorse-1e300]="--view=$SEAHORSE_300,5.5e-300"
    [elephant-1e100-aa-de]="--view=$ELEPHANT_100,1e-75 --antialias --de"
    [burning-ship]="--kind burning-ship"
    [3d]="--rendering-kind 3d --de --pitch -40 --view=-0.75,0,1.5"
    [anim-zoom]="--view=-0.75,0,1.5 --to-view=$SEAHORSE_40,1e-30 --frames 24"
)

quick=0
runs=
rev=
picked=()
while (($#)); do
    case $1 in
        --quick) quick=1 ;;
        --runs) runs=$2; shift ;;
        --rev) rev=$2; shift ;;
        --all) picked+=("${SCENARIOS[@]}" "${SLOW[@]}") ;;
        --list) printf '%s\n' "${SCENARIOS[@]}"; printf '%s (slow)\n' "${SLOW[@]}"; exit 0 ;;
        -h|--help) sed -n '2,16s/^# \{0,1\}//p' "$0"; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *)
            [[ -v ARGS[$1] ]] || { echo "unknown scenario: $1 (try --list)" >&2; exit 2; }
            picked+=("$1") ;;
    esac
    shift
done
((${#picked[@]})) || picked=("${SCENARIOS[@]}")

if ((quick)); then
    width=960 height=540 runs=${runs:-3}
else
    width=1920 height=1080 runs=${runs:-5}
fi

command -v hyperfine >/dev/null || { echo "hyperfine not found (https://github.com/sharkdp/hyperfine)" >&2; exit 1; }

# Headless-only binary: no eframe/egui, faster to build, same render path.
build() { cargo build --release --no-default-features --quiet --manifest-path "$1/Cargo.toml"; }

echo "building working tree…" >&2
build "$root"
new_bin="$root/target/release/mandelbrot"

if [[ -n $rev ]]; then
    rev_sha=$(git -C "$root" rev-parse --short "$rev^{commit}")
    wt="$here/.worktrees/$rev_sha"
    [[ -d $wt ]] || git -C "$root" worktree add --detach --quiet "$wt" "$rev_sha"
    echo "building $rev ($rev_sha)…" >&2
    # Separate target dir so switching revisions doesn't invalidate target/.
    CARGO_TARGET_DIR="$here/.worktrees/target" build "$wt"
    old_bin="$here/.worktrees/target/release/mandelbrot"
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

sha=$(git -C "$root" rev-parse --short HEAD)
git -C "$root" diff --quiet HEAD -- . ':!tools/bench' || sha+=-dirty
out="$here/results/${rev:+$rev_sha-vs-}$sha"
mkdir -p "$out"

# Full headless command line for scenario $2 with binary $1.
command_for() {
    local bin=$1 name=$2 w=$width h=$height dest=$tmp/$2.png
    if [[ $name == anim-* ]]; then
        # Animation: a directory of frames, at a smaller size.
        w=$((width / 3)) h=$((height / 3)) dest=$tmp/$name
    fi
    echo "$bin --headless --width $w --height $h --export-path $dest ${ARGS[$name]}"
}

for name in "${picked[@]}"; do
    echo >&2
    echo "=== $name ===" >&2
    if [[ -n $rev ]]; then
        cmds=(-n "$rev ($rev_sha)" "$(command_for "$old_bin" "$name")"
              -n "working tree" "$(command_for "$new_bin" "$name")")
    else
        cmds=(-n "$name" "$(command_for "$new_bin" "$name")")
    fi
    hyperfine --warmup 1 --runs "$runs" --style basic \
        --export-json "$out/$name.json" --export-markdown "$out/$name.md" \
        "${cmds[@]}"
done

echo
echo "## Summary (${width}×${height}, $runs runs; results in ${out#"$root"/}/)"
for name in "${picked[@]}"; do
    echo
    echo "### $name"
    cat "$out/$name.md"
done
