"""N6 boundary callback: source-derived interface and complete recipe drift gate."""
import re
import subprocess
import sys
from pathlib import Path
import code_anchors

REPORT = Path(__file__).resolve().parent.parent
TARGET = 'book/chapters/14-boundaries/_howto.qmd'
PIN = '0d5caacbd8e365c87a30af6ce043c39ab30c0959'


def read_source(checkout, sha, path):
    return subprocess.run(['git', '-C', str(checkout), 'show', f'{sha}:{path}'],
                          check=True, capture_output=True, text=True).stdout


def generate(report):
    repo, sha = code_anchors.read_pin(report / 'code_pin.yml')
    if sha != PIN:
        raise ValueError('N6 boundary contract needs review when the code pin changes')
    checkout = code_anchors.find_commit(repo, sha)

    def excerpt(path, pattern, label=None, symbol=None):
        """the one match of pattern in path at the pin (the drift gate), and a pinned code citation"""
        text = read_source(checkout, sha, path)
        matches = list(re.finditer(pattern, text, re.M | re.S))
        if len(matches) != 1:
            raise ValueError(f'{path}: expected one match, got {len(matches)}')
        code = '{{< code ' + path + (' ' + symbol if symbol else '') + ' >}}'
        return matches[0][0].strip(), (f'{label}, {code}' if label else code)

    def cite(label, path, pattern, symbol=None):
        return excerpt(path, pattern, label, symbol)[1]

    signature, signature_cite = excerpt('python/snapy/mesh.pyi',
        r'    def set_bfunc\(self,.*?\) -> None: \.\.\.')
    registration, registration_cite = excerpt('python/csrc/pymesh.cpp',
        r'            self->bfuncs\(\)\[face\] = func;')
    arguments, arguments_cite = excerpt('python/csrc/pymesh.cpp',
        r'          py::arg\("dx3"\).*?py::arg\("name"\) = ""\)')
    callback, callback_cite = excerpt('python/csrc/pymesh.cpp',
        r'              func = \[f\]\(torch::Tensor const& var, int id,.*?\n              \};')
    native, native_cite = excerpt('src/bc/bc_func.hpp',
        r'using bcfunc_t =\n[^;]+;', symbol='bcfunc_t')
    ghosts, ghosts_cite = excerpt('src/coord/coordinate.hpp',
        r'  int nc1\(\) const[^\n]+\n  int nc2\(\) const[^\n]+\n  int nc3\(\) const[^\n]+')
    lookup, lookup_cite = excerpt('src/mesh/meshblock_options.cpp',
        r'      name \+= "_" \+ side;.*?op->bfuncs\(\).push_back\(get_bc_func\(\)\[name\]\);',
        symbol='MeshBlockOptionsImpl::from_yaml')
    bc_glob, bc_glob_cite = excerpt('src/CMakeLists.txt',
        r'file\(GLOB src_files\n    #\*\.cpp\n    bc/\*\.cpp')
    # Join the source's error-string pieces around the selected example key.
    error_parts = re.findall(r'"([^"\n]*)"', lookup)
    inner_error = error_parts[1] + 'zero_gradient' + error_parts[0] + 'inner' + error_parts[2]
    links = {
        'shape_cite': cite('shape checks', 'src/mesh/meshblock.cpp',
                           r'"initialize: hydro_w has incorrect shape\."', 'MeshBlockImpl::initialize_local'),
        'slab_cite': cite('slab dispatch', 'src/mesh/meshblock_exchange.cpp',
                          r'          int g = ng < n / 2 \? ng : n / 2;', 'MeshBlockImpl::exchange_ghost_zones'),
        'mapping_cite': cite('mapping', 'python/csrc/pymesh.cpp',
                             r'            int face = -1;.*?face = 5;\n            \}'),
        'dispatch_cite': cite('dispatch', 'src/mesh/meshblock_exchange.cpp',
                              r'      fn\(hydro, dim, op\);\n      if \(has_tracers\) \{.*?\n      \}',
                              'MeshBlockImpl::apply_boundaries'),
        'init_prim_cite': cite('initialization', 'src/mesh/meshblock.cpp',
                               r'  apply_boundaries\(vars, hydro_w, scalar_r, true\);', 'MeshBlockImpl::initialize_local'),
        'init_cons_cite': cite('initialization', 'src/mesh/meshblock.cpp',
                               r'  apply_boundaries\(\n      vars, vars.at\("hydro_u"\),',
                               'MeshBlockImpl::finalize_initialization'),
        'stage_cite': cite('each stage', 'src/mesh/meshblock_advance.cpp',
                           r'void MeshBlockImpl::stage_boundaries\(.*?\n\}', 'MeshBlockImpl::stage_boundaries'),
        'refresh_cite': cite('refresh', 'src/mesh/meshblock_exchange.cpp',
                             r'    auto refresh = \[&\]\(torch::Tensor field, int type\) \{.*?\n    \};',
                             'MeshBlockImpl::exchange_ghost_zones'),
        'repeat_cite': cite('repeat-call requirement', 'src/bc/bc_func.hpp',
                            r'// A face function may be called again.*?\(#264\)\.', 'BC_FUNCTION'),
        'macro_cite': cite('macro and registrar', 'src/bc/bc_func.hpp',
                           r'#define BC_FUNCTION\(name, var, dim, op\).*?op\)$', 'BC_FUNCTION'),
        'binding_cite': cite('binding checks', 'python/csrc/pymesh.cpp',
                             r'            if \(self->bfuncs\(\)\.empty\(\)\) \{.*?"functions\."\);\n            \}'),
        'names_cite': cite('name handling', 'python/csrc/pymesh.cpp',
                           r'            // The name must not outlive.*?self->bcnames\(\)\[face\] = name;\n            \}'),
    }
    replacements = locals()
    text = TEMPLATE
    for key in ('signature', 'signature_cite', 'registration', 'registration_cite',
                'arguments', 'arguments_cite', 'callback', 'callback_cite',
                'native', 'native_cite', 'ghosts', 'ghosts_cite',
                'lookup', 'lookup_cite', 'bc_glob', 'bc_glob_cite', 'inner_error'):
        text = text.replace('@' + key + '@', replacements[key])
    for key, value in links.items():
        text = text.replace('@' + key + '@', value)
    return text


def check(report=REPORT):
    skipped = code_anchors.legacy_chapter_of(TARGET, Path(report))
    if skipped:  # a legacy chapter (tools/code-anchors-legacy.txt): checked in full once its line is deleted
        print(f'[N6 boundary contract] skipped: {skipped} is in tools/code-anchors-legacy.txt')
        return 0
    try:
        expected = generate(Path(report))
        errors = [] if (Path(report) / TARGET).read_text() == expected else [
            'generated boundary signature, registration or recipe differs from the pin']
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        errors = [str(exc)]
    print(f'[N6 boundary contract] {len(errors)} problem(s)')
    for error in errors:
        print('  ' + error)
    return int(bool(errors))


TEMPLATE = r'''<!-- generated reader-reference: boundary-contract -->
<!-- Generated by tools/gen_boundary_contract.py; do not edit. -->
#### How to add one

**A minimal input.** Follow steps 0–5 in one Python file, run from the repository root.

0. Prepare a four-cell Cartesian grid with two x1 ghosts at each end. Use the
   chemistry from `examples/straka_single.yaml`, PLM and one RK1 stage
   (see @sec-ch07-user); reserve both x1 faces with `custom`.

```python
from pathlib import Path
import torch
import snapy
import yaml

deck = yaml.safe_load(Path("examples/straka_single.yaml").read_text())
deck["geometry"] = {
    "type": "cartesian",
    "cells": {"nx1": 4, "nx2": 1, "nx3": 1, "nghost": 2},
    "bounds": {"x1min": 0., "x1max": 1., "x2min": 0.,
               "x2max": 1., "x3min": 0., "x3max": 1.},
}
deck.pop("forcing", None)
deck["dynamics"]["reconstruct"] = {
    "vertical": {"type": "plm"}, "horizontal": {"type": "plm"}}
deck["integration"] = {"type": "rk1"}
deck["boundary-condition"] = {"external": {
    "x1-inner": "custom", "x1-outer": "custom"}}
Path("tiny-boundary.yaml").write_text(yaml.safe_dump(deck))
```

1. Write a callback that copies the nearest interior cell into the selected
   ghost strip. It returns nothing and handles the supplied tensor's extent,
   including a tangential slab. Record metadata only, not borrowed tensors.

```python
calls = []

def zero_gradient(face):
    def fill(var, dim, op):
        ng, n = op.nghost(), var.size(dim)
        lo, src = (0, ng) if face == 0 else (n - ng, n - ng - 1)
        target = var.narrow(dim, lo, ng)
        target.copy_(var.narrow(dim, src, 1).expand_as(target))
        calls.append((face, dim, op.type(), tuple(var.shape),
                      var.dtype, var.device.type))
    return fill
```

2. Register both faces on the options before creating the block. The signature
   and registration below are generated from the pinned Python stub and binding
   (@signature_cite@; @registration_cite@; @arguments_cite@).

```python
opts = snapy.MeshBlockOptions.from_yaml("tiny-boundary.yaml")
opts.set_bfunc(0, 0, -1, zero_gradient(0), name="zero_gradient_inner")
opts.set_bfunc(0, 0, 1, zero_gradient(1), name="zero_gradient_outer")
block = snapy.MeshBlock(opts)
block.to(torch.device("cpu"), torch.float64)
```

3. Initialize the full primitive state, including ghosts. Fill only the interior
   here so that the callback has observable work to do.

```python
coord, eos = block.module("coord"), block.module("hydro.eos")
shape = (eos.nvar(), coord.buffer("x3v").numel(),
         coord.buffer("x2v").numel(), coord.buffer("x1v").numel())
assert shape[1:] == (1, 1, 8)
w = torch.zeros(shape, dtype=torch.float64, device="cpu")
w[snapy.kIDN, ..., 2:6] = 1.
w[snapy.kIPR, ..., 2:6] = 1.e5
interior = w[..., 2:6].clone()
variables, _ = block.initialize({"hydro_w": w})
```

4. Check both ghost strips and the unchanged primitive interior after
   initialization, then advance one RK1 stage.

```python
def check_ghosts(var):
    torch.testing.assert_close(
        var[..., :2], var[..., 2:3].expand_as(var[..., :2]))
    torch.testing.assert_close(
        var[..., -2:], var[..., -3:-2].expand_as(var[..., -2:]))

check_ghosts(w)
torch.testing.assert_close(w[..., 2:6], interior, rtol=0, atol=0)
initial_calls = len(calls)
block.advance_local(variables, 0.001, 0)
```

5. Check that both faces ran again and filled the conserved ghosts.

```python
assert {c[0] for c in calls[initial_calls:]} == {0, 1}
assert all(c[1] == 3 for c in calls)
check_ghosts(variables["hydro_u"])
print("N6 boundary: both x1 faces filled at initialization and after RK1")
```

- **Contract.** Signature: `fill(var, dim, op)` returns no result to the driver;
  the binding acquires the GIL and calls `f(var, id, op)` (@callback_cite@).
  The generated public registration signature is:

  ```text
  @signature@
  ```

- Shapes and ghosts: hydro uses `[nvar, nc3, nc2, nc1]` and scalars use
  `[nscalar, nc3, nc2, nc1]` (@shape_cite@).
  The extents below include ghosts only on non-singleton axes (@ghosts_cite@).
  After exchange, an x1 callback can receive a view whose tangential extent is
  `min(nghost, n/2)` for tangential size `n`; use the supplied sizes, not cached block sizes
  (@slab_cite@).
- Faces and timing: offsets `(dx3, dx2, dx1)` select inner/outer x1 with
  `(0,0,-1)` / `(0,0,1)`, x2 with `(0,-1,0)` / `(0,1,0)` and x3 with
  `(-1,0,0)` / `(1,0,0)` (@mapping_cite@).
  C++ dispatch skips null callbacks and singleton dimensions, passes `dim = 3 - f/2`,
  then calls hydro and, if present, scalars with the corresponding `op.type()`
  (@dispatch_cite@). Calls occur for primitives at
  @init_prim_cite@, conserved state at the end
  of @init_cons_cite@ and
  @stage_cite@, plus a tangential-slab refresh of every installed
  callback except outflow after exchange, on the slabs of the faces exchange owns
  (@refresh_cite@). Do not assume one call per stage.
- Dtype/device: the binding passes the tensor through (@callback_cite@).
  Use tensor-relative operations, as above, and explicitly match any auxiliary
  floating tensors to `var.dtype` and `var.device`. This recipe's runtime
  coverage is CPU; it does not establish CUDA callback behavior.
- Mutation/ownership: fill ghosts in place; rebinding a local Python name will
  not replace the driver's tensor (inference from the void callback interface,
  @native_cite@). Preserve interior cells, do not retain borrowed tensor views,
  and make repeated calls idempotent (@repeat_cite@).
  The binding captures the Python function by value and stores its wrapper in
  the options' callback vector (@callback_cite@; @registration_cite@).
- Registration/refusals: `None` installs a null callback; an empty callback
  vector or a direction unavailable to its size is refused
  (@binding_cite@). The default name is empty;
  registration replaces the previous name. Declare a reflecting name only for
  a matching wall callback, because names participate in the wall coefficient
  (@names_cite@). This zero-gradient recipe
  uses the descriptive names `zero_gradient_inner` and `zero_gradient_outer`;
  these labels do not add functions to the native registry.

The generated ghost extents and binding registration facts are:

```cpp
@ghosts@
@registration@
@arguments@
```

CI checks source-contract drift; the pin-build runtime check covers CPU fills, not MPI corners or CUDA.

**Native C++ alternative.** Implement the generated callback type (@native_cite@):

```cpp
@native@
```

1. Add `src/bc/zero_gradient.cpp` with a suffixed pair. The macro registers
   its exact function name (@macro_cite@).
   For example, the same nearest-interior fill is:

```cpp
#include "bc_func.hpp"

namespace snap {
BC_FUNCTION(zero_gradient_inner, var, dim, op) {
  int ng = op.nghost();
  auto dst = var.narrow(dim, 0, ng);
  dst.copy_(var.narrow(dim, ng, 1).expand_as(dst));
}
BC_FUNCTION(zero_gradient_outer, var, dim, op) {
  int ng = op.nghost();
  int n = var.size(dim);
  auto dst = var.narrow(dim, n - ng, ng);
  dst.copy_(var.narrow(dim, n - ng - 1, 1).expand_as(dst));
}
}  // namespace snap
```

2. Re-run CMake configuration so the new file is picked up by the boundary
   source glob (@bc_glob_cite@), then rebuild the library and use that rebuilt
   package. The generated glob excerpt is:

```cmake
@bc_glob@
```

3. Set the unsuffixed key in YAML:

```yaml
boundary-condition:
  external:
    x1-inner: zero_gradient
    x1-outer: zero_gradient
```

The parser appends `_inner` or `_outer` to each face value, checks the registry,
then installs the function (@lookup_cite@).
See the existing registry explanation in @sec-ch14-registry. The generated
lookup and refusal facts are:

```cpp
@lookup@
```

For the missing inner function, the error text is
`@inner_error@` (the outer lookup reports the `_outer` name).
Both callbacks must obey the same in-place and repeat-call contract. This
native example is source-derived but was not compiled as part of the Python
card's runtime check. The captured Python wrapper is unnecessary on this route.
<!-- end reader-reference: boundary-contract -->
'''

if __name__ == '__main__':
    if '--write' in sys.argv:
        (REPORT / TARGET).write_text(generate(REPORT))
    else:
        sys.exit(check())
