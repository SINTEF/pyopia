# pyopia-gui-support

Staging branch for fixes needed by
[pyopia-gui](https://github.com/nimmo-smith-technologies/pyopia-gui), rebased
regularly against `summer26-features` (currently its furthest-along, not yet
merged, upstream base - see #430/#431), or `main` once that work has landed.

Not a release channel. Each fix lands in `main` via its own PR (`Closes #N`);
this branch is deleted once they have.

Docker images built from this branch use distinct tags, never `latest`.

Scope: #434, #427, #423, #426
