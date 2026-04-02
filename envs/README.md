# Environment Files

- `PHPGAT.yaml`: base pipeline environment
- `dnaberts.yaml`: DNA-BERT embedding environment
- `esm.yaml`: ESM protein embedding environment
- `pharokka.yaml`: lightweight template for Pharokka / Phanotate / Prodigal
- `sourmash.yaml`: lightweight template for sourmash similarity search

The first three files were copied from the working project environment exports and had their machine-specific `prefix:` removed. The last two are repository-local templates because no exported YAML for those environments was available in the source workspace.
