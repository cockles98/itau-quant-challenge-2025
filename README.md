# t_hrp_v3

Estrutura inicial para o desenvolvimento de uma pipeline de Hierarchical Risk Parity (HRP) com modulos de dados, features, portfolio, risco, backtesting, validacao, metricas e relatorios.

## Comecando

1. Crie e ative um ambiente virtual (exemplo com `venv`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\\Scripts\\activate     # Windows
   ```
2. Instale o projeto em modo editavel:
   ```bash
   pip install -e .
   ```
3. Exporte o caminho da pasta `src` para o `PYTHONPATH` antes de rodar os modulos:
   ```bash
   export PYTHONPATH="$(pwd)/src"            # Linux/Mac
   $Env:PYTHONPATH = "$PWD/src"              # Windows PowerShell
   set PYTHONPATH=%CD%\\src                   # Windows CMD
   ```
4. Verifique as importacoes minimas (DoD):
   ```bash
   python -c "import importlib; importlib.import_module('dataio'); importlib.import_module('features')"
   ```
5. Teste rapido do carregador de configuracoes:
   ```bash
   python -c "from dataio import load_config; load_config('configs/base.yaml')"
   ```

## Estrutura de Pastas

```
src/
  dataio/
  features/
  portfolio/
  risk/
  backtest/
  validation/
  metrics/
  reports/
configs/
notebooks/
reports/
artifacts/
```

Utilize `configs/base.yaml` como ponto de partida para configuracoes do projeto e preencha conforme as necessidades do ambiente de execucao.
