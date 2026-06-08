# LLM Quick Map

## Core package
- `reality_stone/python/reality_stone/models/`: hierarchical sentence-topic LLM, manifold learner, top-down decoder, bottom-up encoder, transformer converter.
- `reality_stone/python/reality_stone/layers/`: metric attention, Poincare/Lorentz/Klein/diffusion/suppression/RSULF layers.
- `reality_stone/python/reality_stone/api/`: pipeline, indexing, inference, QA convenience APIs.
- `reality_stone/python/reality_stone/clarus/`: Clarus runtime, CE ops, sleep/replay, agent loop, research probes, reality bridge.
- `reality_stone/src/`: Rust/PyO3 geometry backend plus CUDA kernels.

## LLM design docs
- `docs/7_AGI/9_LLM.md`: CE-LLM build guide and implementation principles.
- `docs/7_AGI/12_Equation.md`: CE-Transformer and energy relaxation architecture.
- `docs/7_AGI/13_Verification.md`: verification, runtime measurements, artifact notes.
- `docs/7_AGI/17_AgentLoop.md`: self-reference, memory, critic, residual loop.
- `docs/7_AGI/18_CodeMap.md`: intended code map.

## Important tests
- `reality_stone/tests/llm/`: model, decoder, metric attention/router, GPT2/manifold, CUDA symbol coverage.
- `reality_stone/tests/api/test_pipeline_api.py`: high-level API regression.

## Suggested validation
```powershell
$env:PYTHONPATH = "reality_stone/python"
.\.venv\Scripts\python.exe -m pytest -q reality_stone\tests\llm reality_stone\tests\api\test_pipeline_api.py
```
