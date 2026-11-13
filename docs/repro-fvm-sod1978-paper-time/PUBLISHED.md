# 发布记录

| 项 | 值 |
|---|---|
| 平台报告 id | 17307 |
| 报告标题 | REPRO-FVM-SOD1978 · 判据时刻 t=0.2 的 L1 验收复核（红灯门重建） |
| mission | 18524 |
| 工件 | `paper_time_study.json`、`REPORT.md`、`figures/*.png`（全部由模块程序化生成） |
| 性质门 | `tests/test_compressible_sod.py`（9 passed）+ `tests/test_repro_fvm_sod1978.py`（6 passed） |
| 锚点状态建议 | 保持 `open`：C1/C2 已达标，C3（全域 L1 阶 ≥1.0）字面仍 FAIL，不覆盖 |

复现：

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src python -m tensorfvm.verification.sod1978_paper_time \
        --out docs/repro-fvm-sod1978-paper-time
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src python -m pytest tests/test_compressible_sod.py -q
