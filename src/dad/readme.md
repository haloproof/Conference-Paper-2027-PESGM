# src/dad 文件说明

以下列出保留的源文件；自动生成的 __pycache__/ 不列入。B3-v4、B5-v4 为当前版本；B4 保留为历史方法。

```text
src/dad/
├── __init__.py — 包说明；导入时不启动任务
├── benchmark_b1.py — B1 通用调度执行、状态恢复与账目基础
├── benchmark_b1_continue.py — B1 日前实验阶段接续与来源核验
├── benchmark_b1_da.py — B1 日前调度入口，固定吞吐量老化成本
├── benchmark_b1_solver.py — B1 充放电互斥 MILP 求解器
├── benchmark_b2_continue.py — B2 日前实验阶段接续与来源核验
├── benchmark_b2_da.py — B2 日前调度与原生教师执行入口
├── benchmark_b2_solver.py — B2 分段线性日历／循环老化 MILP
├── benchmark_b2_tables.py — 读取作者老化表，选择健康列并计算参考缩放系数
├── benchmark_b3_v4_continue.py — B3-v4 已完成阶段的延长与完整状态交接
├── benchmark_b3_v4_da.py — B3-v4 日前调度、执行与检查点入口
├── benchmark_b3_v4_solver.py — B3-v4 十初值固定模式投影梯度／时移搜索
├── benchmark_b4_continue.py — 保留的 B4 阶段接续入口，原 b45 接续逻辑已并入
├── benchmark_b4_da.py — 保留的 B4 调度执行入口，原 b45 执行逻辑已并入
├── benchmark_b4_solver.py — 历史 B4 动态吞吐量成本反馈与历史损伤定价
├── benchmark_b5_v4_continue.py — B5-v4 从已完成阶段准备退役接续
├── benchmark_b5_v4_da.py — B5-v4 日前调度、执行与检查点入口
├── benchmark_b5_v4_eol.py — B5-v4 接续运行至原生 SOH 退役阈值
├── benchmark_b5_v4_solver.py — B5-v4 十档固定吞吐量候选及健康加权选优
├── benchmark_checks.py — 独立核对原生执行边界、交付误差与调度 SOC 带
├── benchmark_hybrid_common.py — B4／B5-v4 共用日账、文件锁、来源签名与阶段核验
├── benchmark_hybrid_solver.py — B4／B5-v4 共用模型加载、前向评价、进程管理与结果核验
├── benchmark_m0_continue.py — M0 纯套利实验阶段接续与来源核验
├── benchmark_m0_da.py — M0 不计老化罚项的日前调度入口
├── benchmark_m0_solver.py — M0 纯套利 MILP，复用 B1 约束
├── benchmark_prices.py — 连续 15 分钟价格窗口与时间索引
├── check_hybrid.py — 混合模型与原生教师的有界短检查
├── common.py — JSON、哈希、随机种子、工作预算与运行目录通用工具
├── contract.py — 持续 AC 功率交付约定的版本标识
├── data.py — 教师证书核验、基础轨迹生成及数据划分读取
├── dispatch_continuous_prices.py — UTC 调度价格审计及历史价格年映射
├── dispatch_health.py — 原生健康状态与残余半循环估值的分开记账
├── estimator_compare.py — 不同老化估计模型的离线训练与比较入口
├── estimator_compare_models.py — 估计模型、输入特征、损失函数与原始单位指标
├── hybrid.py — 物理／热网络混合模型、SOC／温度递推与半循环老化
├── hybrid_data.py — 混合模型训练用历史状态与教师数据生成入口
├── hybrid_train.py — 等效热量网络的离线两阶段训练入口
├── power_limits.py — AC 调度功率与 DC 电池功率上限换算
├── prices.py — 原始 SMARD 电价读取、时区审计与预处理
├── readme.md — 本目录文件树与用途说明
└── teacher.py — 固定版本 SimSES Sony LFP／Naumann 与动态热模型适配
```

共享代码提取仅整理依赖，不改变现行调度算法。已有 runs 中的来源签名与检查点保留原样；源码重组后的新签名不同，旧运行的严格恢复检查不会被跳过。
