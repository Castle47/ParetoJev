# 相同混合后验训练下的Jev增量实验

使用既有36,000组、12,606名用户，复用既有模型，没有新训练。没有新增Jev请求或付费GPU。
**本轮在原外层结果已被检查后设计，只能视为探索性改进，不能称为新的独立确认。**
两组共享训练损失和预算。主比较用于识别Jev增量；原EMER参考和训练方法对照用于区分共享训练修改的收益。
前六项是按原样本权重计算的NDCG@3，最后一项为top3无负反馈比例。表中为三种子均值；区间是固定预测的条件用户bootstrap95%区间。

两组共同使用50%原EMER后验＋50%条件任务平衡后验；该系数固定，没有进行系数搜索。

## 主比较：混合后验EMER＋Jev vs 混合后验EMER

六目标均值：0.301267 → 0.307199，相对+1.97%；绝对变化区间[+0.003325, +0.009242]。

|目标|对照|实验|绝对变化|相对变化|条件95%区间|
|---|---:|---:|---:|---:|---|
|点击|0.450898|0.450729|-0.000169|-0.04%|[-0.002036, +0.001741]|
|长看|0.397073|0.394448|-0.002625|-0.66%|[-0.005047, -0.000377]|
|点赞|0.265463|0.282387|+0.016924|+6.38%|[+0.012805, +0.021200]|
|关注|0.230080|0.235565|+0.005486|+2.38%|[-0.001581, +0.013100]|
|评论|0.235604|0.243233|+0.007629|+3.24%|[+0.002244, +0.012979]|
|转发|0.228485|0.236833|+0.008348|+3.65%|[-0.002600, +0.024033]|
|无负反馈比例|0.999517|0.999521|+0.000003|—|[-0.000006, +0.000013]|

|外层折|均值绝对变化|
|---|---:|
|0|+0.010474|
|1|+0.004988|
|2|+0.002703|

|种子|均值绝对变化|
|---|---:|
|7|+0.015049|
|17|+0.002440|
|29|+0.000307|

|预设条件|通过|
|---|---|
|primary_conditional_ci_positive|是|
|primary_positive_each_seed|是|
|primary_positive_each_fold|是|
|positive_point_guards|是|
|hate_point_guard|是|

预设条件通过5/5项。

## 参考比较：混合后验EMER＋Jev vs 原EMER

六目标均值：0.299541 → 0.307199，相对+2.56%；绝对变化区间[+0.004786, +0.010983]。

|目标|对照|实验|绝对变化|相对变化|条件95%区间|
|---|---:|---:|---:|---:|---|
|点击|0.450583|0.450729|+0.000146|+0.03%|[-0.002069, +0.002366]|
|长看|0.393020|0.394448|+0.001429|+0.36%|[-0.001242, +0.003998]|
|点赞|0.261814|0.282387|+0.020572|+7.86%|[+0.015800, +0.025221]|
|关注|0.231706|0.235565|+0.003859|+1.67%|[-0.004391, +0.012357]|
|评论|0.231880|0.243233|+0.011353|+4.90%|[+0.004932, +0.017790]|
|转发|0.228243|0.236833|+0.008590|+3.76%|[-0.002754, +0.024731]|
|无负反馈比例|0.999518|0.999521|+0.000002|—|[-0.000009, +0.000014]|

|外层折|均值绝对变化|
|---|---:|
|0|+0.010474|
|1|+0.006572|
|2|+0.006305|

|种子|均值绝对变化|
|---|---:|
|7|+0.014972|
|17|+0.004025|
|29|+0.003978|

|预设条件|通过|
|---|---|
|primary_conditional_ci_positive|是|
|primary_positive_each_seed|是|
|primary_positive_each_fold|是|
|positive_point_guards|是|
|hate_point_guard|是|

预设条件通过5/5项。

## 训练方法对照：混合后验EMER vs 原EMER

六目标均值：0.299541 → 0.301267，相对+0.58%；绝对变化区间[+0.000268, +0.003124]。

|目标|对照|实验|绝对变化|相对变化|条件95%区间|
|---|---:|---:|---:|---:|---|
|点击|0.450583|0.450898|+0.000315|+0.07%|[-0.000968, +0.001615]|
|长看|0.393020|0.397073|+0.004054|+1.03%|[+0.002584, +0.005538]|
|点赞|0.261814|0.265463|+0.003648|+1.39%|[+0.000976, +0.006288]|
|关注|0.231706|0.230080|-0.001627|-0.70%|[-0.005903, +0.002507]|
|评论|0.231880|0.235604|+0.003724|+1.61%|[-0.000711, +0.008086]|
|转发|0.228243|0.228485|+0.000242|+0.11%|[-0.004588, +0.005271]|
|无负反馈比例|0.999518|0.999517|-0.000001|—|[-0.000008, +0.000006]|

|外层折|均值绝对变化|
|---|---:|
|0|+0.000000|
|1|+0.001584|
|2|+0.003602|

|种子|均值绝对变化|
|---|---:|
|7|-0.000077|
|17|+0.001585|
|29|+0.003670|

## 选模与证据边界

两项要求的比较同时通过：是。这不自动等价于独立确认或线上收益。
18个模型槽位中有8个使用epoch0原EMER。完整记录和逐目标结果见evaluation.json。
单项守护允许最多0.005绝对下降；通过守护并不代表每个目标都提升。应结合上表查看长看等目标的实际退化。

|折|种子|模型|选中轮次|
|---|---|---|---:|
|0|7|emer|0|
|0|7|emer_jev|10|
|0|17|emer|0|
|0|17|emer_jev|10|
|0|29|emer|0|
|0|29|emer_jev|15|
|1|7|emer|0|
|1|7|emer_jev|2|
|1|17|emer|1|
|1|17|emer_jev|1|
|1|29|emer|0|
|1|29|emer_jev|0|
|2|7|emer|8|
|2|7|emer_jev|2|
|2|17|emer|0|
|2|17|emer_jev|0|
|2|29|emer|4|
|2|29|emer_jev|4|

This training recipe is an EMER adaptation; changes shared by both arms cannot be attributed to Jev.
The cohort and its outer results were already explored. New results remain exploratory even if all conditions pass.
Conditional bootstrap does not capture all training variation; folds overlap in their training users.
No claim of new-date confirmation, strict all-objective dominance, or online business uplift.
The blend was proposed from already-observed inner guard failures after both earlier outer reports were opened; this is another exploratory test.
This is a new deployment/selection policy, not a correction that changes earlier registered results.
All previous outer data and results were already explored; these estimates are exploratory and need independent confirmation.
Reported benefits are conditional offline ranking estimates, not online causal uplift or universal training uncertainty.
A fallback has exactly zero Jev increment for that model slot; no zero is counted as a strict positive gain.

全部检查点和内层选模通过审计，外层预测、指标和区间在独立进程精确重放。原用户留出结果与Apr23–25未打分独立队列保持原样。

匹配回退策略在9个Jev槽位中保留5个Jev检查点，其余4个使用当前匹配无Jev模型。全部选择仅由冻结的内层指标决定，并在新外层计算之前冻结；详情见policy.json。
