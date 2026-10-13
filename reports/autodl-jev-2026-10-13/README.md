# AutoDL Jev 专用实例

用户授权申请新资源，总预算 **47 元**。已创建且通过远端 GPU 检查，
Jev 服务现已通过健康检查，完成336次真实请求；并发分数一致性未过门槛，整批打分被阻止。

|项目|已核验配置|
|---|---|
|实例|`td89w6tsnx-ba43120e` / `JEV-27B-ParetoJev`|
|区域/主机|北京B区 / A19机|
|GPU|RTX PRO 6000 Blackwell Server Edition ×1，97,887 MiB|
|CPU / 内存|25 vCPU / 120GB|
|系统盘 / 数据盘|30GB / 60GB（免费50GB＋付费10GB）|
|基础镜像|PyTorch 2.8.0+cu128、Python 3.12、Ubuntu 22.04|
|驱动|595.71.05|
|GPU 价格|6.98 元/小时|
|附加磁盘|0.0066 元/GB/日，10GB 共0.066元/日，关机后仍计费|
|平台定时关机|**2026-10-14 01:15:00，UTC+08**，已在实例行核验|

定时关机对话框显示预计算力消费 41.02 元。开机不超过约6小时，
为磁盘费用与计费误差预留余量；不是承诺47元能跑完60万次请求。
旧四个实例没有开机、改配置或删除。西北区前两次申请明确返回库存不足，
成功新建的仅此一台北京实例。

## 部署记录

- 经 Chrome 读取控制台，创建、命名、确认平台关机时间；进入新实例 Jupyter 终端。
- `nvidia-smi` 核实型号、显存、驱动；Python 核实 PyTorch/CUDA。
- Hugging Face 直连超时；AutoDL 官方 `source /etc/network_turbo` 后 HTTPS 返回200，未关闭证书验证。
- Chrome 上传入口需要文件 URL 权限，未扩大权限；改在已授权终端输入自有安装脚本。
- 首次脚本粘贴的 Python 换行转义有误，修复并从 Jupyter 编辑器读回完整脚本核对。
- 默认 Aliyun pip 镜像找不到 vLLM 0.31.0；改用官方 PyPI，清除不适用于 pip 的代理。
  加入 `flock` 单进程锁后重新运行，已观察到依赖从 PyPI 正常下载。
- 远端 `/root/bootstrap_jev.sh`；日志 `/root/autodl-tmp/jev-service/setup.log`。
  启动：先清除 shell 代理，再运行 `nohup flock -n /root/jev-bootstrap.lock bash /root/bootstrap_jev.sh`。
- 参考脚本：[bootstrap_jev.sh](../../deployment/autodl/bootstrap_jev.sh)，
  本地版本另内置 pip 前清除代理，避免恢复时依赖调用方环境。

模型固定到 Hugging Face revision `14b425269ed24452e0a5ad5a8f3e51d139647a88`，
vLLM 固定为0.31.0；模型、决策头和校准文件来自同一版本。
计划用 BF16、8,192 上下文、16并发序列起步，待实测后调整吞吐；
官方服务仅监听远端 `127.0.0.1:8000`，尚未对公网开放。

原公共服务的自报 revision 与这个仓库提交不是同一标识。新服务完成前不得修改旧分数身份，
不得直接拼接两批缓存；后续需完成协议适配与同输入对照，再登记新的打分运行。
当前本地基线12次训练已完成，原公共服务打分继续，独立测试未开放。

证据：[实例与预算记录](instance.json)。未保存任何账号密码、密钥或登录令牌。

官方参考：[模型部署](https://huggingface.co/autotrust/JEV-27B)、
[AutoDL 学术加速](https://www.autodl.com/docs/network_turbo/)、
[AutoDL 实例保留规则](https://www.autodl.com/docs/instance_data/)。

## 首轮服务验收（21:28，UTC+08）

- 全部13个权重分片落盘；发布方23个必要文件哈希校验通过后才启动GPU模型。
- vLLM大包下载缓慢，已停止，改用作者公布的PyTorch＋PEFT直接决策头路径，复用Torch 2.8.0+cu128。
- 已完成3.6万组请求的13片上传和压缩/解压双重SHA-256核对，未上传反馈标签或独立测试数据。
- 单并发/8并发/16并发各112个真实请求，吞吐分别约4.18/4.01/4.01次/秒。
- 并发结果相对单请求的最大绝对概率差为0.0370256，高于事前容差0.01；没有放宽门槛，未采集整批训练分数。
- 新缓存与旧公共缓存隔离。待修正批处理、加速后另建运行目录重新验收。
- 本地全部281项测试曾通过，后新增模型哈希测试也通过；这些工程测试不代表真实服务验收通过。
- [实时状态快照](status-2128.json)、[请求包验证](transfer-verified.json)。
