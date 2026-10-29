#!/usr/bin/env python3
"""Publish all seven goals for the fixed 10K exploratory pilot."""
import argparse
from pathlib import Path
import shutil

from paretojev.utils.io import read_json
from paretojev.utils.provenance import sha256_file


def main(args):
    result = read_json(args.source / "evaluation.json")
    replay = read_json(args.source / "replay.json")
    if (not replay["exact_predictions_and_metrics"] or not replay["all_raw_responses_audited_again"] or
            replay["evaluation_sha256"] != sha256_file(args.source / "evaluation.json") or
            replay["predictions_sha256"] != sha256_file(args.source / "predictions.npz")):
        raise ValueError("Exact pilot replay and raw-response audit required")
    names = {
        "is_click": "点击", "long_view": "长看", "is_like": "点赞", "is_follow": "关注",
        "is_comment": "评论", "is_forward": "转发", "is_hate": "无负反馈比例"
    }
    titles = {"signal": "同配方Jev增量", "original_reference": "相对原EMER参考", "training_control": "无Jev训练方法对照"}
    lines = [
        "# 1万组新日期探索性测试", "",
        f"固定抽样{result['support']['slates']:,}组、{result['support']['candidate_exposures']:,}条候选曝光、{result['support']['users']:,}名用户。",
        "该子集在任何新日期Jev响应或模型指标产生前，按原分层和固定SHA256规则选定；剩余40,000组没有在本轮评价。",
        "全部70,000项真实Jev判断通过producer、请求、候选顺序和原始响应审计。结果是探索性方向检查，不替代原5万组正式确认。"
    ]
    for key, title in titles.items():
        report = result["pooled"][key]
        mean = report["positive_mean"]
        low, high = mean["confidence_interval"]
        lines += ["", f"## {title}", "",
            f"六目标均值：{mean['emer']:.6f} → {mean['emer_jev']:.6f}，相对变化{100*mean['absolute_gain']/mean['emer']:+.2f}%；绝对变化条件95%区间[{low:+.6f}, {high:+.6f}]。", "",
            "|目标|对照|实验|绝对变化|相对变化|条件95%区间|", "|---|---:|---:|---:|---:|---|"]
        for goal, row in report["objectives"].items():
            lo, hi = row["confidence_interval"]
            relative = "—" if goal == "is_hate" else f"{100*row['relative_gain']:+.2f}%"
            lines.append(f"|{names[goal]}|{row['emer']:.6f}|{row['emer_jev']:.6f}|{row['absolute_gain']:+.6f}|{relative}|[{lo:+.6f}, {hi:+.6f}]|")
        if key in result["conditions"]:
            lines += ["", "|冻结条件|满足|", "|---|---|"]
            for name, passed in result["conditions"][key].items():
                lines.append(f"|{name}|{'是' if passed else '否'}|")
    lines += ["", "## 解释边界", "",
        f"两项要求比较同时通过：{'是' if result['screen_passed'] else '否'}。不因本结果追加样本、改模型或改阈值。",
        "本轮打开了原5万组中的固定1万组；后续正式确认必须明确排除这些ID，不能再把完整5万组称为未见测试。",
        "全部模型仍是此前冻结的候选；没有新训练。条件区间仅描述固定模型预测，不覆盖完整训练方差，也不是线上因果收益。", ""]
    lines += result["limitations"]
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("registration.json", "input_preflight.json", "response_audit.json", "evaluation.json", "replay.json"):
        shutil.copy2(args.source / name, args.output / name)
    (args.output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
