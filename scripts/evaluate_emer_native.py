#!/usr/bin/env python3
"""Replay selected expanded-development models; never open independent test data.

Validation is reused for selection: intervals are descriptive and conditional on
the fitted models, not confirmatory evidence of generalization after selection.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from paretojev.experiments.comparison import _output_lock, _source_identity
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import sha256_file
from run_emer_jev import dataset, load_model, predict, paired_report
from run_emer_scale import immutable_json


def write_report(folder, result, base, full):
    r = result['results']
    lines = ['# EMER＋Jev 扩大开发集比较', '',
        '训练30,000组，验证6,000组（60,000候选曝光、4,253用户）。'
        '本页是被用于选模型的验证集结果，不是独立测试结果。', '',
        '两臂均43,345参数、3个种子、4组配置、每组30轮；'
        '共同输入、行为模型、标签、初始化种子及选择规则相同。'
        'Jev臂增加真实信号输入，基线将同尺寸Jev输入块置零。', '',
        '前六项目标为NDCG@3；最后一项为前3候选的无负反馈比例，均越高越好。'
        '点估计为三种子均值；区间为2,000次用户聚类成对bootstrap，'
        '仅描述已选模型的差异，未校正同一验证集选模带来的偏差。', '',
        '|目标|EMER|EMER＋Jev|绝对变化|相对变化|绝对变化95%区间|',
        '|---|---:|---:|---:|---:|---|']
    names = dict(is_click='点击', long_view='长看', is_like='点赞', is_follow='关注',
                 is_comment='评论', is_forward='转发', is_hate='无负反馈比例')
    for name, row in r['objectives'].items():
        lo, hi = row['confidence_interval']
        relative = '—' if name == 'is_hate' else f'{100 * row["relative_gain"]:+.2f}%'
        lines.append(f'|{names[name]}|{row["emer"]:.6f}|{row["emer_jev"]:.6f}|'
                     f'{row["absolute_gain"]:+.6f}|{relative}|[{lo:+.6f}, {hi:+.6f}]|')
    mean = r['positive_mean']
    lines += ['', f'六正向目标均值：{mean["emer"]:.6f} → {mean["emer_jev"]:.6f}。', '',
        '|种子|EMER长看|EMER＋Jev长看|绝对变化|', '|---|---:|---:|---:|']
    for seed, row in r['per_seed'].items():
        b, f = row['emer'][1], row['emer_jev'][1]
        lines.append(f'|{seed}|{b:.6f}|{f:.6f}|{f-b:+.6f}|')
    lines += ['', '以下仅把预设门槛应用于开发集作诊断；不能据此宣布正式实验成功。', '',
        '|开发集诊断条件|是否满足|', '|---|---|']
    labels = dict(primary_ci_positive='长看差值区间下限>0', primary_positive_every_seed='每个种子长看均提升',
        positive_mean_noninferior='六正向均值差值区间下限≥0', click_noninferior='点击差值区间下限≥−0.005',
        hate_noninferior='负反馈风险增加区间上限≤0.005')
    for name, passed in r['gates'].items():
        lines.append(f'|{labels[name]}|{"是" if passed else "否"}|')
    lines += ['', '|种子|模型|选中轮次|学习率|先验系数|', '|---|---|---:|---:|---:|']
    for seed in base['selected']:
        for label, selection in [('EMER', base), ('EMER＋Jev', full)]:
            row = selection['selected'][seed]
            lines.append(f'|{seed}|{label}|{row["best_epoch"]}|{row["lr"]}|{row["prior_coefficient"]}|')
    lines += ['', 'Jev信号来自固定版本的自建JEV-27B，25.2万项原始判断已逐条审计；'
        '未拼接旧公共服务分数。行为预测冻结为现有七目标树模型。', '',
        '尚未证明新日期泛化、线上业务提升或完整推荐RT。独立测试窗口仍未开放。', '',
        '数值证据：evaluation.json；重放证据：replay.json。']
    (folder / 'README.md').write_text('\n'.join(lines) + '\n')


def run(a):
    config = read_json(a.config)
    target = read_json(a.target)
    base = read_json(a.common / 'emer_selection.json')
    full = read_json(a.training / 'selection.json')
    registration = full['registration']
    expected = {
        'dataset_config_sha256': a.config,
        'common_manifest_sha256': a.common / 'common_manifest.json',
        'baseline_selection_sha256': a.common / 'emer_selection.json',
        'native_complete_sha256': a.scores / 'complete.json',
        'native_registration_sha256': a.scores / 'registration.json',
        'fit_script_sha256': Path('scripts/run_emer_jev.py'),
    }
    for key, path in expected.items():
        if sha256_file(path) != registration[key]:
            raise ValueError(f'Frozen input differs: {key}')
    if registration['package_sha256'] != _source_identity()['source_sha256']:
        raise ValueError('Package changed after training')
    if full['independent_test_opened'] or registration['public_scores_used']:
        raise ValueError('Unexpected experiment scope')
    if full['baseline_selected'] != base['selected']:
        raise ValueError('Baseline selection differs')
    if registration['training'] != config['training'] or registration['model'] != target['model']:
        raise ValueError('Training budget/model changed')
    for selection, use_jev in ((base, False), (full, True)):
        if len(selection['trials']) != 12:
            raise ValueError('Incomplete equal-budget search')
        for seed in config['training']['seeds']:
            options = [r for r in selection['trials'] if r['seed'] == seed]
            grid = [(lr, coef) for lr in config['training']['learning_rates']
                    for coef in config['training']['prior_coefficients']]
            if [(r['lr'], r['prior_coefficient']) for r in options] != grid:
                raise ValueError('Search grid/order changed')
            for r in options:
                if r['use_jev'] != use_jev or len(r['history']) != config['training']['epochs']:
                    raise ValueError('Training arm/epochs differ')
            best = max(options, key=lambda r: r['validation'])
            selected = selection['selected'][str(seed)]
            if any(selected[k] != best[k] for k in selected):
                raise ValueError('Checkpoint/config selection differs')
    complete = read_json(a.scores / 'complete.json')
    if not complete['complete'] or sha256_file(a.scores / 'scores.npz') != complete['scores_sha256']:
        raise ValueError('Native score file differs')
    manifest = read_json(a.common / 'common_manifest.json')
    path = a.common / 'validation.npz'
    if sha256_file(path) != manifest['files']['validation.npz']:
        raise ValueError('Common validation file differs')
    data = dataset(path)
    with np.load(path, allow_pickle=False) as z, np.load(a.scores / 'scores.npz', allow_pickle=False) as native:
        ids = z['slate_ids']
        if not np.array_equal(z['candidate_ids'], native['candidate_ids'][ids]):
            raise ValueError('Candidate order differs')
        for key, source in [('jev_scores', 'probabilities'), ('jev_entropy', 'entropy'), ('jev_confidence', 'confidence')]:
            data.arrays[key] = torch.from_numpy(native[source][ids].astype('float32'))
    protocol = dict(role='development validation, reused for model selection',
        independent_test_opened=False, confirmatory_success=False,
        baseline_selection_sha256=sha256_file(a.common / 'emer_selection.json'),
        native_selection_sha256=sha256_file(a.training / 'selection.json'),
        validation_sha256=sha256_file(path), scores_sha256=complete['scores_sha256'],
        target_sha256=sha256_file(a.target), evaluator_sha256=sha256_file(Path(__file__)),
        metric_script_sha256=sha256_file(Path('scripts/run_emer_jev.py')),
        interval_caveat='Paired users, conditional on selected fits; does not correct validation selection bias.')
    immutable_json(a.output / 'registration.json', protocol)
    predictions = {}
    for arm, selected, folder in [('emer', base['selected'], a.common),
                                   ('emer_jev', full['selected'], a.training)]:
        for seed, row in selected.items():
            checkpoint = folder / row['path']
            if sha256_file(checkpoint) != row['checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed')
            model = load_model(checkpoint)
            if model.config.use_jev != (arm == 'emer_jev'):
                raise ValueError('Wrong inference arm')
            predictions[f'{arm}/{seed}'] = predict(model, data, config['training']['batch_size'])
    c = dict(config, evaluation=dict(config['evaluation'], **target['effect_gates']))
    report = paired_report(predictions, data, c)
    # Recompute each selected validation score from the saved checkpoint.
    for seed, metrics in report['per_seed'].items():
        for arm, selection in [('emer', base), ('emer_jev', full)]:
            if not np.isclose(np.mean(metrics[arm][:6]), selection['selected'][seed]['validation'], rtol=0, atol=1e-10):
                raise ValueError('Saved validation score does not replay')
    labels = data.arrays['labels'].numpy()
    support = dict(slates=len(data), candidates=int(data.arrays['mask'].sum()),
        users=len(np.unique(data.users)), candidate_positive_counts=labels.sum((0, 1)).astype(int).tolist(),
        slates_with_positive=(labels.sum(1) > 0).sum(0).tolist(),
        weight_kish_effective_slates=float(data.sample_weights.sum() ** 2 / np.square(data.sample_weights).sum()))
    result = dict(protocol=protocol, support=support, results=report,
        interpretation='Development comparison only. Selection on these same 6000 slates invalidates a confirmatory superiority claim.')
    pred_path = a.output / 'predictions.npz'
    if a.replay:
        with np.load(pred_path, allow_pickle=False) as old:
            if set(old.files) != set(predictions) or any(not np.array_equal(old[k], v) for k, v in predictions.items()):
                raise ValueError('Prediction replay differs')
        if read_json(a.output / 'evaluation.json') != result:
            raise ValueError('Metric/interval replay differs')
        atomic_write_json(a.output / 'replay.json', dict(exact_predictions_and_metrics=True,
            predictions_sha256=sha256_file(pred_path), evaluation_sha256=sha256_file(a.output / 'evaluation.json')))
    else:
        if pred_path.exists():
            raise ValueError('Evaluation exists; use --replay')
        np.savez_compressed(pred_path, **predictions)
        atomic_write_json(a.output / 'evaluation.json', result)
        write_report(a.output, result, base, full)
    print('Development gates (not confirmation):', report['gates'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('common', 'training', 'scores', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/emer_jev_scale.json'))
    p.add_argument('--target', type=Path, default=Path('configs/emer_jev_scale_target.json'))
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with _output_lock(args.output), threadpool_limits(limits=4):
        run(args)
