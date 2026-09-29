"""Build a self-contained CartPole rollout viewer from recorded policy steps."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>我到底训练了什么？ · CartPole 策略演示</title>
<style>
:root{color-scheme:light;--ink:#172737;--muted:#596b7d;--line:#dce5eb;--blue:#176ed0;--green:#118366;--bg:#f3f6f9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1160px;margin:auto;padding:32px 24px 56px}h1{font-size:32px;line-height:1.25;margin:7px 0 12px;letter-spacing:-.6px}h2{font-size:20px;margin:0}p{margin:8px 0}.eyebrow{font-weight:700;color:var(--blue);font-size:12px;letter-spacing:1px}.intro{max-width:870px;color:var(--muted);font-size:16px}.notice{background:#fff5da;border:1px solid #efdfa9;padding:12px 16px;border-radius:12px;margin:20px 0;color:#665321}.flow{display:flex;align-items:stretch;gap:12px;margin:24px 0}.flow-block{flex:1;background:white;border:1px solid var(--line);border-radius:12px;padding:13px 16px}.flow-block b{display:block}.flow-block span{font-size:13px;color:var(--muted)}.flow-arrow{align-self:center;color:#91a4b4;font-size:24px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:white;border:1px solid var(--line);border-radius:16px;overflow:hidden;box-shadow:0 4px 18px #18344c05}.card-head{display:flex;align-items:center;justify-content:space-between;padding:18px 20px 10px}.pill{font-size:12px;border-radius:20px;padding:3px 10px;background:#eaf2fc;color:var(--blue)}.pill.ended{background:#edf1f4;color:#637487}.after .pill:not(.ended){background:#e7f5ef;color:var(--green)}canvas{width:100%;height:auto;display:block;aspect-ratio:560/250}.stats{padding:0 20px 18px}.big-stats{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}.metric{font-size:12px;color:var(--muted)}.metric b{display:block;font-size:25px;font-weight:650;color:var(--ink);font-variant-numeric:tabular-nums}.metric b small{font-size:13px;font-weight:400;color:var(--muted)}.prob-labels{display:flex;justify-content:space-between;color:var(--muted);font-size:13px}.prob-bar{display:flex;height:12px;border-radius:8px;overflow:hidden;background:#eaf0f5;margin:6px 0 9px}.prob-left{background:#377ac2}.prob-right{background:#eeae55}.action{font-size:14px;margin-bottom:12px}.action strong{font-size:18px;color:var(--blue)}.state-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;border-top:1px solid var(--line);padding-top:12px}.state-grid div{font-size:11px;color:var(--muted)}.state-grid b{display:block;font-size:13px;color:var(--ink);font-variant-numeric:tabular-nums}.controls{background:white;border:1px solid var(--line);border-radius:14px;padding:16px 20px;margin:18px 0 24px}.buttons{display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,select{font:inherit;border:1px solid #d1dde7;border-radius:8px;background:white;padding:7px 14px;color:var(--ink)}button{cursor:pointer}button.primary{background:var(--blue);color:white;border-color:var(--blue);min-width:96px}button:hover{filter:brightness(.96)}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #9bc9fd;outline-offset:3px}.timeline{display:flex;align-items:center;gap:16px;margin-top:12px}.timeline input{flex:1;accent-color:var(--blue);min-width:0}.timeline output{font-size:13px;font-variant-numeric:tabular-nums;min-width:120px;text-align:right}.control-note{font-size:12px;color:var(--muted);margin:8px 0 0}.bottom{display:grid;grid-template-columns:1fr 1fr;gap:18px}.explain,.evaluation{background:white;border:1px solid var(--line);border-radius:14px;padding:20px}.explain h2,.evaluation h2{font-size:17px;margin-bottom:12px}.explain p{font-size:14px;color:var(--muted)}.explain strong{color:var(--ink)}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:right;padding:9px 10px;border-bottom:1px solid #e8edf1;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}th{font-size:12px;color:var(--muted);font-weight:500}.avg{font-weight:650}footer{font-size:12px;color:var(--muted);margin-top:18px}.legend{font-size:12px;color:var(--muted);margin:6px 0 0}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px;background:#377ac2}.dot.right{background:#eeae55}.speed{font-size:13px;color:var(--muted);margin-left:auto}#training-meta{font-size:12px;color:var(--muted)}@media(max-width:760px){main{padding:22px 14px 40px}h1{font-size:27px}.cards,.bottom{grid-template-columns:1fr}.flow{flex-wrap:wrap;gap:8px}.flow-block{flex-basis:42%;padding:10px}.flow-arrow{display:none}.speed{margin-left:0}.state-grid b{font-size:12px}.timeline output{min-width:100px}}
</style>
</head>
<body>
<main>
<div class="eyebrow">HW2 · 策略梯度 · CARTPOLE</div>
<h1>你训练的是：让小车学会接住杆子</h1>
<p class="intro">杆子往一边倒，小车就需要移动来保持平衡。你写的策略网络读取当前状态，输出“向左推”和“向右推”的概率；训练让能获得更高回报的动作变得更可能。</p>
<div class="notice"><strong>这是你的算法的一次额外演示训练。</strong> <span id="note"></span></div>
<div class="flow" aria-label="一次决策流程">
<div class="flow-block"><b>① 看状态</b><span>车的位置 / 速度<br>杆的角度 / 角速度</span></div><span class="flow-arrow">→</span>
<div class="flow-block"><b>② 策略网络 π(a|s)</b><span>把 4 个数变成<br>左右两个动作的概率</span></div><span class="flow-arrow">→</span>
<div class="flow-block"><b>③ 按概率选动作</b><span>← 向左推小车<br>→ 向右推小车</span></div><span class="flow-arrow">→</span>
<div class="flow-block"><b>④ 环境反馈</b><span>得到新状态与奖励<br>这些经历用于更新网络</span></div>
</div>
<div class="cards" id="cards"></div>
<div class="controls">
<div class="buttons"><button class="primary" id="play" type="button">▶ 播放</button><button id="restart" type="button">↺ 从头重播</button><span id="training-meta"></span><label class="speed">播放速度 <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label></div>
<div class="timeline"><input id="scrubber" type="range" min="0" max="199" value="0" aria-label="同步查看的时间步"><output id="time-label" for="scrubber"></output></div>
<p class="control-note">两边同步到同一个时间步。较短的轨迹结束后会停在最终状态；画面使用真实记录的状态，并非示意动画。</p>
</div>
<div class="bottom">
<section class="explain"><h2>这里该看什么？</h2>
<p><strong>杆子能保持多久？</strong> CartPole 每执行一步获得 +1 奖励，跌倒或小车越界会结束。CartPole-v0 一条轨迹最多 200 步，所以满分为 200。</p>
<p><strong>概率会随状态改变。</strong> 蓝色代表向左，橙色代表向右；箭头是本步实际采样的动作。策略是按概率采样，不一定选择概率较大的一边。</p>
<p><strong>你优化的是整段经历的回报。</strong> advantage 衡量动作表现比预期好多少，用来调整动作概率。baseline 辅助计算 advantage，降低训练噪声，不直接选择动作。</p>
<p>进行时显示动作前状态，累计奖励统计至所选动作完成。轨迹结束后显示最终状态与最后一次动作。</p>
</section>
<section class="evaluation"><h2>换几个初始状态，还能站稳吗？</h2><p class="legend">相同的评估 seed，对比训练前后各一条完整轨迹。</p><table><thead><tr><th>评估 seed</th><th>训练前回报</th><th>训练后回报</th></tr></thead><tbody id="eval-body"></tbody></table><p class="control-note">这里只是 3 个 seed 的小演示，不代替作业的完整训练曲线与正式实验记录。</p></section>
</div>
<footer>此页面离线运行，无需启动服务器。数据来自同目录的 rollouts.json；不会修改你的作业训练日志。</footer>
</main>
<script id="rollout-data" type="application/json">__ROLLOUT_JSON__</script>
<script>
'use strict';
const data = JSON.parse(document.getElementById('rollout-data').textContent);
const keys = ['before', 'after'];
const maxSteps = Math.max(...keys.map(key => data[key].steps.length));
const canvases = {};
let frame = 0, playing = false, previousTime = null, accumulatedMs = 0;
const play = document.getElementById('play');
const scrubber = document.getElementById('scrubber');
scrubber.max = maxSteps - 1;
document.getElementById('note').textContent = data.note || '用于直观看懂策略学到了什么，不是之前正式实验日志的回放。';
const training = data.training || {};
document.getElementById('training-meta').textContent = `${training.iterations ?? '?'} 轮训练 · batch ${training.batch_size ?? '?'} · seed ${training.seed ?? '?'}`;
for (const key of keys) {
  const article = document.createElement('article');
  article.className = `card ${key}`;
  article.innerHTML = `<div class="card-head"><h2 id="${key}-title"></h2><span class="pill" id="${key}-status"></span></div>
    <canvas id="${key}-canvas" width="1120" height="500" aria-label="CartPole 小车与杆子的状态动画"></canvas>
    <div class="stats"><div class="big-stats"><div class="metric">累计奖励<b id="${key}-reward"></b></div><div class="metric">本次完整轨迹回报<b id="${key}-total"></b></div></div>
    <div class="prob-labels"><span id="${key}-left"></span><span id="${key}-right"></span></div>
    <div class="prob-bar" role="img" id="${key}-prob-bar"><div class="prob-left" id="${key}-prob-left"></div><div class="prob-right" id="${key}-prob-right"></div></div>
    <div class="action">实际动作：<strong id="${key}-action"></strong></div>
    <div class="state-grid"><div>位置 x (m)<b id="${key}-x"></b></div><div>车速 (m/s)<b id="${key}-v"></b></div><div>杆角度 (°)<b id="${key}-theta"></b></div><div>角速度 (rad/s)<b id="${key}-omega"></b></div></div></div>`;
  document.getElementById('cards').appendChild(article);
  document.getElementById(`${key}-title`).textContent = data[key].label;
  document.getElementById(`${key}-total`).textContent = data[key].total_reward;
  canvases[key] = document.getElementById(`${key}-canvas`);
  let reward = 0;
  data[key].cumulativeRewards = data[key].steps.map(step => reward += step.reward);
}
const evaluation = data.evaluation || {};
const seeds = evaluation.seeds || [];
function addEvalRow(values, isAverage = false) {
  const row = document.createElement('tr');
  if (isAverage) row.className = 'avg';
  for (const value of values) {
    const td = document.createElement('td');
    td.textContent = value;
    row.appendChild(td);
  }
  document.getElementById('eval-body').appendChild(row);
}
seeds.forEach((seed, i) => addEvalRow([seed, evaluation.before_returns[i], evaluation.after_returns[i]]));
if (seeds.length) {
  const average = values => (values.reduce((a, b) => a + b, 0) / values.length).toFixed(1);
  addEvalRow(['平均', average(evaluation.before_returns), average(evaluation.after_returns)], true);
}
function draw(key, observation, action, ended) {
  const ctx = canvases[key].getContext('2d');
  const [x, , theta] = observation;
  ctx.setTransform(2, 0, 0, 2, 0, 0);
  ctx.clearRect(0, 0, 560, 250);
  const gradient = ctx.createLinearGradient(0, 0, 0, 250);
  gradient.addColorStop(0, '#f9fbfd'); gradient.addColorStop(1, '#edf3f8');
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, 560, 250);
  const ground = 190, center = 280, scale = 85;
  ctx.strokeStyle = '#c8d6e2'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(18, ground); ctx.lineTo(542, ground); ctx.stroke();
  ctx.font = '12px system-ui,sans-serif'; ctx.textAlign = 'center';
  for (const limit of [-2.4, 0, 2.4]) {
    const marker = center + limit * scale;
    ctx.strokeStyle = limit === 0 ? '#d6e0e8' : '#dba99b';
    ctx.setLineDash(limit === 0 ? [3, 5] : [5, 5]);
    ctx.beginPath();ctx.moveTo(marker, 34);ctx.lineTo(marker, ground + 5);ctx.stroke();
    ctx.setLineDash([]);ctx.fillStyle = '#788899';
    ctx.fillText(limit === 0 ? '中心' : `${limit > 0 ? '+' : ''}${limit} m 边界`, marker, 216);
  }
  const cartX = center + x * scale, pivotY = ground - 37;
  ctx.fillStyle = key === 'after' ? '#178369' : '#377ac2';
  ctx.fillRect(cartX - 27, ground - 37, 54, 25);
  ctx.fillStyle = '#32465b';
  for (const offset of [-17, 17]) {ctx.beginPath();ctx.arc(cartX + offset, ground - 8, 7, 0, Math.PI * 2);ctx.fill();}
  const poleLength = 116;
  const tipX = cartX + Math.sin(theta) * poleLength;
  const tipY = pivotY - Math.cos(theta) * poleLength;
  ctx.strokeStyle = '#eda852';ctx.lineWidth = 9;ctx.lineCap = 'round';
  ctx.beginPath();ctx.moveTo(cartX, pivotY);ctx.lineTo(tipX, tipY);ctx.stroke();
  ctx.fillStyle = '#815f32';ctx.beginPath();ctx.arc(cartX, pivotY, 5, 0, Math.PI * 2);ctx.fill();
  const direction = action === 0 ? -1 : 1, arrowY = ground + 42;
  const arrowStart = cartX - direction * 12, arrowEnd = cartX + direction * 28;
  ctx.strokeStyle = action === 0 ? '#377ac2' : '#cf8b2c';ctx.lineWidth = 3;
  ctx.beginPath();ctx.moveTo(arrowStart, arrowY);ctx.lineTo(arrowEnd, arrowY);ctx.moveTo(arrowEnd - direction * 8, arrowY - 6);ctx.lineTo(arrowEnd, arrowY);ctx.lineTo(arrowEnd - direction * 8, arrowY + 6);ctx.stroke();
  ctx.lineCap = 'butt';
  if (ended) {ctx.fillStyle = '#54687a';ctx.font = '12px system-ui,sans-serif';ctx.textAlign = 'right';ctx.fillText('轨迹已结束 · 最终状态', 539, 23);}
}
function update() {
  scrubber.value = frame;
  document.getElementById('time-label').textContent = `第 ${frame + 1} / ${maxSteps} 步`;
  for (const key of keys) {
    const rollout = data[key];
    const i = Math.min(frame, rollout.steps.length - 1);
    const step = rollout.steps[i];
    const ended = frame >= rollout.steps.length - 1;
    const status = document.getElementById(`${key}-status`);
    status.textContent = ended ? `已结束 · ${rollout.steps.length} 步` : `第 ${i + 1} 步`;
    status.classList.toggle('ended', ended);
    document.getElementById(`${key}-reward`).innerHTML = `${rollout.cumulativeRewards[i]} <small>/ 200</small>`;
    const left = step.probabilities[0] * 100, right = step.probabilities[1] * 100;
    document.getElementById(`${key}-left`).textContent = `← 向左 ${left.toFixed(1)}%`;
    document.getElementById(`${key}-right`).textContent = `向右 ${right.toFixed(1)}% →`;
    document.getElementById(`${key}-prob-left`).style.width = `${left}%`;
    document.getElementById(`${key}-prob-right`).style.width = `${right}%`;
    document.getElementById(`${key}-prob-bar`).setAttribute('aria-label', `向左概率 ${left.toFixed(1)}%，向右概率 ${right.toFixed(1)}%`);
    document.getElementById(`${key}-action`).textContent = step.action === 0 ? '← 向左推' : '向右推 →';
    const observation = ended && rollout.final_observation ? rollout.final_observation : step.observation;
    const values = [observation[0], observation[1], observation[2] * 180 / Math.PI, observation[3]];
    ['x', 'v', 'theta', 'omega'].forEach((name, index) => document.getElementById(`${key}-${name}`).textContent = values[index].toFixed(3));
    draw(key, observation, step.action, ended);
  }
}
function setPlaying(value) {playing = value;play.textContent = playing ? 'Ⅱ 暂停' : '▶ 播放';previousTime = null;accumulatedMs = 0;}
play.addEventListener('click', () => {if (!playing && frame === maxSteps - 1) {frame = 0;update();}setPlaying(!playing);});
document.getElementById('restart').addEventListener('click', () => {frame = 0;update();setPlaying(true);});
scrubber.addEventListener('input', () => {frame = Number(scrubber.value);accumulatedMs = 0;update();if (frame === maxSteps - 1) setPlaying(false);});
function tick(time) {
  if (playing) {
    if (previousTime !== null) accumulatedMs += Math.min(time - previousTime, 100) * Number(document.getElementById('speed').value);
    // CartPole advances by 0.02 simulated seconds per action.
    const advance = Math.floor(accumulatedMs / 20);
    if (advance > 0) {accumulatedMs -= advance * 20;frame = Math.min(frame + advance, maxSteps - 1);update();if (frame === maxSteps - 1) setPlaying(false);}
    previousTime = time;
  }
  requestAnimationFrame(tick);
}
update();requestAnimationFrame(tick);
</script>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/policy_demo/rollouts.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/policy_demo/index.html')
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    for key in ('before', 'after'):
        if not data[key]['steps']:
            raise ValueError(f'{key} rollout has no steps')
        for step in data[key]['steps']:
            if len(step['observation']) != 4 or len(step['probabilities']) != 2:
                raise ValueError('Expected 4-dimensional CartPole states and 2 action probabilities')
    # Escape characters that could terminate the JSON script element in HTML.
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False)
    serialized = serialized.replace('&', r'\u0026').replace('<', r'\u003c').replace('>', r'\u003e')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace('__ROLLOUT_JSON__', serialized), encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
