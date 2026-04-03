#!/usr/bin/env python3
"""競馬GB指数 予測Webアプリ - 学習済みモデル読込版（起動高速）"""

import os, json, re, pickle, io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, 'models')

# ── 設定 ──────────────────────────────────────────────────
CONFIGS = {'turf': 'turf_v2_config.json', 'dart': 'dart_v2_config.json'}
PKLS    = {'turf': 'turf_model.pkl',      'dart': 'dart_model.pkl'}

# ── 起動時に学習済みモデルを読み込む（数秒で完了）──────────
_models = {}
for _m in CONFIGS:
    _pkl = os.path.join(MODELS_DIR, PKLS[_m])
    try:
        with open(_pkl, 'rb') as f:
            _models[_m] = pickle.load(f)
        print(f'[{_m}] モデル読込完了')
    except Exception as e:
        print(f'[{_m}] モデル読込失敗: {e}')

# ── ユーティリティ ────────────────────────────────────────
def to_num(x):
    s = re.sub(r'[▲△◇○□☆★\s]', '', str(x))
    try: return float(s)
    except: return np.nan

def to_rank(x):
    try: return int(float(str(x).strip()))
    except: return 99

def load_config(mtype):
    with open(os.path.join(MODELS_DIR, CONFIGS[mtype]), encoding='utf-8') as f:
        return json.load(f)

def restore_le(cfg):
    le_dict = {}
    for col, classes in cfg['le_classes'].items():
        le = LabelEncoder()
        le.classes_ = np.array(classes)
        le_dict[col] = le
    return le_dict

def get_model(mtype):
    if mtype not in _models:
        raise Exception(f'{mtype}モデルが読み込まれていません')
    return _models[mtype]

# ── 特徴量構築 ────────────────────────────────────────────
def build_vec(row, cfg, le_dict):
    med      = cfg['medians']
    num_cols = cfg['num_cols']
    cat_cols = cfg['cat_cols']
    def g(c):
        v = to_num(row.get(c, ''))
        return v if not np.isnan(v) else med.get(c, 0.0)

    mw=g('馬体重'); mwc=g('馬体重増減')
    kw=g('斤量');   pkw=g('前走斤量')
    dist=g('距離'); pdist=g('前距離')
    puri3=g('前走上り3F順'); p_ninki=g('前走人気')
    odds_f=g('単勝オッズ'); ninki_f=g('人気')

    num_vals = [kw/(mw+1e-6)*100 if c=='斤量体重比' else g(c) for c in num_cols]
    prev_r   = float(to_rank(row.get('前走着順', 99)))

    eng = cfg.get('eng_names', [])
    if '馬体重変化率' in eng:
        ev = [pdist-dist, pkw-kw, mwc/(mw+1e-6), odds_f/(ninki_f+1e-6), puri3-p_ninki]
    else:
        ev = [pdist-dist, pkw-kw, odds_f/(ninki_f+1e-6), puri3-p_ninki]

    cv = []
    for c in cat_cols:
        le = le_dict[c]
        try: cv.append(float(le.transform([str(row.get(c, ''))])[0]))
        except: cv.append(0.0)

    return np.nan_to_num(np.array(num_vals+[prev_r]+ev+cv, dtype=np.float32), nan=0.0)

def to_rec(score, is_shin):
    if is_shin:      return '× 新馬'
    if score >= 0.18: return '★ バリューベット'
    if score >= 0.16: return '○ 対抗候補'
    if score >= 0.15: return '△ 検討'
    return ''

def run_predict(horses, mtype):
    cfg     = load_config(mtype)
    le_dict = restore_le(cfg)
    gb      = get_model(mtype)
    vecs    = [build_vec(h, cfg, le_dict) for h in horses]
    probas  = gb.predict_proba(np.array(vecs))[:, 1]
    results = []
    for h, prob in zip(horses, probas):
        s = round(float(prob), 4)
        results.append({
            '馬番':      h.get('馬番', ''),
            '馬名':      h.get('馬名', ''),
            '人気':      h.get('人気', ''),
            '単勝オッズ': h.get('単勝オッズ', ''),
            'GBスコア':  s,
            '予測勝率':  f'{s*100:.1f}%',
            '推奨':      to_rec(s, str(h.get('クラス名', '')) == '新馬'),
        })
    results.sort(key=lambda x: -x['GBスコア'])
    return results

# ── ルーティング ──────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    ready = {m: m in _models for m in CONFIGS}
    return jsonify({'status': 'ok', 'models': ready})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data   = request.get_json()
        horses = data.get('horses', [])
        if not horses:
            return jsonify({'error': '馬データがありません'}), 400
        return jsonify({'results': run_predict(horses, data.get('model', 'dart'))})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/predict_csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'ファイルがありません'}), 400
        raw   = request.files['file'].read()
        mtype = request.form.get('model', 'dart')
        df    = None
        for enc in ['utf-8-sig', 'utf-8', 'cp932', 'shift_jis']:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except: continue
        if df is None:
            return jsonify({'error': 'CSV読み込み失敗'}), 400

        # オッズ・人気の上書き
        for col, key in [('単勝オッズ', 'odds_input'), ('人気', 'ninki_input')]:
            v = request.form.get(key, '')
            if v:
                vals = [x.strip() for x in v.split(',')]
                if len(vals) == len(df):
                    df[col] = vals

        results = run_predict(df.to_dict(orient='records'), mtype)
        return jsonify({'results': results, 'count': len(results)})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'起動: http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
