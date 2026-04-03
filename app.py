#!/usr/bin/env python3
"""競馬GB指数 予測Webアプリ - Render.com対応版"""

import os, json, re, pickle, io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, 'models')
TRAINED_DIR= os.path.join(BASE, 'trained')
os.makedirs(TRAINED_DIR, exist_ok=True)

def to_num(x):
    s = re.sub(r'[▲△◇○□☆★\s]', '', str(x))
    try: return float(s)
    except: return np.nan

def to_rank(x):
    try: return int(float(str(x).strip()))
    except: return 99

CONFIGS = {'turf': 'turf_v2_config.json', 'dart': 'dart_v2_config.json'}
TRAINS  = {'turf': 'turf_v2_train.npz',   'dart': 'dart_v2_train.npz'}
_models = {}

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
    if mtype in _models:
        return _models[mtype]
    pkl = os.path.join(TRAINED_DIR, f'{mtype}_model.pkl')
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            _models[mtype] = pickle.load(f)
        print(f'[{mtype}] loaded')
        return _models[mtype]
    print(f'[{mtype}] training...')
    cfg  = load_config(mtype)
    data = np.load(os.path.join(MODELS_DIR, TRAINS[mtype]))
    p    = cfg['model_params']
    gb   = GradientBoostingClassifier(
        n_estimators=p['n_estimators'], max_depth=p['max_depth'],
        learning_rate=p['learning_rate'], min_samples_leaf=p['min_samples_leaf'],
        subsample=p['subsample'], max_features=p['max_features'],
        random_state=p['random_state'], verbose=0
    )
    gb.fit(data['X'], data['y'])
    with open(pkl, 'wb') as f: pickle.dump(gb, f)
    _models[mtype] = gb
    print(f'[{mtype}] done')
    return gb

def build_vec(row, cfg, le_dict):
    med      = cfg['medians']
    num_cols = cfg['num_cols']
    cat_cols = cfg['cat_cols']
    def g(c): v=to_num(row.get(c,'')); return v if not np.isnan(v) else med.get(c,0.0)

    mw=g('馬体重'); mwc=g('馬体重増減'); kw=g('斤量'); pkw=g('前走斤量')
    dist=g('距離'); pdist=g('前距離'); puri3=g('前走上り3F順')
    p_ninki=g('前走人気'); odds_f=g('単勝オッズ'); ninki_f=g('人気')

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
        try: cv.append(float(le.transform([str(row.get(c,''))])[0]))
        except: cv.append(0.0)

    return np.nan_to_num(np.array(num_vals+[prev_r]+ev+cv, dtype=np.float32), nan=0.0)

def to_rec(score, is_shin):
    if is_shin:    return '× 新馬'
    if score>=0.18: return '★ バリューベット'
    if score>=0.16: return '○ 対抗候補'
    if score>=0.15: return '△ 検討'
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
            '馬番': h.get('馬番',''), '馬名': h.get('馬名',''),
            '人気': h.get('人気',''), '単勝オッズ': h.get('単勝オッズ',''),
            'GBスコア': s, '予測勝率': f'{s*100:.1f}%',
            '推奨': to_rec(s, str(h.get('クラス名',''))=='新馬'),
        })
    results.sort(key=lambda x: -x['GBスコア'])
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify({'status': 'ok'})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data   = request.get_json()
        horses = data.get('horses', [])
        if not horses: return jsonify({'error': '馬データがありません'}), 400
        return jsonify({'results': run_predict(horses, data.get('model','dart'))})
    except Exception as e:
        import traceback; return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/predict_csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'ファイルがありません'}), 400
        raw   = request.files['file'].read()
        mtype = request.form.get('model', 'dart')
        df    = None
        for enc in ['utf-8-sig','utf-8','cp932','shift_jis']:
            try: df = pd.read_csv(io.BytesIO(raw), encoding=enc); break
            except: continue
        if df is None: return jsonify({'error': 'CSV読み込み失敗'}), 400

        for col, key in [('単勝オッズ','odds_input'),('人気','ninki_input')]:
            v = request.form.get(key, '')
            if v:
                vals = [x.strip() for x in v.split(',')]
                if len(vals) == len(df): df[col] = vals

        results = run_predict(df.to_dict(orient='records'), mtype)
        return jsonify({'results': results, 'count': len(results)})
    except Exception as e:
        import traceback; return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

# 起動時学習（gunicorn/直接起動どちらでも動く）
for _m in CONFIGS:
    try: get_model(_m)
    except Exception as _e: print(f'[{_m}] preload skip: {_e}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'起動: http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
