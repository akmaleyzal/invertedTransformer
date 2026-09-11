"""Small CPU parity probes against pinned, inspected official model definitions.

Downloads source to the ignored D: task temp directory. Only named class/function
definitions are loaded; no upstream imports, launchers or training scripts run.
The JSON evidence records revisions, source hashes, config and numerical errors.
"""
import ast
import hashlib
import json
import math
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from itransformer_btc.model import ITransformerConfig
from itransformer_btc.baselines import PatchTSTConfig
from itransformer_btc.train import code_sha256

CACHE = ROOT / '.research/temp/upstream-parity'
CACHE.mkdir(parents=True, exist_ok=True)
REVISIONS = {'thuml/iTransformer': 'c2426e68ca13f74aaec08045c5c724d8ad328124',
             'yuqinie98/PatchTST': '204c21efe0b39603ad6e2ca640ef5896646ab1a9'}
evidence = {'prediction_code_sha256': code_sha256(), 'torch': str(torch.__version__),
            'scope': 'synthetic CPU forward parity for the declared configuration; not training or GPU parity',
            'revisions': REVISIONS, 'sources': [], 'checks': []}


def definitions(repo, path, names, namespace):
    url = f'https://raw.githubusercontent.com/{repo}/{REVISIONS[repo]}/{path}'
    local = CACHE / (repo.replace('/', '_') + '_' + Path(path).name)
    if not local.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'iTransformer-research-parity'})
        with urllib.request.urlopen(request, timeout=30) as response:
            local.write_bytes(response.read())
    source = local.read_text(encoding='utf-8')
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), url, 'exec'), namespace)
    evidence['sources'].append({'url': url, 'sha256': hashlib.sha256(local.read_bytes()).hexdigest()})


def compare(name, actual, expected, tolerance=2e-5):
    error = float((actual-expected).abs().max())
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    evidence['checks'].append({'name': name, 'max_abs_error': error, 'tolerance': tolerance})
    print(name, error, flush=True)


def base_namespace():
    return dict(torch=torch, nn=nn, Tensor=Tensor, F=F, np=np, sqrt=math.sqrt,
                math=math, Optional=Optional, Callable=Callable)


torch.set_num_threads(2)
torch.manual_seed(314159)
evidence['probe_seed'] = 314159
ref = base_namespace()
repo = 'thuml/iTransformer'
definitions(repo, 'layers/Embed.py', ['DataEmbedding_inverted'], ref)
definitions(repo, 'layers/SelfAttention_Family.py', ['FullAttention', 'AttentionLayer'], ref)
definitions(repo, 'layers/Transformer_EncDec.py', ['EncoderLayer', 'Encoder'], ref)
definitions(repo, 'model/iTransformer.py', ['Model'], ref)
cfg = ITransformerConfig(dropout=0.)
local = cfg.build().eval()
upstream = ref['Model'](SimpleNamespace(**asdict(cfg), output_attention=False,
    embed='timeF', freq='h', factor=5, activation='gelu', class_strategy='projection')).eval()
upstream.enc_embedding.value_embedding.load_state_dict(local.embedding.projection.state_dict())
upstream.projector.load_state_dict(local.projection.state_dict())
upstream.encoder.norm.load_state_dict(local.final_norm.state_dict())
for a, b in zip(local.layers, upstream.encoder.attn_layers):
    for ours, theirs in [('q','query_projection'), ('k','key_projection'),
                         ('v','value_projection'), ('out','out_projection')]:
        getattr(b.attention, theirs).load_state_dict(getattr(a.attention, ours).state_dict())
    for n in ('norm1', 'norm2'):
        getattr(b, n).load_state_dict(getattr(a, n).state_dict())
    with torch.no_grad():
        for linear, conv in ((a.ffn[0], b.conv1), (a.ffn[3], b.conv2)):
            conv.weight.copy_(linear.weight.unsqueeze(-1)); conv.bias.copy_(linear.bias)
for k in (1, 4, 8, 12):
    x = torch.randn(3, cfg.seq_len, k)
    with torch.no_grad():
        compare(f'iTransformer eval K={k}', local(x), upstream(x, None, None, None)[:, :, 0])

ref = base_namespace()
repo = 'yuqinie98/PatchTST'
prefix = 'PatchTST_supervised/layers/'
definitions(repo, prefix+'PatchTST_layers.py', ['Transpose', 'get_activation_fn', 'positional_encoding'], ref)
definitions(repo, prefix+'RevIN.py', ['RevIN'], ref)
definitions(repo, prefix+'PatchTST_backbone.py', ['PatchTST_backbone', 'Flatten_Head', 'TSTiEncoder',
            'TSTEncoder', 'TSTEncoderLayer', '_MultiheadAttention', '_ScaledDotProductAttention'], ref)
for objective in ('target', 'all'):
    cfg = PatchTSTConfig(loss_channels=objective)
    local = cfg.build()
    upstream = ref['PatchTST_backbone'](8, cfg.seq_len, cfg.pred_len, cfg.patch_len, cfg.stride,
        n_layers=cfg.e_layers, d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=cfg.d_ff,
        dropout=cfg.dropout, attn_dropout=0., norm='BatchNorm', affine=False,
        head_dropout=0., padding_patch=None, res_attention=True, pre_norm=False)
    upstream.backbone.W_P.load_state_dict(local.embedding.state_dict())
    upstream.head.linear.load_state_dict(local.head.state_dict())
    with torch.no_grad():
        upstream.backbone.W_pos.copy_(local.position)
    for a, b in zip(local.layers, upstream.backbone.encoder.layers):
        for ours, theirs in [('q','W_Q'), ('k','W_K'), ('v','W_V')]:
            getattr(b.self_attn, theirs).load_state_dict(getattr(a, ours).state_dict())
        b.self_attn.to_out[0].load_state_dict(a.out.state_dict())
        b.norm_attn[1].load_state_dict(a.norm1.state_dict())
        b.norm_ffn[1].load_state_dict(a.norm2.state_dict())
        b.ff.load_state_dict(a.ffn.state_dict())
    for mode in ('eval', 'train'):
        local.train(mode == 'train'); upstream.train(mode == 'train')
        x = torch.randn(3, cfg.seq_len, 8)
        selected = x[:, :, :1] if objective == 'target' else x
        with torch.no_grad():
            torch.manual_seed(123)
            actual = local.forecast_target(x)
            torch.manual_seed(123)
            expected = upstream(selected.transpose(1,2))[:, 0, :]
        compare(f'PatchTST {mode} objective={objective}', actual, expected)

output = ROOT / '.research/audit-repair-work/upstream_parity.json'
output.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print(f'All {len(evidence["checks"])} parity checks passed; wrote {output}')
