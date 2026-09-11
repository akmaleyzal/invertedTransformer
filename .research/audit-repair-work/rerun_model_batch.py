from pathlib import Path
exec(Path(__file__).with_name('applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])

replace('model.py', '    uniform_attention: bool = False', '    uniform_attention: bool = False\n    final_norm: bool = True')
replace('model.py', '        self.projection = nn.Linear(cfg.d_model, cfg.pred_len)', '        self.final_norm = nn.LayerNorm(cfg.d_model) if cfg.final_norm else nn.Identity()\n        self.projection = nn.Linear(cfg.d_model, cfg.pred_len)')
replace('model.py', '        out = self.projection(h).permute(0, 2, 1)', '        out = self.projection(self.final_norm(h)).permute(0, 2, 1)')
replace('model.py', '            mean = x.mean(dim=1, keepdim=True)', '            mean = x.mean(dim=1, keepdim=True).detach()')
replace('model.py', '            context = v.mean(dim=2, keepdim=True).expand(-1, -1, n, -1)', '''            weights = torch.full((b, self.n_heads, n, n), 1.0/n, device=x.device, dtype=x.dtype)
            if self.capture:
                self.last_weights = weights.detach().mean(dim=1)
            context = self.dropout(weights) @ v''')
replace('model.py', '        self.k = nn.Linear(d_model, d_model)', '''        self.k = nn.Linear(d_model, d_model)
        if uniform:
            self.q.requires_grad_(False)
            self.k.requires_grad_(False)''')

for name in ('DLinearConfig','PatchTSTConfig'):
    c,node=cell_for('baselines.py',name)
    lines=''.join(c['source']).splitlines(keepends=True)
    body=''.join(lines[node.lineno-1:node.end_lineno])
    body=body.replace('loss_channels: str = "all"', 'loss_channels: str = "target"')
    pos=body.index('    def build(')
    body=body[:pos]+'''    lr: float = 1e-3
    max_epochs: int = 120
    patience: int = 12
    lr_halve_every: int = 20

    def schedule(self) -> "TrainSchedule":
        return TrainSchedule(lr=self.lr, max_epochs=self.max_epochs,
                             patience=self.patience, lr_halve_every=self.lr_halve_every)

'''+body[pos:]
    lines[node.lineno-1:node.end_lineno]=[body+'\n']
    c['source']=''.join(lines).splitlines(keepends=True)
    c['metadata']['itbtc'].setdefault('projection_imports',[]).append('from itransformer_btc.train import TrainSchedule')

layer_source = '''
class PatchEncoderLayer(nn.Module):
    """PatchTST post-norm encoder with BatchNorm and residual attention scores.

    Follows the official supervised backbone configuration (GELU, post-norm,
    residual attention); no reuse of the iTransformer layer. Attention dropout
    is zero, residual/FFN dropout is cfg.dropout. RevIN affine is disabled.
    """
    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        self.heads = cfg.n_heads
        self.width = cfg.d_model // cfg.n_heads
        if cfg.d_model % cfg.n_heads:
            raise ValueError("PatchTST d_model must divide into n_heads")
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.k = nn.Linear(cfg.d_model, cfg.d_model)
        self.v = nn.Linear(cfg.d_model, cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm1 = nn.BatchNorm1d(cfg.d_model)
        self.norm2 = nn.BatchNorm1d(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ffn = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, cfg.d_model))

    def forward(self, x: Tensor, previous_scores: Tensor | None = None):
        b, n, d = x.shape
        def heads(layer):
            return layer(x).reshape(b, n, self.heads, self.width).transpose(1, 2)
        q, k, v = heads(self.q), heads(self.k), heads(self.v)
        scores = q @ k.transpose(-2, -1) / self.width**.5
        if previous_scores is not None:
            scores = scores + previous_scores
        context = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, d)
        x = self.norm1((x + self.dropout(self.out(context))).transpose(1, 2)).transpose(1, 2)
        x = self.norm2((x + self.dropout(self.ffn(x))).transpose(1, 2)).transpose(1, 2)
        return x, scores


'''
c,node=cell_for('baselines.py','PatchTST')
lines=''.join(c['source']).splitlines(keepends=True)
lines[node.lineno-1:node.lineno-1]=[layer_source]
c['source']=''.join(lines).splitlines(keepends=True)
replace('baselines.py','        self.layers = nn.ModuleList(EncoderLayer(block) for _ in range(cfg.e_layers))','        self.layers = nn.ModuleList(PatchEncoderLayer(cfg) for _ in range(cfg.e_layers))')
replace('baselines.py','        for layer in self.layers:\n            h = layer(h)\n        out = self.head(h.reshape(b * n, -1)).reshape(b, n, self.cfg.pred_len)', '''        scores = None
        for layer in self.layers:
            h, scores = layer(h, scores)
        out = self.head(h.transpose(1, 2).reshape(b * n, -1)).reshape(b, n, self.cfg.pred_len)''')
# The objective is explicit at both train and checkpoint selection.
replace('train.py','def _mean_loss(model: nn.Module, x: Tensor, y: Tensor, batch: int = 512) -> float:', 'def _mean_loss(model: nn.Module, x: Tensor, y: Tensor, batch: int = 512, target: str = "target") -> float:')
replace('train.py','            model(x[i : i + batch]), y[i : i + batch], reduction="sum"', '            (model.forecast_target(x[i : i + batch]) if target == "target" else model(x[i : i + batch])), y[i : i + batch], reduction="sum"')
replace('train.py','            loss = nn.functional.mse_loss(model(x_tr[idx]), y_tr[idx])','            fitted = model.forecast_target(x_tr[idx]) if loss_target == "target" else model(x_tr[idx])\n            loss = nn.functional.mse_loss(fitted, y_tr[idx])')
replace('train.py','        val = _mean_loss(model, x_va, y_va)', '        val = _mean_loss(model, x_va, y_va, target=loss_target)')
replace('train.py','        "n_parameters": outcome.n_parameters,', '''        "n_parameters": outcome.n_parameters,
        "n_allocated_parameters": sum(p.numel() for p in model.parameters()) if isinstance(model, nn.Module) else outcome.n_parameters,
        "loss_target": cfg.loss_target() if hasattr(cfg, "loss_target") else "target",
        "reached_epoch_cap": bool(outcome.epochs_run and hasattr(cfg, "schedule") and outcome.epochs_run >= cfg.schedule().max_epochs),''')

for c in nb['cells']:
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
        c['outputs']=[];c['execution_count']=None
PATH.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print('Applied rerun architecture/objective corrections; not exported or trained yet')
