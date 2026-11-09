"""Layer table for an actual forward pass; no extra summary library required."""
import csv
import html
import json
from pathlib import Path


def shape_tree(value):
    if hasattr(value, 'shape'):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [shape_tree(v) for v in value]
    if isinstance(value, dict):
        return {str(k):shape_tree(v) for k,v in value.items()}
    return type(value).__name__


def describe_model(model, sample):
    import torch
    rows, handles = [], []
    modes = [(module, module.training) for module in model.modules()]
    calls = {}

    def record(name):
        def hook(module, inputs, output):
            calls[name] = calls.get(name, 0)+1
            parameters = list(module.named_parameters(recurse=False))
            buffers = list(module.named_buffers(recurse=False))
            rows.append(dict(name=name, call=calls[name], type=type(module).__name__,
                input_shapes=shape_tree(inputs), output_shapes=shape_tree(output),
                learnable_sizes={n:list(p.shape) for n,p in parameters if p.requires_grad},
                frozen_sizes={n:list(p.shape) for n,p in parameters if not p.requires_grad},
                state_sizes={n:list(b.shape) for n,b in buffers},
                parameters=sum(p.numel() for _,p in parameters),
                trainable_parameters=sum(p.numel() for _,p in parameters if p.requires_grad)))
        return hook

    try:
        for name, module in model.named_modules():
            # Leaf modules plus any container that owns parameters directly.
            if name and (not list(module.children()) or list(module.parameters(recurse=False))):
                handles.append(module.register_forward_hook(record(name)))
        model.eval()
        devices = [sample.device.index] if sample.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            output = model(sample)
        report = dict(model_type=type(model).__name__,input_shape=shape_tree(sample),
                      output_shape=shape_tree(output),
                      total_parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      state_elements=sum(b.numel() for b in model.buffers()),layers=rows,
                      note='Shapes are PyTorch batch/channel/height/width. One eval-mode forward pass; no training. '
                           'Rows show executed modules, not functional operations such as torch.cat, residual addition or functional interpolation. '
                           'Repeated calls reuse the same weights; unique model totals must not be inferred by summing all rows.')
        return report
    finally:
        for handle in handles:
            handle.remove()
        # Preserve mixed train/eval modes as well as the root mode.
        for module, mode in modes:
            module.training = mode


def sizes_text(sizes):
    return '\n'.join(f'{name}: {" × ".join(map(str,shape)) if shape else "scalar"}'
                     for name,shape in sizes.items()) or '—'


def table_html(report):
    esc = lambda v:html.escape(str(v))
    headings = ['#','Name','Type','Input shapes','Output / activations','Learnable sizes',
                'Frozen sizes','State sizes','Parameters']
    rows=[]
    entries=[dict(name='input',call=1,type='Input tensor',input_shapes='—',output_shapes=report['input_shape'],
                  learnable_sizes={},frozen_sizes={},state_sizes={},parameters=0)] + report['layers']
    for i,row in enumerate(entries):
        values=[i,row['name']+(f' [call {row["call"]}]' if row['call']>1 else ''),row['type'],
                row['input_shapes'],row['output_shapes'],sizes_text(row['learnable_sizes']),
                sizes_text(row['frozen_sizes']),sizes_text(row['state_sizes']),f'{row["parameters"]:,}']
        rows.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model layer information</title><style>
body{font:14px system-ui;margin:24px;color:#172b40;background:#f5f7fa}h1{font-size:24px}p{line-height:1.6}
.table{overflow:auto;max-height:75vh;background:white;border:1px solid #cbd6e2}table{border-collapse:collapse;width:100%}
th{position:sticky;top:0;background:#456e98;color:white;text-align:left;padding:12px;white-space:nowrap}
td{padding:9px;border-bottom:1px solid #dce3eb;border-right:1px solid #dce3eb;white-space:pre;font:12px ui-monospace,monospace}
tr:nth-child(even){background:#edf3f9}tr:hover{background:#dfeeff}input{padding:10px;width:360px;max-width:100%;margin-bottom:14px}
</style><h1>Model layer information — '''+esc(report['model_type'])+'''</h1><p>Input: '''+esc(report['input_shape'])+''' · Output: '''+esc(report['output_shape'])+'''<br>
Unique parameters: '''+f'{report["total_parameters"]:,}'+''' · Trainable: '''+f'{report["trainable_parameters"]:,}'+'''</p>
<p>'''+esc(report['note'])+'''</p><input id="filter" placeholder="Search layer name or type"><div class="table"><table><thead><tr>'''+''.join('<th>'+esc(h)+'</th>' for h in headings)+'''</tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<script>document.getElementById('filter').addEventListener('input',function(){const q=this.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(q))})</script></html>'''


def save_model_summary(report, destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'model_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (destination/'model_summary.html').write_text(table_html(report),encoding='utf-8')
    fields=['name','call','type','input_shapes','output_shapes','learnable_sizes','frozen_sizes','state_sizes','parameters','trainable_parameters']
    with (destination/'model_summary.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        for row in report['layers']:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    print('\nMODEL LAYER INFORMATION (B, C, H, W)',flush=True)
    print(f'{"Name":52} {"Type":22} {"Output shape":28} {"Parameters":>12}',flush=True)
    for row in report['layers']:
        name=row['name']+(f' #{row["call"]}' if row['call']>1 else '')
        print(f'{name:52} {row["type"]:22} {str(row["output_shapes"]):28} {row["parameters"]:>12,}',flush=True)
    print(f'Unique parameters: {report["total_parameters"]:,}; trainable: {report["trainable_parameters"]:,}',flush=True)
    print(f'Full layer table: {destination/"model_summary.html"}\n',flush=True)
