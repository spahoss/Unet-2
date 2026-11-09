"""Local live training charts and standalone HTML snapshots, using stdlib only."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>U-Net training progress</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#182438;font:15px system-ui,sans-serif}
header{padding:20px 28px;background:white;border-bottom:1px solid #dfe5ed}h1{margin:0 0 6px;font-size:23px}
header p{margin:0;color:#52637b}main{display:grid;grid-template-columns:minmax(0,1fr) 270px;gap:18px;padding:22px}
.chart,aside{background:white;border:1px solid #dfe5ed;border-radius:10px;padding:16px}.chart{margin-bottom:16px}
h2{font-size:16px;margin:0 0 10px}canvas{display:block;width:100%;height:245px}aside{align-self:start;position:sticky;top:18px}
dl{margin:15px 0}dt{color:#52637b;font-size:13px;margin-top:15px}dd{margin:3px 0;font-weight:600;overflow-wrap:anywhere}
progress{width:100%;height:18px;accent-color:#1976d2}.legend{display:flex;gap:20px;font-size:12px;margin:7px 0;color:#52637b}
.blue{color:#1976d2}.orange{color:#e26b22}.green{color:#14815c}.purple{color:#8752c7}.muted{font-size:12px;color:#52637b;line-height:1.6}
@media(max-width:850px){main{grid-template-columns:1fr}aside{position:static}canvas{height:220px}}
</style>
<header><h1>U-Net training progress</h1><p id="subtitle"></p></header>
<main><section>
<div class="chart"><h2>Pixel accuracy (%)</h2><div class="legend"><span class="blue">Training per batch</span><span class="orange">Validation per epoch</span></div><canvas id="accuracy"></canvas></div>
<div class="chart"><h2>Loss — BCE + Dice loss</h2><div class="legend"><span class="blue">Training per batch</span><span class="orange">Validation per epoch</span></div><canvas id="loss"></canvas></div>
<div class="chart"><h2>Validation segmentation overlap (%)</h2><div class="legend"><span class="green">Dice</span><span class="purple">IoU / Jaccard</span></div><canvas id="overlap"></canvas></div>
</section><aside><h2 id="state">Starting</h2><progress id="progress" value="0" max="1"></progress>
<dl id="details"></dl><p id="connection" class="muted"></p><p class="muted">Training metrics use augmented batches before their weight update. Validation runs in evaluation mode once per epoch. Padding is excluded. Pixel accuracy includes background; use Dice and IoU to assess the mask.</p>
</aside></main>
<script>
const LIVE=__LIVE__;
let data=__DATA__;
const colors={train:'#1976d2',val:'#e26b22',dice:'#14815c',iou:'#8752c7'};
function duration(s){return Math.floor(s/3600)+'h '+Math.floor(s%3600/60)+'m '+Math.floor(s%60)+'s'}
function plot(id,series,percentage){
 const canvas=document.getElementById(id),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);
 const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const w=rect.width,h=rect.height;
 const left=50,right=14,top=12,bottom=35,pw=w-left-right,ph=h-top-bottom;
 const maxX=Math.max(1,data.current_iteration),points=series.flatMap(s=>s.points);
 const maxY=percentage?100:Math.max(0.1,...points.map(p=>p[1]))*1.1;
 const X=v=>left+v/maxX*pw,Y=v=>top+(1-v/maxY)*ph;
 ctx.fillStyle='white';ctx.fillRect(0,0,w,h);
 const b=data.info.batches_per_epoch;
 if(b){ctx.fillStyle='#f0f3f7';for(let i=0;i<maxX;i+=2*b)ctx.fillRect(X(i),top,(Math.min(i+b,maxX)-i)/maxX*pw,ph)}
 ctx.font='11px system-ui';ctx.textAlign='right';
 for(let i=0;i<=5;i++){let v=maxY*i/5,y=Y(v);ctx.strokeStyle='#dfe5ed';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-right,y);ctx.stroke();ctx.fillStyle='#52637b';ctx.fillText(percentage?v.toFixed(0):v.toFixed(2),left-8,y+4)}
 ctx.textAlign='center';for(let i=0;i<=5;i++){let v=maxX*i/5;ctx.fillText(v.toFixed(0),X(v),h-18)}
 ctx.fillText('Training iteration (batch)',left+pw/2,h-2);
 for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.beginPath();s.points.forEach((p,i)=>i?ctx.lineTo(X(p[0]),Y(p[1])):ctx.moveTo(X(p[0]),Y(p[1])));ctx.stroke();
 if(s.markers||s.points.length===1){ctx.fillStyle=s.color;for(const p of s.points){ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),3,0,Math.PI*2);ctx.fill()}}}
}
function render(){
 document.getElementById('subtitle').textContent=data.info.run_name+' | Age: '+(data.info.age===null?'all':data.info.age+'h');
 document.getElementById('state').textContent=data.status;
 const progress=document.getElementById('progress');progress.max=data.info.max_iterations;progress.value=data.current_iteration;
 const elapsed=LIVE&&data.status==='Training'?data.elapsed+Math.max(0,(Date.now()-Date.parse(data.updated_at))/1000):data.elapsed;
 const rows=[['Epoch',data.current_epoch+' / '+data.info.epochs],['Iteration',data.current_iteration+' / '+data.info.max_iterations],['Elapsed',duration(elapsed)],['Device',data.info.device],['Learning rate',data.info.lr],['Batch size',data.info.batch_size],['Input size',data.info.size+' × '+data.info.size],['Latest training accuracy',data.batches.length?(data.batches.at(-1).accuracy*100).toFixed(2)+'%':'—'],['Latest validation Dice',data.epochs.length?(data.epochs.at(-1).val_dice*100).toFixed(2)+'%':'—']];
 const dl=document.getElementById('details');dl.replaceChildren();for(const [key,value]of rows){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=value;dl.append(dt,dd)}
 const batch=(key,scale=1)=>data.batches.map(r=>[r.iteration,r[key]*scale]);const val=(key,scale=1)=>data.epochs.map(r=>[r.iteration,r[key]*scale]);
 plot('accuracy',[{points:batch('accuracy',100),color:colors.train},{points:val('val_accuracy',100),color:colors.val,markers:true}],true);
 plot('loss',[{points:batch('loss'),color:colors.train},{points:val('val_loss'),color:colors.val,markers:true}],false);
 plot('overlap',[{points:val('val_dice',100),color:colors.dice,markers:true},{points:val('val_iou',100),color:colors.iou,markers:true}],true);
}
async function refresh(){try{const response=await fetch('/metrics',{cache:'no-store'});if(!response.ok)throw Error();data=await response.json();document.getElementById('connection').textContent='Live update every second';render()}catch{document.getElementById('connection').textContent='Live connection closed. Open training_progress.html in the run folder for the saved results.'}}
window.addEventListener('resize',render);render();if(LIVE){refresh();setInterval(refresh,1000)}else{document.getElementById('connection').textContent='Saved snapshot — '+data.updated_at}
</script></html>'''


def _page(data, live):
    # Safe embedding even when user-supplied run names include HTML characters.
    encoded = json.dumps(data, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return PAGE.replace('__LIVE__', 'true' if live else 'false').replace('__DATA__', encoded)


class TrainingProgress:
    def __init__(self, destination, info, live=False):
        self.destination = Path(destination)
        self.destination.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.server = None
        self.data = dict(info=info, status='Training', elapsed=0., updated_at='',
                         current_epoch=0, current_iteration=0, batches=[], epochs=[])
        self.save()
        if live:
            progress = self

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path not in ('/', '/metrics'):
                        self.send_error(404)
                        return
                    snapshot = progress.snapshot()
                    body = (json.dumps(snapshot, allow_nan=False) if self.path=='/metrics' else _page(snapshot, True)).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8' if self.path=='/metrics' else 'text/html; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *args):
                    pass

            self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            self.url = f'http://localhost:{self.server.server_port}/'
            print(f'Live training charts: {self.url}', flush=True)

    def snapshot(self):
        from datetime import datetime, timezone
        with self.lock:
            self.data['elapsed'] = time.monotonic()-self.started
            self.data['updated_at'] = datetime.now(timezone.utc).isoformat()
            return json.loads(json.dumps(self.data, allow_nan=False))

    def batch(self, epoch, iteration, loss, accuracy):
        with self.lock:
            self.data['current_epoch'] = epoch
            self.data['current_iteration'] = iteration
            self.data['batches'].append(dict(epoch=epoch, iteration=iteration, loss=loss, accuracy=accuracy))

    def epoch(self, metrics):
        with self.lock:
            self.data['epochs'].append(dict(metrics, iteration=self.data['current_iteration']))
        self.save()

    def save(self):
        data = self.snapshot()
        for name, body in [('training_progress.json', json.dumps(data, indent=2, allow_nan=False)),
                           ('training_progress.html', _page(data, False))]:
            target = self.destination/name
            temp = target.with_suffix(target.suffix+'.tmp')
            temp.write_text(body, encoding='utf-8')
            temp.replace(target)

    def status(self, value):
        with self.lock:
            self.data['status'] = value
        self.save()

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def on_exit(self):
        if self.data['status'] != 'Completed':
            self.status('Stopped before completion')
        self.close()
