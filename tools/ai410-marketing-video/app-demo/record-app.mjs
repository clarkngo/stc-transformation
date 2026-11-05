// AI 410 app-demo video: records the HOS 4 reference app running live, scene by scene,
// as numbered JPEG frames (30 fps). Every answer is a real call to the running app;
// only the time spent waiting for each answer is shortened.
//
// Needs the app running locally (API :8000, worker, web chat :8501, Expo web :8081), with
// the rental sheet not yet added: run ./reset-app.sh <app-dir> first.
// Usage: node --experimental-websocket record-app.mjs <frames-dir> <app-dir>
// Then:  ../build.sh <frames-dir> <out.mp4>
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync, copyFileSync, rmSync, readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const OUT = path.resolve(process.argv[2] || 'frames');
const APP = path.resolve(process.argv[3] || '../../../reference/ai410-hos4-app');
const HERE = path.dirname(fileURLToPath(import.meta.url));
const API = 'http://localhost:8000', WEB = 'http://localhost:8501', PHONE = 'http://localhost:8081';
const FPS = 30, W = 1280, H = 720, DPR = 1.5, KEEP_WAIT = 1.2;  // seconds of "thinking" kept per answer
const sleep = ms => new Promise(r => setTimeout(r, ms));
const TOOL_LOG = path.join(APP, 'backend', 'tool-calls.log');
const NEW_DOC = path.join(APP, 'new_docs', 'gear-rentals.md');

for (const u of [`${API}/docs`, WEB, PHONE]) {
  const r = await fetch(u).catch(() => null);
  if (!r || !r.ok) { console.error('not running:', u); process.exit(1); }
}
copyFileSync(path.join(HERE, 'phone-stage.html'), path.join(APP, 'mobile', 'public', 'phone-stage.html'));

rmSync(OUT, { recursive: true, force: true });
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new', '--disable-gpu', '--hide-scrollbars', '--remote-debugging-port=9336',
  `--user-data-dir=${OUT}-profile`, 'about:blank'], { stdio: 'ignore' });
let targets;
for (let i = 0; i < 50; i++) { try { targets = await (await fetch('http://127.0.0.1:9336/json/list')).json(); break; } catch { await sleep(200); } }
const ws = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r));
let id = 0; const pending = new Map(); let onLoad = null;
ws.addEventListener('message', e => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Page.loadEventFired' && onLoad) { onLoad(); onLoad = null; }
});
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const js = async expr => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: DPR, mobile: false });
await send('Page.enable');
await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'light' }] });

// ---- frame helpers ----
let dir, n = 0; const marks = {};
function scene(name) { dir = `${OUT}/${name}`; mkdirSync(dir, { recursive: true }); n = 0; marks[name] = {}; }
const mark = label => { marks[path.basename(dir)][label] = +(n / FPS).toFixed(2); };
const file = k => `${dir}/f${String(k).padStart(5, '0')}.jpg`;
async function frame() {
  const r = await send('Page.captureScreenshot', { format: 'jpeg', quality: 90 });
  writeFileSync(file(++n), Buffer.from(r.result.data, 'base64'));
}
async function hold(sec) { await sleep(120); await frame(); for (let k = 1; k < Math.round(sec * FPS); k++) copyFileSync(file(n), file(++n)); }
const holdTo = sec => hold(Math.max(0.2, sec - n / FPS));
async function load(url) { const done = new Promise(r => { onLoad = r; }); await send('Page.navigate', { url }); await done; await sleep(1200); }
// Wait for a live answer, keeping only the first KEEP_WAIT seconds of the wait on screen.
async function liveWait(check, max = 90) {
  const t0 = Date.now(); let kept = 0;
  while (!(await js(check))) {
    if ((Date.now() - t0) / 1000 > max) throw new Error('timed out waiting for: ' + check);
    if (kept < KEEP_WAIT * FPS) { await frame(); kept++; } else await sleep(200);
  }
  await sleep(500);   // let the answer finish rendering
}
async function type(text) { for (let k = 0; k < text.length; k += 2) { await send('Input.insertText', { text: text.slice(k, k + 2) }); await frame(); } }
const key = async k => { for (const type of ['keyDown', 'keyUp']) await send('Input.dispatchKeyEvent', { type, key: k, code: k, windowsVirtualKeyCode: 13, text: type === 'keyDown' ? '\r' : undefined }); };
const clickAt = async (x, y) => { for (const type of ['mousePressed', 'mouseReleased']) await send('Input.dispatchMouseEvent', { type, x, y, button: 'left', clickCount: 1 }); };

// Caption bar (top, so it never covers the chat box) and side chips for what the app did.
async function caption(text) {
  await js(`(function(){
    var l=document.createElement('link'); l.rel='stylesheet'; l.href='https://fonts.googleapis.com/css2?family=Open+Sans:wght@600;800&display=swap'; document.head.appendChild(l);
    var c=document.getElementById('vid-cap')||document.body.appendChild(Object.assign(document.createElement('div'),{id:'vid-cap'}));
    c.textContent=${JSON.stringify(text)};
    c.style.cssText='position:fixed;left:50%;top:22px;transform:translateX(-50%);z-index:99999;width:max-content;max-width:1100px;background:rgba(0,45,92,0.95);color:#fff;font:600 26px/1.35 "Open Sans",sans-serif;padding:12px 24px;border-left:8px solid #F63E02;border-radius:10px;box-shadow:0 8px 32px rgba(0,0,0,.3)';
  })(); 1`);
  await sleep(500);
}
async function chip(title, body) {
  await js(`(function(){
    var c=document.getElementById('vid-chip')||document.body.appendChild(Object.assign(document.createElement('div'),{id:'vid-chip'}));
    c.innerHTML='<b style="display:block;font:800 13px/1.2 Open Sans,sans-serif;letter-spacing:2px;text-transform:uppercase;color:#F63E02;margin-bottom:8px"></b><span style="white-space:pre-wrap"></span>';
    c.firstChild.textContent=${JSON.stringify(title)}; c.lastChild.textContent=${JSON.stringify(body)};
    c.style.cssText='position:fixed;right:28px;top:150px;z-index:99999;width:250px;background:#fff;color:#002D5C;font:600 17px/1.4 ui-monospace,Menlo,monospace;padding:14px 16px;border:2px solid #002D5C;border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.15)';
  })(); 1`);
}
const lastToolCall = () => existsSync(TOOL_LOG) ? readFileSync(TOOL_LOG, 'utf8').trim().split('\n').pop() : '';

// ---- web chat helpers (Streamlit) ----
const chatCount = `document.querySelectorAll('[data-testid=stChatMessage]').length`;
async function webAsk(question) {
  const before = await js(chatCount);
  await js(`document.querySelector('[data-testid=stChatInputTextArea]').focus(); 1`);
  await type(question); await hold(0.4);
  mark('asked'); await key('Enter');
  // Done when the new reply has text and Streamlit has finished its run.
  await liveWait(`(function(){var m=document.querySelectorAll('[data-testid=stChatMessage]'); if(m.length<${before}+2) return false;
    var md=m[m.length-1].querySelector('[data-testid=stMarkdownContainer]'); return !!md && md.innerText.trim().length>2 && !document.querySelector('[data-testid=stStatusWidget]');})()`);
  await js(`window.scrollTo(0, document.body.scrollHeight); 1`);
  mark('answered');
}
async function openWeb() {
  await load(WEB);
  await js(`(function(){var s=document.createElement('style'); s.textContent='[data-testid=stHeader],[data-testid=stToolbar],[data-testid=stDecoration]{display:none!important}'; document.head.appendChild(s);})(); 1`);
}

// ---- scenes ----
scene('01-hook');                                      // 7 s
await load('file://' + path.join(HERE, 'hook-card.html'));
await hold(7);

scene('02-grounded');                                  // 17 s
await openWeb();
await caption('It answers from the store’s own documents.');
await hold(1.5);
await webAsk('Can I return a tent I have used once?');
await holdTo(17);

scene('03-tool');                                      // 15 s
await openWeb();
await caption('It decides when to use a tool.');
await hold(0.8);
const callsBefore = lastToolCall();
await webAsk('I’m buying a 189-dollar tent and two 64-dollar backpacks. What’s my total with 10% sales tax?');   // no $: Streamlit reads $…$ as math
const call = lastToolCall();
if (call && call !== callsBefore) {
  const [name, args] = call.split('\t'); const expr = JSON.parse(args).expression || args;
  await chip('Tool call', `${name}(\n  ${expr}\n)`);
  mark('tool');
} else console.warn('WARNING: no new tool call logged for scene 3');
await holdTo(15);

scene('04-learn');                                     // 26 s
await openWeb();
await caption('It says when it doesn’t know.');
await hold(1);
await webAsk('Can I rent a tent for a weekend trip?');
await hold(2.5);
await caption('Add a new document, and it learns it.');
const job = await (await fetch(`${API}/documents`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ doc_path: NEW_DOC }) })).json();
mark('doc-added');
let status = 'queued';
await chip('Background job', `gear-rentals.md\n${status}`); await hold(0.6);
for (let i = 0; i < 60 && status !== 'finished'; i++) {
  status = (await (await fetch(`${API}/jobs/${job.job_id}`)).json()).status;
  await chip('Background job', `gear-rentals.md\n${status}`); await frame(); await sleep(250);
}
if (status !== 'finished') throw new Error('ingest job did not finish');
await hold(1.2);
await webAsk('Can I rent a tent for a weekend trip?');
await holdTo(26);

scene('05-phone');                                     // 16 s
await load(`${PHONE}/phone-stage.html`);
await sleep(1500);
await hold(1.2);
const box = await js(`(function(){var f=document.querySelector('iframe'), r=f.getBoundingClientRect(), d=f.contentDocument;
  var i=d.querySelector('input[placeholder^="Ask"]'), ir=i.getBoundingClientRect();
  var s=[].slice.call(d.querySelectorAll('div')).filter(function(e){return e.textContent==='Send';}).pop(), sr=s.getBoundingClientRect();
  return {ix:r.left+ir.left+ir.width/2, iy:r.top+ir.top+ir.height/2, sx:r.left+sr.left+sr.width/2, sy:r.top+sr.top+sr.height/2};})()`);
await clickAt(box.ix, box.iy); await sleep(200);
await type('How long does standard shipping take?'); await hold(0.4);
mark('asked'); await clickAt(box.sx, box.sy);
await liveWait(`(function(){var d=document.querySelector('iframe').contentDocument; return [].slice.call(d.querySelectorAll('div')).filter(function(e){return e.textContent.toLowerCase()==='assistant';}).length>=1;})()`);
mark('answered');
await holdTo(16);

scene('06-units');                                     // 11 s
await load('file://' + path.join(HERE, 'units-card.html'));
await hold(11);

scene('07-close');                                     // 13 s
await load('file://' + path.join(HERE, '..', 'closing-card.html')); await sleep(1300);
await hold(13);

ws.close(); chrome.kill(); await sleep(1500);
try { rmSync(`${OUT}-profile`, { recursive: true, force: true, maxRetries: 5 }); } catch {}
writeFileSync(`${OUT}/marks.json`, JSON.stringify(marks, null, 2));
console.log('recorded scenes into', OUT); console.log(JSON.stringify(marks));
