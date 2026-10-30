// AI 410 marketing video: records each scene as numbered JPEG frames (30 fps).
// Drives headless Chrome over the DevTools protocol against the public student site,
// so every scene can be re-recorded after the site changes.
//
// Usage: node --experimental-websocket record.mjs <frames-dir> [site-url]
// Then:  ./build.sh <frames-dir> <out.mp4>
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync, copyFileSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const OUT = path.resolve(process.argv[2] || 'frames');
const SITE = (process.argv[3] || 'https://cityuseattle.github.io/ai410').replace(/\/$/, '');
const HERE = path.dirname(fileURLToPath(import.meta.url));
const FPS = 30, W = 1280, H = 720, DPR = 1.5;           // 1920x1080 output
const sleep = ms => new Promise(r => setTimeout(r, ms));

rmSync(OUT, { recursive: true, force: true });
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new', '--disable-gpu', '--hide-scrollbars', '--remote-debugging-port=9335',
  `--user-data-dir=${OUT}-profile`, 'about:blank'], { stdio: 'ignore' });
let targets;
for (let i = 0; i < 50; i++) { try { targets = await (await fetch('http://127.0.0.1:9335/json/list')).json(); break; } catch { await sleep(200); } }
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
await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'dark' }, { name: 'prefers-reduced-motion', value: 'reduce' }] });

// ---- frame helpers ----
let dir, n = 0;
function scene(name) { dir = `${OUT}/${name}`; mkdirSync(dir, { recursive: true }); n = 0; }
const file = k => `${dir}/f${String(k).padStart(5, '0')}.jpg`;
async function frame() {
  const r = await send('Page.captureScreenshot', { format: 'jpeg', quality: 90 });
  writeFileSync(file(++n), Buffer.from(r.result.data, 'base64'));
}
async function hold(sec) { await sleep(120); await frame(); for (let k = 1; k < Math.round(sec * FPS); k++) copyFileSync(file(n), file(++n)); }
const ease = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
async function scrollTo(y, sec) {
  const from = await js('scrollY'), steps = Math.round(sec * FPS);
  for (let k = 1; k <= steps; k++) { await js(`window.scrollTo(0, ${from + (y - from) * ease(k / steps)}); 1`); await frame(); }
}
// Page y of an element, leaving room for the site's sticky header and section nav.
const yOf = (sel, pad = 140) => js(`(function(){var e=document.querySelector(${JSON.stringify(sel)});return Math.max(0,e.getBoundingClientRect().top+scrollY-(${pad}));})()`);
// Load a page with a clean localStorage (so guided tours start at step 1), waiting for the
// load event so the browser's scroll restoration can't undo our first scroll.
async function load(url) { const done = new Promise(r => { onLoad = r; }); await send('Page.navigate', { url }); await done; await sleep(900); }
async function go(url) {
  await load(url);
  await js('try{localStorage.clear()}catch(e){}; 1');
  await load(url + (url.includes('?') ? '&' : '?') + 'v=' + Date.now());
}
// Page y of the element whose own text includes `text`.
const yOfText = (text, pad = 140) => js(`(function(){var all=[].slice.call(document.querySelectorAll('h2,h3,p,div'));var e=all.filter(function(x){return x.textContent.indexOf(${JSON.stringify(text)})>=0&&x.children.length<4;}).pop();return e?Math.max(0,e.getBoundingClientRect().top+scrollY-(${pad})):scrollY;})()`);
// Burned-in caption bar in CityU colors (Marine with a Blood Orange edge).
async function caption(text) {
  await js(`(function(){
    var l=document.createElement('link'); l.rel='stylesheet'; l.href='https://fonts.googleapis.com/css2?family=Open+Sans:wght@600&display=swap'; document.head.appendChild(l);
    var c=document.getElementById('vid-cap')||document.body.appendChild(Object.assign(document.createElement('div'),{id:'vid-cap'}));
    c.textContent=${JSON.stringify(text)};
    c.style.cssText='position:fixed;left:50%;bottom:40px;transform:translateX(-50%);z-index:99999;width:max-content;max-width:1100px;background:rgba(0,45,92,0.95);color:#fff;font:600 30px/1.35 "Open Sans",sans-serif;padding:16px 28px;border-left:8px solid #F63E02;border-radius:10px;box-shadow:0 8px 32px rgba(0,0,0,.35);text-align:left';
  })(); 1`);
  await sleep(600);
}
const click = sel => js(`(function(){var e=document.querySelector(${JSON.stringify(sel)}); if(e){e.click(); return 1;} return 0;})()`);

// ---- scenes (durations add up to about 1:45) ----
scene('01-hook');                                      // 8 s
await go(`${SITE}/index.html`);
await caption('Build a real AI app in your first module.');
await hold(1.2);
await scrollTo(820, 5);
await hold(1.8);

scene('02-preview');                                   // 14 s
await go(`${SITE}/module-1/simulator.html`);
await js(`window.scrollTo(0, ${await yOf('#tour', 90)}); 1`);
await caption('See it working before the lecture.');
await hold(2);
await click('.tour-setup'); await hold(1.2);
await scrollTo(await yOfText('Watch the loop', 90), 1.2);
for (let k = 0; k < 4; k++) { await click('#step'); await hold(1.9); }
await hold(Math.max(0.2, 14 - n / FPS));

scene('03-break');                                     // 18 s
await go(`${SITE}/module-1/simulator.html`);
await click('.tour-next');
await js(`window.scrollTo(0, ${await yOf('#tour', 90)}); 1`);
await caption('Learn to catch AI when it’s wrong.');
await hold(3);
await click('.tour-setup'); await hold(1.2);
const verdict = await js(`(function(){var all=[].slice.call(document.querySelectorAll('div,p'));var v=all.filter(function(e){return /Hallucination risk/.test(e.textContent)&&e.children.length<4;}).pop();return v?Math.max(0,v.getBoundingClientRect().top+scrollY-360):scrollY;})()`);
await scrollTo(verdict, 2.5);
await hold(Math.max(0.2, 18 - n / FPS));

scene('04-build');                                     // 15 s
await go(`${SITE}/module-1/hos-guide.html`);
await caption('Build it in GitHub Codespaces. Free tools, no credit card.');
await hold(1.5);
await scrollTo(await yOf('#before'), 3.5); await hold(2.5);
await scrollTo(await yOf('#create'), 3.5);
await hold(Math.max(0.2, 15 - n / FPS));

scene('05-explain');                                   // 17 s
await go(`${SITE}/module-1/hos-guide.html`);
await js(`window.scrollTo(0, ${await yOf('#analyze')}); 1`);
await caption('AI drafts it. You prove you understand it.');
await hold(3);
await scrollTo(await yOf('#refine'), 1.5); await hold(3);
await go(`${SITE}/module-1/index.html`);
await js(`window.scrollTo(0, ${await yOf('#mp1', 330)}); 1`);
await caption('AI drafts it. You prove you understand it.');
await hold(1);
const typed = 'Lecture Part 3, step 5: why does the backend resend the whole history with the function result?';
for (let k = 1; k <= typed.length; k += 2) {
  await js(`(function(){var t=document.getElementById('mp2')||document.getElementById('mp1'); t.value=${JSON.stringify(typed)}.slice(0,${k}); t.dispatchEvent(new Event('input',{bubbles:true})); return 1;})()`);
  await frame();
}
await hold(Math.max(0.2, 17 - n / FPS));

scene('06-project');                                   // 20 s
await go(`${SITE}/module-5/example-project.html`);
await js(`window.scrollTo(0, ${await yOf('#demo', 70)}); 1`);
await caption('Combine everything into an app of your own.');
await hold(2);
await scrollTo(await yOf('#demo', -230), 1.2);
for (let k = 0; k < 10; k++) { await click('#next'); await hold(1.5); }
await hold(Math.max(0.2, 20 - n / FPS));

scene('07-close');                                     // 13 s
await send('Page.navigate', { url: 'file://' + path.join(HERE, 'closing-card.html') }); await sleep(2500);
await hold(13);

ws.close(); chrome.kill(); await sleep(1500);
try { rmSync(`${OUT}-profile`, { recursive: true, force: true, maxRetries: 5 }); } catch {}
console.log('recorded scenes into', OUT);
