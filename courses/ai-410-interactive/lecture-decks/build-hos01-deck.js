const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fi = require("react-icons/fi");
const path = require("path");
const fs = require("fs");
const SKILLS = "/Users/clark/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/96f01b0d-25b3-4f5d-a451-032629840edf/d0551b9f-c46f-4eec-bde6-5f4bd19939c1/skills";
const { applyTheme } = require(path.join(SKILLS, "pptx/scripts/apply_theme.js"));
const LOGO = path.join(SKILLS, "cityu-brand/assets/logo.png");
const OUT = process.argv[2];

// CityU hero scheme: Marine + Blood Orange + Teal
const HEX = { marine: "002D5C", orange: "F63E02", teal: "3BBABD", slate: "3A4A5C", mist: "EEF3F8", line: "C9D4E1", white: "FFFFFF", gray: "6B7A8C" };
const THEME = {
  name: "CityU AI 410",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: { dk1: HEX.marine, lt1: HEX.white, dk2: HEX.slate, lt2: HEX.mist, accent1: HEX.orange, accent2: HEX.teal, accent3: HEX.marine, accent4: HEX.gray, accent5: "8FA3B8", accent6: "1F4E79", hlink: "1F6FB2", folHlink: "5E4B8B" },
};

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625
  pres.title = "AI 410 HOS 1 Lecture: From chat box to agent";
  pres.author = "Clark Ngo";
  pres.company = "City University of Seattle";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  const logoW = 1.25, logoH = logoW * 388 / 1530;
  pres.defineSlideMaster({
    title: "Title",
    background: { color: C.background1 },
    objects: [
      { image: { path: LOGO, x: 0.6, y: 0.5, w: 1.9, h: 1.9 * 388 / 1530, altText: "City University of Seattle" } },
    ],
    placeholders: [],
  });
  pres.defineSlideMaster({
    title: "Content",
    background: { color: C.background1 },
    margin: [0.5, 0.5, 0.6, 0.5],
    objects: [
      { text: { text: "AI 410 · HOS 1 · Foundations & the Agentic Loop", options: { x: 0.5, y: 5.2, w: 6, h: 0.3, fontSize: 10, color: C.accent4, isTextBox: true, margin: 0 } } },
      { image: { path: LOGO, x: 10 - 0.5 - logoW, y: 5.12, w: logoW, h: logoH, altText: "City University of Seattle" } },
      { placeholder: { options: { name: "eyebrow", type: "body", x: 0.5, y: 0.32, w: 9, h: 0.3, fontSize: 12, bold: true, color: C.accent1, margin: 0 }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.6, w: 9, h: 0.7, fontSize: 30, bold: true, color: C.text1, margin: 0, valign: "top", align: "left" }, text: "" } },
    ],
    slideNumber: { x: 7.45, y: 5.2, w: 0.55, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
  });
  pres.defineSlideMaster({
    title: "Close",
    background: { color: C.accent2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.45, w: 8.8, h: 0.7, fontSize: 32, bold: true, color: C.text1, margin: 0, align: "left" }, text: "" } },
    ],
  });

  const I = {
    monitor: await icon(fi.FiMonitor, HEX.marine), server: await icon(fi.FiServer, HEX.white), cpu: await icon(fi.FiCpu, HEX.marine),
    key: await icon(fi.FiKey, HEX.orange), x: await icon(fi.FiX, HEX.white), check: await icon(fi.FiCheck, HEX.white),
    repeat: await icon(fi.FiRepeat, HEX.orange), tool: await icon(fi.FiTool, HEX.marine), play: await icon(fi.FiPlayCircle, HEX.marine),
    help: await icon(fi.FiHelpCircle, HEX.marine), msg: await icon(fi.FiMessageSquare, HEX.marine), code: await icon(fi.FiCode, HEX.marine),
  };
  const card = (slide, o) => slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { rectRadius: 0.12, line: { color: o.line || HEX.line, width: 1 }, fill: { color: o.fill || C.background2 }, ...o.box, objectName: o.name });
  const txt = (slide, t, o) => slide.addText(t, { isTextBox: true, fontSize: 14, color: C.text2, margin: 0, valign: "top", ...o });

  // ---------- 1. Title
  pres.addSection({ title: "Opening" });
  let s = pres.addSlide({ masterName: "Title", sectionTitle: "Opening" });
  txt(s, "AI 410 · HOS 1 LECTURE", { x: 0.6, y: 1.55, w: 8, h: 0.35, fontSize: 14, bold: true, color: C.accent1 });
  txt(s, "From chat box to agent", { x: 0.6, y: 1.95, w: 8.8, h: 0.95, fontSize: 44, bold: true, color: C.text1 });
  txt(s, "Three ideas you'll use all term: the three-piece app, what an LLM actually does, and the agentic loop", { x: 0.6, y: 3.0, w: 7.6, h: 0.75, fontSize: 18, color: C.text2 });
  txt(s, "≈ 7 minutes  ·  Outcome O1  ·  Weeks 1–2", { x: 0.6, y: 4.65, w: 6, h: 0.35, fontSize: 12, color: C.accent4 });
  s.addImage({ data: I.repeat, x: 8.0, y: 3.3, w: 1.4, h: 1.4, transparency: 15, altText: "Loop icon" });
  s.addNotes(
`[About 30 seconds]
Welcome to AI 410. This first lecture is short on purpose: about seven minutes, three ideas.
First, every app you build this term has the same three-piece shape. Second, what a large language model actually does when you send it a message. And third, the one idea that turns a chat box into an agent: the agentic loop.
If something stays fuzzy by the end, good. That's exactly what your Muddiest Point is for. Let's start with the shape of the app.`);

  // ---------- 2. Three pieces
  pres.addSection({ title: "Part 1 · Anatomy" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Part 1 · Anatomy" });
  s.addText("PART 1 · ANATOMY OF A FULL-STACK AI APP", { placeholder: "eyebrow" });
  s.addText("Every app this term has three pieces", { placeholder: "title" });
  const cols = [
    { x: 0.5, ic: I.monitor, h: "Frontend", sub: "Streamlit page now, an Expo phone app in HOS 4", note: "Anyone can inspect it", dark: false },
    { x: 3.65, ic: I.server, h: "Backend", sub: "FastAPI, POST /chat. Holds the API key. Runs the tools.", note: "The only piece with secrets", dark: true },
    { x: 6.8, ic: I.cpu, h: "LLM", sub: "Gemini, through the free API", note: "Sees only what the backend sends", dark: false },
  ];
  cols.forEach((c, i) => {
    card(s, { name: "piece" + i, box: { x: c.x, y: 1.65, w: 2.7, h: 2.85 }, fill: c.dark ? C.text1 : C.background2, line: c.dark ? HEX.marine : HEX.line });
    s.addImage({ data: c.ic, x: c.x + 0.3, y: 1.95, w: 0.6, h: 0.6, altText: c.h + " icon" });
    if (c.dark) s.addImage({ data: I.key, x: c.x + 2.0, y: 1.95, w: 0.45, h: 0.45, altText: "API key icon" });
    txt(s, c.h, { x: c.x + 0.3, y: 2.7, w: 2.2, h: 0.45, fontSize: 22, bold: true, color: c.dark ? C.background1 : C.text1 });
    txt(s, c.sub, { x: c.x + 0.3, y: 3.2, w: 2.2, h: 0.75, fontSize: 14, color: c.dark ? C.background1 : C.text2 });
    txt(s, c.note, { x: c.x + 0.3, y: 4.0, w: 2.2, h: 0.35, fontSize: 12, italic: true, color: c.dark ? C.accent2 : C.accent4 });
  });
  [3.25, 6.4].forEach((x, i) => txt(s, "⇄", { x, y: 2.75, w: 0.35, h: 0.5, fontSize: 24, bold: true, color: C.accent1, align: "center", objectName: "arrow" + i }));
  s.addNotes(
`[About 45 seconds]
Here's the shape of every app you'll build this term.
[Point to Frontend] The frontend collects the user's message and shows the reply. Right now it's a Streamlit page; in HOS 4 it becomes an app on your phone. Remember: anything in the frontend, anyone can inspect.
[Point to Backend] The backend is a FastAPI server with one important route: POST /chat. It receives the message, adds whatever the model needs, calls the model, and runs any tools. It's also the only piece that holds your API key.
[Point to LLM] And the LLM, Google's Gemini, receives text and returns text. It sees only what your backend chooses to send it. Hold on to that last line; it comes back in a minute.`);

  // ---------- 3. Why the key lives in the backend
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Part 1 · Anatomy" });
  s.addText("PART 1 · WORKED EXAMPLE", { placeholder: "eyebrow" });
  s.addText("Why the API key lives in the backend", { placeholder: "title" });
  const rows = [
    { y: 1.6, bad: true, h: "Key pasted into the frontend", flow: "Browser 🔑  →  Gemini", out: "Anyone who opens developer tools can copy the key and spend its quota" },
    { y: 3.2, bad: false, h: "Key kept in the backend", flow: "Browser  →  Backend 🔑  →  Gemini", out: "The app isn't smarter. The secret is just unreachable." },
  ];
  rows.forEach((r, i) => {
    card(s, { name: "case" + i, box: { x: 0.5, y: r.y, w: 9, h: 1.35 }, fill: r.bad ? "FDEEE8" : "E6F6F6", line: r.bad ? "F7B9A1" : "9ED9DA" });
    s.addShape(pres.shapes.OVAL, { x: 0.8, y: r.y + 0.35, w: 0.65, h: 0.65, fill: { color: r.bad ? C.accent1 : C.accent2 }, line: { color: r.bad ? HEX.orange : HEX.teal }, objectName: "badge" + i });
    s.addImage({ data: r.bad ? I.x : I.check, x: 0.92, y: r.y + 0.47, w: 0.41, h: 0.41, altText: r.bad ? "Wrong" : "Right" });
    txt(s, r.h, { x: 1.75, y: r.y + 0.2, w: 3.6, h: 0.4, fontSize: 18, bold: true, color: C.text1 });
    txt(s, r.flow, { x: 1.75, y: r.y + 0.7, w: 4.05, h: 0.4, fontSize: 13, color: C.text2, fontFace: "Courier New" });
    txt(s, r.out, { x: 6.0, y: r.y + 0.25, w: 3.3, h: 0.9, fontSize: 14, color: C.text2, valign: "middle" });
  });
  s.addNotes(
`[About 45 seconds]
Here's why that split matters, with a real kind of mistake.
[Point to the top row] A classmate ships a quick demo that calls Gemini straight from the frontend, with the API key pasted into the page's code. It works. And within an hour of sharing the link, anyone who opens the browser's developer tools can copy that key and spend its quota.
[Point to the bottom row] Moving the call behind a backend endpoint doesn't make the app any smarter. It makes the secret unreachable. That's the first reason the backend exists. The second reason, tools, is where we'll end up in Part 3.`);

  // ---------- 4. Next token
  pres.addSection({ title: "Part 2 · What an LLM does" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Part 2 · What an LLM does" });
  s.addText("PART 2 · WHAT AN LLM ACTUALLY DOES", { placeholder: "eyebrow" });
  s.addText("It predicts the next token, over and over", { placeholder: "title" });
  const toks = ["What's", "342", "times", "87", "?"];
  let tx = 0.5;
  toks.forEach((t, i) => {
    const w = 0.28 + t.length * 0.13;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { rectRadius: 0.08, x: tx, y: 1.75, w, h: 0.5, fill: { color: C.background2 }, line: { color: HEX.line }, objectName: "tok" + i });
    txt(s, t, { x: tx, y: 1.75, w, h: 0.5, fontSize: 16, bold: true, color: C.text1, align: "center", valign: "middle" });
    tx += w + 0.1;
  });
  txt(s, "→ next token?", { x: tx + 0.05, y: 1.75, w: 1.6, h: 0.5, fontSize: 16, bold: true, color: C.accent1, valign: "middle" });
  s.addChart(pres.charts.BAR, [{ name: "Probability", labels: ["29,754", "29,744", "Let", "The"], values: [41, 22, 15, 9] }], {
    x: 0.5, y: 2.45, w: 5.2, h: 2.55, barDir: "bar", chartColors: [HEX.marine],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', dataLabelFontFace: "+mn-lt", dataLabelFontSize: 12, dataLabelColor: HEX.slate,
    catAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 14, catAxisLabelColor: HEX.slate, catAxisOrientation: "maxMin",
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "Illustrative probabilities for the next token", titleFontFace: "+mn-lt", titleFontSize: 12, titleColor: HEX.gray,
  });
  card(s, { name: "takeaway", box: { x: 6.0, y: 2.55, w: 3.5, h: 2.3 } });
  txt(s, "Sampled, not looked up", { x: 6.25, y: 2.75, w: 3.0, h: 0.4, fontSize: 18, bold: true, color: C.text1 });
  txt(s, "The same input can give different outputs. Great for writing, a hazard for software, and why “it worked once” is never a complete test.", { x: 6.25, y: 3.2, w: 3.0, h: 1.5, fontSize: 14 });
  s.addNotes(
`[About 40 seconds]
So what does the model actually do with your message?
[Point to the tokens] It reads a sequence of tokens, which are word pieces, and predicts the next one. Then the next. Over and over until it decides to stop.
[Point to the chart] These numbers are made up, but the idea is real: each next token is sampled from a probability distribution. Notice the right answer isn't guaranteed. A model can be fairly confident and still wrong about arithmetic.
[Point to the box] Because it samples, the same input can produce different outputs. That's lovely for writing and a real hazard for software. It's why HOS 3 adds guardrails, and why "it worked once" is never a complete Evaluate.`);

  // ---------- 5. Context window
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Part 2 · What an LLM does" });
  s.addText("PART 2 · CONTEXT", { placeholder: "eyebrow" });
  s.addText("It only knows what's in this one request", { placeholder: "title" });
  card(s, { name: "window", box: { x: 0.5, y: 1.6, w: 5.2, h: 3.35 }, fill: C.background1, line: HEX.marine });
  txt(s, "One request's context window", { x: 0.75, y: 1.75, w: 4.7, h: 0.35, fontSize: 14, bold: true, color: C.accent1 });
  ["System instruction", "Chat history your backend chose to resend", "Retrieved documents (from HOS 2)", "The user's latest message"].forEach((t, i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { rectRadius: 0.08, x: 0.75, y: 2.2 + i * 0.66, w: 4.7, h: 0.54, fill: { color: i === 3 ? C.text1 : C.background2 }, line: { color: i === 3 ? HEX.marine : HEX.line }, objectName: "layer" + i });
    txt(s, t, { x: 0.95, y: 2.2 + i * 0.66, w: 4.4, h: 0.54, fontSize: 15, bold: i === 3, color: i === 3 ? C.background1 : C.text1, valign: "middle" });
  });
  txt(s, "Not in the box? Then the model doesn't know it. It remembers nothing between requests.", { x: 6.0, y: 1.75, w: 3.5, h: 1.3, fontSize: 18, bold: true, color: C.text1 });
  card(s, { name: "quickcheck", box: { x: 6.0, y: 3.25, w: 3.5, h: 1.7 } });
  s.addImage({ data: I.help, x: 6.2, y: 3.42, w: 0.4, h: 0.4, altText: "Question icon" });
  txt(s, "Quick check", { x: 6.7, y: 3.45, w: 2.6, h: 0.35, fontSize: 14, bold: true, color: C.accent1 });
  txt(s, "Delete the chat-history list in your backend. Does the model still remember the conversation?", { x: 6.2, y: 3.9, w: 3.1, h: 0.95, fontSize: 14 });
  s.addNotes(
`[About 50 seconds]
Here's the most useful sentence in this lecture: the model only knows what's in this one request.
[Point to the box] Everything it "knows" about your conversation is whatever fits in its context window for that single request. The system instruction. The chat history your backend chose to resend. Any documents you retrieved, which is HOS 2. And the user's latest message.
[Point to the right] If it's not in the box, the model doesn't know it. It doesn't remember your last request unless your backend sends it again.
[Pause] Quick check: if you delete the chat-history list in your backend, does the model still remember the conversation? Decide now, then try it in this week's build.`);

  // ---------- 6. Prompt vs context engineering
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Part 2 · What an LLM does" });
  s.addText("PART 2 · TWO KINDS OF ENGINEERING", { placeholder: "eyebrow" });
  s.addText("Prompt engineering vs. context engineering", { placeholder: "title" });
  [
    { x: 0.5, ic: I.msg, h: "Prompt engineering", q: "How should I word the request?", where: "The instruction text", ex: "“Answer in two sentences and cite the policy.”" },
    { x: 5.1, ic: I.code, h: "Context engineering", q: "What should the model see at all?", where: "Backend code that builds each request", ex: "Resend the last 6 turns, attach the 3 most relevant policy chunks, list the tools." },
  ].forEach((c, i) => {
    card(s, { name: "eng" + i, box: { x: c.x, y: 1.6, w: 4.4, h: 3.35 }, fill: i ? C.background2 : C.background1, line: i ? HEX.marine : HEX.line });
    s.addImage({ data: c.ic, x: c.x + 0.3, y: 1.85, w: 0.5, h: 0.5, altText: c.h + " icon" });
    txt(s, c.h, { x: c.x + 0.95, y: 1.88, w: 3.3, h: 0.45, fontSize: 20, bold: true, color: C.text1 });
    txt(s, [{ text: "Asks  ", options: { bold: true, color: HEX.orange } }, { text: c.q }], { x: c.x + 0.3, y: 2.6, w: 3.85, h: 0.6, fontSize: 15 });
    txt(s, [{ text: "Lives in  ", options: { bold: true, color: HEX.orange } }, { text: c.where }], { x: c.x + 0.3, y: 3.25, w: 3.85, h: 0.6, fontSize: 15 });
    txt(s, c.ex, { x: c.x + 0.3, y: 3.95, w: 3.85, h: 0.85, fontSize: 14, italic: true, color: C.accent4 });
  });
  s.addNotes(
`[About 40 seconds]
That gives us two different jobs, and people mix them up.
[Point left] Prompt engineering asks: how should I word the request? It lives in the instruction text. "Answer in two sentences and cite the policy."
[Point right] Context engineering asks a bigger question: what should the model see at all? It lives in your backend code, the code that builds each request. Resend the last six turns. Attach the three most relevant policy chunks. List the tools that are available.
Most of this course is context engineering: tool declarations this HOS, retrieval in HOS 2, and reading traces in HOS 4.`);

  // ---------- 7. The loop
  pres.addSection({ title: "Part 3 · The agentic loop" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Part 3 · The agentic loop" });
  s.addText("PART 3 · THE AGENTIC LOOP", { placeholder: "eyebrow" });
  s.addText("A chat app is one round trip. An agent is a loop.", { placeholder: "title" });
  const nodes = [
    { x: 0.6, y: 1.75, h: "1 · Model decides", b: "Text answer, or a function call?" },
    { x: 3.7, y: 1.75, h: "2 · Function call", b: "A name and arguments. No text yet." },
    { x: 3.7, y: 3.55, h: "3 · Your code runs it", b: "The backend calls the real function" },
    { x: 0.6, y: 3.55, h: "4 · Result goes back", b: "The model reads it and decides again" },
  ];
  nodes.forEach((n, i) => {
    card(s, { name: "node" + i, box: { x: n.x, y: n.y, w: 2.6, h: 1.15 }, fill: i === 2 ? C.text1 : C.background2, line: i === 2 ? HEX.marine : HEX.line });
    txt(s, n.h, { x: n.x + 0.2, y: n.y + 0.15, w: 2.25, h: 0.4, fontSize: 16, bold: true, color: i === 2 ? C.background1 : C.text1 });
    txt(s, n.b, { x: n.x + 0.2, y: n.y + 0.55, w: 2.25, h: 0.5, fontSize: 13, color: i === 2 ? C.background1 : C.text2 });
  });
  [{ x: 3.25, y: 2.1, t: "→" }, { x: 4.85, y: 2.95, t: "↓" }, { x: 3.25, y: 3.9, t: "←" }, { x: 1.75, y: 2.95, t: "↑" }].forEach((a, i) =>
    txt(s, a.t, { x: a.x, y: a.y, w: 0.4, h: 0.5, fontSize: 24, bold: true, color: C.accent1, align: "center", valign: "middle", objectName: "loopArrow" + i }));
  s.addImage({ data: I.repeat, x: 2.2, y: 3.0, w: 0.5, h: 0.5, altText: "Repeat icon" });
  card(s, { name: "exit", box: { x: 6.75, y: 1.75, w: 2.75, h: 2.95 }, fill: C.background1, line: HEX.teal });
  txt(s, "The loop ends", { x: 7.0, y: 1.95, w: 2.3, h: 0.4, fontSize: 18, bold: true, color: C.text1 });
  txt(s, "when the model answers in plain text. That answer goes back to the frontend.", { x: 7.0, y: 2.4, w: 2.3, h: 1.0, fontSize: 14 });
  txt(s, "Your backend also sets a limit, so a confused loop can't run forever.", { x: 7.0, y: 3.45, w: 2.3, h: 1.1, fontSize: 13, italic: true, color: C.accent4 });
  s.addNotes(
`[About 45 seconds]
Now the idea that turns a chat app into an agent.
A plain chat app is one round trip: message in, text out. An agent adds a loop.
[Trace the loop with your cursor] Your backend tells the model which functions exist. The model decides: answer in text, or ask for a function? If it asks, it doesn't reply with text. It replies with a function call: a name and arguments. Your code runs the real function. The result goes back to the model, and it decides again.
[Point to the right] The loop ends when the model answers in plain text. And your backend sets a limit on the number of turns, so a confused loop can't run forever.`);

  // ---------- 8. Step by step
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Part 3 · The agentic loop" });
  s.addText("PART 3 · ONE REQUEST, STEP BY STEP", { placeholder: "eyebrow" });
  s.addText("“What's 342 times 87?”", { placeholder: "title" });
  const steps = [
    "The frontend POSTs the message to /chat",
    "The backend sends it to Gemini with the calculate tool's declaration",
    "Gemini replies with a function call: calculate(\"342*87\"). No text yet.",
    "The backend runs the real Python function: 29754",
    "The backend sends that result back to Gemini, with the history so far",
    "Gemini answers in text: “342 × 87 = 29,754.” The frontend shows it.",
  ];
  steps.forEach((t, i) => {
    const col = i < 3 ? 0 : 1, row = i % 3;
    const x = 0.5 + col * 4.6, y = 1.65 + row * 1.12;
    const hot = i === 3;
    s.addShape(pres.shapes.OVAL, { x, y: y + 0.08, w: 0.55, h: 0.55, fill: { color: hot ? C.accent1 : C.text1 }, line: { color: hot ? HEX.orange : HEX.marine }, objectName: "num" + i });
    txt(s, String(i + 1), { x, y: y + 0.08, w: 0.55, h: 0.55, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
    txt(s, t, { x: x + 0.75, y, w: 3.65, h: 0.9, fontSize: 15, bold: hot, color: hot ? C.text1 : C.text2, valign: "middle" });
  });
  s.addNotes(
`[About 45 seconds]
Let's walk one request all the way through.
[Point to each number as you go] One: the user types "What's 342 times 87?" and the frontend posts it to /chat. Two: the backend sends that message to Gemini, along with the declaration of a calculate tool. Three: Gemini doesn't answer. It replies with a function call: calculate, with the expression 342 times 87.
[Point to step 4] Four is the step to remember. Your backend runs the real Python function and gets 29,754.
Five: the backend sends that result back to Gemini, with the conversation so far. That's the loop, and it can repeat. Six: Gemini answers in text, and the frontend shows it.`);

  // ---------- 9. The model never runs the function
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Part 3 · The agentic loop" });
  s.addText("PART 3 · THE KEY PROPERTY", { placeholder: "eyebrow" });
  s.addText("The model never runs the function. It only asks.", { placeholder: "title" });
  card(s, { name: "codebox", box: { x: 0.5, y: 1.6, w: 5.0, h: 3.35 }, fill: C.text1, line: HEX.marine });
  txt(s, "A tool declaration (simplified)", { x: 0.75, y: 1.75, w: 4.5, h: 0.35, fontSize: 12, bold: true, color: C.accent2 });
  txt(s,
`calculate_tool = {
  "name": "calculate",
  "description": "Evaluate a basic
     arithmetic expression.",
  "parameters": {
    "expression": "e.g. '342*87'"
  }
}`, { x: 0.75, y: 2.15, w: 4.6, h: 2.6, fontSize: 14, color: C.background1, fontFace: "Courier New" });
  s.addImage({ data: I.tool, x: 5.9, y: 1.7, w: 0.55, h: 0.55, altText: "Tool icon" });
  txt(s, "The model sees only this description", { x: 6.6, y: 1.72, w: 2.9, h: 0.6, fontSize: 16, bold: true, color: C.text1, valign: "middle" });
  txt(s, "It reads the name, description and parameters to decide when to ask. Your code decides what actually happens.", { x: 5.9, y: 2.45, w: 3.6, h: 1.1, fontSize: 14 });
  card(s, { name: "judgment", box: { x: 5.9, y: 3.7, w: 3.6, h: 1.25 } });
  txt(s, "So which functions you expose is an engineering judgment. You'll practice it in Application Challenge 1.", { x: 6.1, y: 3.85, w: 3.25, h: 1.0, fontSize: 14, bold: true, color: C.text1 });
  s.addNotes(
`[About 40 seconds]
Notice what the model never does in all of that. It never runs the function. It only asks.
[Point to the code] This is roughly what the model sees: a name, a description, and the parameters it can fill in. That's it. It reads this to decide when to ask. Your code decides what actually happens.
That's what makes tool calling safe to reason about. And it's why deciding which functions to expose is a real engineering judgment, not a detail. Could the model call a "send email" tool? A "refund" tool? You'll practice exactly that call in Application Challenge 1.`);

  // ---------- 10. This week
  pres.addSection({ title: "Closing" });
  s = pres.addSlide({ masterName: "Close", sectionTitle: "Closing" });
  s.addText("Your next steps this week", { placeholder: "title" });
  [
    { ic: I.play, h: "Try the simulator", b: "Step through “342 times 87”, then “Tell me a joke”, then “12 / (3 -”. Watch where each one leaves the loop." },
    { ic: I.help, h: "Muddiest Point, by Wednesday", b: "The one place you got stuck. Specific beats general." },
    { ic: I.msg, h: "Concept Debate", b: "Who should decide when to call a tool: the model, or your backend code?" },
    { ic: I.code, h: "Build it in Codespaces", b: "Scaffold the app with an AI assistant, then test it on purpose." },
  ].forEach((c, i) => {
    const x = 0.6 + (i % 2) * 4.5, y = 1.45 + Math.floor(i / 2) * 1.75;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { rectRadius: 0.12, x, y, w: 4.25, h: 1.5, fill: { color: C.background1 }, line: { color: HEX.white }, objectName: "next" + i });
    s.addImage({ data: c.ic, x: x + 0.25, y: y + 0.25, w: 0.5, h: 0.5, altText: c.h + " icon" });
    txt(s, c.h, { x: x + 0.95, y: y + 0.25, w: 3.15, h: 0.45, fontSize: 17, bold: true, color: C.text1, valign: "middle" });
    txt(s, c.b, { x: x + 0.95, y: y + 0.72, w: 3.15, h: 0.7, fontSize: 13, color: C.text2 });
  });
  txt(s, "AI 410 · HOS 1", { x: 0.6, y: 5.1, w: 4, h: 0.3, fontSize: 10, color: C.text1 });
  s.addNotes(
`[About 40 seconds]
That's the lecture. Here's what to do next.
[Point to each card] Open the Agentic Loop Simulator and step through three messages: "What's 342 times 87," "Tell me a joke," and "Calculate 12 divided by 3 minus," with the expression left unfinished. Watch where each one leaves the loop.
Post your Muddiest Point by Wednesday. The one place you got stuck, as specifically as you can.
Then the Concept Debate: who should decide when to call a tool, the model or your backend code? You'll argue both sides.
And then build it in Codespaces. You'll have an AI assistant scaffold the app, and then you'll test it on purpose. See you in the debate.`);

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
