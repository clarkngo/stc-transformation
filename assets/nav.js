(function () {
  // Determine path depth relative to the assets/ directory
  var scripts = document.getElementsByTagName('script');
  var navScript = null;
  for (var i = 0; i < scripts.length; i++) {
    if (scripts[i].src && scripts[i].src.indexOf('nav.js') !== -1) {
      navScript = scripts[i];
      break;
    }
  }

  // assets/nav.js is always at <root>/assets/nav.js.
  // Derive root by going up from the script's src until we reach the root.
  var root = '';
  if (navScript) {
    var src = navScript.getAttribute('src') || '';
    // src may be relative (e.g. "assets/nav.js", "../assets/nav.js", "../../assets/nav.js")
    var depth = (src.match(/\.\.\//g) || []).length;
    for (var d = 0; d < depth; d++) root += '../';
  }

  var nav = document.createElement('nav');
  nav.setAttribute('aria-label', 'Site navigation');
  nav.style.cssText = [
    'font-family:Inter,system-ui,sans-serif',
    'font-size:0.85rem',
    'background:#fff',
    'border-bottom:1px solid #e2e8f0',
    'padding:10px 24px',
    'display:flex',
    'align-items:center',
    'gap:4px',
    'flex-wrap:wrap',
    'position:sticky',
    'top:0',
    'z-index:100',
    'box-shadow:0 1px 4px rgba(15,23,42,0.06)'
  ].join(';');

  var linkStyle = 'color:#1e8449;text-decoration:none;font-weight:500;padding:4px 8px;border-radius:6px';
  var sepStyle = 'color:#d1d5db;padding:0 2px';

  nav.innerHTML = [
    '<a href="' + root + 'index.html" style="' + linkStyle + '">🔄 STC Hub</a>',
    '<span style="' + sepStyle + '">|</span>',
    '<a href="' + root + 'bs-ai-aac-04142026.html" style="' + linkStyle + '">🎓 BSAI Program</a>',
    '<span style="' + sepStyle + '">|</span>',
    '<a href="' + root + 'courses/index.html" style="' + linkStyle + '">📚 Courses</a>',
    '<span style="' + sepStyle + '">|</span>',
    '<a href="' + root + 'strategy/index.html" style="' + linkStyle + '">📊 Strategy</a>',
    '<span style="' + sepStyle + '">|</span>',
    '<a href="' + root + 'stc-strategic-alignment.html" style="' + linkStyle + '">🗺️ Alignment</a>'
  ].join('');

  // Insert as first child of body
  var body = document.body;
  if (body.firstChild) {
    body.insertBefore(nav, body.firstChild);
  } else {
    body.appendChild(nav);
  }
})();
