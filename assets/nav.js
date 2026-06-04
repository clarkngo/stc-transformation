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

  // ── Section Navigator HUD ─────────────────────────────────────────────────
  // Only activate when the page has at least 2 navigable stops.
  function buildSectionNav() {
    function getTargets() {
      var hero     = document.querySelector('.hero, .doc-hero');
      var sections = Array.from(document.querySelectorAll('.section-block'));
      var all = hero ? [hero].concat(sections) : sections;
      return all.length >= 2 ? all : [];
    }

    var targets = getTargets();
    if (targets.length < 2) return; // not a multi-section briefing

    // Detect accent colour from CSS variable so HUD counter matches the page theme
    var accentColor = getComputedStyle(document.documentElement)
      .getPropertyValue('--accent').trim() || '#60a5fa';

    // Build HUD
    var hud = document.createElement('div');
    hud.id = 'sec-nav-hud';
    hud.setAttribute('role', 'navigation');
    hud.setAttribute('aria-label', 'Section navigation');
    hud.style.cssText = [
      'position:fixed', 'bottom:24px', 'left:50%', 'transform:translateX(-50%)',
      'background:rgba(15,23,42,0.88)', 'backdrop-filter:blur(10px)',
      '-webkit-backdrop-filter:blur(10px)', 'color:#fff',
      'border-radius:40px', 'padding:10px 20px',
      'display:flex', 'align-items:center', 'gap:14px',
      'font-family:Inter,system-ui,sans-serif', 'font-size:0.82rem', 'font-weight:600',
      'box-shadow:0 4px 24px rgba(0,0,0,0.3)',
      'border:1px solid rgba(255,255,255,0.12)',
      'user-select:none', 'z-index:998', 'transition:opacity 0.3s',
      'white-space:nowrap'
    ].join(';');

    var btnStyle = [
      'background:rgba(255,255,255,0.12)',
      'border:1px solid rgba(255,255,255,0.2)',
      'color:#fff', 'border-radius:50%',
      'width:30px', 'height:30px',
      'display:flex', 'align-items:center', 'justify-content:center',
      'cursor:pointer', 'font-size:0.9rem',
      'transition:background 0.15s', 'flex-shrink:0',
      'line-height:1'
    ].join(';');

    hud.innerHTML =
      '<button id="snav-prev" title="Previous section (← or ↑)" style="' + btnStyle + '">←</button>' +
      '<div style="display:flex;align-items:center;gap:8px;">' +
        '<span id="snav-label" style="opacity:0.7;max-width:240px;overflow:hidden;text-overflow:ellipsis;"></span>' +
        '<span style="opacity:0.3;">·</span>' +
        '<span id="snav-counter" style="font-variant-numeric:tabular-nums;"></span>' +
      '</div>' +
      '<button id="snav-next" title="Next section (→ or ↓)" style="' + btnStyle + '">→</button>' +
      '<div style="width:1px;height:18px;background:rgba(255,255,255,0.15);"></div>' +
      '<span style="opacity:0.35;font-size:0.75rem;letter-spacing:0.03em;">← → keys</span>';

    document.body.appendChild(hud);

    var prevBtn = document.getElementById('snav-prev');
    var nextBtn = document.getElementById('snav-next');
    var labelEl = document.getElementById('snav-label');
    var counterEl = document.getElementById('snav-counter');
    counterEl.style.color = accentColor;

    var current = 0;

    function getLabel(el, idx) {
      if (el.classList.contains('hero') || el.classList.contains('doc-hero')) return 'Overview';
      var lbl = el.querySelector('.section-label, .eyebrow');
      var ttl = el.querySelector('.section-title, h2');
      if (lbl && ttl) return lbl.textContent.trim() + ' — ' + ttl.textContent.trim();
      if (ttl) return ttl.textContent.trim();
      return 'Section ' + idx;
    }

    function refreshHud() {
      var t = getTargets();
      labelEl.textContent   = getLabel(t[current], current + 1);
      counterEl.textContent = (current + 1) + ' / ' + t.length;
      prevBtn.style.opacity = current === 0 ? '0.3' : '1';
      nextBtn.style.opacity = current === t.length - 1 ? '0.3' : '1';
    }

    function goTo(idx) {
      var t = getTargets();
      if (idx < 0 || idx >= t.length) return;
      current = idx;
      t[current].scrollIntoView({ behavior: 'smooth', block: 'start' });
      setTimeout(function () { window.scrollBy({ top: -16, behavior: 'smooth' }); }, 350);
      refreshHud();
    }

    // Sync HUD when user scrolls manually
    var scrollTimer;
    window.addEventListener('scroll', function () {
      clearTimeout(scrollTimer);
      scrollTimer = setTimeout(function () {
        var t = getTargets();
        var best = 0, bestDist = Infinity;
        t.forEach(function (el, i) {
          var d = Math.abs(el.getBoundingClientRect().top);
          if (d < bestDist) { bestDist = d; best = i; }
        });
        if (best !== current) { current = best; refreshHud(); }
      }, 80);
    }, { passive: true });

    // Keyboard
    document.addEventListener('keydown', function (e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' ||
          e.target.isContentEditable) return;
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault(); goTo(current + 1);
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault(); goTo(current - 1);
      }
    });

    prevBtn.addEventListener('click', function () { goTo(current - 1); });
    nextBtn.addEventListener('click', function () { goTo(current + 1); });

    [prevBtn, nextBtn].forEach(function (btn) {
      btn.addEventListener('mouseenter', function () {
        if (parseFloat(btn.style.opacity) !== 0.3)
          btn.style.background = 'rgba(255,255,255,0.22)';
      });
      btn.addEventListener('mouseleave', function () {
        btn.style.background = 'rgba(255,255,255,0.12)';
      });
    });

    refreshHud();
  }

  // Run after DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', buildSectionNav);
  } else {
    buildSectionNav();
  }
})();
