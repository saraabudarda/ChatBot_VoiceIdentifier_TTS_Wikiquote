/* ── Animations & Interactions for the Presentation Page ── */

// ── NAV: scroll class + active section highlight ─────────────
const nav = document.getElementById('main-nav');
const sections = document.querySelectorAll('section[id], header[id]');
const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');

const onScroll = () => {
  nav.classList.toggle('scrolled', window.scrollY > 40);

  let current = '';
  sections.forEach(s => {
    if (window.scrollY >= s.offsetTop - 120) current = s.id;
  });
  navLinks.forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === `#${current}`);
  });
};
window.addEventListener('scroll', onScroll, { passive: true });
onScroll();

// ── Hamburger menu ─────────────────────────────────────────────
const hamburger = document.getElementById('hamburger');
const navLinksList = document.getElementById('nav-links');
hamburger.addEventListener('click', () => {
  navLinksList.classList.toggle('open');
});
document.addEventListener('click', e => {
  if (!nav.contains(e.target)) navLinksList.classList.remove('open');
});

// ── Floating particles ─────────────────────────────────────────
const particleContainer = document.getElementById('particles');
const PARTICLE_COUNT = 40;
for (let i = 0; i < PARTICLE_COUNT; i++) {
  const p = document.createElement('div');
  p.className = 'particle';
  const x = Math.random() * 100;
  const dur = 6 + Math.random() * 10;
  const delay = Math.random() * 10;
  const colors = ['#60a5fa', '#818cf8', '#a78bfa', '#22d3ee', '#2dd4bf'];
  const color = colors[Math.floor(Math.random() * colors.length)];
  p.style.cssText = `
    left: ${x}%;
    bottom: ${Math.random() * 60}%;
    --dur: ${dur}s;
    --delay: ${-delay}s;
    background: ${color};
    width: ${2 + Math.random() * 3}px;
    height: ${2 + Math.random() * 3}px;
  `;
  particleContainer.appendChild(p);
}

// ── Intersection Observer: animate elements on scroll ─────────
const observer = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      const delay = parseInt(el.dataset.delay ?? 0);

      setTimeout(() => {
        el.classList.add('visible');

        // Animate metric bars when visible
        if (el.classList.contains('metric-card')) {
          const fill = el.querySelector('.mc-fill');
          if (fill) {
            const targetWidth = fill.style.getPropertyValue('--fill');
            fill.style.setProperty('--fill', '0%');
            requestAnimationFrame(() => {
              requestAnimationFrame(() => {
                fill.style.setProperty('--fill', targetWidth);
              });
            });
          }
        }
      }, delay);

      observer.unobserve(el);
    });
  },
  { threshold: 0.12 }
);

document.querySelectorAll(
  '.feature-card, .stack-card, .metric-card, .pip-step'
).forEach(el => observer.observe(el));

// ── Waveform ring: subtle interactive effect ───────────────────
const micIcon = document.querySelector('.mic-icon');
if (micIcon) {
  document.addEventListener('mousemove', e => {
    const cx = window.innerWidth / 2;
    const cy = window.innerHeight / 2;
    const dx = (e.clientX - cx) / cx;
    const dy = (e.clientY - cy) / cy;
    micIcon.style.transform = `translate(${dx * 6}px, ${dy * 6}px)`;
  });
}

// ── Smooth active link on click ────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      navLinksList.classList.remove('open');
    }
  });
});

// ── Demo button: pulse effect on hover ────────────────────────
['demo-btn', 'cta-demo-btn'].forEach(id => {
  const btn = document.getElementById(id);
  if (!btn) return;
  btn.addEventListener('mouseenter', () => {
    btn.style.boxShadow = '0 12px 48px rgba(96,165,250,0.5)';
  });
  btn.addEventListener('mouseleave', () => {
    btn.style.boxShadow = '';
  });
});

// ── Typing quote animation in hero ────────────────────────────
const quotes = [
  'To be, or not to be…',
  'Imagination is everything.',
  'Be the change you wish to see.',
  'In the beginning was the Word.',
];
const fq1 = document.querySelector('.fq1');
if (fq1) {
  let qi = 0;
  const rotateQuote = () => {
    fq1.style.opacity = '0';
    fq1.style.transition = 'opacity 0.5s ease';
    setTimeout(() => {
      qi = (qi + 1) % quotes.length;
      fq1.firstChild.textContent = `"${quotes[qi]}"`;
      fq1.style.opacity = '1';
    }, 500);
  };
  setInterval(rotateQuote, 4000);
}

console.log('%c🎙️ WikiQuote NLP — Sara Abudarda', 'color:#60a5fa;font-size:14px;font-weight:bold;');
