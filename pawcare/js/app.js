/* js/app.js  — site bootstrap + Golden membership state */

window.PAWCARE = window.PAWCARE || (function(){
  const GOLD_KEY = 'pawcare_is_golden';
  const GOLD_SINCE = 'pawcare_gold_since';

  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // Put your Razorpay TEST/PROD key here (or leave blank for demo)
  const RAZORPAY_KEY = ''; // e.g. 'rzp_test_abc123XYZ'
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

  function isGolden(){
    return localStorage.getItem(GOLD_KEY) === '1';
  }
  function setGolden(v){
    localStorage.setItem(GOLD_KEY, v ? '1' : '0');
    if (v && !localStorage.getItem(GOLD_SINCE)) {
      localStorage.setItem(GOLD_SINCE, new Date().toISOString());
    }
    document.documentElement.classList.toggle('gold-active', !!v);

    // Toggle any elements that declare gold gating
    document.querySelectorAll('[data-if-gold]').forEach(el => el.hidden = !v);
    document.querySelectorAll('[data-if-free]').forEach(el => el.hidden = v);

    // Keep Bookings page checkbox in sync if present
    const chk = document.getElementById('goldBooking');
    if (chk) chk.checked = !!v;

    // Update navbar button label
    document.querySelectorAll('.btn-gold').forEach(a => {
      a.textContent = v ? 'Golden Member' : 'Join Golden';
      a.href = v ? 'index.html#golden' : 'join-golden.html?return=' + encodeURIComponent(location.pathname + location.hash);
    });
  }

  function initHeader(){
    // Mobile menu
    const btn = document.getElementById('hamburger');
    const menu = document.getElementById('menu') || document.querySelector('.menu');
    if (btn && menu){
      btn.addEventListener('click', () => menu.classList.toggle('open'));
    }
    // Year in footers
    const y = document.getElementById('year');
    if (y) y.textContent = new Date().getFullYear();
  }

  function boot(){
    initHeader();
    setGolden(isGolden()); // apply current state on load
  }

  document.addEventListener('DOMContentLoaded', boot);

  return { isGolden, setGolden, RAZORPAY_KEY };
})();
