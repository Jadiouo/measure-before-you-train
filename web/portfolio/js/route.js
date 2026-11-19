/* The route map follows the reader: it marks the station being read and slides "you are here" along the line.
   Without scripting the map is an ordinary table of contents. */
(function () {
  'use strict';
  document.documentElement.classList.add('js');
  function init() {
    var nav = document.querySelector('.route');
    if (!nav) return;
    var stops = Array.prototype.slice.call(nav.querySelectorAll('li a'));
    var secs = stops.map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); });
    var readout = nav.querySelector('.readout');
    var here = nav.getAttribute('data-here') || '';
    var cur = -1, queued = false;

    function set(i) {
      if (i === cur) return;
      cur = i;
      nav.style.setProperty('--pos', i);
      stops.forEach(function (a, k) { if (k === i) a.setAttribute('aria-current', 'location'); else a.removeAttribute('aria-current'); });
      if (readout) {
        var nm = stops[i].querySelector('.nm').textContent;
        readout.innerHTML = '';
        readout.appendChild(document.createTextNode(here + '　'));
        var b = document.createElement('b');
        b.textContent = (i + 1) + ' / ' + stops.length;
        readout.appendChild(b);
        readout.appendChild(document.createTextNode('　' + nm));
      }
    }
    function update() {
      queued = false;
      var line = window.innerHeight * 0.4, i = 0;
      for (var k = 0; k < secs.length; k++) { if (secs[k].getBoundingClientRect().top <= line) i = k; }
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) i = secs.length - 1;
      set(i);
      nav.classList.toggle('stuck', nav.getBoundingClientRect().top <= 0 && window.scrollY > 0);
    }
    function queue() { if (!queued) { queued = true; window.requestAnimationFrame(update); } }
    window.addEventListener('scroll', queue, { passive: true });
    window.addEventListener('resize', queue);
    update();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
