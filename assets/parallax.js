/* ---------------------------------------------------------------
   parallax.js — считает сдвиг каждой секции в rAF-цикле,
   привязанном к скроллу. Работает в iOS Safari (в отличие от
   background-attachment: fixed) и на всех современных браузерах.

   Подключать в конце body:  script src="assets/parallax.js"
   --------------------------------------------------------------- */
(function () {
  "use strict";

  document.documentElement.classList.add("px-js");

  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  var panels = [];
  var ticking = false;

  function collect() {
    panels = Array.prototype.map.call(
      document.querySelectorAll(".px-panel"),
      function (el) {
        return {
          el: el,
          media: el.querySelector(".px-panel__media"),
          copy: el.querySelector(".px-panel__copy"),
          // data-depth — насколько фон отстаёт от скролла (0 = не двигается)
          depth: parseFloat(el.getAttribute("data-depth")) || 0.15
        };
      }
    );
  }

  function render() {
    ticking = false;
    var vh = window.innerHeight;

    for (var i = 0; i < panels.length; i++) {
      var p = panels[i];
      var r = p.el.getBoundingClientRect();

      // секция далеко за кадром — не трогаем
      if (r.bottom < -vh * 0.3 || r.top > vh * 1.3) continue;

      // progress: 1 — секция только снизу, 0 — по центру, -1 — ушла вверх
      var progress = (r.top + r.height / 2 - vh / 2) / vh;
      if (progress > 1.4) progress = 1.4;
      if (progress < -1.4) progress = -1.4;

      if (p.media) {
        p.media.style.transform =
          "translate3d(0," + (progress * p.depth * vh).toFixed(2) + "px,0)";
      }
      // текст идёт чуть быстрее полосы — отсюда ощущение глубины
      if (p.copy) {
        p.copy.style.transform =
          "translate3d(0," + (progress * -p.depth * 0.34 * vh).toFixed(2) + "px,0)";
      }
    }
  }

  function onScroll() {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(render);
    }
  }

  function watch() {
    if (!("IntersectionObserver" in window)) {
      Array.prototype.forEach.call(
        document.querySelectorAll(".px-panel"),
        function (el) { el.classList.add("px-in"); }
      );
      return;
    }
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            e.target.classList.add("px-in");
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.28 }
    );
    Array.prototype.forEach.call(
      document.querySelectorAll(".px-panel"),
      function (el) { io.observe(el); }
    );
  }

  function order() {
    Array.prototype.forEach.call(
      document.querySelectorAll(".px-panel__copy"),
      function (copy) {
        Array.prototype.forEach.call(
          copy.querySelectorAll("[data-reveal]"),
          function (node, i) { node.style.setProperty("--px-i", i); }
        );
      }
    );
  }

  function init() {
    order();
    watch();
    if (reduce.matches) return;
    collect();
    render();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", function () { collect(); onScroll(); });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
