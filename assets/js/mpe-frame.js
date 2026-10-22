(function () {
  "use strict";

  const frame = document.getElementById("mpe-frame");
  if (!frame) return;

  let cleanup = function () {};

  function connect() {
    cleanup();

    let doc;
    try {
      doc = frame.contentDocument;
      if (!doc || !doc.body) return;
    } catch {
      return;
    }

    frame.contentWindow.scrollTo(0, 0);

    const style = doc.createElement("style");
    style.textContent = `
      html {
        height: auto !important;
        min-height: 0 !important;
        overflow-y: hidden !important;
      }
      body {
        height: auto !important;
        min-height: 0 !important;
        max-height: none !important;
        margin: 0 !important;
        display: flow-root !important;
        overflow-y: hidden !important;
      }
      body .markdown-preview {
        position: relative !important;
        height: auto !important;
        min-height: 0 !important;
        max-height: none !important;
        overflow-y: visible !important;
      }
    `;
    doc.head.appendChild(style);

    let pending = 0;
    let active = true;
    let firstMeasurement = true;

    function scrollToHash(hash) {
      if (!hash || hash === "#") return;
      let id;
      try {
        id = decodeURIComponent(hash.slice(1));
      } catch {
        return;
      }

      const target =
        doc.getElementById(id) || Array.from(doc.getElementsByName(id))[0];
      if (!target) return;

      frame.contentWindow.scrollTo(0, 0);
      const top =
        window.scrollY +
        frame.getBoundingClientRect().top +
        target.getBoundingClientRect().top;
      window.scrollTo({ top: Math.max(0, top), behavior: "auto" });
    }

    function measure() {
      pending = 0;
      if (!active) return;

      const rect = doc.body.getBoundingClientRect();
      const height = Math.ceil(
        Math.max(rect.height, doc.body.scrollHeight) +
          Math.max(0, rect.top) +
          2
      );

      if (frame.style.height !== height + "px") {
        frame.style.height = height + "px";
      }

      if (firstMeasurement) {
        firstMeasurement = false;
        scrollToHash(window.location.hash);
      }
    }

    function schedule() {
      if (active && !pending) pending = requestAnimationFrame(measure);
    }

    function loadImages() {
      doc.querySelectorAll('img[loading="lazy"]').forEach(function (image) {
        image.loading = "eager";
      });
    }

    function handleClick(event) {
      const link = event.target.closest
        ? event.target.closest("a[href]")
        : null;
      if (
        !link ||
        event.button !== 0 ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        event.altKey
      ) {
        return;
      }

      const href = link.getAttribute("href");
      if (!href || !href.startsWith("#")) return;

      event.preventDefault();
      event.stopImmediatePropagation();
      scrollToHash(href);
    }

    function handleHashChange() {
      scrollToHash(window.location.hash);
    }

    const resizeObserver = new ResizeObserver(schedule);
    resizeObserver.observe(doc.body);

    const mutationObserver = new MutationObserver(function () {
      loadImages();
      schedule();
    });
    mutationObserver.observe(doc.body, {
      subtree: true,
      childList: true,
      attributes: true,
      characterData: true
    });

    doc.addEventListener("load", schedule, true);
    doc.addEventListener("error", schedule, true);
    doc.addEventListener("click", handleClick, true);
    window.addEventListener("resize", schedule);
    window.addEventListener("hashchange", handleHashChange);

    if (doc.fonts) doc.fonts.ready.then(schedule);
    loadImages();
    schedule();

    cleanup = function () {
      active = false;
      cancelAnimationFrame(pending);
      resizeObserver.disconnect();
      mutationObserver.disconnect();
      doc.removeEventListener("load", schedule, true);
      doc.removeEventListener("error", schedule, true);
      doc.removeEventListener("click", handleClick, true);
      window.removeEventListener("resize", schedule);
      window.removeEventListener("hashchange", handleHashChange);
      style.remove();
    };
  }

  frame.addEventListener("load", connect);
  connect();
})();
