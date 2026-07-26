window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Material loads pages via XHR when navigation.instant is on, so typesetting
// has to be re-run per navigation rather than once on load.
document$.subscribe(() => {
  MathJax.startup?.output?.clearCache?.();
  MathJax.typesetClear?.();
  MathJax.texReset?.();
  MathJax.typesetPromise?.();
});
