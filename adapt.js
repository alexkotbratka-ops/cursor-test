(() => {
  const filters = document.querySelectorAll(".a-filters button");
  const cards = Array.from(document.querySelectorAll(".p-card"));
  const grid = document.getElementById("productGrid");
  const hint = document.getElementById("filterHint");
  const sortSelect = document.getElementById("sortSelect");
  const stickyLinks = document.querySelectorAll(".a-sticky-nav a");
  const sections = Array.from(stickyLinks)
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);
  const burger = document.getElementById("burger");
  const header = document.querySelector(".a-header");
  const toTop = document.getElementById("toTop");
  const form = document.getElementById("requestForm");
  const toast = document.getElementById("toast");

  let activeFilter = "all";

  function updateHint(visibleCount) {
    if (!hint) return;
    if (activeFilter === "all") {
      hint.textContent = `Показаны все типоразмеры · ${visibleCount}`;
    } else {
      hint.textContent = `ЗПУ ${activeFilter} мм · ${visibleCount} моделей`;
    }
  }

  function applyFilterAndSort() {
    const sort = sortSelect ? sortSelect.value : "zpu-vol";
    const visible = [];

    cards.forEach((card) => {
      const zpu = card.dataset.zpu;
      const match = activeFilter === "all" || zpu === activeFilter;
      card.classList.toggle("is-hidden", !match);
      if (match) visible.push(card);
    });

    visible.sort((a, b) => {
      const av = Number(a.dataset.vol);
      const bv = Number(b.dataset.vol);
      const az = Number(a.dataset.zpu);
      const bz = Number(b.dataset.zpu);
      const as = Number(a.dataset.series);
      const bs = Number(b.dataset.series);
      if (sort === "vol") return av - bv || az - bz;
      if (sort === "series") return as - bs || av - bv || az - bz;
      return az - bz || av - bv || as - bs;
    });

    visible.forEach((card) => grid.appendChild(card));
    updateHint(visible.length);
  }

  filters.forEach((btn) => {
    btn.addEventListener("click", () => {
      filters.forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      activeFilter = btn.dataset.filter;
      applyFilterAndSort();
    });
  });

  if (sortSelect) {
    sortSelect.addEventListener("change", applyFilterAndSort);
  }

  applyFilterAndSort();

  // Scroll spy for sticky section nav
  const spy = () => {
    const offset = 120;
    let current = sections[0];
    sections.forEach((section) => {
      if (section.getBoundingClientRect().top - offset <= 0) current = section;
    });
    stickyLinks.forEach((link) => {
      const active = link.getAttribute("href") === `#${current.id}`;
      link.classList.toggle("is-active", active);
    });
  };
  window.addEventListener("scroll", spy, { passive: true });
  spy();

  // Smooth scroll for in-page anchors
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      const id = anchor.getAttribute("href");
      if (!id || id === "#") return;
      const target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      if (header && header.classList.contains("is-open")) {
        header.classList.remove("is-open");
        burger?.setAttribute("aria-expanded", "false");
      }
    });
  });

  // Mobile menu
  burger?.addEventListener("click", () => {
    const open = header.classList.toggle("is-open");
    burger.setAttribute("aria-expanded", String(open));
  });

  // Back to top
  const onScrollTop = () => {
    toTop?.classList.toggle("is-visible", window.scrollY > 600);
  };
  window.addEventListener("scroll", onScrollTop, { passive: true });
  toTop?.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));

  // Request form
  form?.addEventListener("submit", (e) => {
    e.preventDefault();
    if (!form.checkValidity()) {
      form.reportValidity();
      return;
    }
    toast.hidden = false;
    form.reset();
    setTimeout(() => {
      toast.hidden = true;
    }, 2500);
  });
})();
