const { test, expect } = require("@playwright/test");
const AxeBuilder = require("@axe-core/playwright").default;

const appBaseUrl = process.env.PLAYWRIGHT_BASE_URL || "http://127.0.0.1:5055";
const brandAssetPaths = [
  "/static/brand/evidence-aperture-color.svg",
  "/static/brand/evidence-aperture-mono-light.svg",
  "/static/brand/evidence-aperture-mono-dark.svg",
  "/static/brand/evidence-aperture-on-dark.svg",
  "/static/brand/evidence-aperture-on-light.svg",
  "/static/brand/evidence-aperture-scale-sheet.svg",
];
const legacyBrandAssetPaths = [
  "/static/brand/evidence-trace-color.svg",
  "/static/brand/evidence-trace-mono-dark.svg",
  "/static/brand/evidence-trace-monochrome.svg",
  "/static/brand/evidence-trace-on-dark.svg",
  "/static/brand/evidence-trace-on-light.svg",
  "/static/brand/evidence-trace-scale-sheet.svg",
];
const brandScaleSizes = [16, 20, 24, 32, 64, 128];
const brandPartSelectors = [
  ".brand-mark__rear-frame",
  ".brand-mark__review-frame",
  ".brand-mark__window",
  ".brand-mark__passage",
];

const visualViewports = [
  { name: "320x568", width: 320, height: 568 },
  { name: "390x844", width: 390, height: 844 },
  { name: "768x1024", width: 768, height: 1024 },
  { name: "1280x720", width: 1280, height: 720 },
  { name: "1440x900", width: 1440, height: 900 },
];

async function installMotionProbe(page) {
  await page.addInitScript(() => {
    const nativeRequestAnimationFrame = window.requestAnimationFrame.bind(window);
    const nativeCancelAnimationFrame = window.cancelAnimationFrame.bind(window);
    const pendingFrames = new Set();

    window.__motionProbe = {
      canceled: 0,
      executed: 0,
      pending: 0,
      scheduled: 0,
    };

    window.requestAnimationFrame = (callback) => {
      window.__motionProbe.scheduled += 1;
      let frameId;
      frameId = nativeRequestAnimationFrame((time) => {
        if (pendingFrames.delete(frameId)) window.__motionProbe.pending -= 1;
        window.__motionProbe.executed += 1;
        callback(time);
      });
      pendingFrames.add(frameId);
      window.__motionProbe.pending += 1;
      return frameId;
    };

    window.cancelAnimationFrame = (frameId) => {
      if (pendingFrames.delete(frameId)) {
        window.__motionProbe.pending -= 1;
        window.__motionProbe.canceled += 1;
      }
      nativeCancelAnimationFrame(frameId);
    };

    window.__layoutShiftTotal = 0;
    if (typeof PerformanceObserver === "function") {
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (!entry.hadRecentInput) window.__layoutShiftTotal += entry.value;
          }
        });
        observer.observe({ type: "layout-shift", buffered: true });
      } catch (_error) {
        // Older engines can omit the layout-shift entry type.
      }
    }
  });
}

async function canvasSignature(canvas) {
  return canvas.evaluate((element) => element.toDataURL("image/png"));
}

async function depthOffsets(canvas) {
  return canvas.evaluate((element) => ({
    x: Number(element.dataset.depthX),
    y: Number(element.dataset.depthY),
  }));
}

function expectDepthWithinCap(offsets) {
  expect(Number.isFinite(offsets.x)).toBe(true);
  expect(Number.isFinite(offsets.y)).toBe(true);
  expect(Math.abs(offsets.x)).toBeLessThanOrEqual(3);
  expect(Math.abs(offsets.y)).toBeLessThanOrEqual(3);
}

async function scrollFlowToRatio(page, ratio) {
  await page.locator(".architecture-flow[data-reveal]").evaluate((flow, visibleRatio) => {
    const scroller = flow.closest(".overview-scroll");
    const scrollerRect = scroller.getBoundingClientRect();
    const flowRect = flow.getBoundingClientRect();
    const flowTop = flowRect.top - scrollerRect.top + scroller.scrollTop;
    const visiblePixels = flowRect.height * visibleRatio;
    scroller.scrollTop = Math.max(0, flowTop - scroller.clientHeight + visiblePixels);
  }, ratio);
}

test("Evidence Aperture has one accessible brand name and deployable SVG assets", async ({ page }) => {
  const runtimeErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") runtimeErrors.push(message.text());
  });
  page.on("pageerror", (error) => runtimeErrors.push(error.message));
  await installMotionProbe(page);
  await page.goto("/", { waitUntil: "domcontentloaded" });

  const brandHome = page.getByRole("link", { name: "Clinical Evidence Assistant home", exact: true });
  const mark = page.locator("#brand-mark");
  await expect(brandHome).toHaveCount(1);
  await expect(mark).toHaveCount(1);
  await expect(mark).toHaveAttribute("aria-hidden", "true");
  await expect(mark).toHaveAttribute("viewBox", "0 0 32 32");
  await expect(mark).toHaveAttribute("width", "32");
  await expect(mark).toHaveAttribute("height", "32");
  await expect(brandHome.getByText("Clinical Evidence Assistant", { exact: true })).toHaveCount(1);
  await expect(mark.locator("title, script, style, filter, mask, foreignObject, image")).toHaveCount(0);
  for (const selector of brandPartSelectors) {
    await expect(mark.locator(selector), `${selector} should identify one controlled logo element`).toHaveCount(1);
  }
  expect(await mark.getAttribute("role")).toBeNull();
  expect(await mark.getAttribute("aria-label")).toBeNull();
  expect(await mark.evaluate((element) => new TextEncoder().encode(element.outerHTML).byteLength)).toBeLessThan(1200);

  const beforeBox = await mark.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    return { height: rect.height, width: rect.width, x: rect.x, y: rect.y };
  });
  const initialAnimationsByPart = await mark.evaluate((element, selectors) => selectors.map((selector) => {
    const part = element.querySelector(selector);
    return part.getAnimations().map((animation) => {
      const timing = animation.effect.getComputedTiming();
      return {
        delay: Number(timing.delay) || 0,
        duration: Number(timing.duration) || 0,
        iterations: Number(timing.iterations) || 0,
      };
    });
  }), brandPartSelectors);
  expect(initialAnimationsByPart.every((animations) => animations.length > 0)).toBe(true);
  const initialAnimations = initialAnimationsByPart.flat();
  expect(new Set(initialAnimations.map((animation) => animation.delay)).size).toBeGreaterThan(1);
  for (const animation of initialAnimations) {
    expect(animation.iterations).toBe(1);
    expect(animation.delay + animation.duration).toBeLessThanOrEqual(260);
  }

  await page.waitForTimeout(350);
  const afterBox = await mark.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    return { height: rect.height, width: rect.width, x: rect.x, y: rect.y };
  });
  expect(afterBox.width).toBeCloseTo(beforeBox.width, 1);
  expect(afterBox.height).toBeCloseTo(beforeBox.height, 1);
  expect(afterBox.x).toBeCloseTo(beforeBox.x, 1);
  expect(afterBox.y).toBeCloseTo(beforeBox.y, 1);
  expect(await page.evaluate(() => window.__layoutShiftTotal)).toBeLessThanOrEqual(0.02);

  await page.getByRole("tab", { name: "System Overview" }).click();
  await page.getByRole("tab", { name: "Clinical Assistant" }).click();
  await expect.poll(() => mark.evaluate((element) => (
    element.getAnimations({ subtree: true }).filter((animation) => animation.playState === "running").length
  ))).toBe(0);

  const faviconHref = await page.locator('link[rel~="icon"]').getAttribute("href");
  expect(faviconHref).toBeTruthy();
  expect(new URL(faviconHref, page.url()).pathname).toBe("/static/favicon.svg");
  await expect(page.locator(".empty-aperture")).toHaveAttribute("src", /\/static\/brand\/evidence-aperture-color\.svg$/);
  await expect(page.locator(".empty-aperture")).toHaveAttribute("width", "56");
  await expect(page.locator(".empty-aperture")).toHaveAttribute("height", "56");
  await expect(page.locator(".overview-watermark")).toHaveAttribute("src", /\/static\/brand\/evidence-aperture-on-dark\.svg$/);
  await expect(page.locator(".overview-watermark")).toHaveAttribute("width", "320");
  await expect(page.locator(".overview-watermark")).toHaveAttribute("height", "320");
  const assetPaths = [new URL(faviconHref, page.url()).pathname, ...brandAssetPaths];
  const assetSources = new Map();
  let familyBytes = 0;
  for (const path of assetPaths) {
    const response = await page.request.get(new URL(path, appBaseUrl).href);
    expect(response.ok(), `${path} should load`).toBe(true);
    expect(response.headers()["content-type"]).toContain("image/svg+xml");
    const source = await response.text();
    assetSources.set(path, source);
    familyBytes += Buffer.byteLength(source, "utf8");
    expect(source).toMatch(/<svg\b/);
    expect(source).not.toMatch(/<(?:script|style|filter|mask|foreignObject|image)\b/i);
    expect(source).not.toMatch(/\sstyle=["']/i);
    expect(source).not.toMatch(/(?:href|xlink:href|src)=["'](?:https?:|\/\/|data:)/i);
  }
  expect(familyBytes).toBeLessThan(12 * 1024);

  const identitySignatures = await page.evaluate(({ primarySource, selectors, sources }) => {
    const geometryAttributes = ["d", "x", "y", "width", "height", "rx", "ry", "cx", "cy", "r", "x1", "y1", "x2", "y2"];
    const normalizeValue = (value) => value.replace(/[\s,]+/g, "");
    const geometry = (element) => ({
      attributes: Object.fromEntries(geometryAttributes
        .filter((attribute) => element.hasAttribute(attribute))
        .map((attribute) => [attribute, normalizeValue(element.getAttribute(attribute))])),
      tag: element.tagName.toLowerCase(),
    });
    const inheritedPresentation = (element, property) => {
      for (let current = element; current?.nodeType === Node.ELEMENT_NODE; current = current.parentElement) {
        if (current.hasAttribute(property)) return current.getAttribute(property);
      }
      return property === "fill" ? "black" : "none";
    };
    const hasPaint = (value) => !["none", "transparent", "rgba(0, 0, 0, 0)"].includes(value.trim().toLowerCase());
    const parsedSvg = (source) => new DOMParser().parseFromString(source, "image/svg+xml").documentElement;
    const shapeElements = (root) => Array.from(root.querySelectorAll("path, rect, circle, line, polyline, polygon"));

    const primaryRoot = parsedSvg(primarySource);
    const primaryElements = shapeElements(primaryRoot);
    const primary = primaryElements.map((element) => ({
      ...geometry(element),
      paint: {
        fill: hasPaint(inheritedPresentation(element, "fill")),
        stroke: hasPaint(inheritedPresentation(element, "stroke")),
      },
    }));
    const inline = selectors.map((selector) => {
      const element = document.querySelector(`#brand-mark ${selector}`);
      const style = getComputedStyle(element);
      return {
        ...geometry(element),
        paint: { fill: hasPaint(style.fill), stroke: hasPaint(style.stroke) },
      };
    });
    const exportedGeometry = Object.fromEntries(sources.map(([path, source]) => [
      path,
      shapeElements(parsedSvg(source)).map(geometry),
    ]));
    return {
      exportedGeometry,
      inline,
      primary,
      primaryGeometry: primaryElements.map(geometry),
    };
  }, {
    primarySource: assetSources.get(brandAssetPaths[0]),
    selectors: brandPartSelectors,
    sources: Array.from(assetSources.entries()),
  });
  expect(identitySignatures.inline).toEqual(identitySignatures.primary);
  const canonicalGeometry = JSON.stringify(identitySignatures.primaryGeometry);
  for (const [path, shapes] of Object.entries(identitySignatures.exportedGeometry)) {
    const containsCanonicalMark = shapes.some((_, index) => (
      JSON.stringify(shapes.slice(index, index + identitySignatures.primaryGeometry.length)) === canonicalGeometry
    ));
    expect(containsCanonicalMark, `${path} should contain the canonical Evidence Aperture geometry`).toBe(true);
  }

  const renderedSizes = await page.evaluate(async ({ sizes, source }) => {
    const results = [];
    for (const size of sizes) {
      const image = document.createElement("img");
      image.alt = "";
      image.src = source;
      image.width = size;
      image.height = size;
      image.style.position = "fixed";
      image.style.inset = "auto auto -9999px -9999px";
      document.body.append(image);
      await image.decode();
      const box = image.getBoundingClientRect();
      results.push({
        complete: image.complete,
        height: box.height,
        naturalHeight: image.naturalHeight,
        naturalWidth: image.naturalWidth,
        width: box.width,
      });
      image.remove();
    }
    return results;
  }, {
    sizes: brandScaleSizes,
    source: new URL(brandAssetPaths[0], appBaseUrl).href,
  });
  expect(renderedSizes).toHaveLength(brandScaleSizes.length);
  renderedSizes.forEach((rendered, index) => {
    expect(rendered.complete).toBe(true);
    expect(rendered.naturalWidth).toBeGreaterThan(0);
    expect(rendered.naturalHeight).toBeGreaterThan(0);
    expect(rendered.width).toBe(brandScaleSizes[index]);
    expect(rendered.height).toBe(brandScaleSizes[index]);
  });
  for (const path of legacyBrandAssetPaths) {
    const response = await page.request.get(new URL(path, appBaseUrl).href);
    expect(response.status(), `${path} should be retired`).toBe(404);
  }
  expect(runtimeErrors).toEqual([]);
});

test("Evidence Aperture remains visible in forced colors", async ({ page }) => {
  await page.emulateMedia({ forcedColors: "active", reducedMotion: "reduce" });
  await page.goto("/");

  const mark = page.locator("#brand-mark");
  await expect(mark).toBeVisible();
  const partStyles = await mark.evaluate((element, selectors) => selectors.map((selector) => {
    const part = element.querySelector(selector);
    const style = getComputedStyle(part);
    return { opacity: style.opacity, stroke: style.stroke, visibility: style.visibility };
  }), brandPartSelectors);
  for (const style of partStyles) {
    expect(style.visibility).toBe("visible");
    expect(style.opacity).toBe("1");
    expect(style.stroke).not.toBe("none");
    expect(style.stroke).not.toBe("rgba(0, 0, 0, 0)");
  }
});

test("initial workspace is keyboard-operable and axe-clean", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Trace every answer back to its source" })).toBeVisible();

  const assistantTab = page.getByRole("tab", { name: "Clinical Assistant" });
  const overviewTab = page.getByRole("tab", { name: "System Overview" });
  await assistantTab.focus();
  await page.keyboard.press("ArrowRight");
  await expect(overviewTab).toBeFocused();
  await expect(overviewTab).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("ArrowLeft");
  await expect(assistantTab).toBeFocused();

  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});

test("answer exposes one selected passage and returns focus from mobile sheet", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByLabel("Ask a clinical question").fill("What discharge diagnoses are documented?");
  await page.getByRole("button", { name: "Send" }).click();

  await expect(page.getByText("Source-support check passed")).toBeVisible();
  const evidenceTrigger = page.getByRole("button", { name: /Review 5 evidence passages/ });
  await evidenceTrigger.click();
  await expect(page.getByRole("dialog", { name: "Evidence inspector" })).toBeVisible();
  await expect(page.locator(".evidence-passage")).toHaveCount(1);
  await page.getByRole("tab", { name: "Evidence passage 2" }).click();
  await expect(page.getByText("Passage content 2.", { exact: false })).toBeVisible();
  const sheetAxeResults = await new AxeBuilder({ page }).include("#evidence-inspector").analyze();
  expect(sheetAxeResults.violations).toEqual([]);

  await page.getByRole("button", { name: "Close evidence inspector" }).click();
  await expect(evidenceTrigger).toBeFocused();
  await expect(page.locator("#evidence-inspector")).toHaveAttribute("hidden", "");
});

test("loading uses an honest indeterminate state and blocks double submission", async ({ page }) => {
  await page.route("**/ask", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 1200));
    await route.continue();
  });
  await page.goto("/");
  await page.getByLabel("Ask a clinical question").fill("What medications were prescribed at discharge?");
  await page.getByRole("button", { name: "Send" }).click();

  await expect(page.getByRole("status")).toContainText("Processing locally");
  await expect(page.locator("#assistant-workspace")).toHaveAttribute("aria-busy", "true");
  await expect(page.getByRole("button", { name: "Send" })).toBeDisabled();
  await expect(page.locator(".user-message")).toHaveCount(1);
  await expect(page.locator("#assistant-workspace")).toHaveAttribute("aria-busy", "false", { timeout: 10000 });
});

test("Evidence Field completes once and uses capped depth only for fine pointers", async ({ page }) => {
  await installMotionProbe(page);
  await page.goto("/");

  const canvas = page.locator("#evidence-canvas");
  await expect(canvas).toBeVisible();
  expect(await page.evaluate(() => matchMedia("(pointer: fine)").matches)).toBe(true);
  await expect(canvas).toHaveAttribute("data-depth-enabled", "true");
  await expect(canvas).toHaveAttribute("data-animation-state", "static", { timeout: 2500 });

  const settledProbe = await page.evaluate(() => ({ ...window.__motionProbe }));
  await page.waitForTimeout(350);
  const idleProbe = await page.evaluate(() => ({ ...window.__motionProbe }));
  expect(idleProbe.pending).toBe(0);
  expect(idleProbe.scheduled).toBe(settledProbe.scheduled);

  const beforePointer = await canvasSignature(canvas);
  const field = page.locator(".constellation");
  const box = await field.boundingBox();
  await field.hover({ position: { x: box.width - 4, y: 4 } });
  await expect.poll(() => depthOffsets(canvas)).not.toEqual({ x: 0, y: 0 });
  const offsets = await depthOffsets(canvas);
  expectDepthWithinCap(offsets);
  await expect.poll(() => canvasSignature(canvas)).not.toBe(beforePointer);

  await page.waitForTimeout(100);
  const afterPointerProbe = await page.evaluate(() => ({ ...window.__motionProbe }));
  await page.waitForTimeout(350);
  const pointerIdleProbe = await page.evaluate(() => ({ ...window.__motionProbe }));
  expect(pointerIdleProbe.pending).toBe(0);
  expect(pointerIdleProbe.scheduled).toBe(afterPointerProbe.scheduled);
  await expect(canvas).toHaveAttribute("data-animation-state", "static");
});

test("coarse pointers cannot activate Evidence Field depth", async ({ browser }) => {
  const context = await browser.newContext({
    baseURL: appBaseUrl,
    hasTouch: true,
    isMobile: true,
    viewport: { width: 390, height: 844 },
  });
    const page = await context.newPage();
    try {
      await page.goto("/");
      const canvas = page.locator("#evidence-canvas");
      expect(await page.evaluate(() => matchMedia("(pointer: coarse)").matches)).toBe(true);
      await expect(canvas).toHaveAttribute("data-depth-enabled", "false");
      await canvas.scrollIntoViewIfNeeded();
      await expect(canvas).toHaveAttribute("data-animation-state", "static", { timeout: 2500 });
    expect(await depthOffsets(canvas)).toEqual({ x: 0, y: 0 });
    const beforePointer = await canvasSignature(canvas);
    const box = await canvas.boundingBox();
    await page.mouse.move(box.x + box.width - 2, box.y + 2);
    await page.waitForTimeout(100);
    expect(await depthOffsets(canvas)).toEqual({ x: 0, y: 0 });
    expect(await canvasSignature(canvas)).toBe(beforePointer);
  } finally {
    await context.close();
  }
});

test("reduced motion renders all identity graphics in their final static state", async ({ page }) => {
  await installMotionProbe(page);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  expect(await page.evaluate(() => matchMedia("(prefers-reduced-motion: reduce)").matches)).toBe(true);

  const mark = page.locator("#brand-mark");
  const canvas = page.locator("#evidence-canvas");
  await expect(canvas).toBeVisible();
  await expect(canvas).toHaveAttribute("data-animation-state", "static");
  await expect(canvas).toHaveAttribute("data-depth-enabled", "false");
  expect(await depthOffsets(canvas)).toEqual({ x: 0, y: 0 });
  expect(await mark.evaluate((element) => element.getAnimations({ subtree: true }).length)).toBe(0);
  const markPartStyles = await mark.evaluate((element, selectors) => selectors.map((selector) => {
    const style = getComputedStyle(element.querySelector(selector));
    return { opacity: style.opacity, transform: style.transform };
  }), brandPartSelectors);
  expect(markPartStyles).toEqual(brandPartSelectors.map(() => ({ opacity: "1", transform: "none" })));

  const beforePointer = await canvasSignature(canvas);
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width - 2, box.y + 2);
  await page.waitForTimeout(100);
  expect(await canvasSignature(canvas)).toBe(beforePointer);
  expect(await depthOffsets(canvas)).toEqual({ x: 0, y: 0 });

  await page.getByRole("tab", { name: "System Overview" }).click();
  await expect(page.getByRole("heading", { name: "Understand the evidence path and its limits" })).toBeVisible();
  const flow = page.locator(".architecture-flow[data-reveal]");
  await scrollFlowToRatio(page, 0.4);
  await expect(flow).toHaveAttribute("data-reveal-state", "revealed");
  const stageStyles = await flow.locator("li").evaluateAll((stages) => stages.map((stage) => ({
    opacity: getComputedStyle(stage).opacity,
    transform: getComputedStyle(stage).transform,
  })));
  for (const style of stageStyles) {
    expect(style.opacity).toBe("1");
    expect(style.transform).toBe("none");
  }
  expect(await flow.evaluate((element) => element.getAnimations({ subtree: true }).length)).toBe(0);
});

test("architecture trace reveals at 20 percent visibility and does not replay", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 500 });
  await page.goto("/");
  await page.getByRole("tab", { name: "System Overview" }).click();

  const flow = page.locator(".architecture-flow[data-reveal]");
  await expect(flow).toHaveAttribute("data-reveal-state", "pending");
  await scrollFlowToRatio(page, 0.1);
  await page.waitForTimeout(120);
  await expect(flow).toHaveAttribute("data-reveal-state", "pending");

  await scrollFlowToRatio(page, 0.35);
  await expect(flow).toHaveAttribute("data-reveal-state", "revealed");
  const timings = await flow.locator("li").evaluateAll((stages) => stages.map((stage) => {
    const transition = getComputedStyle(stage);
    return {
      delay: transition.transitionDelay,
      duration: transition.transitionDuration,
    };
  }));
  expect(timings).toHaveLength(5);
  expect(timings.map((timing) => timing.duration)).toEqual(Array(5).fill("0.22s, 0.22s"));
  expect(timings.map((timing) => timing.delay.split(",")[0].trim())).toEqual(["0s", "0.04s", "0.08s", "0.12s", "0.16s"]);

  await page.waitForTimeout(450);
  const firstAnimations = await flow.evaluate((element) => element.getAnimations({ subtree: true }).map((animation) => ({
    currentTime: animation.currentTime,
    playState: animation.playState,
    startTime: animation.startTime,
  })));
  await page.locator(".overview-scroll").evaluate((scroller) => { scroller.scrollTop = 0; });
  await page.waitForTimeout(120);
  await expect(flow).toHaveAttribute("data-reveal-state", "revealed");
  await scrollFlowToRatio(page, 0.45);
  await page.waitForTimeout(120);
  await expect(flow).toHaveAttribute("data-reveal-state", "revealed");
  const secondAnimations = await flow.evaluate((element) => element.getAnimations({ subtree: true }).map((animation) => ({
    currentTime: animation.currentTime,
    playState: animation.playState,
    startTime: animation.startTime,
  })));
  expect(secondAnimations).toEqual(firstAnimations);
});

test("identity and overview refinements make no remote requests", async ({ page }) => {
  const remoteRequests = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (["http:", "https:"].includes(url.protocol) && url.origin !== new URL(appBaseUrl).origin) {
      remoteRequests.push(request.url());
    }
  });

  await page.goto("/");
  await expect(page.locator("#evidence-canvas")).toHaveAttribute("data-animation-state", "static", { timeout: 2500 });
  await page.getByRole("tab", { name: "System Overview" }).click();
  await scrollFlowToRatio(page, 0.45);
  await expect(page.locator(".architecture-flow[data-reveal]")).toHaveAttribute("data-reveal-state", "revealed");
  await page.getByRole("tab", { name: "Clinical Assistant" }).click();
  await page.getByLabel("Ask a clinical question").fill("What discharge diagnoses are documented?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.locator(".response-answer")).toBeVisible();
  await expect(page.getByText("Source-support check passed")).toBeVisible();
  expect(remoteRequests).toEqual([]);
});

test("long local processing produces one stable announcement", async ({ page }) => {
  await page.clock.install();
  let releaseRequest;
  await page.route("**/ask", async (route) => {
    await new Promise((resolve) => {
      releaseRequest = resolve;
    });
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        answer: "Grounded answer.",
        route: "retrieve",
        tool_used: "Retrieved 1 evidence passage",
        citations: [],
        reflection: { supported: true, unsupported_claims: [] },
        needs_clarification: false,
        fda_result: null,
      }),
    });
  });

  await page.goto("/");
  await page.getByLabel("Ask a clinical question").fill("What is documented at discharge?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByRole("status")).toContainText("Processing locally");
  await expect.poll(() => Boolean(releaseRequest)).toBe(true);

  await page.clock.fastForward(21000);
  const announcer = page.locator("#global-announcer");
  await expect(announcer).toHaveText("Still processing locally. CPU generation can take longer.");
  await page.clock.fastForward(60000);
  await expect(announcer).toHaveText("Still processing locally. CPU generation can take longer.");

  releaseRequest();
  await expect(page.locator("#assistant-workspace")).toHaveAttribute("aria-busy", "false");
});

test("layout has no horizontal overflow at 200 percent text size", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.evaluate(() => {
    document.documentElement.style.fontSize = "200%";
  });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});

test("clarification, FDA, support-gap, and server-error states remain distinct", async ({ page }) => {
  await page.goto("/");

  const ask = async (question, selector) => {
    await page.getByLabel("Ask a clinical question").fill(question);
    await page.getByRole("button", { name: "Send" }).click();
    await expect(page.locator(selector)).toBeVisible();
  };

  await ask("Please clarify the admission", ".response-clarification");
  await expect(page.locator(".response-clarification")).toContainText("Clarification needed");

  await page.getByRole("button", { name: "New session" }).click();
  await ask("Show the FDA label for metformin", ".response-fda");
  await expect(page.locator(".response-fda")).toContainText(
    "FDA label excerpts; formatting normalized for readability. Not personalized medical advice.",
  );
  await expect(page.getByRole("link", { name: "Open FDA label" })).toHaveAttribute("href", /labels\.fda\.gov/);

  await page.getByRole("button", { name: "New session" }).click();
  await ask("Give an unsupported answer", ".response-refusal");
  await expect(page.getByText("Potential source-support gap", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "New session" }).click();
  await ask("Trigger a server error", ".response-error");
  await expect(page.getByRole("alert")).toContainText("We could not complete this request");
  await expect(page.getByRole("alert")).not.toContainText(/Traceback|RuntimeError|Exception:/);
});

test("network failure and long answers with missing metadata remain contained", async ({ page }) => {
  await page.route("**/ask", (route) => route.abort("failed"));
  await page.goto("/");
  await page.getByLabel("Ask a clinical question").fill("What is documented?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByRole("alert")).toContainText("Unable to reach the local service");

  await page.unroute("**/ask");
  const longAnswer = "The record contains a detailed discharge narrative for clinical review. ".repeat(90);
  await page.route("**/ask", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({ answer: longAnswer }),
  }));
  await page.getByRole("button", { name: "New session" }).click();
  await page.getByLabel("Ask a clinical question").fill("Return a long answer");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.locator(".response-answer")).toContainText("detailed discharge narrative");
  await expect(page.locator(".response-meta")).toHaveCount(0);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});

test("full loads and repeated resets create isolated client thread IDs", async ({ page }) => {
  const requestBodies = [];
  page.on("request", (request) => {
    if (new URL(request.url()).pathname === "/ask") requestBodies.push(request.postDataJSON());
  });

  const ask = async () => {
    await page.getByLabel("Ask a clinical question").fill("What is documented at discharge?");
    await page.getByRole("button", { name: "Send" }).click();
    await expect(page.locator(".assistant-message")).toHaveCount(1);
  };

  await page.goto("/");
  await ask();
  await page.getByRole("button", { name: "New session" }).click();
  await expect(page.locator(".message")).toHaveCount(0);
  await ask();
  await page.reload();
  await ask();

  const threadIds = requestBodies.map((body) => body.thread_id);
  expect(threadIds).toHaveLength(3);
  expect(new Set(threadIds).size).toBe(3);
  for (const threadId of threadIds) {
    expect(threadId).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  }
});

for (const viewport of visualViewports) {
  test(`visual baseline ${viewport.name}`, async ({ page }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.goto("/");
    const evidenceCanvas = page.locator("#evidence-canvas");
    await expect(evidenceCanvas).toHaveAttribute(
      "data-animation-state",
      /^(?:paused|static)$/,
      { timeout: 2500 },
    );
    await expect(page).toHaveScreenshot(`workspace-${viewport.name}.png`, {
      animations: "disabled",
      fullPage: true,
    });
  });
}
