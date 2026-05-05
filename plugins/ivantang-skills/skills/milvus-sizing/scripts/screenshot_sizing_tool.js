#!/usr/bin/env node
/**
 * Opens https://milvus.io/tools/sizing, fills in the given parameters,
 * waits for results, and saves a screenshot.
 *
 * Usage:
 *   node screenshot_sizing_tool.js --vectors 1000000 --dim 1536 [--out /tmp/milvus_sizing.png]
 *
 * Requires: playwright (CLI/Node)  +  Chromium installed at /opt/pw-browsers/chromium-1194
 */

const { chromium } = require("playwright");
const path = require("path");

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, def) => {
    const i = args.indexOf(flag);
    return i !== -1 ? args[i + 1] : def;
  };
  return {
    vectors: parseInt(get("--vectors", "1000000"), 10),
    dim: parseInt(get("--dim", "1536"), 10),
    out: get("--out", "/tmp/milvus_sizing.png"),
  };
}

async function delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/** Clear an <input> and type a new value, triggering React's synthetic events. */
async function fillInput(page, selector, value) {
  const el = await page.waitForSelector(selector, { timeout: 15000 });
  await el.click({ clickCount: 3 });
  await el.fill(String(value));
  await el.press("Tab");
  await delay(300);
}

async function main() {
  const { vectors, dim, out } = parseArgs();
  console.log(`\nMilvus Sizing Tool — fetching results for:`);
  console.log(`  vectors : ${vectors.toLocaleString()}`);
  console.log(`  dim     : ${dim}`);
  console.log(`  index   : HNSW (default)`);
  console.log(`  mode    : Distributed (default)\n`);

  const browser = await chromium.launch({
    executablePath: "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  const page = await browser.newPage();
  await page.setViewportSize({ width: 1440, height: 900 });

  console.log("Navigating to https://milvus.io/tools/sizing …");
  await page.goto("https://milvus.io/tools/sizing", {
    waitUntil: "networkidle",
    timeout: 60000,
  });
  await delay(2000);

  // ── Dismiss any cookie / popup banners ───────────────────────────────────
  for (const sel of [
    'button:has-text("Accept")',
    'button:has-text("Got it")',
    'button:has-text("OK")',
    '[aria-label="Close"]',
  ]) {
    const btn = await page.$(sel);
    if (btn) {
      await btn.click().catch(() => {});
      await delay(500);
    }
  }

  // ── Scroll the tool into view ─────────────────────────────────────────────
  await page.evaluate(() => window.scrollTo(0, 0));
  await delay(500);

  // ── Dump all input fields to understand the page structure ────────────────
  const inputs = await page.$$eval("input", (els) =>
    els.map((el) => ({
      id: el.id,
      name: el.name,
      type: el.type,
      placeholder: el.placeholder,
      value: el.value,
      ariaLabel: el.getAttribute("aria-label"),
    }))
  );
  console.log("Inputs found on page:");
  inputs.forEach((inp) => console.log(" ", JSON.stringify(inp)));

  // ── Dump visible select / radio labels ───────────────────────────────────
  const labels = await page.$$eval("label", (els) =>
    els.map((el) => el.textContent.trim()).filter(Boolean)
  );
  console.log("\nLabels found:", labels.slice(0, 30));

  // ── Set vector count ──────────────────────────────────────────────────────
  // Try common selector patterns used by React/Next.js numeric inputs
  const vectorSelectors = [
    'input[id*="vector" i]',
    'input[name*="vector" i]',
    'input[placeholder*="vector" i]',
    'input[aria-label*="vector" i]',
  ];
  let vectorSet = false;
  for (const sel of vectorSelectors) {
    const el = await page.$(sel);
    if (el) {
      console.log(`\nSetting vector count via: ${sel}`);
      await fillInput(page, sel, vectors);
      vectorSet = true;
      break;
    }
  }
  if (!vectorSet) {
    // Fall back: first numeric input
    const numInputs = await page.$$('input[type="number"]');
    if (numInputs.length > 0) {
      console.log("\nSetting vector count via first number input (fallback)");
      await numInputs[0].click({ clickCount: 3 });
      await numInputs[0].fill(String(vectors));
      await numInputs[0].press("Tab");
      await delay(300);
    }
  }

  // ── Set dimension ─────────────────────────────────────────────────────────
  const dimSelectors = [
    'input[id*="dim" i]',
    'input[name*="dim" i]',
    'input[placeholder*="dim" i]',
    'input[aria-label*="dim" i]',
  ];
  let dimSet = false;
  for (const sel of dimSelectors) {
    const el = await page.$(sel);
    if (el) {
      console.log(`Setting dimension via: ${sel}`);
      await fillInput(page, sel, dim);
      dimSet = true;
      break;
    }
  }
  if (!dimSet) {
    const numInputs = await page.$$('input[type="number"]');
    if (numInputs.length > 1) {
      console.log("Setting dimension via second number input (fallback)");
      await numInputs[1].click({ clickCount: 3 });
      await numInputs[1].fill(String(dim));
      await numInputs[1].press("Tab");
      await delay(300);
    }
  }

  // ── Ensure Distributed mode is selected ──────────────────────────────────
  // Look for a "Distributed" button/tab/radio
  for (const sel of [
    'button:has-text("Distributed")',
    '[role="radio"]:has-text("Distributed")',
    'label:has-text("Distributed")',
    'input[value*="distributed" i]',
  ]) {
    const el = await page.$(sel);
    if (el) {
      console.log(`Clicking Distributed mode via: ${sel}`);
      await el.click();
      await delay(500);
      break;
    }
  }

  // ── Wait for results to render ────────────────────────────────────────────
  await delay(3000);

  // ── Scroll to show the full tool area ────────────────────────────────────
  await page.evaluate(() => {
    const el =
      document.querySelector('[class*="sizing"]') ||
      document.querySelector('[class*="result"]') ||
      document.querySelector("main");
    if (el) el.scrollIntoView({ behavior: "instant", block: "start" });
  });
  await delay(500);

  // ── Scrape result numbers for comparison ─────────────────────────────────
  const bodyText = await page.evaluate(() => document.body.innerText);
  console.log("\n── Page text (first 3000 chars) ──");
  console.log(bodyText.substring(0, 3000));

  // ── Screenshot ───────────────────────────────────────────────────────────
  await page.screenshot({ path: out, fullPage: false });
  console.log(`\nScreenshot saved: ${out}`);

  await browser.close();
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
