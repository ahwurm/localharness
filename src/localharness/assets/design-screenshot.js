#!/usr/bin/env node
// Screenshot helper for the frontend-designer subagent.
// Usage: node design-screenshot.js <html-file> <output-prefix> [--viewport w,h]... [--wait ms]
// Captures 1440x900 + 375x812 by default. Waits for network idle, web fonts, and a
// settle delay (default 2000ms) so CSS animations/entrance transitions finish before
// capture — a screenshot taken at load time misses animated-in content entirely.
const path = require('path');
const os = require('os');

function resolvePlaywright() {
  const candidates = [
    'playwright',
    path.join(os.homedir(), '.localharness', 'tools', 'node_modules', 'playwright'),
  ];
  for (const c of candidates) {
    try { return require(c); } catch (_) { /* try next */ }
  }
  console.error('playwright not installed — run: cd ~/.localharness/tools && npm install playwright');
  process.exit(2);
}
const { chromium } = resolvePlaywright();

async function takeScreenshot(htmlFile, outputPath, viewport, waitMs) {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport });
  await page.goto('file://' + path.resolve(htmlFile), { waitUntil: 'networkidle' });
  try { await page.evaluate(() => document.fonts && document.fonts.ready); } catch (_) { /* no Font API */ }
  await page.waitForTimeout(waitMs);
  await page.screenshot({ path: outputPath, fullPage: true });
  await browser.close();
  console.log(`Screenshot saved: ${outputPath} (${viewport.width}x${viewport.height})`);
}

(async () => {
  const args = process.argv.slice(2);
  const htmlFile = args[0];
  const outputPrefix = args[1] && !args[1].startsWith('--') ? args[1] : '/tmp/designs/screenshot';

  if (!htmlFile || htmlFile.startsWith('--')) {
    console.error('Usage: node design-screenshot.js <html-file> <output-prefix> [--viewport w,h]... [--wait ms]');
    process.exit(1);
  }

  const viewports = [];
  let waitMs = 2000;
  for (let i = 2; i < args.length; i++) {
    if (args[i] === '--viewport' && args[i + 1]) {
      const [w, h] = args[i + 1].split(',').map(Number);
      viewports.push({ width: w, height: h });
      i++;
    } else if (args[i] === '--wait' && args[i + 1]) {
      waitMs = Number(args[i + 1]) || waitMs;
      i++;
    }
  }
  if (viewports.length === 0) {
    viewports.push({ width: 1440, height: 900 }, { width: 375, height: 812 });
  }

  for (const vp of viewports) {
    await takeScreenshot(htmlFile, `${outputPrefix}-${vp.width}x${vp.height}.png`, vp, waitMs);
  }
  console.log('All screenshots complete.');
})().catch((e) => { console.error(e); process.exit(1); });
