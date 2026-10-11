// Captures portfolio screenshots of the running SentinelGraph app (2x DPR).
import { chromium } from "playwright-core";
import path from "node:path";

const OUT = process.argv[2];
const WEB = "http://localhost:5173";
const W = 1440, H = 868; // 5:3, fits marketplace gig image ratio (1280x769)

const browser = await chromium.launch({ executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe" });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 2, colorScheme: "dark" });
const page = await ctx.newPage();
const shot = async (name) => {
  await page.mouse.move(W - 4, H - 4); // park the cursor so no hover tooltips leak into shots
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(OUT, name) });
  console.log("saved", name);
};
const scrollTo = async (heading, offset = 20) => {
  await page.evaluate(([h, off]) => {
    const el = [...document.querySelectorAll("h2")].find((e) => e.textContent.trim().startsWith(h));
    el.closest("section").scrollIntoView({ block: "start" });
    window.scrollBy(0, -off);
  }, [heading, offset]);
  await page.waitForTimeout(600);
};
const settle = (ms = 1500) => page.waitForTimeout(ms);

// 1. Overview
await page.goto(WEB + "/", { waitUntil: "networkidle" });
await page.waitForSelector(".hero");
await settle();
await shot("01-overview-dashboard.png");

// 3. Alert queue (before the live ring, so it shows a realistic mix)
await page.goto(WEB + "/alerts?sort=recent", { waitUntil: "networkidle" });
await settle();
await shot("03-alert-queue.png");
await page.request.post(`${WEB}/api/simulate`, { data: { scenario: "card_testing" } });
await page.goto(WEB + "/", { waitUntil: "networkidle" });
await page.waitForSelector(".hero");
await settle();

// 2. Inject a fraud ring live and capture the detection toast + suspected rings
const [resp] = await Promise.all([
  page.waitForResponse((r) => r.url().endsWith("/api/simulate")),
  page.getByRole("button", { name: "Fraud ring" }).click(),
]);
const sim = await resp.json();
await page.waitForSelector(".toast");
await settle(5500); // let the 5s poll refresh the panels
await scrollTo("Alerts per day");
await shot("02-live-attack-detected.png");

// pick the riskiest alert from the injected ring
const alerts = await (await page.request.get(`${WEB}/api/alerts?status=open&sort=risk&limit=200`)).json();
const ringAlert = alerts.items.find((a) => sim.alert_ids.includes(a.id)) ?? alerts.items[0];


// 4-6. Investigation
await page.goto(`${WEB}/alerts/${ringAlert.id}`, { waitUntil: "networkidle" });
await page.waitForSelector(".factor");
await settle(7000); // force graph layout settles + zoom-to-fit
await shot("04-explainable-alert.png");
await scrollTo("Linked entities", 70);
await settle(1000);
await shot("05-linked-entity-graph.png");
// timeline: show the run-up to the alert (fraud confirmations, linked activity, the flag itself)
await page.evaluate(() => {
  const ul = document.querySelector(".timeline");
  const items = [...ul.querySelectorAll("li")];
  const firstFraud = items.findIndex((li) => li.querySelector(".pin.critical"));
  const target = items[Math.max(0, firstFraud - 1)];
  ul.scrollTop += target.getBoundingClientRect().top - ul.getBoundingClientRect().top;
});
await scrollTo("Context", 20);
await shot("06-investigation-timeline.png");

// 7-8. Models
await page.goto(WEB + "/models", { waitUntil: "networkidle" });
await settle(2000);
await shot("07-model-evaluation.png");
await scrollTo("Model comparison", 20);
await shot("08-model-comparison.png");

// 9. API docs
await page.goto("http://localhost:8000/docs", { waitUntil: "networkidle" });
await settle(1500);
await shot("09-fastapi-docs.png");

console.log(JSON.stringify({ ringAlert: ringAlert.id, sim: { tx: sim.transactions, alerts: sim.alerts } }));
await browser.close();
