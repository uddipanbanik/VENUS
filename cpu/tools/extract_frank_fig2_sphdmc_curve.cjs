#!/usr/bin/env node
// Extract red stroked curve paths from a selected PDF page through PDF.js.
// Inputs are a PDF filename, one-based page number and optional --full flag.
// Output JSON contains page-coordinate extents and optional vector path samples.

// Extract candidate red vector paths from Frank & Tokayer Figure 2.
// This operates on the PDF drawing operators, avoiding raster/color-overlap
// ambiguity in the crowded central-density panel.

const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

// Keep the entry point in CommonJS format for direct execution across Node
// installations while dynamically importing PDF.js's ESM implementation.
async function main() {
  // Validate the command line before resolving optional dependencies so an
  // incomplete invocation always produces the actionable usage contract.
  const pdfPath = process.argv[2];
  const pageNumber = Number(process.argv[3] || 8);
  if (!pdfPath) {
    throw new Error(
      "usage: extract_frank_fig2_sphdmc_curve.cjs PAPER.pdf [PAGE] [--full]\n"
        + "Set PDFJS_DIST_MODULE to override the pdfjs-dist module path.",
    );
  }

  // Resolve PDF.js from the caller's environment so the extractor can run on
  // clusters without embedding a private copy of the PDF parsing library.
  const configuredPdfjs = process.env.PDFJS_DIST_MODULE;
  const pdfjsSpecifier = configuredPdfjs
    ? (
      path.isAbsolute(configuredPdfjs) || configuredPdfjs.startsWith(".")
        ? pathToFileURL(path.resolve(configuredPdfjs)).href
        : configuredPdfjs
    )
    : "pdfjs-dist/legacy/build/pdf.mjs";
  let pdfjs;
  try {
    pdfjs = await import(pdfjsSpecifier);
  } catch (error) {
    throw new Error(
      "Cannot load pdfjs-dist. Install it in the Node environment or set "
        + "PDFJS_DIST_MODULE to its legacy/build/pdf.mjs module.",
      { cause: error },
    );
  }
  const { getDocument, OPS } = pdfjs;

  // Load one paper page and request its vector drawing operations. Reading
  // vectors preserves plotted-curve coordinates and avoids raster noise.
  const bytes = new Uint8Array(fs.readFileSync(pdfPath));
  const document = await getDocument({ data: bytes, disableWorker: true }).promise;
  const page = await document.getPage(pageNumber);
  const operatorList = await page.getOperatorList();

  // Track the PDF graphics state while paths are reconstructed in page
  // coordinates. Stroke colour identifies the red SphDMC comparison curve.
  let stroke = [0, 0, 0];
  let fill = [0, 0, 0];
  let lineWidth = 1;
  const candidates = [];
  let transform = [1, 0, 0, 1, 0, 0];
  const transformStack = [];

  // PDF colour operators may use unit channels, byte channels, or hexadecimal
  // strings; normalize all representations before testing for a red stroke.
  function channel(value) {
    return value > 1 ? value / 255 : value;
  }

  function normalizedColor(args) {
    const flat = Array.from(args).flat();
    if (typeof flat[0] === "string" && /^#[0-9a-f]{6}$/i.test(flat[0])) {
      return [1, 3, 5].map((offset) => parseInt(flat[0].slice(offset, offset + 2), 16) / 255);
    }
    const values = flat.slice(0, 3).map(channel);
    return values.length === 3 ? values : [0, 0, 0];
  }

  function isRed(color) {
    return color[0] > 0.65 && color[0] > 1.8 * color[1]
      && color[0] > 1.8 * color[2];
  }

  // Compose affine coordinate transforms in PDF graphics-state order, then
  // apply the current transform to every path point.
  function multiply(left, right) {
    // An affine transform stores a two-by-two linear map plus a page-coordinate translation.
    const [a, b, c, d, e, f] = left;
    const [g, h, i, j, k, l] = right;
    return [
      a * g + c * h,
      b * g + d * h,
      a * i + c * j,
      b * i + d * j,
      a * k + c * l + e,
      b * k + d * l + f,
    ];
  }

  function transformPoint(point) {
    const [a, b, c, d, e, f] = transform;
    return [a * point[0] + c * point[1] + e, b * point[0] + d * point[1] + f];
  }

  // Decode move, line, cubic, and close-path operations. Cubic control points
  // are skipped because only plotted endpoints are exported as curve samples.
  function parsePath(args) {
    const data = Array.from(args[1][0]);
    const subpaths = [];
    let current = [];
    let start = null;
    for (let index = 0; index < data.length;) {
      const operation = data[index++];
      if (operation === 0) {
        if (current.length) subpaths.push(current);
        current = [];
        const point = transformPoint([data[index++], data[index++]]);
        current.push(point);
        start = point;
      } else if (operation === 1) {
        current.push(transformPoint([data[index++], data[index++]]));
      } else if (operation === 2) {
        index += 4;
        current.push(transformPoint([data[index++], data[index++]]));
      } else if (operation === 3) {
        index += 2;
        current.push(transformPoint([data[index++], data[index++]]));
      } else if (operation === 4 && start) {
        current.push(start);
      } else {
        throw new Error(`unrecognized DrawOPS value ${operation}`);
      }
    }
    if (current.length) subpaths.push(current);
    return subpaths;
  }

  // Replay the page operators while maintaining colour, line width, and
  // transform stacks. Retain red stroked paths long enough to represent a curve.
  for (let index = 0; index < operatorList.fnArray.length; ++index) {
    const operation = operatorList.fnArray[index];
    const args = operatorList.argsArray[index];
    if (operation === OPS.setStrokeRGBColor) {
      stroke = normalizedColor(args);
    } else if (operation === OPS.setStrokeGray) {
      const gray = channel(Number(args[0]));
      stroke = [gray, gray, gray];
    } else if (operation === OPS.setStrokeCMYKColor) {
      const [c, m, y, k] = Array.from(args).flat().slice(0, 4).map(channel);
      stroke = [1 - Math.min(1, c + k), 1 - Math.min(1, m + k), 1 - Math.min(1, y + k)];
    } else if (operation === OPS.setFillRGBColor) {
      fill = normalizedColor(args);
    } else if (operation === OPS.setLineWidth) {
      lineWidth = Number(args[0]);
    } else if (operation === OPS.save) {
      transformStack.push(transform.slice());
    } else if (operation === OPS.restore) {
      transform = transformStack.pop() || [1, 0, 0, 1, 0, 0];
    } else if (operation === OPS.transform) {
      transform = multiply(transform, Array.from(args));
    } else if (operation === OPS.constructPath) {
      // A constructed path counts only when its paint operator strokes it with the selected red color.
      const paintOperation = args[0];
      if (
        isRed(stroke)
        && [OPS.stroke, OPS.closeStroke, OPS.fillStroke, OPS.eoFillStroke].includes(paintOperation)
      ) {
        for (const points of parsePath(args)) {
          if (points.length < 3) continue;
          const xs = points.map((point) => point[0]);
          const ys = points.map((point) => point[1]);
          candidates.push({
            operatorIndex: index,
            stroke,
            fill,
            lineWidth,
            pointCount: points.length,
            xmin: Math.min(...xs),
            xmax: Math.max(...xs),
            ymin: Math.min(...ys),
            ymax: Math.max(...ys),
            points,
          });
        }
      }
    }
  }

  // The compact default is an audit inventory. --full additionally exports all
  // reconstructed points for later axis calibration and curve digitization.
  console.log(JSON.stringify({
    pageNumber,
    pageView: page.view,
    operatorCount: operatorList.fnArray.length,
    redStrokeCandidates: process.argv.includes("--full") ? candidates : candidates.map((candidate) => ({
      operatorIndex: candidate.operatorIndex,
      lineWidth: candidate.lineWidth,
      pointCount: candidate.pointCount,
      xmin: candidate.xmin,
      xmax: candidate.xmax,
      ymin: candidate.ymin,
      ymax: candidate.ymax,
    })),
  }, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? (error.stack || error.message) : String(error));
  process.exitCode = 1;
});
