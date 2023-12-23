import * as d3 from "d3";
import { wasm_memory, LCNonlinear, LCFeatures } from "algs";

const InputNumPoints = <HTMLInputElement>document.getElementById("num-points");
const InputNumRuns = <HTMLInputElement>document.getElementById("num-runs");
const SelectFeatures = <HTMLSelectElement>(
  document.getElementById("choose-features")
);
const ButtonRun = <HTMLButtonElement>document.getElementById("run-bias-sim");

const nMin = parseInt(InputNumPoints.min);
const nMax = parseInt(InputNumPoints.max);
const runsMin = parseInt(InputNumRuns.min);
const runsMax = parseInt(InputNumRuns.max);

const width = 600;
const height = 600;
const marginTop = 20;
const marginRight = 20;
const marginBottom = 30;
const marginLeft = 40;

interface Point {
  x: number;
  y: number;
  i: number;
}

// Set axis to [-1, 1] x [-1, 1]
const x = d3
  .scaleLinear()
  .domain([-1, 1])
  .range([marginLeft, width - marginRight]);

const y = d3
  .scaleLinear()
  .domain([-1, 1])
  .range([height - marginBottom, marginTop]);

const svg = d3
  .select("#nonlinearPlot")
  .attr("width", width)
  .attr("height", height);

// Mask out plot overflowing into axis
svg
  .append("defs")
  .append("SVG:clipPath")
  .attr("id", "clip")
  .append("SVG:rect")
  .attr("x", marginLeft)
  .attr("y", marginTop)
  .attr("width", width - marginLeft - marginRight)
  .attr("height", height - marginBottom - marginTop);

const contourSample: Point[] = [];
let i = 0;
d3.range(-1.5, 1.6, 0.1).forEach((y) => {
  d3.range(-1.5, 1.6, 0.1).forEach((x) => {
    contourSample.push({ x, y, i: i });
    i += 1;
  });
});

// Compute the density contours given [-1.5, 1.5] x [-1.5, 1.5] sample points
// of f, for every (x, y) with gap 0.1 each.
// There is probably a better method but this is fast enough.
function draw_contours(f: Float64Array, color: string, opacity: number) {
  let contours = d3
    .contourDensity<Point>()
    .x((d) => x(d.x))
    .y((d) => y(d.y))
    .size([width, height])
    .bandwidth(30)
    .weight((d) => f[d.i])
    .thresholds([0])(contourSample);

  svg
    .append("g")
    .attr("fill", "none")
    .attr("stroke", color)
    .attr("stroke-linejoin", "round")
    .attr("stroke-opacity", opacity)
    .selectAll()
    .data(contours)
    .join("path")
    .attr("clip-path", "url(#clip)")
    .attr("stroke-width", 2)
    .attr("d", d3.geoPath());
}

function plot_lc_bias(n: number, runs: number, features: LCFeatures) {
  // Clear previous runs
  svg.selectAll("path").remove();

  // Draw x and y axis
  svg
    .append("g")
    .attr("transform", `translate(0, ${height - marginBottom})`)
    .call(d3.axisBottom(x));
  svg
    .append("g")
    .attr("transform", `translate(${marginLeft}, 0)`)
    .call(d3.axisLeft(y));

  let runner = LCNonlinear.new();
  runner.set_features(features);

  for (let i = 0; i < runs; i += 1) {
    setTimeout(() => {
      runner.get_prediction(n);
      let g = new Float64Array(wasm_memory().buffer, runner.get_g(), 961);
      draw_contours(g, "grey", 0.1);
    }, 0);
  }

  setTimeout(() => {
    let f = new Float64Array(wasm_memory().buffer, runner.get_f(), 961);
    draw_contours(f, "black", 1.0);
    runner.free();
  }, 0);
}

function get_features() {
  let features_index = SelectFeatures.selectedIndex;
  if (features_index == 0) {
    return LCFeatures.Linear;
  } else if (features_index == 1) {
    return LCFeatures.Quadratic;
  }
  console.log("Missing LCFeature, Defaulting to Linear");
  return LCFeatures.Linear;
}

ButtonRun.onclick = () => {
  let n = InputNumPoints.valueAsNumber;
  let runs = InputNumRuns.valueAsNumber;
  if (isNaN(n) || n < nMin || n > nMax) {
    alert(`Size of sample must be an integer from ${nMin} to ${nMax}`);
  } else if (isNaN(runs) || runs < runsMin || runs > runsMax) {
    alert(`Number of runs must be an integer from ${runsMin} to ${runsMax}`);
  } else {
    plot_lc_bias(n, runs, get_features());
  }
};

plot_lc_bias(
  InputNumPoints.valueAsNumber,
  InputNumRuns.valueAsNumber,
  get_features()
);
