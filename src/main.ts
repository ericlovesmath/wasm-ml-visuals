import * as d3 from "d3";
import "./style.css";
import { wasm_memory, LCInSampleError, LCNonlinear, LCFeatures } from "algs";

const InputNumPoints = <HTMLInputElement>document.getElementById("num-points");
const InputNumRuns = <HTMLInputElement>document.getElementById("num-runs");
const ButtonRunBiasSim = <HTMLButtonElement>(
  document.getElementById("run-bias-sim")
);
const SelectLCFeatures = <HTMLSelectElement>(
  document.getElementById("choose-features")
);

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

function plot_lc_in_sample_error() {
  // Declare the x (horizontal position) scale.
  const x = d3
    .scaleLinear()
    .domain([100, 1000])
    .range([marginLeft, width - marginRight]);

  // Declare the y (vertical position) scale.
  const y = d3
    .scaleLinear()
    .domain([0, 0.05])
    .range([height - marginBottom, marginTop]);

  // Create the SVG container.
  const svg = d3
    .select("#linePlot")
    .attr("width", width)
    .attr("height", height);

  // Create x and y axis
  svg
    .append("g")
    .attr("transform", `translate(0, ${height - marginBottom})`)
    .call(d3.axisBottom(x));
  svg
    .append("g")
    .attr("transform", `translate(${marginLeft}, 0)`)
    .call(d3.axisLeft(y));

  //////////////////////////////////

  interface DataPoint {
    x: number;
    mean: number;
    std: number;
  }

  const mean = d3
    .line<DataPoint>()
    .x((d) => x(d.x))
    .y((d) => y(d.mean));

  const std = d3
    .area<DataPoint>()
    .x((d) => x(d.x))
    .y0((d) => y(d.mean - d.std)) // Bottom edge of the area
    .y1((d) => y(d.mean + d.std)); // Top edge of the area

  let data: DataPoint[] = [];
  let runner = LCInSampleError.new();
  for (let n = 100; n <= 1000; n += 10) {
    setTimeout(() => {
      runner.run(n);
      data.push({ x: n, mean: runner.mean, std: runner.std });

      svg.selectAll("path").remove();

      svg
        .datum(data)
        .append("path")
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("stroke-width", 2)
        .attr("d", mean);

      svg
        .datum(data)
        .append("path")
        .attr("fill", "lightblue")
        .attr("opacity", 0.5)
        .attr("d", std);
    }, 0);
  }

  //////////////////////////////////

  setTimeout(() => runner.free(), 0);
}

function plot_lc_bias(n: number, runs: number, features: LCFeatures) {
  // Declare the x (horizontal position) scale.
  const x = d3
    .scaleLinear()
    .domain([-1, 1])
    .range([marginLeft, width - marginRight]);

  // Declare the y (vertical position) scale.
  const y = d3
    .scaleLinear()
    .domain([-1, 1])
    .range([height - marginBottom, marginTop]);

  // Create the SVG container.
  const svg = d3
    .select("#nonlinearPlot")
    .attr("width", width)
    .attr("height", height);

  svg.selectAll("path").remove();

  // Create x and y axis
  svg
    .append("g")
    .attr("transform", `translate(0, ${height - marginBottom})`)
    .call(d3.axisBottom(x));
  svg
    .append("g")
    .attr("transform", `translate(${marginLeft}, 0)`)
    .call(d3.axisLeft(y));

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

  //////////////////////////////////

  interface Point {
    x: number;
    y: number;
    i: number;
  }

  const contourSample: Point[] = [];
  let i = 0;
  d3.range(-1.5, 1.6, 0.1).forEach((y) => {
    d3.range(-1.5, 1.6, 0.1).forEach((x) => {
      contourSample.push({ x, y, i: i });
      i += 1;
    });
  });

  // Compute the density contours.
  const draw_contours = (f: Float64Array, color: string, opacity: number) => {
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
  };

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

const get_features = () => {
  let features_index = SelectLCFeatures.selectedIndex;
  if (features_index == 0) {
    return LCFeatures.Linear;
  } else if (features_index == 1) {
    return LCFeatures.Quadratic;
  }
  console.log("Missing LCFeature, Defaulting to Linear");
  return LCFeatures.Linear;
};

plot_lc_bias(
  InputNumPoints.valueAsNumber,
  InputNumRuns.valueAsNumber,
  get_features()
);
plot_lc_in_sample_error();

ButtonRunBiasSim.onclick = () => {
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
