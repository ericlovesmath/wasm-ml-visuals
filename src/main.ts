import * as d3 from "d3";
import "./style.css";
import Chart from "chart.js/auto";
import {
  wasm_memory,
  LCInSampleError,
  LCBiasVariance,
  LCNonlinear,
} from "algs";

const InputNumPoints = <HTMLInputElement>document.getElementById("num-points");
const InputNumRuns = <HTMLInputElement>document.getElementById("num-runs");
const ButtonRunBiasSim = <HTMLButtonElement>(
  document.getElementById("run-bias-sim")
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

let biasSimChart = new Chart("bias-sim-canvas", {
  type: "scatter",
  data: {
    datasets: [
      {
        type: "line",
        borderColor: "Black",
        label: "Hypothesis",
        data: [] as any[],
      },
    ],
  },
  options: {
    devicePixelRatio: 2,
    scales: {
      x: { min: -1, max: 1 },
      y: { min: -1, max: 1 },
    },
    plugins: {
      legend: { labels: { filter: (item) => item.text !== undefined } },
    },
  },
});

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

function plot_lc_bias(n: number, runs: number) {
  let runner = LCBiasVariance.new();

  // Clear previous run
  biasSimChart.data.datasets.length = 1;
  biasSimChart.data.datasets[0].data = [
    { x: -2, y: runner.f_neg },
    { x: 2, y: runner.f_pos },
  ];

  for (let i = 0; i < runs; i += 1) {
    runner.run(n);
    biasSimChart.data.datasets.push({
      type: "line",
      borderColor: "rgba(100, 100, 100, 0.1)",
      data: [
        { x: -2, y: runner.g_neg },
        { x: 2, y: runner.g_pos },
      ],
    });
  }

  biasSimChart.update("none");
  runner.free();
}

function plot_lc_nonlinear(n: number, runs: number) {
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
    .attr("width", width - marginLeft - marginRight)
    .attr("height", height - marginBottom - marginTop)
    .attr("x", marginLeft)
    .attr("y", marginTop);

  //////////////////////////////////

  interface DataPoint {
    x: number;
    y: number;
  }

  const line = d3
    .line<DataPoint>()
    .x((d) => x(d.x))
    .y((d) => y(d.y));

  let runner = LCNonlinear.new();

  let f = new Float64Array(wasm_memory().buffer, runner.set_features(0), 201);
  const f_data = [...Array(201).keys()].map((_, n) => ({
    x: (n - 100) * 0.01,
    y: f[n],
  }));

  for (let i = 0; i < runs; i += 1) {
    let g = new Float64Array(
      wasm_memory().buffer,
      runner.get_prediction(n),
      201
    );
    let data = [...Array(201).keys()].map((_, n) => ({
      x: (n - 100) * 0.01,
      y: g[n],
    }));
    svg
      .append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", `steelblue`)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("stroke-width", 2)
      .attr("stroke-opacity", 0.1)
      .attr("clip-path", "url(#clip)")
      .attr("d", line);
  }

  svg
    .append("path")
    .datum(f_data)
    .attr("fill", "none")
    .attr("stroke", `black`)
    .attr("stroke-linejoin", "round")
    .attr("stroke-linecap", "round")
    .attr("stroke-width", 2)
    .attr("clip-path", "url(#clip)")
    .attr("d", line);

  runner.free();
}

plot_lc_in_sample_error();
plot_lc_bias(InputNumPoints.valueAsNumber, InputNumRuns.valueAsNumber);
plot_lc_nonlinear(200, 200);

ButtonRunBiasSim.onclick = () => {
  let n = InputNumPoints.valueAsNumber;
  let runs = InputNumRuns.valueAsNumber;
  if (isNaN(n) || n < nMin || n > nMax) {
    alert(`Size of sample must be an integer from ${nMin} to ${nMax}`);
  } else if (isNaN(runs) || runs < runsMin || runs > runsMax) {
    alert(`Number of runs must be an integer from ${runsMin} to ${runsMax}`);
  } else {
    plot_lc_bias(n, runs);
  }
};
